"""
run_skillspan.py

Driver: load SkillSpan -> call the LLM -> convert spans to BIO -> score ->
report. Raw LLM responses are cached to outputs/cache/ so a bug in
bio_converter never costs another round of API calls.

ASSUMPTIONS ABOUT bio_converter.py'S INTERFACE (adjust the two calls below
if your real signatures differ):
    spans_to_bio(skills: list[str], tokens: list[str]) -> tuple[list[str], int]
        Returns (bio_tags, unaligned_count) where unaligned_count is how
        many entries in `skills` could not be matched onto `tokens`.
    gold_to_bio(tags_skill: list[str]) -> list[str]
        Standardizes SkillSpan's raw "B"/"I"/"O" gold labels into
        "B-SKILL"/"I-SKILL"/"O".
"""

import json
import os
import time
from datetime import datetime
from typing import List

from llm_client import extract_skills, tokens_to_sentence, api_key
from bio_converter import spans_to_bio, gold_to_bio
from evaluate import score_sentence, aggregate, error_breakdown, score_sentence_relaxed
from extract_skill_logprobs import extract_skill_logprobs, assign_token_level_logprobs

CACHE_DIR = "outputs/cache"
RESULTS_DIR = "outputs"
os.makedirs(CACHE_DIR, exist_ok=True)

# --- Rate limiting ---
# NVIDIA NIM (and similarly Groq free tier) cap requests per minute.
# MIN_SECONDS_BETWEEN_CALLS enforces a floor delay between calls so we
# never exceed that, rather than firing calls in a tight loop and hoping.
# 40 rpm -> 60/40 = 1.5s/call floor; padded slightly for safety margin.
RPM_LIMIT = 40
MIN_SECONDS_BETWEEN_CALLS = 60.0 / RPM_LIMIT * 1.1  # ~1.65s

# --- Retry on transient errors (rate limit, CDN/edge errors, timeouts) ---
MAX_RETRIES = 4
RETRY_BASE_DELAY = 5.0  # seconds; doubles each retry (exponential backoff)


def call_with_rate_limit_and_retry(
    sentence: str, api_key: str, last_call_time: List[float]
):
    """Wraps extract_skills with (a) a floor delay since the previous call
    and (b) exponential-backoff retry on transient failures (rate limits,
    CDN edge errors like Akamai/Edgesuite, timeouts).

    last_call_time is a 1-element list used as a mutable float holder so
    the caller's timestamp persists across calls without needing a class.
    """
    elapsed = time.time() - last_call_time[0]
    if elapsed < MIN_SECONDS_BETWEEN_CALLS:
        time.sleep(MIN_SECONDS_BETWEEN_CALLS - elapsed)

    delay = RETRY_BASE_DELAY
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = extract_skills(sentence, api_key)
            last_call_time[0] = time.time()
            return result
        except Exception as e:
            last_exc = e
            msg = str(e).lower()
            # Retry on things that look transient: rate limits, edge/CDN
            # errors, timeouts, connection resets. Don't retry on things
            # like bad auth (that will never succeed on retry).
            transient = any(
                kw in msg
                for kw in [
                    "rate limit",
                    "429",
                    "timeout",
                    "edgesuite",
                    "connection",
                    "502",
                    "503",
                    "504",
                ]
            )
            last_call_time[0] = time.time()
            if not transient or attempt == MAX_RETRIES:
                raise
            print(
                f"  transient error (attempt {attempt}/{MAX_RETRIES}), retrying in {delay:.0f}s: {e}"
            )
            time.sleep(delay)
            delay *= 2

    raise last_exc


os.makedirs(RESULTS_DIR, exist_ok=True)


def build_full_token_row(
    tokens: list,
    pred_bio: list,
    matched_spans: list,
    skills: list,
    logprobs_content,
) -> list:
    """
    Builds one row per DATASET TOKEN (including O tokens), showing what
    the LLM predicted and its confidence -- gives full visibility into
    where the LLM is right/wrong, and feeds Component C's per-token
    feature table directly.

    Returns a list of dicts, one per token position:
        {"position": int, "token": str, "llm_label": "B-SKILL"/"I-SKILL"/"O",
            "logprob": float or None}
    """
    # Get per-skill confidence, then expand it down to individual tokens.
    if logprobs_content:
        skill_confidences = extract_skill_logprobs(skills, logprobs_content)
        token_logprob_map = assign_token_level_logprobs(
            skill_confidences, matched_spans, logprobs_content
        )
    else:
        token_logprob_map = {}

    rows = []
    for position, (token, label) in enumerate(zip(tokens, pred_bio)):
        rows.append(
            {
                "position": position,
                "token": token,
                "llm_label": label,
                "logprob": token_logprob_map.get(position),  # None if not covered
            }
        )
    return rows


def is_degenerate_sentence(tokens: List[str]) -> bool:
    """
    Detects sentences with no real extractable content -- placeholder-only
    tokens (e.g. '<DESCRIPTION>', '<ORGANIZATION>') or pure punctuation/
    symbol tokens (e.g. '*') with nothing else.

    Why this matters: an LLM given only '<DESCRIPTION>' or '*' has no real
    text to extract skills from. If it invents skills anyway (a known
    failure mode -- see Groq's behavior), that's a fair thing to note
    separately, but it shouldn't be silently mixed into the main
    extraction-quality score, since there's no real task being tested.

    A sentence is "degenerate" if EVERY token is either:
        - a placeholder tag like <SOMETHING>, or
        - pure punctuation/symbols (no letters or digits at all)
    """
    import re

    if not tokens:
        return True

    for tok in tokens:
        is_placeholder = bool(re.fullmatch(r"<[A-Z]+>", tok))
        is_pure_symbol = not re.search(r"[A-Za-z0-9]", tok)
        if not (is_placeholder or is_pure_symbol):
            return False  # found at least one token with real content
    return True


def load_dataset(split: str = "test"):
    from datasets import load_dataset as hf_load_dataset

    path_map = {
        "test": "D:/Conference/jobBERTa/nnose/data/skillspan/test.jsonl",
        "dev": "D:/Conference/jobBERTa/nnose/data/skillspan/dev.jsonl",
        "train": "D:/Conference/jobBERTa/nnose/data/skillspan/train.jsonl",
    }
    if split not in path_map:
        raise ValueError(f"Unknown split '{split}', expected one of {list(path_map)}")

    ds = hf_load_dataset(
        "json",
        data_files=path_map[split],
        split="train",  # local single-file loads always land under "train" internally
    )
    return ds


def run(
    split: str = "test", limit: int = None, cache_run_name: str = None, start: int = 0
):
    dataset = load_dataset(split)
    end = min(start + limit, len(dataset)) if limit else len(dataset)
    dataset = dataset.select(range(start, end))

    run_name = cache_run_name or f"{split}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    cache_path = os.path.join(CACHE_DIR, f"{run_name}.jsonl")
    relaxed_tp_total = 0
    relaxed_fp_total = 0
    relaxed_fn_total = 0
    sentence_results = []
    n_invalid_json = 0
    n_failed_calls = 0
    last_call_time = [0.0]  # mutable holder, see call_with_rate_limit_and_retry

    with open(cache_path, "w", encoding="utf-8") as cache_f:
        for local_idx, sample in enumerate(dataset):
            idx = start + local_idx
            tokens: List[str] = sample["tokens"]
            gold_raw = sample["tags_skill"]

            if is_degenerate_sentence(tokens):
                print(
                    f"[{idx}] skipped -- degenerate/placeholder-only sentence: {tokens}"
                )
                continue  # don't call the LLM, don't score this sentence at all

            sentence = tokens_to_sentence(tokens)

            # --- Call the LLM (rate-limited + retried) ---
            try:
                result = call_with_rate_limit_and_retry(
                    sentence, api_key, last_call_time
                )
            except Exception as e:
                n_failed_calls += 1
                print(f"[{idx}] LLM call failed after retries: {e}")
                # Treat as zero predictions: every gold span becomes a
                # false negative naturally, no special-casing needed.
                result = None

            skills = result.skills if result else []
            if result and result.invalid_json:
                n_invalid_json += 1

            # --- Convert to BIO ---
            # spans_to_bio returns (bio_tags, matched_spans) -- matched_spans
            # is the list of SUCCESSFULLY aligned spans, not failures. The
            # unaligned count is whatever the LLM proposed minus what
            # actually got matched -- compute it directly here so we're
            # never relying on a name that could be misread either way.
            pred_bio, matched_spans = spans_to_bio(skills, tokens)
            unaligned_count = len(skills) - len(matched_spans)
            gold_bio = gold_to_bio(gold_raw)
            logprobs_content = result.raw_logprobs if result and hasattr(result, "raw_logprobs") else None
            llm_labels = build_full_token_row(tokens, pred_bio, matched_spans, skills, logprobs_content)
            # --- Score this sentence ---
            sent_result = score_sentence(
                tokens=tokens,
                gold_bio=gold_bio,
                pred_bio=pred_bio,
                unaligned_count=unaligned_count,
            )
            sentence_results.append(sent_result)

            # --- Cache the raw response for replay/debugging ---
            cache_f.write(
                json.dumps(
                    {
                        "idx": idx,
                        "tokens": tokens,
                        "sentence": sentence,
                        "skills_raw": skills,
                        "confidence": result.confidence if result else None,
                        "latency_ms": result.latency_ms if result else None,
                        "invalid_json": result.invalid_json if result else True,
                        "llm_labels": llm_labels if result else None,
                    },
                )
                + "\n"
            )
            relaxed = score_sentence_relaxed(sent_result)
            relaxed_tp_total += relaxed["tp"]
            relaxed_fp_total += relaxed["fp"]
            relaxed_fn_total += relaxed["fn"]
            if idx % 25 == 0 or idx < 10:
                print(f"[{idx}/{len(dataset)}] processed")

            # Periodic checkpoint: write partial results every 200 examples
            # so a crash late in a run doesn't lose everything processed so far.
            if idx > 0 and idx % 200 == 0:
                partial_report = aggregate(sentence_results)
                checkpoint_path = os.path.join(
                    RESULTS_DIR, f"{run_name}_checkpoint.json"
                )
                with open(checkpoint_path, "w", encoding="utf-8") as cf:
                    json.dump(
                        {
                            "examples_processed": idx + 1,
                            "precision": partial_report.precision,
                            "recall": partial_report.recall,
                            "f1": partial_report.f1,
                        },
                        cf,
                        indent=2,
                    )

    report = aggregate(sentence_results)
    breakdown = error_breakdown(report)

    print()
    print(report.summary())
    print()
    print("=== Error Breakdown (FN causes) ===")
    print(
        f"Missed entirely (no overlapping prediction): {breakdown['missed_entirely']}"
    )
    print(
        f"Boundary mismatch (overlap, wrong span):      {breakdown['boundary_mismatch']}"
    )
    print(
        f"Unaligned LLM spans (never mapped to tokens):  {breakdown['unaligned_llm_spans']}"
    )
    print()
    print(f"Invalid JSON responses: {n_invalid_json}")
    print(f"Failed LLM calls:       {n_failed_calls}")
    relaxed_precision = (
        relaxed_tp_total / (relaxed_tp_total + relaxed_fp_total)
        if (relaxed_tp_total + relaxed_fp_total) > 0
        else 0.0
    )
    relaxed_recall = (
        relaxed_tp_total / (relaxed_tp_total + relaxed_fn_total)
        if (relaxed_tp_total + relaxed_fn_total) > 0
        else 0.0
    )
    relaxed_f1 = (
        2 * relaxed_precision * relaxed_recall / (relaxed_precision + relaxed_recall)
        if (relaxed_precision + relaxed_recall) > 0
        else 0.0
    )

    print("\n--- RELAXED (overlap) scoring ---")
    print(f"Precision: {relaxed_precision:.4f}")
    print(f"Recall:    {relaxed_recall:.4f}")
    print(f"F1:        {relaxed_f1:.4f}")
    # --- Persist results ---
    results_path = os.path.join(RESULTS_DIR, f"{run_name}_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_name": run_name,
                "split": split,
                "n_examples": len(dataset),
                "precision": report.precision,
                "recall": report.recall,
                "f1": report.f1,
                "tp": report.tp,
                "fp": report.fp,
                "fn": report.fn,
                "total_gold_spans": report.total_gold_spans,
                "total_pred_spans": report.total_pred_spans,
                "total_unaligned": report.total_unaligned,
                "error_breakdown": breakdown,
                "n_invalid_json": n_invalid_json,
                "n_failed_calls": n_failed_calls,
                "cache_file": cache_path,
            },
            f,
            indent=2,
        )
    print(f"\nResults written to {results_path}")
    print(f"Raw responses cached to {cache_path}")

    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run LLM skill extraction over SkillSpan"
    )
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"])
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Index to start from (for running in batches)",
    )    
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples (for quick/mini-eval runs)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Name for this run (used in cache/results filenames)",
    )
    args = parser.parse_args()

    run(split=args.split, limit=args.limit, cache_run_name=args.name, start=args.start)
