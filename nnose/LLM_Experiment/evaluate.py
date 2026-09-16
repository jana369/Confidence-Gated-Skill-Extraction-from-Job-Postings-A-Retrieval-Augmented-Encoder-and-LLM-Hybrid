"""
evaluate.py

Span-level precision / recall / F1 for BIO-tagged skill extraction.

Policy (agreed): if bio_converter fails to align a predicted span back onto
the token list, that span is NOT silently dropped. It is counted as a false
negative — the gold span existed, the model output couldn't be mapped to it,
so from the pipeline's perspective the skill was missed. This keeps
precision honest instead of quietly inflating it whenever alignment fails.

Evaluation is span-level (exact boundary match on B-SKILL/I-SKILL runs),
not token-level accuracy, since that's what SkillSpan's own eval reports
and what the paper baselines are comparable against.
"""

from dataclasses import dataclass, field
from typing import List, Tuple

Span = Tuple[int, int]  # (start_token_idx, end_token_idx) inclusive


@dataclass
class SentenceResult:
    tokens: List[str]
    gold_bio: List[str]
    pred_bio: List[str]
    gold_spans: List[Span]
    pred_spans: List[Span]
    true_positives: List[Span]
    false_positives: List[Span]
    false_negatives: List[Span]
    unaligned_count: (
        int  # number of spans the LLM proposed but bio_converter couldn't place
    )


@dataclass
class EvalReport:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    total_gold_spans: int
    total_pred_spans: int
    total_unaligned: int
    sentence_results: List[SentenceResult] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=== Evaluation Summary ===",
            f"Gold spans:      {self.total_gold_spans}",
            f"Predicted spans: {self.total_pred_spans}",
            f"  of which unaligned (counted as FP): {self.total_unaligned}",
            f"TP: {self.tp}   FP: {self.fp}   FN: {self.fn}",
            f"Precision: {self.precision:.4f}",
            f"Recall:    {self.recall:.4f}",
            f"F1:        {self.f1:.4f}",
        ]
        return "\n".join(lines)


def bio_to_spans(bio_tags: List[str]) -> List[Span]:
    """Convert a BIO tag sequence into a list of (start, end) inclusive spans.

    Expects tags in {"O", "B-SKILL", "I-SKILL"}. Any I-SKILL not preceded by
    a B-SKILL/I-SKILL is treated as a new span start (defensive against
    malformed sequences from either gold data quirks or conversion bugs).
    """
    spans: List[Span] = []
    start = None
    for i, tag in enumerate(bio_tags):
        if tag == "B-SKILL":
            if start is not None:
                spans.append((start, i - 1))
            start = i
        elif tag == "I-SKILL":
            if start is None:
                start = i  # malformed I- with no preceding B-; recover gracefully
        else:  # "O"
            if start is not None:
                spans.append((start, i - 1))
                start = None
    if start is not None:
        spans.append((start, len(bio_tags) - 1))
    return spans


def score_sentence(
    tokens: List[str],
    gold_bio: List[str],
    pred_bio: List[str],
    unaligned_count=0,
) -> SentenceResult:
    """Score one sentence at the span level.

    unaligned_count: number of LLM-proposed spans that bio_converter could
    not map onto the token list at all (never made it into pred_bio). These
    are added directly to false_negatives below, per the FN policy above —
    they're gold-independent misses in the sense that we don't know which
    gold span (if any) they correspond to, so we report them as a separate
    count AND fold them into FN so recall reflects the true miss rate.
    """
    # unaligned_count may come in as an int (a plain count) or a list
    # (the actual unaligned spans/skills) depending on bio_converter's
    # return type. Normalize to an int here so downstream aggregation
    # never has to care which one it got.
    if isinstance(unaligned_count, (list, tuple, set)):
        unaligned_count = len(unaligned_count)

    gold_spans = bio_to_spans(gold_bio)
    pred_spans = bio_to_spans(pred_bio)

    gold_set = set(gold_spans)
    pred_set = set(pred_spans)

    tp_spans = sorted(gold_set & pred_set)
    fp_spans = sorted(pred_set - gold_set)
    fn_spans = sorted(gold_set - pred_set)

    return SentenceResult(
        tokens=tokens,
        gold_bio=gold_bio,
        pred_bio=pred_bio,
        gold_spans=gold_spans,
        pred_spans=pred_spans,
        true_positives=tp_spans,
        false_positives=fp_spans,
        false_negatives=fn_spans,
        unaligned_count=unaligned_count,
    )


def score_sentence_relaxed(sent_result: "SentenceResult") -> dict:
    """
    Relaxed (overlap-based) scoring for ONE sentence.

    Takes an already-computed SentenceResult (the output of score_sentence())
    and re-judges it with a looser rule:

        STRICT rule (score_sentence):  gold span and predicted span must
            have the EXACT SAME (start, end) to count as a match.

        RELAXED rule (this function): a gold span counts as "found" if ANY
            predicted span shares at least one token position with it.
            We don't care if the boundaries are slightly different --
            only whether the model was "pointing at the same skill."

    Why build this separately instead of changing score_sentence():
        We want to report BOTH numbers side by side in the paper. Strict
        F1 is the honest, defensible number. Relaxed F1 shows how much of
        the gap is boundary/tokenization noise vs. genuine misses. Keeping
        them as two separate functions means neither one can accidentally
        break the other.

    Returns a dict of relaxed counts for this one sentence:
        { "tp": int, "fp": int, "fn": int }
    """
    gold_spans = sent_result.gold_spans
    pred_spans = sent_result.pred_spans

    def overlaps(span_a: Span, span_b: Span) -> bool:
        # Two spans overlap if they share at least one token index.
        # (start_a, end_a) and (start_b, end_b) -- classic interval overlap check.
        return span_a[0] <= span_b[1] and span_b[0] <= span_a[1]

    relaxed_tp = 0
    relaxed_fn = 0

    # For each GOLD span: was it "found" by at least one prediction?
    for g in gold_spans:
        found = any(overlaps(g, p) for p in pred_spans)
        if found:
            relaxed_tp += 1
        else:
            relaxed_fn += 1

    # For each PREDICTED span: did it overlap NO gold span at all?
    # If so, it's a wrong guess with zero basis -- a false positive.
    relaxed_fp = 0
    for p in pred_spans:
        matched_something = any(overlaps(p, g) for g in gold_spans)
        if not matched_something:
            relaxed_fp += 1

    return {"tp": relaxed_tp, "fp": relaxed_fp, "fn": relaxed_fn}


def aggregate(results: List[SentenceResult]) -> EvalReport:
    tp = sum(len(r.true_positives) for r in results)
    fp = sum(len(r.false_positives) for r in results)
    fn = sum(len(r.false_negatives) for r in results)
    unaligned = sum(r.unaligned_count for r in results)

    # Unaligned LLM spans: the model proposed a skill span we couldn't place
    # on the token list. These are spans the model asserted exist -- if they
    # can't be verified against gold, that's a false positive (the model
    # claimed a skill span that isn't a confirmed hit), not a false negative
    # (which means "gold had something we missed"). Counting them as FN was
    # wrong -- it let wrong/unmatchable predictions hide as if they were
    # simply missed gold spans, which silently protected precision instead
    # of penalizing the bad prediction where it belongs.
    fp += unaligned

    total_gold = sum(len(r.gold_spans) for r in results)
    total_pred = sum(len(r.pred_spans) for r in results)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return EvalReport(
        precision=precision,
        recall=recall,
        f1=f1,
        tp=tp,
        fp=fp,
        fn=fn,
        total_gold_spans=total_gold,
        total_pred_spans=total_pred,
        total_unaligned=unaligned,
        sentence_results=results,
    )


def error_breakdown(report: EvalReport, tokens_key=lambda r: r.tokens) -> dict:
    """Lightweight error analysis: buckets FN into 'missed entirely' vs
    'boundary mismatch' (gold and pred overlap but don't match exactly),
    since these indicate different LLM weaknesses (recall vs. precision
    of span boundaries).
    """
    missed_entirely = 0
    boundary_mismatch = 0

    for r in report.sentence_results:
        pred_token_positions = set()
        for s, e in r.pred_spans:
            pred_token_positions.update(range(s, e + 1))

        for fn_span in r.false_negatives:
            s, e = fn_span
            overlap = any(i in pred_token_positions for i in range(s, e + 1))
            if overlap:
                boundary_mismatch += 1
            else:
                missed_entirely += 1

    return {
        "missed_entirely": missed_entirely,
        "boundary_mismatch": boundary_mismatch,
        "unaligned_llm_spans": report.total_unaligned,
    }
