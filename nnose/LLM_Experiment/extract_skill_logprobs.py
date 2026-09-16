"""
extract_skill_logprobs.py

Maps NVIDIA NIM's raw per-output-token logprobs back onto individual
extracted skill strings, WITHOUT changing the JSON-array output format
your pipeline already uses (prompts.py / llm_client.py stay as-is).

This solves the "per-token confidence" need from Component C's gate
without reverting to the direct-BIO-output format we deliberately moved
away from earlier in the project (that format caused malformed output
and wrapper text -- see test_api_NVIDIA.py's own output, which included
an unrequested "Here is the precise..." preamble and explanation).
"""

from dataclasses import dataclass
from typing import List


@dataclass
class SkillWithConfidence:
    skill_text: str  # e.g. "communication skills"
    mean_logprob: float  # average logprob across the tokens spelling this skill
    min_logprob: float  # the WORST (lowest-confidence) token in this skill span
    token_count: int  # how many raw API output-tokens made up this skill's text


def extract_skill_logprobs(
    skills: List[str],
    logprobs_content: list,
) -> List[SkillWithConfidence]:
    """
    Walks through the raw API response's logprobs.content (a flat list of
    every token the model generated, each with its own logprob) and finds
    which stretch of those tokens spells out each skill string in `skills`.

    Args:
        skills: the parsed JSON list of skill strings, e.g.
                ["Python", "communication skills"]
        logprobs_content: response.choices[0].logprobs.content -- the raw
                list of ChatCompletionTokenLogprob objects from the API,
                covering the ENTIRE response (including the JSON brackets,
                quotes, commas -- not just the skill words themselves).

    Returns:
        One SkillWithConfidence per skill in `skills`, in the same order.
        If a skill's text cannot be located in the raw token stream (rare,
        but possible if the model's JSON formatting is unusual), that
        skill is skipped with a printed warning -- consistent with how
        bio_converter.py already handles unmatchable spans elsewhere in
        this pipeline, rather than silently guessing.
    """
    # Step 1: flatten the raw API tokens into one long string, while
    # remembering exactly which (start_char, end_char, logprob) each
    # original API token covers in that combined string. We need this
    # because the API's tokens rarely line up with whole words -- e.g.
    # "communication" might arrive as ONE token, but "JavaScript" might
    # arrive as THREE tokens ("Java", "Sc", "ript"). Character position is
    # the only reliable common ground between "skill text" and "API tokens".
    full_text = ""
    token_spans = []  # list of (start_char, end_char, logprob, token_text)
    for tok in logprobs_content:
        start = len(full_text)
        full_text += tok.token
        end = len(full_text)
        token_spans.append((start, end, tok.logprob, tok.token))

    results = []
    search_from = 0  # only search forward, so repeated skill strings match
    # their correct (later) occurrence, not the first one
    # every time -- mirrors the used_positions logic in
    # bio_converter.py's own span matching

    for skill in skills:
        # Step 2: find where this skill's exact text appears in the
        # combined API output string.
        idx = full_text.find(skill, search_from)
        if idx == -1:
            # Fallback: the API sometimes emits the JSON with escaped
            # quotes or slightly different whitespace than the parsed
            # Python string. Try a case-insensitive search as a fallback,
            # same philosophy as bio_converter.py's fuzzy-matching pass.
            idx = full_text.lower().find(skill.lower(), search_from)
            if idx == -1:
                print(
                    f"  [logprob WARNING] Could not locate skill '{skill}' "
                    f"in the raw API token stream -- skipping confidence "
                    f"for this skill."
                )
                continue

        skill_end = idx + len(skill)

        # Step 3: collect every API token that OVERLAPS this character
        # range. A single skill can span multiple API tokens (e.g.
        # "JavaScript" = "Java" + "Sc" + "ript"), so we gather all of them.
        matching_logprobs = [
            lp
            for (start, end, lp, _tok) in token_spans
            if start < skill_end and end > idx
        ]

        if not matching_logprobs:
            print(
                f"  [logprob WARNING] Found '{skill}' in text but no "
                f"overlapping API tokens -- skipping."
            )
            continue

        mean_lp = sum(matching_logprobs) / len(matching_logprobs)
        min_lp = min(matching_logprobs)

        results.append(
            SkillWithConfidence(
                skill_text=skill,
                mean_logprob=mean_lp,
                min_logprob=min_lp,
                token_count=len(matching_logprobs),
            )
        )

        search_from = skill_end  # advance past this match for next search

    return results


def assign_token_level_logprobs(
    skill_confidences: List[SkillWithConfidence],
    matched_spans: list,
    logprobs_content: list,
) -> dict:
    """
    Assigns a DISTINCT logprob to EACH individual dataset token, rather
    than copying one averaged phrase-level number across every token in
    a multi-word skill.

    FALLBACK POLICY: if a specific word's logprob can't be located (the
    word-count/token-count mismatch case, or a lookup failure), we do NOT
    leave the token blank. Instead we fall back to the SENTENCE-WIDE
    average logprob across the entire raw API response -- not just the
    matched skills within it. This keeps Component C's feature table
    complete (no nulls for a simple classifier to choke on) while staying
    honest: an unmatched skill still reflects "how confident was the
    model in this response overall", which is a real, if imperfect,
    signal -- better than fabricating precision we don't have.

    Example: for the skill "people management" covering dataset token
    positions 3 ("people") and 4 ("management"), this returns TWO
    different numbers -- one reflecting only the API tokens that spelled
    "people", another reflecting only the API tokens that spelled
    "management" -- not the same blended average for both.

    Args:
        skill_confidences: output of extract_skill_logprobs() -- used
            here only to look up WHERE each skill's text sits in the
            full reconstructed API output string.
        matched_spans: output of bio_converter.py's extract_skill_spans()
            -- MatchedSpan objects with .text, .start, .end (dataset
            TOKEN positions, not characters).
        logprobs_content: response.choices[0].logprobs.content -- the
            same raw per-API-token logprob list used in
            extract_skill_logprobs().

    Returns:
        dict mapping dataset token_position (int) -> mean_logprob (float)
        for JUST that word's own API tokens, or the sentence-wide average
        as a fallback when a specific word can't be located. Token
        positions not covered by any matched skill are absent (predicted
        "O" -- no confidence signal to report).
    """
    # Rebuild the same full-text + character-span map used in
    # extract_skill_logprobs(), so we can independently locate each
    # DATASET TOKEN's own words (not the whole skill phrase) inside the
    # raw API output.
    full_text = ""
    token_char_spans = []  # (start_char, end_char, logprob)
    for tok in logprobs_content:
        start = len(full_text)
        full_text += tok.token
        end = len(full_text)
        token_char_spans.append((start, end, tok.logprob))

    # Sentence-wide fallback: the average logprob across EVERY token in
    # the raw API response for this sentence, computed once up front.
    all_logprobs = [lp for (_s, _e, lp) in token_char_spans]
    sentence_avg_logprob = (
        sum(all_logprobs) / len(all_logprobs) if all_logprobs else None
    )


    def logprobs_for_substring(substring: str, search_from: int):
        """Finds `substring` in full_text (starting the search at
        search_from) and returns (mean_logprob_for_just_this_word, end_char)
        or (None, search_from) if it can't be located.

        Tries three passes, in order, before giving up:
        1. Exact match.
        2. Case-insensitive match.
        3. Loosened match -- strips spaces and common punctuation from
            BOTH the search text and a sliding window of full_text, to
            survive minor API formatting quirks (stray quotes, escaping,
            extra whitespace) without falling back to the sentence-wide
            average too early.
        """
        idx = full_text.find(substring, search_from)
        if idx == -1:
            idx = full_text.lower().find(substring.lower(), search_from)
        if idx == -1:
            # Loosened pass: strip spaces/punctuation from the target word,
            # then scan forward through full_text looking for a window whose
            # own stripped form matches.
            import re

            target_stripped = re.sub(r"[\s\"'.,;:()\[\]{}]", "", substring).lower()
            if target_stripped:
                window = len(substring) + 4  # small slack for stripped chars
                for start in range(search_from, max(len(full_text) - 1, search_from)):
                    candidate = full_text[start : start + window]
                    candidate_stripped = re.sub(
                        r"[\s\"'.,;:()\[\]{}]", "", candidate
                    ).lower()
                    if candidate_stripped.startswith(target_stripped):
                        idx = start
                        substring = candidate[: len(substring)]  # approx match length
                        break
        if idx == -1:
            return None, search_from
        end = idx + len(substring)
        overlapping = [lp for (s, e, lp) in token_char_spans if s < end and e > idx]
        if not overlapping:
            return None, end
        return sum(overlapping) / len(overlapping), end
    token_logprob_map = {}

    for span in matched_spans:
        skill_words = span.text.split()
        dataset_positions = list(range(span.start, span.end + 1))

        if len(skill_words) != len(dataset_positions):
            print(
                f"  [token-logprob WARNING] Word count ({len(skill_words)}) "
                f"doesn't match token span length ({len(dataset_positions)}) "
                f"for '{span.text}' -- using sentence-wide average "
                f"confidence for its tokens instead of per-word values."
            )
            mean_lp, _ = logprobs_for_substring(span.text, 0)
            fallback_value = mean_lp if mean_lp is not None else sentence_avg_logprob
            for pos in dataset_positions:
                token_logprob_map[pos] = fallback_value
            continue

        search_from = 0
        for word, position in zip(skill_words, dataset_positions):
            mean_lp, search_from = logprobs_for_substring(word, search_from)
            if mean_lp is None:
                print(
                    f"  [token-logprob WARNING] Could not locate word "
                    f"'{word}' (from skill '{span.text}') in the raw "
                    f"API token stream -- using sentence-wide average "
                    f"confidence ({sentence_avg_logprob}) instead."
                )
                token_logprob_map[position] = sentence_avg_logprob
                continue
            token_logprob_map[position] = mean_lp

    return token_logprob_map
