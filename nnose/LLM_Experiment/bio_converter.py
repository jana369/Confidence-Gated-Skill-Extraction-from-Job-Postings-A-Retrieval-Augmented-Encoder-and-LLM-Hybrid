"""
bio_converter.py
----------------
Converts LLM-extracted skill span strings into BIO tag sequences
aligned to the original SkillSpan token list.

Two-step process:
1. extract_skill_spans(): takes a list of span strings and the original
    token list, and finds each span's start/end token indices.
2. assign_bio(): converts those (start, end) indices into a BIO tag list.

Design decisions:
- Matching is case-insensitive but the original token casing is preserved.
- Punctuation-attached spans are handled by matching on joined variants.
- If the same span string appears multiple times in the sentence, we match
    greedily from left to right (first unmatched occurrence wins).
- If a span cannot be found at all, it's silently skipped (logged as a warning).
    This is preferable to crashing, since LLM outputs are occasionally slightly
    off from the exact tokenization (e.g. "Python3" vs "Python" + "3").
- Overlapping spans are resolved by keeping the first match encountered.
"""

import re
from dataclasses import dataclass
from typing import Optional

# ── Data structures ────────────────────────────────────────────────────────────


@dataclass
class MatchedSpan:
    """A skill span successfully aligned to token indices."""

    text: str  # original span string from LLM
    start: int  # inclusive start token index
    end: int  # inclusive end token index


# ── Main functions ─────────────────────────────────────────────────────────────


def extract_skill_spans(
    skills: list[str],
    tokens: list[str],
) -> list[MatchedSpan]:
    """
    Align extracted skill span strings to their positions in the token list.

    Args:
        skills: List of skill span strings returned by the LLM.
                e.g. ["Python", "communication skills", "CI/CD pipelines"]
        tokens: Original word-level token list from the SkillSpan dataset.
                e.g. ["Experience", "with", "Python", "and", "strong", ...]

    Returns:
        List of MatchedSpan objects for spans that were successfully located.
        Spans that could not be matched are skipped.
    """
    matched = []
    used_positions = set()  # token indices already claimed by a matched span

    for span_text in skills:
        match, overlapped = _find_span_in_tokens(span_text, tokens, used_positions)
        if match is not None:
            matched.append(match)
            for idx in range(match.start, match.end + 1):
                used_positions.add(idx)
            continue
        if overlapped:
            print(
                f"  [bio_converter Info] Span '{span_text}' overlaps an "
                f"already-matched span -- skipped as duplicate/nested."
            )
            continue
        words = span_text.split()
        partial_matches = []
        if len(words) > 1:
            for word in words:
                word_match, _ = _find_span_in_tokens(word, tokens, used_positions)
                if word_match is not None:
                    partial_matches.append(word_match)
                    for idx in range(word_match.start, word_match.end + 1):
                        used_positions.add(idx)

        if partial_matches:
            matched.extend(partial_matches)
            found_words = [m.text for m in partial_matches]
            print(
                f"  [bio_converter WARNING] Could not align full span "
                f"'{span_text}' as one phrase -- recovered {len(partial_matches)}"
                f"/{len(words)} individual word(s) instead: {found_words}"
            )
        else:
            print(
                f"  [bio_converter WARNING] Could not align span '{span_text}' "
                f"in tokens: {tokens}"
            )

    return matched


def assign_bio(
    tokens: list[str],
    matched_spans: list[MatchedSpan],
) -> list[str]:
    """
    Convert a list of MatchedSpan objects into a BIO tag sequence.

    Args:
        tokens:        Original token list.
        matched_spans: Output of extract_skill_spans().

    Returns:
        List of BIO tags, one per token.
        Tags are: "B-SKILL", "I-SKILL", "O"
    """
    tags = ["O"] * len(tokens)

    for span in matched_spans:
        tags[span.start] = "B-SKILL"
        for i in range(span.start + 1, span.end + 1):
            tags[i] = "I-SKILL"

    return tags


def spans_to_bio(
    skills: list[str],
    tokens: list[str],
) -> tuple[list[str], list[MatchedSpan]]:
    """
    Convenience function: runs extract_skill_spans() + assign_bio() in one call.

    Args:
        skills: LLM-extracted span strings.
        tokens: Original SkillSpan token list.

    Returns:
        (bio_tags, matched_spans)
        bio_tags: one tag per token
        matched_spans: for debugging / evaluation
    """
    matched = extract_skill_spans(skills, tokens)
    tags = assign_bio(tokens, matched)
    return tags, matched


_SUBWORD_RE = re.compile(r"\.\w+|\w+(?:[-']\w+)*|[^\w\s]")


def tokenize_like_dataset(text: str) -> list[str]:
    """Split LLM span text the same way SkillSpan tokens are split,
    so word counts line up with real token windows.
    e.g. "Train (ART)" -> ["Train", "(", "ART", ")"]  (not one glued word)
    """
    return _SUBWORD_RE.findall(text.strip())


# ── Internal matching logic ────────────────────────────────────────────────────


def normalize_token(text: str) -> str:
    """
    Single, centralized normalization rule used everywhere we compare
    text for matching purposes. This does NOT touch the original LLM
    span text stored in MatchedSpan -- it's only ever used as a
    comparison key, never as stored/displayed output.

    Rules: lowercase, strip surrounding whitespace, strip common
    surrounding punctuation (but NOT internal punctuation like "/" or
    "." inside words -- that's handled separately by the glued-token
    logic, not here).
    """
    text = text.lower().strip()
    text = text.strip(".,;:!?()[]{}\"'")
    return text


def _find_span_in_tokens(
    span_text: str,
    tokens: list[str],
    used_positions: set[int],
) -> tuple[Optional[MatchedSpan], bool]:
    """
    Find the first occurrence of span_text in tokens that doesn't overlap
    with already-used positions.

    Matching strategy (in order of preference):
    1. Exact token-sequence match (case-insensitive).
    2. Whole-span-as-one-glued-token match.
    3. Fuzzy: try matching after normalizing punctuation differences.
    4. Single-word span hiding inside a glued/slashed token.

    Args:
        span_text:      The span string to find (e.g. "communication skills").
        tokens:         Full token list.
        used_positions: Token indices already claimed; skip overlapping matches.

    Returns:
        (MatchedSpan or None, overlapped_only)
        overlapped_only is True when the text genuinely matched somewhere
        in the sentence but every such match collided with an already-used
        position (e.g. a shorter span nested inside an earlier, longer
        match) -- distinct from the text simply not appearing at all.
        Callers use this to log an accurate reason for a miss.
    """
    span_words = tokenize_like_dataset(span_text)
    n_span = len(span_words)
    n_tokens = len(tokens)

    if n_span == 0:
        return None, False

    found_overlap = (
        False  # set True the moment any tier matches text but loses to used_positions
    )

    # Try every possible start position
    for start in range(n_tokens - n_span + 1):
        end = start + n_span - 1
        candidate = tokens[start : end + 1]
        if not _tokens_match(span_words, candidate):
            continue
        if any(i in used_positions for i in range(start, end + 1)):
            found_overlap = True
            continue
        return MatchedSpan(text=span_text, start=start, end=end), False
    # New: try matching the WHOLE span text as a single glued token
    # (handles cases like "Java/J2EE" where the dataset kept it as ONE
    # token, unlike ".NET/Azure" where the dataset split the dot off)
    for i, tok in enumerate(tokens):
        if normalize_token(tok) != normalize_token(span_text):
            continue
        if i in used_positions:
            found_overlap = True
            continue
        return MatchedSpan(text=span_text, start=i, end=i), False
    # Fallback: try matching after stripping punctuation from both sides
    # This handles cases like LLM returning "CI/CD" but token is "CI/CD,"
    for start in range(n_tokens - n_span + 1):
        end = start + n_span - 1
        candidate = tokens[start : end + 1]
        if not _tokens_match_fuzzy(span_words, candidate):
            continue
        if any(i in used_positions for i in range(start, end + 1)):
            found_overlap = True
            continue
        return MatchedSpan(text=span_text, start=start, end=end), False
    # Fallback: handle "Azure" hiding inside a glued token like ".NET/Azure"
    # Only for single-word spans, so we don't wrongly match multi-word spans.
    if n_span == 1:
        needle = span_words[0].lower()
        needle_no_dot = needle.lstrip(".")  # dataset sometimes splits a
        # leading "." into its own token, leaving the word without its dot
        # e.g. ".NET/Azure" -> tokens: ".", "NET/Azure"
        for i, tok in enumerate(tokens):
            parts = re.split(r"[/\-,&()]", normalize_token(tok))
            if needle not in parts and needle_no_dot not in parts:
                continue
            if i in used_positions:
                found_overlap = True
                continue
            return MatchedSpan(text=span_text, start=i, end=i), False
    return None, found_overlap


def _tokens_match(span_words: list[str], candidate: list[str]) -> bool:
    """Exact case-insensitive token-by-token match."""
    if len(span_words) != len(candidate):
        return False
    return all(
        normalize_token(s) == normalize_token(c) for s, c in zip(span_words, candidate)
    )


def _tokens_match_fuzzy(span_words: list[str], candidate: list[str]) -> bool:
    """
    Fuzzy match: strip leading/trailing punctuation from each token before comparing.
    Handles cases where the LLM returned "skills" but the token is "skills."
    """
    if len(span_words) != len(candidate):
        return False

    def normalize(w: str) -> str:
        return re.sub(r"^[^\w]+|[^\w]+$", "", w).lower()

    return all(normalize(s) == normalize(c) for s, c in zip(span_words, candidate))


# ── Gold label conversion ──────────────────────────────────────────────────────


def gold_to_bio(tags_skill: list[str]) -> list[str]:
    """
    Convert SkillSpan gold labels to BIO format.

    SkillSpan uses "B", "I", "O" internally.
    This function maps them to "B-SKILL", "I-SKILL", "O" for consistency
    with the format produced by assign_bio().

    Args:
        tags_skill: List of raw gold tags from the SkillSpan dataset.

    Returns:
        List of standardized BIO tags.
    """
    mapping = {"B": "B-SKILL", "I": "I-SKILL", "O": "O"}
    return [mapping.get(t, "O") for t in tags_skill]
