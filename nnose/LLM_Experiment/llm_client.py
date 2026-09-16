"""
llm_client.py
-------------
Provider-agnostic LLM client for Component B skill extraction.

Supports three providers through a unified interface:
- NVIDIA NIM  (base_url: https://integrate.api.nvidia.com/v1)
- Groq        (base_url: https://api.groq.com/openai/v1)
- OpenRouter  (base_url: https://openrouter.ai/api/v1)

All three use the OpenAI-compatible chat completions API, so only
the base_url and api_key need to change between providers.

Returns a structured ExtractionResult with:
- skills: list of extracted span strings (from JSON parsing)
- raw_response: the full text output from the LLM
- confidence: average logprob over the label tokens (if logprobs supported)
- latency_ms: wall-clock inference time in milliseconds
- invalid_json: whether the model's output failed to parse as valid JSON
"""

import json
import time
import re
from dataclasses import dataclass, field
from typing import Optional
from openai import OpenAI
import os
from prompts import SYSTEM_PROMPT, build_prompt
from dotenv import load_dotenv

# ── Provider configurations ──────────────────────────────────────────────────
# Change ACTIVE_PROVIDER to switch between providers.
# Only the base_url and api_key need to change; everything else stays the same.
load_dotenv()
os.path.join(os.path.dirname(__file__), ".env")
PROVIDERS = {
    "nvidia": {
        "base_url": "https://integrate.api.nvidia.com/v1",
        "model": "nvidia/llama-3.3-nemotron-super-49b-v1.5",
        "extra_body": {"chat_template_kwargs": {"thinking": False}},
        "supports_logprobs": True,
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "model": "qwen/qwen3.6-27b",  
        "extra_body": None,
        "supports_logprobs": False,  
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "model": "google/gemma-3-27b-it",  
        "extra_body": None,
        "supports_logprobs": False,  
    },
}

ACTIVE_PROVIDER = "openrouter"  # ← change this line to switch providers
api_key = os.getenv(str(ACTIVE_PROVIDER.upper()) + "_API_KEY")

# ── Result dataclass ──────────────────────────────────────────────────────────


@dataclass
class ExtractionResult:
    skills: list[str]  # extracted span strings; empty list if none found
    raw_response: str  # full LLM text output, for debugging
    confidence: Optional[float]  # mean logprob of label tokens; None if not available
    latency_ms: float  # wall-clock inference time
    invalid_json: bool  # True if output failed JSON parsing
    raw_logprobs: list = None # raw logprobs for each generated token; None if not available


# ── Core extraction function ──────────────────────────────────────────────────


def extract_skills(
    sentence: str,
    api_key: str,
    provider: str = ACTIVE_PROVIDER,
    max_tokens: int = 256,
    temperature: float = 0.0,
) -> ExtractionResult:
    """
    Extract skill spans from a single sentence using the configured LLM provider.

    Args:
        sentence:    Reconstructed plain-text sentence (joined from token list).
        api_key:     API key for the active provider.
        provider:    Provider name (must be a key in PROVIDERS dict).
        max_tokens:  Max tokens to generate. 256 is generous for a JSON span list.
        temperature: 0.0 for deterministic output (recommended for structured extraction).

    Returns:
        ExtractionResult with parsed skills, confidence, latency, and error flags.
    """
    config = PROVIDERS[provider]
    client = OpenAI(
        base_url=config["base_url"],
        api_key=api_key,
    )

    prompt = build_prompt(sentence)
    use_logprobs = config["supports_logprobs"]

    # Build the request kwargs
    request_kwargs = dict(
        model=config["model"],
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        logprobs=use_logprobs,
        top_logprobs=1 if use_logprobs else None,
        extra_body=config["extra_body"],
    )

    # Inference with wall-clock timing
    t0 = time.time()
    response = client.chat.completions.create(**request_kwargs)
    latency_ms = (time.time() - t0) * 1000

    raw_text = response.choices[0].message.content.strip()

    # ── Parse JSON ────────────────────────────────────────────────────────────
    skills, invalid_json = _parse_skills_json(raw_text)

    # ── Compute confidence from logprobs ──────────────────────────────────────
    confidence = None
    raw_logprobs = None
    if use_logprobs and response.choices[0].logprobs:
        raw_logprobs = response.choices[0].logprobs.content
        confidence = _mean_logprob(raw_logprobs)

    return ExtractionResult(
        skills=skills,
        raw_response=raw_text,
        confidence=confidence,
        latency_ms=latency_ms,
        invalid_json=invalid_json,
        raw_logprobs=raw_logprobs,
    )


# ── Helper functions ──────────────────────────────────────────────────────────


def _parse_skills_json(raw_text: str) -> tuple[list[str], bool]:
    """
    Parse the LLM's text output into a list of skill span strings.

    Tries strict JSON parsing first, then falls back to a regex extraction
    of the JSON object in case the model added surrounding text.

    Returns:
        (skills list, invalid_json flag)
        invalid_json is True if we couldn't parse anything useful.
    """
    # 1. Try direct parse
    try:
        parsed = json.loads(raw_text)
        skills = parsed.get("skills", [])
        if isinstance(skills, list):
            return [str(s).strip() for s in skills if s], False
    except json.JSONDecodeError:
        pass

    # 2. Try extracting a JSON object embedded in surrounding text
    match = re.search(r'\{.*?"skills"\s*:\s*\[.*?\]\s*\}', raw_text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            skills = parsed.get("skills", [])
            if isinstance(skills, list):
                return [str(s).strip() for s in skills if s], False
        except json.JSONDecodeError:
            pass

    # 3. Give up — flag as invalid JSON, return empty
    return [], True


def _mean_logprob(token_logprobs) -> float:
    """
    Compute the mean log-probability across all generated tokens.
    This serves as a single scalar confidence score for the whole extraction:
    higher (less negative) = more confident output.

    Note: this is averaged over ALL generated tokens (including structural JSON
    tokens like {, ", [, etc.), not just the skill-text tokens.
    For the gate's feature, this is fine — we want overall output confidence,
    not per-span confidence, since the gate operates at sentence level.
    """
    if not token_logprobs:
        return 0.0
    total = sum(t.logprob for t in token_logprobs)
    return total / len(token_logprobs)


# ── Convenience: reconstruct sentence from token list ─────────────────────────


def tokens_to_sentence(tokens: list[str]) -> str:
    """
    Reconstruct a plain-text sentence from a SkillSpan-style token list.
    Handles basic punctuation attachment so the LLM sees natural text.

    SkillSpan tokens are already individual words, so simple space-joining
    works for most cases. Punctuation tokens that should attach left
    (. , ! ? ; :) are handled by stripping the space before them.

    Args:
        tokens: List of word strings from the SkillSpan dataset.

    Returns:
        Reconstructed sentence string.
    """
    if not tokens:
        return ""

    sentence = " ".join(tokens)

    # Attach punctuation to the preceding word
    sentence = re.sub(r" ([.,!?;:)])", r"\1", sentence)
    sentence = re.sub(r"([(]) ", r"\1", sentence)

    return sentence.strip()


