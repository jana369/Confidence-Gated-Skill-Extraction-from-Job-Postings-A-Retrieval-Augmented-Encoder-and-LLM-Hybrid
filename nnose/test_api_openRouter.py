import requests
import json
import time

import os
api_key = os.environ["GROQ_API_KEY"]


SYSTEM_PROMPT = """
/no_think

Role:
You are a precise skill-extraction tagger.

Output exactly one BIO tag per word:

B-SKILL: first word of a skill mention
I-SKILL: continuation word of a skill mention
O: not a skill

Rules:
- Output ONLY word and tag separated by tab.
- One word per line.
- No explanations.
- Preserve word order.
- Punctuation receives O.

Include:
- Python
- SQL
- machine learning
- communication skills
- problem solving
- project management

Do NOT tag:
- experience
- knowledge
- ability
- job titles
- degrees
- companies
"""


sentence = (
    "Experience with Python and strong communication skills , and 5 years of experience in machine learning. "
)
# google/gemma-4-26b-a4b-it:free
# google/gemma-4-31b-it:free
# nvidia/nemotron-nano-9b-v2:free

payload = {
    "model": "nvidia/nemotron-nano-9b-v2:free",
    "messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Tag this sentence:\n{sentence}"},
    ],
    "temperature": 0.0,
    "max_tokens": 200,
    "reasoning": {"enabled": False},
}


start = time.perf_counter()


response = requests.post(
    "https://openrouter.ai/api/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    },
    data=json.dumps(payload),
)


end = time.perf_counter()


result = response.json()

print(json.dumps(result, indent=2))
print("---")

print(result["choices"][0]["message"]["content"])

print("---")
print(f"Time taken: {end-start:.3f} seconds")
