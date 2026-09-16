from openai import OpenAI
import time

import os

api_key = os.environ["NVIDIA_API_KEY"]

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key,
)


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

Tag:
- Programming languages
- Frameworks
- Libraries
- Software tools
- Platforms
- Soft skills

Do NOT tag:
- Job titles
- Degrees
- Companies
- Dates
- Locations
- Responsibilities
- Generic words like experience, knowledge, ability
"""


sentence = (
    "Experience with Python and strong communication skills, "
    "and 5 years of experience in machine learning."
)


start = time.perf_counter()

response = client.chat.completions.create(
    model="nvidia/llama-3.3-nemotron-super-49b-v1.5",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Tag this sentence:\n{sentence}"},
    ],
    logprobs=True,
    top_p=1.0,
    top_logprobs=1,
    temperature=0.0,
    max_tokens=100,
)


end = time.perf_counter()
print("FULL RESPONSE:")
print(response)

print("\nCHOICES:")
print(response.choices)

if response.choices:
    print("\nCONTENT:")
    print(response.choices[0].message.content)

    print("\nLOGPROBS:")
    print(response.choices[0].logprobs)

print("---")
print(f"Time taken: {end-start:.3f} seconds")
