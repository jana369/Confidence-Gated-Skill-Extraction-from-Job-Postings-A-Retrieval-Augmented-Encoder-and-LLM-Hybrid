from groq import Groq
import time
import os
api_key = os.environ["GROQ_API_KEY"]

client = Groq(api_key=api_key)


SYSTEM_PROMPT = """
/no_think

Role:
You are a precise skill-extraction tagger.

Given a sentence, output exactly one BIO tag per word.

Labels:
- B-SKILL: first word of a skill mention
- I-SKILL: continuation word of a skill mention
- O: not part of a skill

Rules:
- Output ONLY word and tag separated by tab.
- One word per line.
- No explanations.
- Preserve exact word order.
- Punctuation receives O.
- Adjectives describing skills are not tagged.

Boundary rules:
- Do not tag connecting words or prepositions before skills.
Examples:
"with Python" -> with O, Python B-SKILL
"in machine learning" -> in O, machine B-SKILL, learning I-SKILL

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
FEW_SHOT_EXAMPLES = """Example 1:
Sentence: Must have experience with Python and SQL.
Must\tO
have\tO
experience\tO
with\tO
Python\tB-SKILL
and\tO
SQL\tB-SKILL
.\tO

Example 2:
Sentence: Looking for someone with strong communication skills and teamwork.
Looking\tO
for\tO
someone\tO
with\tO
strong\tO
communication\tB-SKILL
skills\tI-SKILL
and\tO
teamwork\tB-SKILL
.\tO

Example 3:
Sentence: Familiar with Kubernetes, Docker, and CI/CD pipelines.
Familiar\tO
with\tO
Kubernetes\tB-SKILL
,\tO
Docker\tB-SKILL
,\tO
and\tO
CI/CD\tB-SKILL
pipelines\tI-SKILL
.\tO

Example 4:
Sentence: Requires 5 years of experience and technical knowledge.
Requires\tO
5\tO
years\tO
of\tO
experience\tO
and\tO
technical\tO
knowledge\tO
.\tO

Example 5:
Sentence: Experience in machine learning and deep learning.
Experience\tO
in\tO
machine\tB-SKILL
learning\tI-SKILL
and\tO
deep\tB-SKILL
learning\tI-SKILL
.\tO
"""

sentence = (
    "Experience with Python and strong communication skills, "
    "and 5 years of experience in machine learning."
)

def build_prompt(sentence):
    return f"{FEW_SHOT_EXAMPLES}\n\nTag this sentence:\n{sentence}"

start = time.perf_counter()
# llama-3.3-70b-versatile
# llama-3.1-8b-instant
# gemma2-9b-it
# qwen3.6-27b

completion = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(sentence)},
    ],
    temperature=0.0,
    max_completion_tokens=80,
    top_p=1,
)


end = time.perf_counter()


print(completion.choices[0].message.content)

print("---")
print(f"Time taken: {end-start:.3f} seconds")
