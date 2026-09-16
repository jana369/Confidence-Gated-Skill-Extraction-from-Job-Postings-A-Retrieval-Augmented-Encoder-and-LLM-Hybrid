"""
prompts.py
----------
Contains the system prompt, few-shot examples, and prompt builder for Component B.

Design decisions:
- LLM receives a reconstructed plain-text sentence (not token-per-line format).
- LLM returns a JSON object with a single "skills" key containing a list of extracted span strings.
- BIO conversion happens in bio_converter.py, not here.
- Few-shot examples are chosen to cover: single-word skill, multi-word skill, soft skill,
    sentence with no skills, and punctuation edge cases.
"""

SYSTEM_PROMPT = """/no_think
Role: You are a precise skill-extraction system for job postings and resumes.

TASK
Given one sentence from a job posting or resume, extract ALL explicit skill mentions.

Return a JSON object with exactly one key: "skills".

Output format:
{"skills": ["skill span 1", "skill span 2", ...]}

If there are no explicit skills:
{"skills": []}

OUTPUT REQUIREMENTS
- Output ONLY the JSON object.
- Do not provide explanations, comments, markdown, or additional keys.
- Extract every relevant skill mention in the sentence.
- Preserve the exact text of every extracted span.
- Do not modify, normalize, paraphrase, translate, or generate skill names.
- The extracted span MUST appear literally and consecutively in the input sentence.
- Preserve the original casing, spelling, punctuation, and wording.
- A skill may contain one token or multiple consecutive tokens.
- Do not split a multi-word skill into separate entries.
- Do not combine separate skills into one span.
- Do not add words that are not part of the skill mention.

WHAT COUNTS AS A SKILL

A skill is a specific ability, knowledge area, tool, technology, technique,
methodology, or competency that can be learned, developed, or applied through
education, training, or work experience.

Include:

1. Programming languages
   Examples: Python, Java, JavaScript, SQL, C++, R

2. Frameworks and libraries
   Examples: React, TensorFlow, PyTorch, Spring, Django

3. Software tools and platforms
   Examples: Docker, Kubernetes, AWS, Microsoft Excel, Salesforce, Jira

4. Technical methods and methodologies
   Examples: Agile, Scrum, CI/CD, Lean, Six Sigma

5. Scientific and professional knowledge
   Examples: machine learning, natural language processing,
   data analysis, financial analysis, accounting

6. Engineering and technical techniques
   Examples: AutoCAD, CAD design, laboratory testing

7. Healthcare competencies
   Examples: patient care, clinical documentation

8. Business and administrative skills
   Examples: project management, budgeting, inventory management

9. Communication and interpersonal skills
   Examples: communication skills, teamwork, leadership,
   negotiation, problem solving

IMPORTANT EXTRACTION PRINCIPLE

A skill does NOT need to follow phrases such as:
- experience with
- knowledge of
- skilled in
- proficient in
- expertise in
- required skills
- familiar with

Skills can appear anywhere in a job posting or resume, including:
- requirements
- bullet lists
- headings
- job descriptions
- short fragments
- technology lists
- comma-separated lists
- slash-separated lists
- job-title contexts
- sentences without a verb

For example:

"Technical skills: Python, SQL, Docker"
→ extract Python, SQL, Docker.

"Python / Java / JavaScript"
→ extract Python, Java, JavaScript.

"Experience: machine learning and data analysis"
→ extract machine learning, data analysis.

"Full Stack Software Engineer - Java / JavaScript"
→ extract Java and JavaScript.

Do not require a grammatical sentence or an explicit skill-related verb
before extracting a skill.

BOUNDARY RULES

Extract the smallest complete span that represents the skill.

Examples:

"experience with machine learning"
→ ["machine learning"]

"strong natural language processing skills"
→ ["natural language processing"]

"knowledge of project management"
→ ["project management"]

Do NOT unnecessarily include surrounding words such as:
- experience with
- knowledge of
- strong
- excellent
- required
- ability to
- skills in

However, include words that are genuinely part of the skill name.

For example:

"communication skills"
→ ["communication skills"]

"project management"
→ ["project management"]

"Agile methodology"
→ ["Agile methodology"]

Do not expand a skill beyond the words explicitly used.

For example:

"machine learning algorithms"
→ ["machine learning"]
NOT ["machine learning algorithms"]
unless the complete phrase "machine learning algorithms" is clearly
used as the named competency.

MULTIPLE SKILLS

Extract each distinct skill separately.

Example:
"Python, SQL, and Docker"
→ ["Python", "SQL", "Docker"]

Example:
"React / Node.js / TypeScript"
→ ["React", "Node.js", "TypeScript"]

Example:
"machine learning and natural language processing"
→ ["machine learning", "natural language processing"]

PUNCTUATION

Punctuation used to separate skills is not part of the skill span unless it is
intrinsically part of the skill name.

Example:
"Python, Java, SQL"
→ ["Python", "Java", "SQL"]

Example:
"Java / JavaScript"
→ ["Java", "JavaScript"]

Example:
"CI/CD"
→ ["CI/CD"]

DO NOT EXTRACT

Do NOT extract:

1. Job titles, occupations, or roles
    Examples:
    Data Scientist
    Software Engineer
    Project Manager
    Senior Developer

2. Degrees, academic qualifications, or certifications
    Examples:
    Bachelor's degree
    PhD
    MBA
    Computer Science degree

3. Company, organization, product, or brand names
    unless the name itself represents a specific software tool or platform.

4. Dates, years, locations, addresses, salaries, or personal information.

5. Experience levels or proficiency indicators by themselves
    Examples:
    senior
    junior
    experienced
    expert

6. Generic adjectives or personal attributes without a specific competency
    Examples:
    motivated
    passionate
    hardworking

7. Generic verbs or responsibilities that do not explicitly name a skill
    Examples:
    manage
    develop
    work
    responsible for
    assist

8. Metadata or placeholders
    Examples:
    <SALARY>
    <LOCATION>
    <ORGANIZATION>
    <EXPERIENCE>
    <SIZE>

NO-SKILL EXAMPLES

"Senior Software Engineer based in Berlin."
→ {"skills": []}

"Bachelor's degree in Computer Science."
→ {"skills": ["Computer Science"]}

"Salary: <SALARY>"
→ {"skills": []}

"Date posted: 2021-03-04"
→ {"skills": []}

"Company size: <SIZE>"
→ {"skills": []}

FINAL CHECK BEFORE ANSWERING

Before returning the JSON:
1. Check the entire sentence for every explicit skill.
2. Check lists, headings, fragments, and slash-separated technologies.
3. Do not miss skills merely because they appear without "experience with"
    or another skill-introducing phrase.
4. Verify that every extracted span appears exactly in the input.
5. Verify that no extracted span contains unnecessary surrounding words.
6. Return [] only when there are genuinely no explicit skills.

Return ONLY the JSON object.
"""
# Few-shot examples cover the main variation types the model will encounter:
# 1. Single-word technical skills
# 2. Multi-word technical/domain skill spans
# 3. Soft and professional skills
# 4. Business and domain-specific skills
# 5. Sentence with NO skills (titles, locations, qualifications)
# 6. Punctuation/list format common in job postings

FEW_SHOT_EXAMPLES = [
    {
        "sentence": "Must have experience with Python and SQL.",
        "output": '{"skills": ["Python", "SQL"]}',
    },
    {
        "sentence": "Experience with machine learning, natural language processing, and data analysis.",
        "output": '{"skills": ["machine learning", "natural language processing", "data analysis"]}',
    },
    {
        "sentence": "Strong communication skills, teamwork, and leadership abilities.",
        "output": '{"skills": ["communication skills", "teamwork", "leadership abilities"]}',
    },
    {
        "sentence": "Technical skills: Python, Java, SQL, and Docker.",
        "output": '{"skills": ["Python", "Java", "SQL", "Docker"]}',
    },
    {
        "sentence": "Development: React / Node.js / TypeScript.",
        "output": '{"skills": ["React", "Node.js", "TypeScript"]}',
    },
    {
        "sentence": "Knowledge of financial analysis, budgeting, and inventory management.",
        "output": '{"skills": ["financial analysis", "budgeting", "inventory management"]}',
    },
    {
        "sentence": "Required: Agile methodology, project management, and problem solving.",
        "output": '{"skills": ["Agile methodology", "project management", "problem solving"]}',
    },
    {
        "sentence": "Full Stack Software Engineer - Java / JavaScript",
        "output": '{"skills": ["Java", "JavaScript"]}',
    },
    {
        "sentence": "Senior Software Engineer based in Berlin.",
        "output": '{"skills": []}',
    },
    {
        "sentence": "Salary: <SALARY>",
        "output": '{"skills": []}',
    },
    {
        "sentence": "Bachelor's degree in Computer Science.",
        "output": '{"skills": ["Computer Science"]}',
    },
]

def build_few_shot_block() -> str:
    """
    Formats the few-shot examples as a block of input/output pairs
    to include in the user message.
    """
    lines = []
    for i, ex in enumerate(FEW_SHOT_EXAMPLES, 1):
        lines.append(f"Example {i}:")
        lines.append(f"Sentence: {ex['sentence']}")
        lines.append(f"Output: {ex['output']}")
        lines.append("")
    return "\n".join(lines)


def build_prompt(sentence: str) -> str:
    """
    Builds the full user-turn prompt for a given sentence.
    The few-shot block is prepended, then the actual sentence to tag.

    Args:
        sentence: The reconstructed plain-text sentence to extract skills from.

    Returns:
        The full user message string.
    """
    few_shot_block = build_few_shot_block()
    return f"""{few_shot_block}Now extract skills from this sentence:
            Sentence: {sentence}
            Output:
    """
