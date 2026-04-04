# =============================================================
# 3Qverse AI Backend — main.py
# INTELLIGENCE LAYER v2.0
# "The system that thinks before it speaks"
# =============================================================

import os
import re
import time
import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
from google import genai
from dotenv import load_dotenv

# ── Load env ──────────────────────────────────────────────────
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

# ── Logging ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[3Q AI] %(asctime)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger("3qverse")

logger.info("Starting 3Qverse AI backend v2.0 — Intelligence Layer active")
logger.info(f"GEMINI_API_KEY present: {bool(API_KEY)}")
logger.info(f"GEMINI_API_KEY length: {len(API_KEY) if API_KEY else 0}")

# ── Gemini Client ─────────────────────────────────────────────
if not API_KEY:
    logger.error("❌ GEMINI_API_KEY not set!")
    client = None
else:
    client = genai.Client(api_key=API_KEY)
    logger.info("✅ Gemini client initialized")

# ── App ───────────────────────────────────────────────────────
app = FastAPI(
    title="3Qverse AI Backend",
    description="Intelligent B.Tech learning assistant — v2.0",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request timing middleware ─────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = round((time.time() - start) * 1000, 2)
    logger.info(f"{request.method} {request.url.path} → {response.status_code} ({duration}ms)")
    return response

# ── Shared code formatting rule ───────────────────────────────
CODE_RULE = """
CRITICAL FORMATTING RULES:
- Every code example MUST be inside a fenced code block.
- ALWAYS put the language name right after the opening triple backticks on the SAME line.
- Correct:   ```python
             # code here
             ```
- NEVER use ``` alone without a language name.
- NEVER write raw code outside a fenced block.
"""

# =============================================================
# Request Models
# =============================================================

class ChatMessage(BaseModel):
    role: str
    content: str

class AskRequest(BaseModel):
    question: str
    history: Optional[List[ChatMessage]] = []

class ConceptRequest(BaseModel):
    concept: str

class StudyPlanRequest(BaseModel):
    tech: str
    level: str
    days: int

class RoadmapRequest(BaseModel):
    tech: str
    level: str

class CodeAnalyzeRequest(BaseModel):
    code: str
    lang: str

class ExamAnswerRequest(BaseModel):
    question: str
    subject: str
    marks: str  # "5" or "10" — string

class LastNightRequest(BaseModel):
    subject: str
    time: str    # e.g. "1 hour", "2 hours"
    topics: Optional[str] = ""

# =============================================================
# ╔══════════════════════════════════════════════════════════╗
# ║          INTELLIGENCE LAYER — THE BRAIN                  ║
# ║  This is what separates 3Qverse from a plain AI wrapper  ║
# ╚══════════════════════════════════════════════════════════╝
# =============================================================

# ── LAYER 1: Question Type Detector ──────────────────────────
# Understands WHAT the student is asking before answering
def detect_question_type(question: str) -> str:
    """
    Classify the exam question into one of 8 types.
    This drives prompt adaptation — different types get different strategies.
    """
    q = question.lower().strip()

    # ORDER MATTERS — more specific patterns first
    if any(w in q for w in ["compare", "difference between", "distinguish", "differentiate", "vs ", "versus"]):
        return "comparison"
    elif any(w in q for w in ["draw", "sketch", "diagram", "illustrate", "show with figure"]):
        return "diagram"
    elif any(w in q for w in ["advantages and disadvantages", "pros and cons", "merits and demerits", "advantages", "disadvantages"]):
        return "pros_cons"
    elif any(w in q for w in ["how does", "how do", "how is", "how are", "working of", "mechanism", "process of"]):
        return "working"
    elif any(w in q for w in ["algorithm", "write algorithm", "steps to", "procedure"]):
        return "algorithm"
    elif any(w in q for w in ["write a program", "write code", "implement", "coding"]):
        return "code"
    elif any(w in q for w in ["define", "what is", "what are", "state", "give the definition"]):
        return "definition"
    elif any(w in q for w in ["explain", "describe", "elaborate", "discuss", "write short note", "write a note"]):
        return "explanation"
    else:
        return "general"


# ── LAYER 2: Subject Intelligence ────────────────────────────
# Knows the exam landscape of each B.Tech subject
SUBJECT_INTELLIGENCE: Dict[str, Dict] = {
    # Operating Systems
    "os": {
        "full_name": "Operating Systems",
        "hot_topics": ["deadlock", "scheduling", "memory management", "paging", "semaphore", "process", "thread", "ipc"],
        "exam_weight": "very_high",
        "tip": "Always mention real OS examples (Linux, Windows). Diagrams for process states and memory are expected."
    },
    "operating systems": {
        "full_name": "Operating Systems",
        "hot_topics": ["deadlock", "scheduling", "memory management", "paging", "semaphore", "process", "thread"],
        "exam_weight": "very_high",
        "tip": "Always mention real OS examples. State transition diagrams score extra marks."
    },
    # DBMS
    "dbms": {
        "full_name": "Database Management Systems",
        "hot_topics": ["normalization", "er diagram", "sql", "transaction", "acid", "joins", "indexing", "b-tree"],
        "exam_weight": "very_high",
        "tip": "Always write SQL queries when asked. Normalization needs step-by-step with examples."
    },
    "database": {
        "full_name": "Database Management Systems",
        "hot_topics": ["normalization", "er diagram", "sql", "transaction", "acid", "joins"],
        "exam_weight": "very_high",
        "tip": "SQL queries and ER diagrams are expected. Always show functional dependencies."
    },
    # Computer Networks
    "computer networks": {
        "full_name": "Computer Networks",
        "hot_topics": ["tcp/ip", "osi model", "routing", "subnetting", "congestion", "http", "dns", "topology"],
        "exam_weight": "high",
        "tip": "OSI/TCP-IP layer diagrams are expected. Always mention protocol names and port numbers."
    },
    "cn": {
        "full_name": "Computer Networks",
        "hot_topics": ["tcp", "osi", "routing", "subnetting", "congestion", "topology"],
        "exam_weight": "high",
        "tip": "Layer diagrams always score marks. Mention specific protocols."
    },
    "computer networking": {
        "full_name": "Computer Networks",
        "hot_topics": ["tcp", "osi", "routing", "topology", "congestion", "dns"],
        "exam_weight": "high",
        "tip": "Topology diagrams and layer models are expected in answers."
    },
    # Data Structures
    "data structures": {
        "full_name": "Data Structures & Algorithms",
        "hot_topics": ["trees", "graphs", "sorting", "searching", "hashing", "stack", "queue", "linked list"],
        "exam_weight": "very_high",
        "tip": "Always write code + trace through example. Time complexity is mandatory for algorithms."
    },
    "dsa": {
        "full_name": "Data Structures & Algorithms",
        "hot_topics": ["trees", "sorting", "hashing", "graphs", "complexity"],
        "exam_weight": "very_high",
        "tip": "Code + dry run + time complexity = full marks formula."
    },
    # Software Engineering
    "software engineering": {
        "full_name": "Software Engineering",
        "hot_topics": ["sdlc", "agile", "waterfall", "testing", "uml", "design patterns", "requirements"],
        "exam_weight": "medium",
        "tip": "Diagrams (UML, flowcharts) and model comparisons score highly."
    },
    # Computer Architecture
    "computer organization": {
        "full_name": "Computer Organization & Architecture",
        "hot_topics": ["pipeline", "cache", "instruction cycle", "addressing modes", "alu", "registers"],
        "exam_weight": "high",
        "tip": "Timing diagrams and block diagrams are expected. Show numerical examples."
    },
    "coa": {
        "full_name": "Computer Organization & Architecture",
        "hot_topics": ["pipeline", "cache", "instruction cycle", "alu"],
        "exam_weight": "high",
        "tip": "Block diagrams with labels score extra marks."
    },
}

def get_subject_intel(subject: str) -> Optional[Dict]:
    """Get subject intelligence data. Fuzzy match by checking substrings."""
    s = subject.lower().strip()
    # Exact match first
    if s in SUBJECT_INTELLIGENCE:
        return SUBJECT_INTELLIGENCE[s]
    # Partial match
    for key, val in SUBJECT_INTELLIGENCE.items():
        if key in s or s in key:
            return val
    return None


# ── LAYER 3: Topic Importance Scorer ─────────────────────────
# Detects if this is a high-frequency exam question that needs extra depth
UNIVERSALLY_IMPORTANT = [
    # OS
    "deadlock", "semaphore", "mutex", "scheduling", "paging", "segmentation",
    "virtual memory", "thrashing", "critical section", "process synchronization",
    # DBMS
    "normalization", "er diagram", "acid properties", "transaction", "join",
    "b+ tree", "indexing", "concurrency control",
    # Networks
    "tcp/ip", "osi model", "subnetting", "congestion control", "routing algorithm",
    "ip addressing", "dns", "http", "three way handshake",
    # DSA
    "binary search tree", "avl tree", "dijkstra", "dynamic programming",
    "sorting algorithm", "time complexity", "hash table",
    # General CS
    "recursion", "polymorphism", "inheritance", "design pattern",
]

def score_question_importance(question: str, subject_intel: Optional[Dict]) -> str:
    """
    Returns 'critical' | 'high' | 'normal'
    Critical = frequently asked, must score full marks
    """
    q = question.lower()

    # Check universal importance
    if any(topic in q for topic in UNIVERSALLY_IMPORTANT):
        return "critical"

    # Check subject-specific hot topics
    if subject_intel:
        hot = subject_intel.get("hot_topics", [])
        if any(topic in q for topic in hot):
            return "high"

    return "normal"


# ── LAYER 4: Prompt Strategy Builder ─────────────────────────
# Builds the right prompt strategy based on everything we know
def build_exam_prompt_strategy(
    question: str,
    subject: str,
    marks: str,
    q_type: str,
    importance: str,
    subject_intel: Optional[Dict]
) -> str:
    """
    Assembles a context-aware, adaptive prompt.
    Every input shapes the output differently.
    """
    # Base character of the AI for this subject
    subject_context = ""
    if subject_intel:
        subject_context = f"""
You are an expert professor of {subject_intel['full_name']}.
Examiner's tip for this subject: {subject_intel['tip']}
"""
    else:
        subject_context = f"You are an expert professor of {subject}."

    # Importance booster
    importance_instruction = ""
    if importance == "critical":
        importance_instruction = """
⚠️  CRITICAL EXAM TOPIC DETECTED:
This question is extremely frequently asked in university exams.
- Emphasize all keywords — examiners look for these specifically
- Every bullet point must be a mark-earning statement
- Do NOT miss any standard sub-topic for this concept
- Include the standard definition that examiners expect
"""
    elif importance == "high":
        importance_instruction = """
📌  HIGH-VALUE QUESTION:
- This is a commonly tested topic — be thorough
- Include all standard components examiners expect
- Keywords and technical terms must be precise
"""

    # Question-type specific strategy
    type_instruction = ""
    if q_type == "comparison":
        type_instruction = """
📊  COMPARISON QUESTION STRATEGY:
- Present differences in a clear TABLE format:
  | Parameter | [Term A] | [Term B] |
  |-----------|----------|----------|
- Cover at least 6-8 comparison parameters
- Parameters: Definition, Purpose, Speed, Memory, Use case, Example
- Examiners give marks per row — more rows = more marks
"""
    elif q_type == "diagram":
        type_instruction = """
🖼️  DIAGRAM QUESTION STRATEGY:
- Diagram section is MANDATORY and worth 3-4 marks alone
- Draw using ASCII art, clearly labeled
- Every component must be labeled
- Show direction of flow with arrows (→, ↓, ←)
- Add a brief description of each labeled component
"""
    elif q_type == "pros_cons":
        type_instruction = """
⚖️  ADVANTAGES/DISADVANTAGES STRATEGY:
- Use clear two-column or two-section format
- ADVANTAGES: numbered list, each point one mark
- DISADVANTAGES: numbered list, each point one mark
- Be specific — avoid vague points like "it is good"
- Give technical reasons for each advantage/disadvantage
"""
    elif q_type == "algorithm":
        type_instruction = """
⚙️  ALGORITHM QUESTION STRATEGY:
- Write algorithm in numbered steps (standard format)
- After algorithm, show a dry run/trace with example input
- State time complexity: O(?) and space complexity: O(?)
- Include code implementation in a fenced code block
"""
    elif q_type == "code":
        type_instruction = f"""
💻  CODE QUESTION STRATEGY:
- Write clean, working code in the most appropriate language
- Add comments explaining each significant line
- Show sample input and expected output
- Mention time and space complexity
- {CODE_RULE}
"""
    elif q_type == "working":
        type_instruction = """
⚙️  HOW IT WORKS — STRATEGY:
- Explain step-by-step, not just what it is
- Use numbered steps for the process/mechanism
- Include a diagram showing the working
- Give a real-world analogy to make it memorable
- Mention edge cases or failure scenarios
"""
    elif q_type == "definition":
        type_instruction = """
📖  DEFINITION QUESTION STRATEGY:
- Start with the standard textbook definition (examiners check this)
- Follow with expanded explanation
- Give one concrete real-world example
- List key characteristics as bullet points
"""

    # Marks-based length guidance
    if marks == "5":
        length_guide = """
📏  5-MARK ANSWER FORMAT:
- Length: approximately 150-200 words
- Definition: 1-2 crisp lines
- Key Points: EXACTLY 4-5 bullet points (each = ~1 mark)
- Example: 2-3 lines maximum
- Every sentence must earn marks — no filler
"""
    else:
        length_guide = """
📏  10-MARK ANSWER FORMAT:
- Length: approximately 400-500 words
- Introduction/Definition: 2-3 lines
- Detailed Explanation: 3-4 paragraphs OR structured sections
- Key Points: 6-8 detailed bullet points (each = ~1 mark)
- Example: Fully explained with specifics
- Diagram: Labeled ASCII diagram (if applicable)
- Conclusion: 1-2 lines about real-world significance
- Every section must be clearly headed for easy marking
"""

    # Assemble the full prompt
    prompt = f"""{subject_context}

Subject: {subject}
Question: {question}
Marks: {marks}
Question Type Detected: {q_type}

{importance_instruction}
{type_instruction}
{length_guide}

Write the exam answer in EXACTLY this format:

## 5 Mark Answer
[Write the 5-mark answer here — even for 10-mark questions, include a concise version]

## 10 Mark Answer
[Write the detailed 10-mark answer here]

## Keywords
[List 6-8 key terms, one per line starting with -]

## Diagram
[ASCII diagram if helpful, or write: Not required]

ABSOLUTE RULES:
- Use EXACTLY these headings: ## 5 Mark Answer, ## 10 Mark Answer, ## Keywords, ## Diagram
- No preamble. No commentary outside sections. No meta-text.
- Answers must be exam-ready — a student should be able to copy and score marks
- Technical terms must be precise and correctly used
{CODE_RULE}
"""
    return prompt


# ── LAYER 5: Output Intelligence Cleaner ─────────────────────
def clean_output(text: str) -> str:
    """
    Post-processing: clean Gemini's output before parsing.
    Fixes common formatting artifacts that break extraction.
    """
    # Fix double spaces after **
    text = re.sub(r'\*\*\s+', '**', text)
    # Fix missing space before **
    text = re.sub(r'([a-z])\*\*', r'\1 **', text)
    # Normalize multiple blank lines to double
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remove trailing whitespace on lines
    text = '\n'.join(line.rstrip() for line in text.split('\n'))
    # Ensure ## headings have space after ##
    text = re.sub(r'##([^ ])', r'## \1', text)
    return text.strip()


# ── LAYER 6: Smart Metadata Builder ──────────────────────────
def build_answer_metadata(
    question: str,
    subject: str,
    q_type: str,
    importance: str,
    subject_intel: Optional[Dict],
    response_time_ms: float
) -> Dict:
    """
    Build metadata returned with every exam answer.
    This powers frontend features like "why this matters" and tips.
    """
    metadata = {
        "question_type": q_type,
        "importance_level": importance,
        "response_time_ms": round(response_time_ms, 2),
    }

    if subject_intel:
        metadata["subject_tip"] = subject_intel.get("tip", "")
        metadata["exam_weight"] = subject_intel.get("exam_weight", "normal")

        # Check if question matches hot topics
        hot = subject_intel.get("hot_topics", [])
        q_lower = question.lower()
        matched_topics = [t for t in hot if t in q_lower]
        if matched_topics:
            metadata["matched_hot_topics"] = matched_topics

    # Add study tip based on question type
    type_tips = {
        "comparison": "Tables get more marks than paragraph comparisons in this topic.",
        "diagram": "A labeled diagram alone can earn 3-4 marks — always draw it.",
        "algorithm": "Writing the dry run trace alongside the algorithm doubles your marks.",
        "code": "Comments in code + time complexity = examiner's favorite answer.",
        "pros_cons": "Equal number of advantages and disadvantages shows balanced understanding.",
        "definition": "Start with the exact textbook definition — examiners check first lines.",
        "working": "Step-by-step numbered explanation shows process understanding.",
    }
    if q_type in type_tips:
        metadata["exam_tip"] = type_tips[q_type]

    if importance == "critical":
        metadata["importance_note"] = "🔥 This is a very frequently asked exam topic — learn this thoroughly."
    elif importance == "high":
        metadata["importance_note"] = "📌 Commonly tested — make sure you understand all aspects."

    return metadata


# =============================================================
# Helpers (parsing, history, Gemini)
# =============================================================

def call_gemini(prompt: str) -> str:
    """Call Gemini 2.5 Flash. Raises on failure."""
    if not client:
        raise Exception("Gemini client not initialized — GEMINI_API_KEY missing")
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    if not response.text:
        raise Exception("Gemini returned empty response")
    return response.text


def build_history_context(history: List[ChatMessage]) -> str:
    """Format chat history for Gemini prompt context."""
    if not history:
        return ""
    lines = ["Previous conversation (use this as context for follow-up questions):"]
    for msg in history[-10:]:
        label = "Student" if msg.role == "user" else "3Q AI"
        lines.append(f"{label}: {msg.content}")
    return "\n".join(lines) + "\n\n"


def extract_section(text: str, heading: str, next_headings: List[str]) -> str:
    """Robustly extract a section from markdown text using regex."""
    escaped = re.escape(heading)
    next_pattern = '|'.join(re.escape(h) for h in next_headings)
    pattern = rf"(?:##\s*{escaped}|##\s*\d+\s*{escaped}|\*\*{escaped}\*\*)[^\n]*\n(.*?)(?=(?:##\s*(?:{next_pattern})|\*\*(?:{next_pattern})\*\*)|\Z)"
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""


def extract_keywords(text: str) -> List[str]:
    """Extract keywords list from Keywords section."""
    section = extract_section(text, "Keywords", ["Diagram", "END"])
    if not section:
        return []
    keywords = []
    for line in section.split("\n"):
        line = line.strip()
        if not line:
            continue
        clean = re.sub(r"^[-*•\d.]\s*", "", line).strip()
        if clean and len(clean) < 60 and not clean.startswith("#"):
            keywords.append(clean)
    return keywords[:10]


def extract_diagram(text: str) -> str:
    """Extract diagram section, return empty if not required."""
    section = extract_section(text, "Diagram", ["END"])
    if not section:
        return ""
    lower = section.lower()
    if any(p in lower for p in ["not required", "not applicable", "n/a", "no diagram", "not needed"]):
        return ""
    return section.strip()


# =============================================================
# Routes
# =============================================================

@app.get("/")
def home():
    return {
        "success": True,
        "data": "3Qverse AI backend is running",
        "version": "2.0.0",
        "intelligence_layers": 6,
        "gemini_ready": bool(client),
    }


@app.get("/debug")
def debug():
    return {
        "gemini_api_key_set": bool(API_KEY),
        "gemini_client_ready": bool(client),
        "key_prefix": API_KEY[:8] + "..." if API_KEY else "NOT SET",
        "version": "2.0.0",
    }


# ── /ask ──────────────────────────────────────────────────────
@app.post("/ask")
def ask_ai(data: AskRequest):
    if not data.question.strip():
        return {"success": False, "error": "Question cannot be empty"}

    logger.info(f"[/ask] q='{data.question[:60]}' history_len={len(data.history or [])}")

    try:
        history_context = build_history_context(data.history or [])
        prompt = f"""You are 3Q AI, a smart B.Tech learning assistant.
Always use the previous conversation context to understand follow-up questions.
If a student asks "give me code" or "example" without specifying a topic,
look at the previous conversation to know what topic they mean.

{history_context}Student's new question: {data.question}

Answer clearly for a B.Tech student using this format:
1. **Simple Explanation** (2-3 lines)
2. **Real-world Analogy**
3. **Technical Breakdown** (bullet points)
4. **Example** (code if applicable)
5. **Interview Questions** (2 questions)

{CODE_RULE}"""

        result = call_gemini(prompt)
        logger.info(f"[/ask] success, response_len={len(result)}")
        return {"success": True, "data": result}

    except Exception as e:
        logger.error(f"[/ask] FAILED: {type(e).__name__}: {e}")
        return {"success": False, "error": f"AI error: {str(e)}"}


# ── /concept ──────────────────────────────────────────────────
@app.post("/concept")
def explain_concept(data: ConceptRequest):
    if not data.concept.strip():
        return {"success": False, "error": "Concept cannot be empty"}

    logger.info(f"[/concept] concept='{data.concept}'")

    try:
        prompt = f"""You are a B.Tech professor explaining a concept to a first-year student.
Concept: {data.concept}

Return STRICTLY in this format:

**Title:** {data.concept}

**Explanation:**
(2-3 sentence simple explanation)

**Analogy:**
(one real-world analogy that makes it click instantly)

**Technical Breakdown:**
- (point 1)
- (point 2)
- (point 3)

**Code Example:**
(working code in a fenced code block with correct language tag)

**Interview Questions:**
1. (question 1)
2. (question 2)

{CODE_RULE}"""

        result = call_gemini(prompt)
        return {"success": True, "data": result}

    except Exception as e:
        logger.error(f"[/concept] FAILED: {type(e).__name__}: {e}")
        return {"success": False, "error": f"AI error: {str(e)}"}


# ── /study-plan ───────────────────────────────────────────────
@app.post("/study-plan")
def generate_study_plan(data: StudyPlanRequest):
    if not data.tech.strip() or not data.level.strip():
        return {"success": False, "error": "Tech and level cannot be empty"}
    if data.days < 1 or data.days > 60:
        return {"success": False, "error": "Days must be between 1 and 60"}

    logger.info(f"[/study-plan] tech={data.tech} level={data.level} days={data.days}")

    try:
        prompt = f"""Create a {data.days}-day study plan for a B.Tech student.
Technology: {data.tech}, Level: {data.level}

Return STRICTLY as a list (one line per day, no extra text):
Day 1 | Topic Title | What to study today in one sentence
Day 2 | Topic Title | What to study today in one sentence
...Day {data.days} | Topic Title | What to study today in one sentence

Rules: Exactly {data.days} lines, progress basics to advanced, match {data.level} level."""

        raw = call_gemini(prompt)
        days = []
        for line in raw.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) == 3:
                try:
                    day_num = int(parts[0].lower().replace("day", "").strip())
                    days.append({
                        "day": day_num,
                        "title": parts[1],
                        "description": f"{data.level} level: {parts[2]}",
                    })
                except ValueError:
                    continue

        return {"success": True, "data": days}

    except Exception as e:
        logger.error(f"[/study-plan] FAILED: {type(e).__name__}: {e}")
        return {"success": False, "error": f"AI error: {str(e)}"}


# ── /roadmap ──────────────────────────────────────────────────
@app.post("/roadmap")
def generate_roadmap(data: RoadmapRequest):
    if not data.tech.strip() or not data.level.strip():
        return {"success": False, "error": "Tech and level cannot be empty"}

    logger.info(f"[/roadmap] tech={data.tech} level={data.level}")

    try:
        prompt = f"""Create a learning roadmap for a B.Tech student.
Technology: {data.tech}, Level: {data.level}

Return STRICTLY as a numbered list (5-7 steps):
Step 1 – (what to learn and why, one sentence)
...

No extra text. Practical and interview-focused."""

        raw = call_gemini(prompt)
        steps = [
            line.strip()
            for line in raw.strip().split("\n")
            if line.strip() and (
                line.strip()[0].isdigit() or
                line.strip().lower().startswith("step")
            )
        ]
        return {"success": True, "data": steps}

    except Exception as e:
        logger.error(f"[/roadmap] FAILED: {type(e).__name__}: {e}")
        return {"success": False, "error": f"AI error: {str(e)}"}


# ── /code-analyze ─────────────────────────────────────────────
@app.post("/code-analyze")
def analyze_code(data: CodeAnalyzeRequest):
    if not data.code.strip():
        return {"success": False, "error": "Code cannot be empty"}
    if not data.lang.strip():
        return {"success": False, "error": "Language cannot be empty"}

    logger.info(f"[/code-analyze] lang={data.lang} lines={len(data.code.splitlines())}")

    try:
        prompt = f"""You are a senior engineer reviewing a B.Tech student's {data.lang} code.

Code:
```{data.lang}
{data.code}
```

Review format:
## Code Analysis ({data.lang})

**Summary:** (1-2 sentences)

**Observations:**
- (point 1)
- (point 2)
- (point 3)

**Issues Found:**
- (bug/bad practice, or "No major issues found")

**Suggestions:**
1. (improvement + reason)
2. (improvement + reason)
3. (improvement + reason)

**Improved Snippet:**
(corrected code in fenced block)

{CODE_RULE}"""

        result = call_gemini(prompt)
        return {"success": True, "data": result}

    except Exception as e:
        logger.error(f"[/code-analyze] FAILED: {type(e).__name__}: {e}")
        return {"success": False, "error": f"AI error: {str(e)}"}


# ── /exam-answer ──────────────────────────────────────────────
# The crown jewel — full intelligence pipeline
@app.post("/exam-answer")
def generate_exam_answer(data: ExamAnswerRequest):

    # ── Validation ────────────────────────────────────────────
    if not data.question.strip():
        return {"success": False, "error": "Question cannot be empty"}
    if not data.subject.strip():
        return {"success": False, "error": "Subject cannot be empty"}
    if data.marks not in ["5", "10"]:
        return {"success": False, "error": "Marks must be '5' or '10'"}

    start_time = time.time()

    # ══════════════════════════════════════════════════════════
    # INTELLIGENCE PIPELINE — runs before touching Gemini
    # ══════════════════════════════════════════════════════════

    # Layer 1: Detect what TYPE of question this is
    q_type = detect_question_type(data.question)

    # Layer 2: Load subject intelligence
    subject_intel = get_subject_intel(data.subject)

    # Layer 3: Score importance (critical/high/normal)
    importance = score_question_importance(data.question, subject_intel)

    logger.info(
        f"[/exam-answer] subject='{data.subject}' marks={data.marks} "
        f"type={q_type} importance={importance} "
        f"subject_known={bool(subject_intel)} "
        f"q='{data.question[:60]}'"
    )

    try:
        # Layer 4: Build the adaptive prompt
        prompt = build_exam_prompt_strategy(
            question=data.question,
            subject=data.subject,
            marks=data.marks,
            q_type=q_type,
            importance=importance,
            subject_intel=subject_intel
        )

        # Call Gemini with our intelligent prompt
        raw = call_gemini(prompt)

        # Layer 5: Clean the output
        raw = clean_output(raw)

        logger.info(f"[/exam-answer] Gemini response: {len(raw)} chars")

        # Layer 6: Parse sections
        five_mark = extract_section(raw, "5 Mark Answer",  ["10 Mark Answer", "Keywords", "Diagram"])
        ten_mark  = extract_section(raw, "10 Mark Answer", ["Keywords", "Diagram"])
        keywords  = extract_keywords(raw)
        diagram   = extract_diagram(raw)

        # Fallback: if parsing fails, return raw (frontend handles it)
        if not five_mark and not ten_mark:
            logger.warning("[/exam-answer] Extraction failed — returning raw")
            five_mark = raw
            ten_mark  = ""

        # Build metadata for frontend
        response_time_ms = (time.time() - start_time) * 1000
        metadata = build_answer_metadata(
            question=data.question,
            subject=data.subject,
            q_type=q_type,
            importance=importance,
            subject_intel=subject_intel,
            response_time_ms=response_time_ms
        )

        logger.info(
            f"[/exam-answer] ✅ Done in {round(response_time_ms)}ms — "
            f"5mark={len(five_mark)}c 10mark={len(ten_mark)}c "
            f"keywords={len(keywords)} diagram={'yes' if diagram else 'no'}"
        )

        return {
            "success": True,
            "data": {
                "five_mark":  five_mark,
                "ten_mark":   ten_mark,
                "keywords":   keywords,
                "diagram":    diagram,
                "metadata":   metadata,  # bonus: frontend can show tips, importance, etc.
            }
        }

    except Exception as e:
        logger.error(f"[/exam-answer] FAILED: {type(e).__name__}: {e}")
        return {"success": False, "error": f"AI error: {str(e)}"}


# ── /last-night ───────────────────────────────────────────────
@app.post("/last-night")
def last_night_plan(data: LastNightRequest):
    if not data.subject.strip():
        return {"success": False, "error": "Subject cannot be empty"}
    if not data.time.strip():
        return {"success": False, "error": "Time cannot be empty"}

    logger.info(f"[/last-night] subject='{data.subject}' time='{data.time}' topics='{data.topics}'")

    try:
        prompt = f"""
You are a ruthless exam strategist.

A student has ONLY {data.time} to prepare for {data.subject}.

Focus topics: {data.topics if data.topics else 'None'}

Your goal: MAXIMUM MARKS, MINIMUM TIME.

STRICT FORMAT:

🔥 HIGH SCORING TOPICS
- Only high probability topics
- Mention WHY they matter

⏱ EXACT TIME PLAN
- Break into minutes (0–20, 20–40)

❌ SKIP THESE
- Low ROI topics

✍️ HOW TO WRITE ANSWERS
- Keywords examiner expects
- Structure (intro, diagram, points, conclusion)

⚡ MEMORY HACKS
- Mnemonics / shortcuts

🚨 LAST 10 MIN STRATEGY
- What to revise
- What to ignore

🎯 EXPECTED QUESTIONS
- 3–5 probable questions

RULES:
- Bullet points only
- No long explanations
- Practical, exam-focused
"""

        raw = call_gemini(prompt)
        response = clean_output(raw)

        # Split sections for better structure parsing (optional but strong)
        sections = {}
        current_section = None
        for line in response.split("\n"):
            if line.startswith("🔥") or line.startswith("⏱") or line.startswith("❌") or line.startswith("✍️") or line.startswith("⚡") or line.startswith("🚨") or line.startswith("🎯"):
                current_section = line.strip()
                sections[current_section] = []
            elif current_section and line.strip():
                sections[current_section].append(line.strip())

        logger.info(f"[/last-night] success, response_len={len(response)}, sections_parsed={len(sections)}")
        return {
            "success": True,
            "result": response.strip()
        }

    except Exception as e:
        logger.error(f"[/last-night] FAILED: {type(e).__name__}: {e}")
        return {"success": False, "error": f"AI error: {str(e)}"}