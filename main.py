# =============================================================
# 3Qverse AI Backend — main.py
# FIX 1: /ask now accepts full conversation history
# FIX 2: Prompts force fenced code blocks with language tags
# Run: uvicorn main:app --reload
# =============================================================

import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from google import genai
from dotenv import load_dotenv

# ── Load env ─────────────────────────────────────────────────
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise Exception("❌ GEMINI_API_KEY not found in .env file")

# ── Logging ───────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="[3Q AI] %(asctime)s — %(message)s")
logger = logging.getLogger("3qverse")

# ── Gemini Client ─────────────────────────────────────────────
client = genai.Client(api_key=API_KEY)

# ── App ───────────────────────────────────────────────────────
app = FastAPI(title="3Qverse AI Backend")

# ── CORS ──────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── CODE BLOCK RULE ───────────────────────────────────────────
CODE_RULE = """
CRITICAL FORMATTING RULES — follow exactly:
- Every code example MUST be inside a fenced code block.
- ALWAYS put the language name right after the opening backticks on the SAME line.
- Correct format:
  ```python
  # your code here
  ```
- NEVER write ``` alone without a language name.
- NEVER write the code outside of a fenced block.
- Supported language tags: python, javascript, typescript, java, c, cpp, go, sql, bash, html, css, json
"""

# =============================================================
# Models
# =============================================================

# A single message in the conversation
class ChatMessage(BaseModel):
    role: str        # "user" or "ai"
    content: str

# FIX: /ask now accepts full history, not just one question
class AskRequest(BaseModel):
    question: str
    history: Optional[List[ChatMessage]] = []   # ← previous messages

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

# =============================================================
# Gemini Helper
# =============================================================

def call_gemini(prompt: str) -> str:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return response.text if response.text else "No response generated"

def build_history_text(history: List[ChatMessage]) -> str:
    """Converts chat history list into a readable string for the prompt."""
    if not history:
        return ""
    lines = ["Previous conversation:"]
    for msg in history[-10:]:  # only last 10 messages to stay within token limits
        label = "Student" if msg.role == "user" else "3Q AI"
        lines.append(f"{label}: {msg.content}")
    return "\n".join(lines) + "\n\n"

# =============================================================
# Routes
# =============================================================

@app.get("/")
def home():
    return {"success": True, "data": "3Qverse AI backend is running"}


# ── /ask — Chat with full conversation memory ─────────────────
@app.post("/ask")
def ask_ai(data: AskRequest):
    if not data.question.strip():
        return {"success": False, "error": "Question cannot be empty"}

    logger.info(f"[/ask] history_len={len(data.history or [])} q={data.question[:60]}")

    try:
        # Build history context so Gemini remembers what was discussed
        history_context = build_history_text(data.history or [])

        prompt = f"""You are 3Q AI, a smart B.Tech learning assistant.
You are in an ongoing conversation with a student. Always use the previous conversation context to understand follow-up questions.

{history_context}Student's new question: {data.question}

Answer clearly for a B.Tech student using this format:
1. **Simple Explanation** (2-3 lines)
2. **Real-world Analogy**
3. **Technical Breakdown** (bullet points)
4. **Example** (code if applicable)
5. **Interview Questions** (2 questions)

{CODE_RULE}
"""
        return {"success": True, "data": call_gemini(prompt)}

    except Exception as e:
        logger.error(f"[/ask] ERROR: {e}")
        return {"success": False, "error": "AI request failed. Try again."}


# ── /concept ──────────────────────────────────────────────────
@app.post("/concept")
def explain_concept(data: ConceptRequest):
    if not data.concept.strip():
        return {"success": False, "error": "Concept cannot be empty"}

    logger.info(f"[/concept] {data.concept}")

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

{CODE_RULE}
"""
        return {"success": True, "data": call_gemini(prompt)}

    except Exception as e:
        logger.error(f"[/concept] ERROR: {e}")
        return {"success": False, "error": "AI request failed. Try again."}


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
Technology: {data.tech}
Level: {data.level}

Return STRICTLY as a list (one line per day):
Day 1 | Topic Title | What to study today in one sentence
Day 2 | Topic Title | What to study today in one sentence
...
Day {data.days} | Topic Title | What to study today in one sentence

Rules:
- Exactly {data.days} lines, nothing else
- Progress from basics to advanced
- Match difficulty to {data.level} level
"""
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
        logger.error(f"[/study-plan] ERROR: {e}")
        return {"success": False, "error": "AI request failed. Try again."}


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

No extra text outside the list. Practical and interview-focused.
"""
        raw = call_gemini(prompt)
        steps = [
            line.strip()
            for line in raw.strip().split("\n")
            if line.strip() and (line.strip()[0].isdigit() or line.strip().lower().startswith("step"))
        ]
        return {"success": True, "data": steps}

    except Exception as e:
        logger.error(f"[/roadmap] ERROR: {e}")
        return {"success": False, "error": "AI request failed. Try again."}


# ── /code-analyze ─────────────────────────────────────────────
@app.post("/code-analyze")
def analyze_code(data: CodeAnalyzeRequest):
    if not data.code.strip():
        return {"success": False, "error": "Code cannot be empty"}
    if not data.lang.strip():
        return {"success": False, "error": "Language cannot be empty"}

    logger.info(f"[/code-analyze] lang={data.lang} lines={len(data.code.splitlines())}")

    try:
        prompt = f"""You are a senior software engineer reviewing a B.Tech student's code.
Language: {data.lang}

Code:
```{data.lang}
{data.code}
```

Review in this format:

## Code Analysis ({data.lang})

**Summary:**
(1-2 sentence overview)

**Observations:**
- (observation 1)
- (observation 2)
- (observation 3)

**Issues Found:**
- (bug or bad practice, or "No major issues found")

**Suggestions:**
1. (improvement with reason)
2. (improvement with reason)
3. (improvement with reason)

**Improved Snippet:**
(corrected code in a fenced block with correct language tag)

{CODE_RULE}
"""
        return {"success": True, "data": call_gemini(prompt)}

    except Exception as e:
        logger.error(f"[/code-analyze] ERROR: {e}")
        return {"success": False, "error": "AI request failed. Try again."}