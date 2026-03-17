# =============================================================
# 3Qverse AI Backend — main.py
# Run locally:  uvicorn main:app --reload
# Deploy:       Push to Render, set GEMINI_API_KEY env var
# =============================================================

import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
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

# Log startup state so you can see it in Render logs
logger.info(f"Starting 3Qverse backend...")
logger.info(f"GEMINI_API_KEY present: {bool(API_KEY)}")
logger.info(f"GEMINI_API_KEY length: {len(API_KEY) if API_KEY else 0}")

# ── Gemini Client ─────────────────────────────────────────────
if not API_KEY:
    logger.error("❌ GEMINI_API_KEY not set! Set it in Render Environment Variables.")
    client = None
else:
    client = genai.Client(api_key=API_KEY)
    logger.info("✅ Gemini client initialized")

# ── App ───────────────────────────────────────────────────────
app = FastAPI(title="3Qverse AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── CODE RULE (injected into every prompt) ────────────────────
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
# Models
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

# =============================================================
# Helpers
# =============================================================

def call_gemini(prompt: str) -> str:
    """Call Gemini and return text. Raises on failure."""
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

# =============================================================
# Routes
# =============================================================

@app.get("/")
def home():
    return {
        "success": True,
        "data": "3Qverse AI backend is running",
        "gemini_ready": bool(client),
    }

# ── /debug — shows env state (remove after fixing) ────────────
@app.get("/debug")
def debug():
    """Temporary debug endpoint — remove after confirming setup."""
    return {
        "gemini_api_key_set": bool(API_KEY),
        "gemini_client_ready": bool(client),
        "key_prefix": API_KEY[:8] + "..." if API_KEY else "NOT SET",
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
            if line.strip() and (line.strip()[0].isdigit() or line.strip().lower().startswith("step"))
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