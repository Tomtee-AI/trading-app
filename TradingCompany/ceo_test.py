import os
import json
import time
import re
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
import smtplib
from email.mime.text import MIMEText
import sys

sys.stdout.reconfigure(encoding='utf-8')
load_dotenv()

# ─────────────────────────────────────────────
# LLM (Ollama Local)
# ─────────────────────────────────────────────
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="llama3.2:3b",
    base_url="http://localhost:11434/v1",
    temperature=0.2  # lower = more deterministic
)

# ─────────────────────────────────────────────
# Persistent State (Idempotency)
# ─────────────────────────────────────────────
STATE_FILE = "event_state.json"

def load_state():
    if not os.path.exists(STATE_FILE):
        return {}
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except:
        return {}

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

def already_processed(event_id):
    state = load_state()
    return state.get(event_id, False)

def mark_processed(event_id):
    state = load_state()
    state[event_id] = True
    save_state(state)

# ─────────────────────────────────────────────
# Email Execution (Deterministic)
# ─────────────────────────────────────────────
def actually_send_email(subject: str, body: str):
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = os.getenv('SMTP_USER')
    msg['To'] = os.getenv('USER_EMAIL')

    with smtplib.SMTP(os.getenv('SMTP_SERVER'), int(os.getenv('SMTP_PORT'))) as server:
        server.starttls()
        server.login(os.getenv('SMTP_USER'), os.getenv('SMTP_PASS'))
        server.sendmail(os.getenv('SMTP_USER'), os.getenv('USER_EMAIL'), msg.as_string())

    print(f"[EMAIL SENT] {subject}")

# ─────────────────────────────────────────────
# Robust JSON Extractor (handles messy LLM output)
# ─────────────────────────────────────────────
def extract_json(text):
    try:
        return json.loads(text)
    except:
        # Try to extract JSON block using regex
        match = re.search(r'\{.*\}', str(text), re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except:
                pass
    return None

# ─────────────────────────────────────────────
# CEO Agent (NO TOOLS)
# ─────────────────────────────────────────────
ceo = Agent(
    role='CEO',
    goal='Produce structured JSON decisions for system execution.',
    backstory=(
        'You are a deterministic trading system controller. '
        'You NEVER call tools. You ONLY output valid JSON.'
    ),
    llm=llm,
    verbose=True,
    max_iterations=1
)

# ─────────────────────────────────────────────
# Task (STRICT JSON ONLY)
# ─────────────────────────────────────────────
startup_task = Task(
    description=(
        'Output ONLY a valid JSON object.\n\n'
        'Do NOT include explanations, text, or markdown.\n\n'
        'Format:\n'
        '{\n'
        '  "event_id": "startup_v1",\n'
        '  "notify": true,\n'
        '  "subject": "CEO Agent Startup Confirmation",\n'
        '  "body": "Hello Tomtee.eth, the CEO agent is now up and running. Ready for trading decisions."\n'
        '}\n'
    ),
    agent=ceo,
    expected_output='Strict JSON only.'
)

# ─────────────────────────────────────────────
# Crew
# ─────────────────────────────────────────────
crew = Crew(
    agents=[ceo],
    tasks=[startup_task],
    verbose=True
)

# ─────────────────────────────────────────────
# Main Execution
# ─────────────────────────────────────────────
if __name__ == "__main__":
    crew_output = crew.kickoff()

    # Extract raw text safely
    raw_result = crew_output.raw if hasattr(crew_output, "raw") else str(crew_output)

    print("\n[RAW OUTPUT]\n", raw_result)

    # Parse JSON robustly
    decision = extract_json(raw_result)

    if not decision:
        print("[ERROR] Could not extract valid JSON from LLM output.")
        decision = {"notify": False}

    print("\n[PARSED DECISION]\n", decision)

    # Extract fields safely
    event_id = decision.get("event_id", f"event_{int(time.time())}")
    should_notify = decision.get("notify", False)
    subject = decision.get("subject", "No Subject")
    body = decision.get("body", "")

    # Idempotent execution
    if should_notify:
        if already_processed(event_id):
            print(f"[SKIP] Event already processed: {event_id}")
        else:
            actually_send_email(subject, body)
            mark_processed(event_id)
    else:
        print("[INFO] No notification requested.")
