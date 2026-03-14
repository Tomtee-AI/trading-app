import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM  # Add LLM import here
from crewai.tools import tool
import smtplib
from email.mime.text import MIMEText
import sys
sys.stdout.reconfigure(encoding='utf-8')  # Force UTF-8 output

load_dotenv()

# Use CrewAI's LLM for Ollama (no OpenAI needed)
llm = LLM(
    model="llama3.2:3b",  # Or "llama3.2:3b-q4_0" if you pulled that
    base_url="http://localhost:11434",  # Ollama server URL
    temperature=0.3
)

# Custom Tool for Email (CEO uses this)
@tool
def send_email_to_user(subject: str, body: str) -> str:
    """Send communication email to Tomtee (user)."""
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = os.getenv('SMTP_USER')
    msg['To'] = os.getenv('USER_EMAIL')
    
    with smtplib.SMTP(os.getenv('SMTP_SERVER'), os.getenv('SMTP_PORT')) as server:
        server.starttls()
        server.login(os.getenv('SMTP_USER'), os.getenv('SMTP_PASS'))
        server.sendmail(os.getenv('SMTP_USER'), os.getenv('USER_EMAIL'), msg.as_string())
    return "Email sent successfully."

# CEO Agent (Isolated for test)
ceo = Agent(
    role='CEO',
    goal='Synthesize data from all other agents, make a final yes/no decision on trades, and provide communications to Tomtee (the user).',
    backstory='Expert trader who reviews inputs from Intern (research), Coder (implementation), and Restraint (compliance). Decides yes/no based on overall viability using advanced concepts like unusual options activity, earnings trades/underpriced volatility, expected moves, commodity correlations, 3-2-1 crack spread, volatility curves/VIX term structure, delta weighting, and tranching. Communicates decisions/summaries to Tomtee via email.',
    tools=[send_email_to_user],
    llm=llm,
    verbose=True
)

# Single Task: Send startup email
startup_task = Task(
    description='Send an email to Tomtee confirming the CEO agent is up and running. Use the email tool with subject "CEO Agent Startup Confirmation" and body "Hello Tomtee.eth, the CEO agent is now up and running. Ready for trading decisions."',
    agent=ceo,
    expected_output='Confirmation that email was sent.'
)

# Minimal Crew (just CEO)
ceo_crew = Crew(agents=[ceo], tasks=[startup_task], verbose=True)

if __name__ == "__main__":
    result = ceo_crew.kickoff()
    print(result)