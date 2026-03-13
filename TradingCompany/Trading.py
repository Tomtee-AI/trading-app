import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from langchain_ollama import ChatOllama
from crewai_tools import tool
import yfinance as yf
import talib
from polygon import RESTClient  # For options data
import pandas as pd
import sqlite3
from apscheduler.schedulers.background import BackgroundScheduler

load_dotenv()
polygon_client = RESTClient(api_key=os.getenv('POLYGON_API_KEY'))
llm = ChatOllama(model="llama3.2:3b-instruct", temperature=0.3)  # Local LLM

# Database for logging trades (for taxes/protection)
conn = sqlite3.connect('trading_log.db')
conn.execute('''CREATE TABLE IF NOT EXISTS trades (id INTEGER PRIMARY KEY, timestamp TEXT, symbol TEXT, action TEXT, size REAL, price REAL, risk_reward REAL, notes TEXT)''')

# Custom Tools (CEO uses these for analysis)
@tool
def get_unusual_options_activity(symbol: str) -> str:
    """Detect unusual options volume (high above average) for potential sentiment."""
    options = polygon_client.get_snapshot_options_chain(symbol)
    avg_vol = pd.DataFrame([o.volume for o in options.options]).mean()
    current_vol = sum(o.volume for o in options.options)
    if current_vol > 2 * avg_vol:  # Threshold for 'unusual'
        return f"Unusual activity detected: {current_vol} vs avg {avg_vol}. Bullish if calls dominate."
    return "No unusual activity."

@tool
def calculate_expected_move(symbol: str, expiration: str) -> str:
    """Calculate expected move from ATM straddle (85% of premium)."""
    chain = polygon_client.get_snapshot_options_chain(symbol, expiration=expiration)
    atm_strike = yf.Ticker(symbol).info['regularMarketPrice']
    straddle = next((o for o in chain.options if o.strike == atm_strike and o.option_type == 'call'), None).ask + \
               next((o for o in chain.options if o.strike == atm_strike and o.option_type == 'put'), None).ask
    return f"Expected move: ±{0.85 * straddle}%"

@tool
def commodity_correlations(commodities: list) -> str:
    """Calculate correlations; identify cheap/expensive for bias (e.g., via RSI). Express with options."""
    data = yf.download(commodities, period='1y')['Adj Close']
    corr = data.corr().to_string()
    rsi = {c: talib.RSI(data[c])[-1] for c in commodities}  # RSI <30 cheap, >70 expensive
    bias = {c: 'Cheap (buy calls)' if r < 30 else 'Expensive (buy puts)' if r > 70 else 'Neutral' for c, r in rsi.items()}
    return f"Correlations:\n{corr}\nBias: {bias}"

@tool
def calculate_crack_spread() -> str:
    """3-2-1 Crack Spread for refiner margins (high = profitable refiners)."""
    cl = yf.Ticker('CL=F').history(period='1d')['Close'][0]  # Crude
    rb = yf.Ticker('RB=F').history(period='1d')['Close'][0]  # Gasoline
    ho = yf.Ticker('HO=F').history(period='1d')['Close'][0]  # Heating Oil
    spread = (2 * rb + ho - 3 * cl) / 3
    return f"3-2-1 Crack Spread: ${spread:.2f}/bbl (high >$15 indicates strong refiner margins)"

@tool
def vix_term_structure() -> str:
    """Fetch VIX curve (contango/backwardation) for volatility bias."""
    vix_futures = ['VXH24', 'VXJ24']  # Example symbols; use real from CBOE
    data = yf.download(vix_futures, period='1d')['Adj Close']
    if data.iloc[0,1] > data.iloc[0,0]:  # Contango
        return "Contango: Expect volatility rise; consider long VIX calls."
    return "Backwardation: Volatility peaking; consider shorts."

@tool
def delta_weight_portfolio(portfolio: dict, benchmark: str = 'SPY') -> str:
    """Beta-weighted delta for overall bias (neutral ~0)."""
    spy_beta = 1.0  # Benchmark
    total_delta = sum(pos['shares'] * pos['beta'] * spy_beta for pos in portfolio.values())  # Example calc
    return f"Portfolio delta-weighted to {benchmark}: {total_delta} (positive = bullish bias)"

@tool
def tranche_position(symbol: str, total_size: float, tranches: int = 3) -> str:
    """Enter position in tranches (e.g., 3 slices) to average in."""
    tranche_size = total_size / tranches
    return f"Tranching plan for {symbol}: {tranches} entries of {tranche_size} each over time/price levels."

# Agents
ceo = Agent(
    role='CEO',
    goal='Make informed trade decisions to grow/protect account using advanced concepts like unusual options activity (high volume signals sentiment), earnings trades (buy underpriced volatility pre-earnings), expected moves (from ATM straddle for range), commodity correlations (identify cheap/expensive for options bias), 3-2-1 crack spread (refiner margins proxy), volatility curves (VIX contango for vol plays), delta weighting (neutralize portfolio bias), and tranching (gradual entry).',
    backstory='Expert trader analyzing markets for optimal entries/exits.',
    tools=[get_unusual_options_activity, calculate_expected_move, commodity_correlations, calculate_crack_spread, vix_term_structure, delta_weight_portfolio, tranche_position],
    llm=llm,
    verbose=True
)

coder = Agent(
    role='Coder',
    goal='Implement approved trades via code (simulated execution).',
    backstory='Builds algorithmic trade logic.',
    llm=llm,
    verbose=True
)

intern = Agent(
    role='Intern',
    goal='Research market data/symbols.',
    backstory='Gathers real-time info.',
    llm=llm,
    verbose=True
)

restraint = Agent(
    role='Restraint',
    goal='Enforce ethics/risk: ≤10% portfolio per monthly expiration; risk $1 to make ≥$5 (5:1 reward:risk); no unethical trades; consider taxes (log for 15-20% capital gains).',
    backstory='Compliance checker vetoing high-risk proposals.',
    llm=llm,
    verbose=True
)

# Tasks (sequential workflow)
research_task = Task(description='Research symbols (e.g., SPY, CL=F) for opportunities using tools.', agent=intern)
strategy_task = Task(description='Propose trades based on concepts (e.g., earnings volatility, crack spread).', agent=ceo)
ethics_task = Task(description='Review proposal: Check allocation ≤10% per expiration, 5:1 risk-reward, tax log.', agent=restraint, expected_output='Approved/Vetoed with reason')
code_task = Task(description='If approved, code/simulate execution (e.g., buy call via Alpaca sim). Log to DB.', agent=coder)

# Crew
trading_crew = Crew(agents=[intern, ceo, restraint, coder], tasks=[research_task, strategy_task, ethics_task, code_task], verbose=2)

# Scheduler (run daily)
def run_crew():
    result = trading_crew.kickoff(inputs={'portfolio_value': 100000})  # Example input
    print(result)
    # Log to DB (tax tracking)
    conn.execute("INSERT INTO trades (...) VALUES (...)")  # Add actual trade data
    conn.commit()

scheduler = BackgroundScheduler()
scheduler.add_job(run_crew, 'cron', hour=9)  # Daily at 9 AM
scheduler.start()

if __name__ == "__main__":
    run_crew()  # Initial run
