import os
import json
import time
import re
import sys
from datetime import datetime, timedelta

import requests
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from langchain_ollama import ChatOllama

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()

# ============================================================
# CONFIG
# ============================================================
FRED_API_KEY = os.getenv("FRED_API_KEY")

if not FRED_API_KEY:
    raise ValueError("Missing FRED_API_KEY in .env")

# Yahoo symbols used for testing.
# These usually work, but Yahoo symbols can occasionally change.
QQQ_SYMBOL = "QQQ"
VIX_SYMBOL = "^VIX"
VVIX_SYMBOL = "^VVIX"
VIX3M_SYMBOL = "^VIX3M"

# FRED series
FRED_SERIES = {
    "DGS2": "DGS2",          # 2-Year Treasury Constant Maturity
    "DGS5": "DGS5",          # 5-Year Treasury Constant Maturity
    "DEXJPUS": "DEXJPUS",    # Japanese Yen to U.S. Dollar Spot Exchange Rate
    "RRPONTSYD": "RRPONTSYD" # Reverse Repo
}

# ============================================================
# LLM
# ============================================================
llm = ChatOllama(
    model="llama3.2:3b",
    base_url="http://localhost:11434/v1",
    temperature=0.2
)

# ============================================================
# HELPERS
# ============================================================
def extract_json(text):
    if text is None:
        return None

    if not isinstance(text, str):
        text = str(text)

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None
    return None


def safe_float(value, default=None):
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def score_to_confidence(price_score, vol_score):
    strength = abs(price_score) + abs(vol_score)
    confidence = min(0.95, 0.50 + strength * 0.05)
    return round(confidence, 2)


def pct_change(new, old):
    if old in (None, 0):
        return None
    return round(((new / old) - 1.0) * 100.0, 4)


def fred_series_observations(series_id, limit=30):
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "sort_order": "desc",
        "limit": limit
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()

    obs = []
    for row in data.get("observations", []):
        value = row.get("value")
        if value == ".":
            continue
        obs.append({
            "date": row.get("date"),
            "value": float(value)
        })
    return obs


# ============================================================
# DATA FETCH
# ============================================================
def fetch_qqq_snapshot():
    """
    Pull QQQ intraday data from yfinance.
    Uses 60m bars over the last ~10 days so we can estimate:
    - current price
    - today's high/low
    - weekly high/low
    """
    hist = yf.Ticker(QQQ_SYMBOL).history(period="10d", interval="60m", auto_adjust=False)

    if hist.empty:
        raise ValueError("No QQQ data returned from yfinance")

    hist = hist.dropna().copy()
    if hist.empty:
        raise ValueError("QQQ history is empty after dropna")

    hist.index = pd.to_datetime(hist.index)

    latest_ts = hist.index[-1]
    latest_close = safe_float(hist["Close"].iloc[-1])

    # "Today" based on the date of the latest bar returned
    current_day = latest_ts.date()
    day_mask = hist.index.date == current_day
    day_df = hist.loc[day_mask]

    daily_high = safe_float(day_df["High"].max())
    daily_low = safe_float(day_df["Low"].min())

    # Approximate "weekly" with the last 5 trading dates present in the frame
    unique_days = sorted(pd.Series(hist.index.date).unique())
    last_5_days = set(unique_days[-5:])
    week_df = hist.loc[[d in last_5_days for d in hist.index.date]]

    weekly_high = safe_float(week_df["High"].max())
    weekly_low = safe_float(week_df["Low"].min())

    # Simple structure flags using the last 3 completed hourly highs/lows
    recent = hist.tail(4)
    higher_highs = False
    higher_lows = False
    if len(recent) >= 3:
        highs = recent["High"].tail(3).tolist()
        lows = recent["Low"].tail(3).tolist()
        higher_highs = highs[2] > highs[1] > highs[0]
        higher_lows = lows[2] > lows[1] > lows[0]

    return {
        "last": latest_close,
        "daily_high": daily_high,
        "daily_low": daily_low,
        "weekly_high": weekly_high,
        "weekly_low": weekly_low,
        "higher_highs": bool(higher_highs),
        "higher_lows": bool(higher_lows),
        "as_of": str(latest_ts)
    }


def fetch_vol_snapshot():
    """
    Pull ^VIX, ^VVIX, ^VIX3M from yfinance.
    Uses daily bars for regime testing.
    """
    symbols = [VIX_SYMBOL, VVIX_SYMBOL, VIX3M_SYMBOL]
    out = {}

    for symbol in symbols:
        hist = yf.Ticker(symbol).history(period="3mo", interval="1d", auto_adjust=False)
        hist = hist.dropna()
        if hist.empty:
            out[symbol] = {"last": None, "prev_5d": None}
            continue

        last = safe_float(hist["Close"].iloc[-1])
        prev_5d = safe_float(hist["Close"].iloc[-6]) if len(hist) >= 6 else None

        out[symbol] = {
            "last": last,
            "prev_5d": prev_5d
        }

    vvix_trend = "NEUTRAL"
    vvix_last = out[VVIX_SYMBOL]["last"]
    vvix_prev_5d = out[VVIX_SYMBOL]["prev_5d"]

    if vvix_last is not None and vvix_prev_5d is not None:
        if vvix_last > vvix_prev_5d:
            vvix_trend = "RISING"
        elif vvix_last < vvix_prev_5d:
            vvix_trend = "FALLING"

    return {
        "vix_front": out[VIX_SYMBOL]["last"],
        "vix_3m": out[VIX3M_SYMBOL]["last"],
        "vvix": vvix_last,
        "vvix_prev_5d": vvix_prev_5d,
        "vvix_trend": vvix_trend
    }


def fetch_macro_snapshot():
    dgs2 = fred_series_observations(FRED_SERIES["DGS2"], limit=10)
    dgs5 = fred_series_observations(FRED_SERIES["DGS5"], limit=10)
    dexjpus = fred_series_observations(FRED_SERIES["DEXJPUS"], limit=10)
    rrp = fred_series_observations(FRED_SERIES["RRPONTSYD"], limit=25)

    latest_dgs2 = dgs2[0]["value"] if dgs2 else None
    latest_dgs5 = dgs5[0]["value"] if dgs5 else None
    spread_2y_5y_bps = None
    change_5d_bps = None

    if latest_dgs2 is not None and latest_dgs5 is not None:
        current_spread = latest_dgs5 - latest_dgs2
        spread_2y_5y_bps = round(current_spread * 100.0, 2)

        if len(dgs2) >= 6 and len(dgs5) >= 6:
            old_spread = dgs5[5]["value"] - dgs2[5]["value"]
            change_5d_bps = round((current_spread - old_spread) * 100.0, 2)

    latest_fx = dexjpus[0]["value"] if dexjpus else None
    fx_5d = dexjpus[5]["value"] if len(dexjpus) >= 6 else None
    fx_daily_change = pct_change(latest_fx, dexjpus[1]["value"]) if len(dexjpus) >= 2 else None
    fx_5d_change = pct_change(latest_fx, fx_5d)

    latest_rrp = rrp[0]["value"] if rrp else None
    rrp_5d_old = rrp[5]["value"] if len(rrp) >= 6 else None
    rrp_20d_old = rrp[20]["value"] if len(rrp) >= 21 else None

    return {
        "rates": {
            "spread_2y_5y_bps": spread_2y_5y_bps,
            "change_5d_bps": change_5d_bps
        },
        "fx": {
            "usdjpy_proxy": latest_fx,
            "daily_change_pct": fx_daily_change,
            "change_5d_pct": fx_5d_change
        },
        "liquidity": {
            "rrp_balance_bil": latest_rrp,
            "change_5d_bil": round(latest_rrp - rrp_5d_old, 2) if latest_rrp is not None and rrp_5d_old is not None else None,
            "change_20d_bil": round(latest_rrp - rrp_20d_old, 2) if latest_rrp is not None and rrp_20d_old is not None else None
        }
    }


# ============================================================
# FEATURE INTERPRETATION
# ============================================================
def interpret_qqq_structure(qqq):
    last_price = qqq["last"]
    daily_high = qqq["daily_high"]
    daily_low = qqq["daily_low"]
    weekly_high = qqq["weekly_high"]
    weekly_low = qqq["weekly_low"]
    higher_highs = qqq.get("higher_highs", False)
    higher_lows = qqq.get("higher_lows", False)

    if None in [last_price, daily_high, daily_low, weekly_high, weekly_low]:
        return "NEUTRAL", 0

    daily_range = max(daily_high - daily_low, 0.0001)
    weekly_range = max(weekly_high - weekly_low, 0.0001)

    daily_pos = (last_price - daily_low) / daily_range
    weekly_pos = (last_price - weekly_low) / weekly_range

    if daily_pos >= 0.6 and weekly_pos >= 0.6 and higher_highs and higher_lows:
        return "BULLISH", 2

    if daily_pos <= 0.4 and weekly_pos <= 0.4 and (not higher_highs) and (not higher_lows):
        return "BEARISH", -2

    if daily_pos > 0.5 and weekly_pos > 0.5:
        return "LEAN_BULLISH", 1

    if daily_pos < 0.5 and weekly_pos < 0.5:
        return "LEAN_BEARISH", -1

    return "NEUTRAL", 0


def interpret_volatility(vol):
    vix_front = vol["vix_front"]
    vix_3m = vol["vix_3m"]
    vvix = vol["vvix"]
    vvix_trend = str(vol["vvix_trend"]).upper()

    if vix_front is None or vix_3m is None:
        return {
            "vix_term_structure": "UNKNOWN",
            "vvix_signal": "UNKNOWN",
            "vol_bias": "NEUTRAL",
            "vol_score": 0
        }

    if vix_3m > vix_front:
        term_signal = "CONTANGO"
        term_score = -2
    elif vix_front > vix_3m:
        term_signal = "BACKWARDATION"
        term_score = 2
    else:
        term_signal = "FLAT"
        term_score = 0

    vvix_signal = "NEUTRAL"
    vvix_score = 0

    if vvix is not None:
        if vvix_trend == "FALLING" and vvix < 100:
            vvix_signal = "CALMING"
            vvix_score = -1
        elif vvix_trend == "RISING" and vvix >= 100:
            vvix_signal = "STRESS_RISING"
            vvix_score = 1

    total_vol_score = term_score + vvix_score

    if total_vol_score <= -2:
        vol_bias = "BEARISH"
    elif total_vol_score >= 2:
        vol_bias = "BULLISH"
    else:
        vol_bias = "NEUTRAL"

    return {
        "vix_term_structure": term_signal,
        "vvix_signal": vvix_signal,
        "vol_bias": vol_bias,
        "vol_score": total_vol_score
    }


def interpret_curve(rates):
    spread = rates["spread_2y_5y_bps"]
    change_5d = rates["change_5d_bps"]

    if spread is None or change_5d is None:
        return "NEUTRAL", 0

    if change_5d >= 3:
        return "RISK_ON_STEEPENING", 1
    if change_5d <= -3:
        return "RISK_OFF_FLATTENING", -1

    if spread > 0:
        return "MILD_RISK_ON", 1
    if spread < 0:
        return "MILD_RISK_OFF", -1

    return "NEUTRAL", 0


def interpret_yen_carry(fx):
    daily_change = fx["daily_change_pct"]
    change_5d = fx["change_5d_pct"]

    if daily_change is None or change_5d is None:
        return "NEUTRAL", 0

    # DEXJPUS is yen per dollar proxy handling can vary; for testing we use your sign rules as-is.
    if daily_change > 0 and change_5d > 0.5:
        return "RISK_ON", 1
    if daily_change < 0 and change_5d < -0.5:
        return "RISK_OFF", -1
    return "NEUTRAL", 0


def interpret_liquidity(liq):
    change_5d = liq["change_5d_bil"]
    change_20d = liq["change_20d_bil"]

    if change_5d is None or change_20d is None:
        return "NEUTRAL", 0

    if change_5d < 0 and change_20d < 0:
        return "LIQUIDITY_SUPPORTIVE", 1
    if change_5d > 0 and change_20d > 0:
        return "LIQUIDITY_DRAINING", -1
    return "NEUTRAL", 0


def build_market_inputs():
    qqq = fetch_qqq_snapshot()
    vol = fetch_vol_snapshot()
    macro = fetch_macro_snapshot()

    qqq_signal, qqq_score = interpret_qqq_structure(qqq)
    vol_info = interpret_volatility(vol)
    curve_signal, curve_score = interpret_curve(macro["rates"])
    yen_signal, yen_score = interpret_yen_carry(macro["fx"])
    liquidity_signal, liquidity_score = interpret_liquidity(macro["liquidity"])

    price_score = qqq_score + curve_score + yen_score + liquidity_score
    vol_score = vol_info["vol_score"]

    if price_score >= 2:
        price_bias = "BULLISH"
    elif price_score <= -2:
        price_bias = "BEARISH"
    else:
        price_bias = "NEUTRAL"

    if vol_score >= 2:
        vol_bias = "BULLISH"
    elif vol_score <= -2:
        vol_bias = "BEARISH"
    else:
        vol_bias = "NEUTRAL"

    if price_bias == "BULLISH" and vol_bias == "BEARISH":
        regime = "BULLISH_PRICE_BEARISH_VOL"
    elif price_bias == "BEARISH" and vol_bias == "BULLISH":
        regime = "BEARISH_PRICE_BULLISH_VOL"
    elif price_bias == "NEUTRAL" and vol_bias == "NEUTRAL":
        regime = "NEUTRAL"
    else:
        regime = "MIXED"

    confidence = score_to_confidence(price_score, vol_score)

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "qqq": {
            **qqq,
            "signal": qqq_signal
        },
        "volatility": {
            **vol,
            "vix_term_structure": vol_info["vix_term_structure"],
            "vvix_signal": vol_info["vvix_signal"]
        },
        "rates": {
            **macro["rates"],
            "signal": curve_signal
        },
        "fx": {
            **macro["fx"],
            "signal": yen_signal
        },
        "liquidity": {
            **macro["liquidity"],
            "signal": liquidity_signal
        },
        "precomputed_scores": {
            "price_score": price_score,
            "vol_score": vol_score,
            "preliminary_price_bias": price_bias,
            "preliminary_vol_bias": vol_bias,
            "preliminary_regime": regime,
            "preliminary_confidence": confidence
        }
    }


# ============================================================
# AGENT
# ============================================================
market_analyst = Agent(
    role="Market Analyst",
    goal=(
        "Classify the current market regime using structured QQQ, volatility, "
        "rates, FX, and liquidity inputs. Output only valid JSON."
    ),
    backstory=(
        "You are the Market Analyst for an AI wealth management system. "
        "You do not execute trades. You do not call tools. "
        "You only interpret the structured inputs you are given."
    ),
    llm=llm,
    verbose=True,
    max_iterations=1
)


def build_task(market_inputs):
    description = f"""
You are given structured market inputs below.

Your job:
1. Determine the current price bias: BULLISH, BEARISH, or NEUTRAL
2. Determine the current volatility bias: BULLISH, BEARISH, or NEUTRAL
3. Determine the regime:
   - BULLISH_PRICE_BEARISH_VOL
   - BEARISH_PRICE_BULLISH_VOL
   - MIXED
   - NEUTRAL
4. Output ONLY valid JSON
5. Do not include markdown, comments, or explanation outside the JSON

Required output schema:
{{
  "agent": "market_analyst",
  "event_id": "<string>",
  "price_bias": "BULLISH|BEARISH|NEUTRAL",
  "vol_bias": "BULLISH|BEARISH|NEUTRAL",
  "regime": "BULLISH_PRICE_BEARISH_VOL|BEARISH_PRICE_BULLISH_VOL|MIXED|NEUTRAL",
  "confidence": <float between 0 and 1>,
  "factors": {{
    "qqq_structure": "<string>",
    "vix_term_structure": "<string>",
    "vvix": "<string>",
    "curve_signal": "<string>",
    "yen_usd_signal": "<string>",
    "rrp_liquidity_signal": "<string>"
  }},
  "summary": "<short string>"
}}

Use the precomputed scores as anchors, but apply judgment to create the final JSON.

Market Inputs:
{json.dumps(market_inputs, indent=2)}
""".strip()

    return Task(
        description=description,
        agent=market_analyst,
        expected_output="Strict JSON only."
    )


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    try:
        market_inputs = build_market_inputs()

        print("\n[STRUCTURED MARKET INPUTS]")
        print(json.dumps(market_inputs, indent=2))

        market_task = build_task(market_inputs)
        crew = Crew(
            agents=[market_analyst],
            tasks=[market_task],
            verbose=True
        )

        crew_output = crew.kickoff()
        raw_result = crew_output.raw if hasattr(crew_output, "raw") else str(crew_output)

        print("\n[RAW OUTPUT]")
        print(raw_result)

        decision = extract_json(raw_result)
        if not decision:
            raise ValueError("Could not parse JSON from Market Analyst output")

        print("\n[PARSED DECISION]")
        print(json.dumps(decision, indent=2))

    except Exception as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)