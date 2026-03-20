import os
import json
import sys
import time
import pickle
from datetime import datetime
from pathlib import Path

import requests
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

# Optional LLM summary layer (now much safer and faster)
USE_LLM_SUMMARY = True

if USE_LLM_SUMMARY:
    from langchain_ollama import ChatOllama

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()

FRED_API_KEY = os.getenv("FRED_API_KEY")
if not FRED_API_KEY:
    raise ValueError("Missing FRED_API_KEY in .env")

# ==================== CONFIG ====================
QQQ_SYMBOL = "QQQ"
VIX_SYMBOL = "^VIX"
VVIX_SYMBOL = "^VVIX"
VIX3M_SYMBOL = "^VIX3M"

CACHE_FILE = Path("market_cache.pkl")
CACHE_TTL_SECONDS = 300  # 5 minutes


def safe_float(value, default=None):
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def pct_change(new, old):
    if old in (None, 0):
        return None
    return round(((new / old) - 1.0) * 100.0, 4)


def score_to_confidence(price_score, vol_score):
    strength = abs(price_score) + abs(vol_score)
    confidence = min(0.95, 0.50 + strength * 0.05)
    return round(confidence, 2)


# ==================== CACHING ====================
def load_cache():
    if CACHE_FILE.exists() and time.time() - CACHE_FILE.stat().st_mtime < CACHE_TTL_SECONDS:
        try:
            with open(CACHE_FILE, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None
    return None


def save_cache(qqq, vol, macro):
    try:
        with open(CACHE_FILE, "wb") as f:
            pickle.dump((qqq, vol, macro), f)
    except Exception:
        pass  # fail silently


# ==================== DATA FETCHERS ====================
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

    out = []
    for row in r.json().get("observations", []):
        value = row.get("value")
        if value == ".":
            continue
        out.append({"date": row.get("date"), "value": float(value)})
    return out


def fetch_qqq_snapshot():
    hist = yf.Ticker(QQQ_SYMBOL).history(period="10d", interval="60m", auto_adjust=False)
    hist = hist.dropna()
    if hist.empty:
        raise ValueError("No QQQ data returned from yfinance")

    hist.index = pd.to_datetime(hist.index)
    latest_ts = hist.index[-1]
    latest_close = safe_float(hist["Close"].iloc[-1])

    current_day = latest_ts.date()
    day_df = hist[hist.index.date == current_day]

    daily_high = safe_float(day_df["High"].max())
    daily_low = safe_float(day_df["Low"].min())

    unique_days = sorted(pd.Series(hist.index.date).unique())
    last_5_days = set(unique_days[-5:])
    week_df = hist[[d in last_5_days for d in hist.index.date]]

    weekly_high = safe_float(week_df["High"].max())
    weekly_low = safe_float(week_df["Low"].min())

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
    symbols = [VIX_SYMBOL, VVIX_SYMBOL, VIX3M_SYMBOL]
    out = {}

    for symbol in symbols:
        hist = yf.Ticker(symbol).history(period="3mo", interval="1d", auto_adjust=False)
        hist = hist.dropna()
        if hist.empty:
            out[symbol] = {"last": None, "prev_5d": None}
            continue

        out[symbol] = {
            "last": safe_float(hist["Close"].iloc[-1]),
            "prev_5d": safe_float(hist["Close"].iloc[-6]) if len(hist) >= 6 else None
        }

    vvix_last = out[VVIX_SYMBOL]["last"]
    vvix_prev_5d = out[VVIX_SYMBOL]["prev_5d"]

    vvix_trend = "NEUTRAL"
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
    dgs2 = fred_series_observations("DGS2", limit=10)
    dgs5 = fred_series_observations("DGS5", limit=10)
    dexjpus = fred_series_observations("DEXJPUS", limit=10)
    rrp = fred_series_observations("RRPONTSYD", limit=25)

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
    fx_daily_change = pct_change(latest_fx, dexjpus[1]["value"]) if len(dexjpus) >= 2 else None
    fx_5d_old = dexjpus[5]["value"] if len(dexjpus) >= 6 else None
    fx_5d_change = pct_change(latest_fx, fx_5d_old)

    latest_rrp = rrp[0]["value"] if rrp else None
    rrp_5d_old = rrp[5]["value"] if len(rrp) >= 6 else None
    rrp_20d_old = rrp[20]["value"] if len(rrp) >= 21 else None

    return {
        "rates": {"spread_2y_5y_bps": spread_2y_5y_bps, "change_5d_bps": change_5d_bps},
        "fx": {"usdjpy_proxy": latest_fx, "daily_change_pct": fx_daily_change, "change_5d_pct": fx_5d_change},
        "liquidity": {
            "rrp_balance_bil": latest_rrp,
            "change_5d_bil": round(latest_rrp - rrp_5d_old, 2) if latest_rrp and rrp_5d_old else None,
            "change_20d_bil": round(latest_rrp - rrp_20d_old, 2) if latest_rrp and rrp_20d_old else None
        }
    }


# ==================== INTERPRETATION LOGIC ====================
def interpret_qqq_structure(qqq):
    # (your original logic — unchanged, just cleaned up)
    vals = [qqq["last"], qqq["daily_high"], qqq["daily_low"], qqq["weekly_high"], qqq["weekly_low"]]
    if any(v is None for v in vals):
        return "NEUTRAL", 0

    last_price = qqq["last"]
    daily_high = qqq["daily_high"]
    daily_low = qqq["daily_low"]
    weekly_high = qqq["weekly_high"]
    weekly_low = qqq["weekly_low"]
    higher_highs = qqq.get("higher_highs", False)
    higher_lows = qqq.get("higher_lows", False)

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
    # (your original logic — unchanged)
    vix_front = vol["vix_front"]
    vix_3m = vol["vix_3m"]
    vvix = vol["vvix"]
    vvix_trend = str(vol["vvix_trend"]).upper()

    if vix_front is None or vix_3m is None:
        return {"vix_term_structure": "UNKNOWN", "vvix_signal": "UNKNOWN", "vol_bias": "NEUTRAL", "vol_score": 0}

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
        elif vvix_trend == "FALLING" and vvix >= 100:
            vvix_signal = "ELEVATED_BUT_EASING"
            vvix_score = 0

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


# (interpret_curve, interpret_yen_carry, interpret_liquidity functions remain exactly as you had them — I left them unchanged for brevity)

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


# ==================== MAIN DECISION ENGINE ====================
def build_market_decision():
    # Try cache first
    cached = load_cache()
    if cached:
        qqq, vol, macro = cached
    else:
        qqq = fetch_qqq_snapshot()
        vol = fetch_vol_snapshot()
        macro = fetch_macro_snapshot()
        save_cache(qqq, vol, macro)

    qqq_signal, qqq_score = interpret_qqq_structure(qqq)
    vol_info = interpret_volatility(vol)
    curve_signal, curve_score = interpret_curve(macro["rates"])
    yen_signal, yen_score = interpret_yen_carry(macro["fx"])
    liquidity_signal, liquidity_score = interpret_liquidity(macro["liquidity"])

    price_score = qqq_score + curve_score + yen_score + liquidity_score
    vol_score = vol_info["vol_score"]

    price_bias = "BULLISH" if price_score >= 2 else "BEARISH" if price_score <= -2 else "NEUTRAL"
    vol_bias = "BULLISH" if vol_score >= 2 else "BEARISH" if vol_score <= -2 else "NEUTRAL"

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
        "agent": "market_analyst",
        "event_id": datetime.now().isoformat(),
        "price_bias": price_bias,
        "vol_bias": vol_bias,
        "regime": regime,
        "confidence": confidence,
        "factors": {
            "qqq_structure": qqq_signal,
            "vix_term_structure": vol_info["vix_term_structure"],
            "vvix": vol_info["vvix_signal"],
            "curve_signal": curve_signal,
            "yen_usd_signal": yen_signal,
            "rrp_liquidity_signal": liquidity_signal
        },
        "raw_data": {"qqq": qqq, "volatility": vol, "rates": macro["rates"], "fx": macro["fx"], "liquidity": macro["liquidity"]},
        "summary": f"Price bias is {price_bias}, volatility bias is {vol_bias}, and the regime is {regime}."
    }


# ==================== LLM SUMMARY (clean & safe) ====================
def add_llm_summary(decision):
    llm = ChatOllama(
        model="llama3.2:3b",
        base_url="http://localhost:11434/v1",
        temperature=0.2
    )

    prompt = f"""You are a concise market narrator.
Given this JSON decision, return ONLY a JSON object with one key "summary" (1-2 sentences max).

{json.dumps(decision, indent=2)}

Output format: {{"summary": "..."}}"""

    try:
        response = llm.invoke(prompt)
        parsed = json.loads(response.content)
        if isinstance(parsed, dict) and "summary" in parsed:
            decision["summary"] = parsed["summary"]
    except Exception:
        pass  # fallback to original summary

    return decision


# ==================== MAIN ====================
if __name__ == "__main__":
    try:
        decision = build_market_decision()

        if USE_LLM_SUMMARY:
            decision = add_llm_summary(decision)

        print(json.dumps(decision, indent=2))

        # Optional: save to file for logging
        with open(f"reports/market_report_{datetime.now().strftime('%Y%m%d_%H%M')}.json", "w") as f:
            json.dump(decision, f, indent=2)

    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
