#!/usr/bin/env python3
"""
trade_analyst_metals.py

Standalone Trade Analyst for metals relative-value trading.

Focus
-----
- Metals complex only:
    GC=F  Gold
    SI=F  Silver
    HG=F  Copper
    PL=F  Platinum
    PA=F  Palladium
- Relative-value / mean-reversion style
- Uses current price when available
- Falls back to previous daily close when current price is unavailable
- Outputs JSON in a structure similar to the market analyst

Run
---
python trade_analyst_metals.py
"""

import os
import json
import sys
import time
import pickle
import math
import itertools
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

# -------------------- OPTIONAL LLM SUMMARY --------------------
# Kept off by default so this remains runnable without extra dependencies.
USE_LLM_SUMMARY = False
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

if USE_LLM_SUMMARY:
    from langchain_ollama import ChatOllama

# -------------------- BOOTSTRAP --------------------
sys.stdout.reconfigure(encoding="utf-8")

# -------------------- CONFIG --------------------
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

CACHE_FILE = Path("trade_analyst_metals_cache.pkl")
CACHE_TTL_SECONDS = 300

METALS_SYMBOLS = {
    "GC=F": "gold",
    "SI=F": "silver",
    "HG=F": "copper",
    "PL=F": "platinum",
    "PA=F": "palladium",
}

DAILY_PERIOD = "2y"
DAILY_INTERVAL = "1d"

CURRENT_PRICE_ATTEMPTS: List[Tuple[str, str]] = [
    ("1d", "1m"),
    ("5d", "5m"),
    ("5d", "15m"),
]

CORR_WINDOW = 126
BETA_WINDOW = 126
ZSCORE_WINDOW = 63
MIN_CORRELATION = 0.45
MIN_ABS_ZSCORE = 1.25
MAX_CANDIDATES = 10


# -------------------- HELPERS --------------------
def now_iso() -> str:
    return datetime.now().isoformat()


def safe_float(value, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def load_cache():
    if CACHE_FILE.exists() and time.time() - CACHE_FILE.stat().st_mtime < CACHE_TTL_SECONDS:
        try:
            with open(CACHE_FILE, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None
    return None


def save_cache(payload):
    try:
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(payload, f)
    except Exception:
        pass


def compare_level(price: Optional[float], level: Optional[float], near_pct: float = 0.01) -> str:
    if price is None or level is None or level == 0:
        return "UNKNOWN"
    diff_pct = (price / level) - 1.0
    if abs(diff_pct) <= near_pct:
        return "NEAR"
    return "ABOVE" if diff_pct > 0 else "BELOW"


def fetch_history(symbol: str, period: str = DAILY_PERIOD, interval: str = DAILY_INTERVAL) -> pd.DataFrame:
    df = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=False)
    df = df.dropna(how="all")
    if df.empty:
        raise ValueError(f"No history returned for {symbol}")
    df.index = pd.to_datetime(df.index)
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    return df


def try_fetch_current_price(symbol: str) -> Tuple[Optional[float], Optional[str], str]:
    """
    Returns:
        price, timestamp_utc, source_label
    """
    ticker = yf.Ticker(symbol)

    for period, interval in CURRENT_PRICE_ATTEMPTS:
        try:
            df = ticker.history(period=period, interval=interval, auto_adjust=False, prepost=True)
            if df is None or df.empty or "Close" not in df.columns:
                continue

            close_series = df["Close"].dropna()
            if close_series.empty:
                continue

            px = safe_float(close_series.iloc[-1])
            if px is None or px <= 0:
                continue

            ts = close_series.index[-1]
            ts_utc = None
            try:
                ts = pd.Timestamp(ts)
                if ts.tzinfo is None:
                    ts = ts.tz_localize("UTC")
                else:
                    ts = ts.tz_convert("UTC")
                ts_utc = ts.isoformat()
            except Exception:
                ts_utc = None

            return px, ts_utc, "current"

        except Exception:
            continue

    return None, None, "previous_close"


def build_symbol_features(symbol: str, df: pd.DataFrame) -> Dict[str, object]:
    close_series = df["Close"].dropna()
    high_series = df["High"].dropna() if "High" in df.columns else pd.Series(dtype=float)
    low_series = df["Low"].dropna() if "Low" in df.columns else pd.Series(dtype=float)

    previous_close = safe_float(close_series.iloc[-1]) if not close_series.empty else None
    daily_high = safe_float(high_series.iloc[-1]) if not high_series.empty else None
    daily_low = safe_float(low_series.iloc[-1]) if not low_series.empty else None

    sma_21 = safe_float(close_series.rolling(21).mean().iloc[-1]) if len(close_series) >= 21 else None
    sma_63 = safe_float(close_series.rolling(63).mean().iloc[-1]) if len(close_series) >= 63 else None

    current_price, current_ts, source = try_fetch_current_price(symbol)
    effective_price = current_price if current_price is not None else previous_close

    return {
        "symbol": symbol,
        "label": METALS_SYMBOLS[symbol],
        "as_of": str(df.index[-1]),
        "rows": int(len(df)),
        "previous_close": previous_close,
        "effective_price": safe_float(effective_price),
        "price_source": source,
        "source_timestamp_utc": current_ts,
        "daily_high": daily_high,
        "daily_low": daily_low,
        "sma_21": sma_21,
        "sma_63": sma_63,
        "vs_sma_21": compare_level(effective_price, sma_21),
        "vs_sma_63": compare_level(effective_price, sma_63),
    }


def fetch_market_snapshot() -> Dict[str, object]:
    symbol_frames: Dict[str, pd.DataFrame] = {}
    symbols_summary: Dict[str, Dict[str, object]] = {}
    errors: Dict[str, str] = {}

    for symbol in METALS_SYMBOLS:
        try:
            df = fetch_history(symbol)
            symbol_frames[symbol] = df
            symbols_summary[symbol] = build_symbol_features(symbol, df)
        except Exception as exc:
            errors[symbol] = str(exc)

    return {
        "fetched_at": now_iso(),
        "symbols": symbols_summary,
        "frames": symbol_frames,
        "errors": errors,
    }


# -------------------- PAIR ANALYTICS --------------------
def calculate_pair_signal(
    symbol_a: str,
    symbol_b: str,
    closes: pd.DataFrame,
    symbol_features: Dict[str, Dict[str, object]],
) -> Optional[Dict[str, object]]:
    pair_df = closes[[symbol_a, symbol_b]].dropna()
    if len(pair_df) < max(CORR_WINDOW, BETA_WINDOW, ZSCORE_WINDOW) + 5:
        return None

    log_px = np.log(pair_df)
    log_ret = log_px.diff().dropna()

    corr_sample = log_ret.tail(CORR_WINDOW)
    corr = safe_float(corr_sample[symbol_a].corr(corr_sample[symbol_b]))
    if corr is None or corr < MIN_CORRELATION:
        return None

    beta_sample = log_px.tail(BETA_WINDOW)
    x = beta_sample[symbol_b].values
    y = beta_sample[symbol_a].values

    var_x = np.var(x, ddof=1)
    if not np.isfinite(var_x) or var_x <= 0:
        return None

    cov_xy = np.cov(x, y, ddof=1)[0, 1]
    beta = safe_float(cov_xy / var_x)
    if beta is None or not np.isfinite(beta):
        return None

    hist_spread = (log_px[symbol_a] - beta * log_px[symbol_b]).dropna()
    z_window = hist_spread.tail(ZSCORE_WINDOW)
    if len(z_window) < ZSCORE_WINDOW:
        return None

    mu = safe_float(z_window.mean())
    sigma = safe_float(z_window.std(ddof=1))
    if mu is None or sigma is None or sigma <= 1e-10:
        return None

    last_close_spread = safe_float(hist_spread.iloc[-1])
    last_close_z = safe_float((last_close_spread - mu) / sigma)

    price_a = symbol_features[symbol_a].get("effective_price")
    price_b = symbol_features[symbol_b].get("effective_price")
    if price_a in (None, 0) or price_b in (None, 0):
        return None

    current_spread = math.log(float(price_a)) - beta * math.log(float(price_b))
    current_z = safe_float((current_spread - mu) / sigma)
    if current_z is None:
        return None

    if current_z > 0:
        expensive_symbol = symbol_a
        cheap_symbol = symbol_b
    else:
        expensive_symbol = symbol_b
        cheap_symbol = symbol_a

    z_strength = min(abs(current_z) / 3.0, 1.0)
    corr_strength = min(max((corr - MIN_CORRELATION) / (1.0 - MIN_CORRELATION + 1e-9), 0.0), 1.0)
    confidence = round(0.55 * z_strength + 0.45 * corr_strength, 2)
    score = round(abs(current_z) * corr, 4)

    recommendation = f"SELL {expensive_symbol} / BUY {cheap_symbol}"
    rationale = (
        f"{METALS_SYMBOLS[expensive_symbol]} screens expensive relative to {METALS_SYMBOLS[cheap_symbol]}. "
        f"The hedge-adjusted spread is {current_z:.2f} standard deviations from its {ZSCORE_WINDOW}-day mean "
        f"with {corr:.2f} rolling return correlation."
    )

    return {
        "pair": f"{symbol_a}__{symbol_b}",
        "pair_label": f"{METALS_SYMBOLS[symbol_a]}__{METALS_SYMBOLS[symbol_b]}",
        "symbol_a": symbol_a,
        "symbol_b": symbol_b,
        "label_a": METALS_SYMBOLS[symbol_a],
        "label_b": METALS_SYMBOLS[symbol_b],
        "correlation": round(float(corr), 4),
        "hedge_ratio_beta": round(float(beta), 4),
        "last_close_zscore": round(float(last_close_z), 4),
        "current_zscore": round(float(current_z), 4),
        "expensive_symbol": expensive_symbol,
        "cheap_symbol": cheap_symbol,
        "expensive_label": METALS_SYMBOLS[expensive_symbol],
        "cheap_label": METALS_SYMBOLS[cheap_symbol],
        "recommendation": recommendation,
        "regime": "MEAN_REVERSION_SHORT_EXPENSIVE_LONG_CHEAP",
        "confidence": confidence,
        "score": score,
        "rationale": rationale,
        "inputs": {
            "price_a": round(float(price_a), 4),
            "price_b": round(float(price_b), 4),
            "price_a_source": symbol_features[symbol_a].get("price_source"),
            "price_b_source": symbol_features[symbol_b].get("price_source"),
        },
    }


def build_pair_snapshot(market_snapshot: Dict[str, object]) -> Dict[str, object]:
    symbol_features = market_snapshot["symbols"]
    available_symbols = [s for s in METALS_SYMBOLS if s in symbol_features]

    if len(available_symbols) < 2:
        raise ValueError("Need at least two metals symbols with valid data")

    close_map = {}
    for symbol in available_symbols:
        frame = market_snapshot["frames"].get(symbol)
        if frame is not None and not frame.empty:
            close_map[symbol] = frame["Close"]

    closes = pd.DataFrame(close_map).dropna(how="all")
    if closes.shape[1] < 2:
        raise ValueError("Not enough close history available to build pair analytics")

    all_pairs: List[Dict[str, object]] = []
    for symbol_a, symbol_b in itertools.combinations(closes.columns.tolist(), 2):
        signal = calculate_pair_signal(symbol_a, symbol_b, closes, symbol_features)
        if signal is not None:
            all_pairs.append(signal)

    all_pairs.sort(key=lambda x: x["score"], reverse=True)
    candidates = [p for p in all_pairs if abs(p["current_zscore"]) >= MIN_ABS_ZSCORE][:MAX_CANDIDATES]

    return {
        "fetched_at": now_iso(),
        "all_pairs": all_pairs,
        "candidates": candidates,
    }


def build_trade_summary(candidates: List[Dict[str, object]]) -> str:
    if not candidates:
        return (
            "No metals relative-value candidate cleared the minimum correlation and z-score thresholds. "
            "Current conditions do not justify a mean-reversion trade signal."
        )

    top = candidates[0]
    return (
        f"Top metals RV opportunity is {top['recommendation']} in {top['pair_label']}. "
        f"Current spread z-score is {top['current_zscore']:.2f} with {top['correlation']:.2f} correlation and confidence {top['confidence']:.2f}."
    )


# -------------------- DECISION BUILDER --------------------
def build_trade_decision() -> Dict[str, object]:
    cached = load_cache()
    cache_hit = bool(cached)

    if cached:
        market_snapshot = cached["market_snapshot"]
        pair_snapshot = cached["pair_snapshot"]
    else:
        market_snapshot = fetch_market_snapshot()
        pair_snapshot = build_pair_snapshot(market_snapshot)
        save_cache({
            "market_snapshot": market_snapshot,
            "pair_snapshot": pair_snapshot,
        })

    candidates = pair_snapshot["candidates"]
    top_candidate = candidates[0] if candidates else None

    data_quality = {
        "cache_hit": cache_hit,
        "fetched_at_market": market_snapshot.get("fetched_at"),
        "fetched_at_pairs": pair_snapshot.get("fetched_at"),
        "missing_symbols": sorted(list(market_snapshot.get("errors", {}).keys())),
        "available_symbols": sorted(list(market_snapshot.get("symbols", {}).keys())),
        "price_sources": {
            METALS_SYMBOLS[s]: market_snapshot["symbols"][s].get("price_source")
            for s in market_snapshot.get("symbols", {})
        },
    }

    return {
        "agent": "trade_analyst",
        "event_id": now_iso(),
        "data_quality": data_quality,
        "decision": {
            "complex": "metals",
            "strategy_family": "relative_value_mean_reversion",
            "candidate_count": len(candidates),
            "top_candidate": top_candidate,
            "thresholds": {
                "min_correlation": MIN_CORRELATION,
                "min_abs_zscore": MIN_ABS_ZSCORE,
                "corr_window_days": CORR_WINDOW,
                "beta_window_days": BETA_WINDOW,
                "zscore_window_days": ZSCORE_WINDOW,
            },
        },
        "components": {
            "metals": market_snapshot["symbols"],
            "pair_analytics": {
                pair["pair_label"]: pair
                for pair in pair_snapshot["all_pairs"]
            },
            "trade_candidates": candidates,
        },
        "summary": build_trade_summary(candidates),
    }


# -------------------- OPTIONAL LLM SUMMARY --------------------
def add_llm_summary(decision: Dict[str, object]) -> Dict[str, object]:
    llm = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.2,
    )

    prompt_payload = {
        "decision": decision["decision"],
        "summary": decision["summary"],
    }

    prompt = (
        "You are a concise trade analyst narrator. "
        "Return ONLY valid JSON with one key named summary. "
        "Keep it to 2 sentences max and do not change the deterministic view.\n\n"
        f"{json.dumps(prompt_payload, indent=2)}\n\n"
        'Output format: {"summary": "..."}'
    )

    try:
        response = llm.invoke(prompt)
        parsed = json.loads(response.content)
        if isinstance(parsed, dict) and "summary" in parsed:
            decision["summary"] = parsed["summary"]
    except Exception:
        pass

    return decision


# -------------------- MAIN --------------------
if __name__ == "__main__":
    try:
        decision = build_trade_decision()

        if USE_LLM_SUMMARY:
            decision = add_llm_summary(decision)

        print(json.dumps(decision, indent=2))

        report_path = REPORTS_DIR / f"trade_report_metals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(decision, f, indent=2)

        print(f"\n[INFO] Report written to {report_path}")

    except Exception as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)
