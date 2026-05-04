#!/usr/bin/env python3
"""
market_analyst_v2.py

Cleaner dual-horizon market analyst with simplified final regime output.

Final public regimes are intentionally limited to:
- BULLISH_PRICE_BEARISH_VOL
- NEUTRAL
- BEARISH_PRICE_BULLISH_VOL

Design goals
------------
- Preserve the useful dual-horizon structure from market_analyst_dual_horizon.py
- Keep deterministic data collection and factor transparency
- Simplify final regime logic for easier Trade Coordinator consumption
- Avoid hard-failing when macro data is unavailable
- Keep output compatible with the current coordinator schema
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import yfinance as yf

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# -------------------- OPTIONAL LLM SUMMARY --------------------
USE_LLM_SUMMARY = False
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

if USE_LLM_SUMMARY:
    from langchain_ollama import ChatOllama

# -------------------- CONFIG --------------------
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

CACHE_FILE = Path("market_cache_v2.pkl")
CACHE_TTL_SECONDS = 300

FRED_API_KEY = os.getenv("FRED_API_KEY")
USE_MACRO = bool(FRED_API_KEY)
USE_FUTURES_CONTEXT = True
FUTURES_SYMBOL = "NQ=F"

QQQ_SYMBOL = "QQQ"
MOVE_SYMBOL = "^MOVE"
VVIX_SYMBOL = "^VVIX"
VOL_CURVE_SYMBOLS = [
    "^VIX9D",
    "^VIN",
    "^VIX",
    "^VIF",
    "^VIX3M",
    "^VIX6M",
    "^VIX1Y",
]

ALL_YF_SYMBOLS = [QQQ_SYMBOL, VVIX_SYMBOL, MOVE_SYMBOL] + VOL_CURVE_SYMBOLS + ([FUTURES_SYMBOL] if USE_FUTURES_CONTEXT else [])

SYMBOL_LABELS = {
    "QQQ": "qqq",
    "^VVIX": "vvix",
    "^MOVE": "move",
    "^VIX9D": "vix9d",
    "^VIN": "vin",
    "^VIX": "vix",
    "^VIF": "vif",
    "^VIX3M": "vix3m",
    "^VIX6M": "vix6m",
    "^VIX1Y": "vix1y",
    "NQ=F": "nq_futures",
}

FRED_SERIES = {
    "DGS2": "DGS2",
    "DGS5": "DGS5",
    "DEXJPUS": "DEXJPUS",
    "RRPONTSYD": "RRPONTSYD",
}


@dataclass
class RuntimeConfig:
    qqq_period: str = "2y"
    qqq_interval: str = "1d"
    cache_ttl_seconds: int = CACHE_TTL_SECONDS
    price_bias_threshold: int = 2
    vol_bias_threshold: int = 2
    dominance_gap: int = 2
    strong_alignment_threshold: int = 2


# -------------------- HELPERS --------------------
def now_iso() -> str:
    return datetime.now().isoformat()


def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def pct_change(new: Optional[float], old: Optional[float]) -> Optional[float]:
    if new is None or old in (None, 0):
        return None
    return round(((new / old) - 1.0) * 100.0, 4)


def compare_level(close: Optional[float], level: Optional[float], near_pct: float = 0.01) -> str:
    if close is None or level is None or level == 0:
        return "UNKNOWN"
    diff_pct = (close / level) - 1.0
    if abs(diff_pct) <= near_pct:
        return "NEAR"
    return "ABOVE" if diff_pct > 0 else "BELOW"


def distance_pct(close: Optional[float], level: Optional[float]) -> Optional[float]:
    if close is None or level in (None, 0):
        return None
    return round(((close / level) - 1.0) * 100.0, 4)


def range_position(close: Optional[float], low: Optional[float], high: Optional[float]) -> Optional[float]:
    if close is None or low is None or high is None or high == low:
        return None
    return round((close - low) / (high - low), 4)


def range_zone(position: Optional[float]) -> str:
    if position is None:
        return "UNKNOWN"
    if position >= 0.80:
        return "NEAR_HIGH"
    if position <= 0.20:
        return "NEAR_LOW"
    return "MID_RANGE"


def week_period(index: pd.DatetimeIndex) -> pd.PeriodIndex:
    return index.to_period("W-FRI")


def latest_two_weeks(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    weeks = week_period(df.index)
    unique_weeks = list(pd.Series(weeks).drop_duplicates())
    if not unique_weeks:
        return pd.DataFrame(), pd.DataFrame()
    current_week = unique_weeks[-1]
    current_df = df[weeks == current_week]
    prev_df = pd.DataFrame()
    if len(unique_weeks) >= 2:
        prev_df = df[weeks == unique_weeks[-2]]
    return current_df, prev_df


def load_cache() -> Optional[Dict[str, Any]]:
    if CACHE_FILE.exists() and time.time() - CACHE_FILE.stat().st_mtime < CACHE_TTL_SECONDS:
        try:
            with open(CACHE_FILE, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None
    return None


def save_cache(payload: Dict[str, Any]) -> None:
    try:
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(payload, f)
    except Exception:
        pass


# -------------------- DATA FETCH --------------------
def fred_series_observations(series_id: str, limit: int = 30) -> List[Dict[str, float]]:
    if not FRED_API_KEY:
        return []

    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "sort_order": "desc",
        "limit": limit,
    }

    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()

    observations: List[Dict[str, float]] = []
    for row in r.json().get("observations", []):
        value = row.get("value")
        if value == ".":
            continue
        observations.append({"date": row.get("date"), "value": float(value)})
    return observations


def fetch_history(symbol: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    df = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=False)
    df = df.dropna(how="all")
    if df.empty:
        raise ValueError(f"No history returned for {symbol}")
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    df.index = idx
    return df


def build_symbol_features(symbol: str, df: pd.DataFrame, include_sma_100_200: bool = False) -> Dict[str, object]:
    close_series = df["Close"]
    high_series = df["High"]
    low_series = df["Low"]
    last_row = df.iloc[-1]

    current_week_df, previous_week_df = latest_two_weeks(df)

    sma21 = safe_float(close_series.rolling(21).mean().iloc[-1]) if len(df) >= 21 else None
    sma100 = safe_float(close_series.rolling(100).mean().iloc[-1]) if include_sma_100_200 and len(df) >= 100 else None
    sma200 = safe_float(close_series.rolling(200).mean().iloc[-1]) if include_sma_100_200 and len(df) >= 200 else None

    range_21_high = safe_float(high_series.tail(21).max()) if len(df) >= 21 else None
    range_21_low = safe_float(low_series.tail(21).min()) if len(df) >= 21 else None
    range_63_high = safe_float(high_series.tail(63).max()) if len(df) >= 63 else None
    range_63_low = safe_float(low_series.tail(63).min()) if len(df) >= 63 else None
    range_126_high = safe_float(high_series.tail(126).max()) if len(df) >= 126 else None
    range_126_low = safe_float(low_series.tail(126).min()) if len(df) >= 126 else None

    close = safe_float(last_row["Close"])
    daily_high = safe_float(last_row["High"])
    daily_low = safe_float(last_row["Low"])

    prev_week_high = safe_float(previous_week_df["High"].max()) if not previous_week_df.empty else None
    prev_week_low = safe_float(previous_week_df["Low"].min()) if not previous_week_df.empty else None
    curr_week_high = safe_float(current_week_df["High"].max()) if not current_week_df.empty else None
    curr_week_low = safe_float(current_week_df["Low"].min()) if not current_week_df.empty else None

    return {
        "symbol": symbol,
        "label": SYMBOL_LABELS.get(symbol, symbol),
        "as_of": str(df.index[-1].date()),
        "rows": int(len(df)),
        "close": close,
        "daily_high": daily_high,
        "daily_low": daily_low,
        "previous_week_high": prev_week_high,
        "previous_week_low": prev_week_low,
        "current_week_high": curr_week_high,
        "current_week_low": curr_week_low,
        "daily_position": range_position(close, daily_low, daily_high),
        "current_week_position": range_position(close, curr_week_low, curr_week_high),
        "range_21_position": range_position(close, range_21_low, range_21_high),
        "range_63_position": range_position(close, range_63_low, range_63_high),
        "range_126_position": range_position(close, range_126_low, range_126_high),
        "sma_21": sma21,
        "sma_100": sma100,
        "sma_200": sma200,
        "vs_sma_21": compare_level(close, sma21),
        "vs_sma_100": compare_level(close, sma100) if include_sma_100_200 else "UNKNOWN",
        "vs_sma_200": compare_level(close, sma200) if include_sma_100_200 else "UNKNOWN",
        "distance_to_sma_21_pct": distance_pct(close, sma21),
        "distance_to_sma_100_pct": distance_pct(close, sma100) if include_sma_100_200 else None,
        "distance_to_sma_200_pct": distance_pct(close, sma200) if include_sma_100_200 else None,
        "vs_previous_week": (
            "ABOVE_PREVIOUS_WEEK_HIGH" if close is not None and prev_week_high is not None and close > prev_week_high
            else "BELOW_PREVIOUS_WEEK_LOW" if close is not None and prev_week_low is not None and close < prev_week_low
            else "INSIDE_PREVIOUS_WEEK"
        ),
        "daily_zone": range_zone(range_position(close, daily_low, daily_high)),
        "current_week_zone": range_zone(range_position(close, curr_week_low, curr_week_high)),
        "range_21_zone": range_zone(range_position(close, range_21_low, range_21_high)),
        "range_63_zone": range_zone(range_position(close, range_63_low, range_63_high)),
        "range_126_zone": range_zone(range_position(close, range_126_low, range_126_high)),
    }


def fetch_market_snapshot() -> Dict[str, object]:
    data: Dict[str, object] = {}
    errors: Dict[str, str] = {}

    for symbol in ALL_YF_SYMBOLS:
        try:
            df = fetch_history(symbol, period="2y", interval="1d")
            data[symbol] = build_symbol_features(
                symbol,
                df,
                include_sma_100_200=(symbol == QQQ_SYMBOL or symbol == FUTURES_SYMBOL),
            )
        except Exception as exc:
            errors[symbol] = str(exc)

    return {
        "fetched_at": now_iso(),
        "symbols": data,
        "errors": errors,
    }


def fetch_macro_snapshot() -> Dict[str, object]:
    if not USE_MACRO:
        return {
            "fetched_at": now_iso(),
            "status": "UNAVAILABLE",
            "rates": {},
            "fx": {},
            "liquidity": {},
        }

    try:
        dgs2 = fred_series_observations(FRED_SERIES["DGS2"], limit=10)
        dgs5 = fred_series_observations(FRED_SERIES["DGS5"], limit=10)
        dexjpus = fred_series_observations(FRED_SERIES["DEXJPUS"], limit=10)
        rrp = fred_series_observations(FRED_SERIES["RRPONTSYD"], limit=25)
    except Exception as exc:
        return {
            "fetched_at": now_iso(),
            "status": "ERROR",
            "error": str(exc),
            "rates": {},
            "fx": {},
            "liquidity": {},
        }

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
        "fetched_at": now_iso(),
        "status": "OK",
        "rates": {
            "spread_2y_5y_bps": spread_2y_5y_bps,
            "change_5d_bps": change_5d_bps,
        },
        "fx": {
            "series": "DEXJPUS",
            "description": "Japanese Yen to 1 U.S. Dollar",
            "yen_per_usd": latest_fx,
            "daily_change_pct": fx_daily_change,
            "change_5d_pct": fx_5d_change,
        },
        "liquidity": {
            "series": "RRPONTSYD",
            "rrp_balance_bil": latest_rrp,
            "change_5d_bil": round(latest_rrp - rrp_5d_old, 2) if latest_rrp is not None and rrp_5d_old is not None else None,
            "change_20d_bil": round(latest_rrp - rrp_20d_old, 2) if latest_rrp is not None and rrp_20d_old is not None else None,
        },
    }


# -------------------- SCORING --------------------
def generic_price_structure_score(features: Dict[str, object], include_long_window: bool = False) -> Tuple[int, Dict[str, object]]:
    score = 0
    details: Dict[str, object] = {}

    prev_week_state = features.get("vs_previous_week")
    if prev_week_state == "ABOVE_PREVIOUS_WEEK_HIGH":
        score += 1
    elif prev_week_state == "BELOW_PREVIOUS_WEEK_LOW":
        score -= 1
    details["vs_previous_week"] = prev_week_state

    week_pos = features.get("current_week_position")
    if week_pos is not None:
        if week_pos >= 0.75:
            score += 1
        elif week_pos <= 0.25:
            score -= 1
    details["current_week_zone"] = features.get("current_week_zone")

    daily_pos = features.get("daily_position")
    if daily_pos is not None:
        if daily_pos >= 0.67:
            score += 1
        elif daily_pos <= 0.33:
            score -= 1
    details["daily_zone"] = features.get("daily_zone")

    vs_sma_21 = features.get("vs_sma_21")
    if vs_sma_21 == "ABOVE":
        score += 1
    elif vs_sma_21 == "BELOW":
        score -= 1
    details["vs_sma_21"] = vs_sma_21

    if include_long_window:
        range_63_pos = features.get("range_63_position")
        if range_63_pos is not None:
            if range_63_pos >= 0.60:
                score += 1
            elif range_63_pos <= 0.40:
                score -= 1
        details["range_63_zone"] = features.get("range_63_zone")

    return score, details


def score_qqq_short(features: Dict[str, object]) -> Tuple[int, Dict[str, object], float]:
    score, details = generic_price_structure_score(features, include_long_window=False)
    penalty = 0.0

    near_100 = features.get("vs_sma_100") == "NEAR"
    near_200 = features.get("vs_sma_200") == "NEAR"
    expansion_risk = near_100 or near_200
    if expansion_risk:
        penalty += 0.05

    details["vs_sma_100"] = features.get("vs_sma_100")
    details["vs_sma_200"] = features.get("vs_sma_200")
    details["expansion_risk_near_big_ma"] = expansion_risk
    return score, details, penalty


def score_qqq_long(features: Dict[str, object]) -> Tuple[int, Dict[str, object], float]:
    score, details = generic_price_structure_score(features, include_long_window=True)
    penalty = 0.0

    if features.get("vs_sma_100") == "ABOVE":
        score += 1
    elif features.get("vs_sma_100") == "BELOW":
        score -= 1

    if features.get("vs_sma_200") == "ABOVE":
        score += 1
    elif features.get("vs_sma_200") == "BELOW":
        score -= 1

    sma100 = features.get("sma_100")
    sma200 = features.get("sma_200")
    if sma100 is not None and sma200 is not None:
        if sma100 > sma200:
            score += 1
            details["sma_100_vs_200"] = "ABOVE"
        elif sma100 < sma200:
            score -= 1
            details["sma_100_vs_200"] = "BELOW"
        else:
            details["sma_100_vs_200"] = "FLAT"
    else:
        details["sma_100_vs_200"] = "UNKNOWN"

    expansion_risk = features.get("vs_sma_100") == "NEAR" or features.get("vs_sma_200") == "NEAR"
    if expansion_risk:
        penalty += 0.05

    details["vs_sma_100"] = features.get("vs_sma_100")
    details["vs_sma_200"] = features.get("vs_sma_200")
    details["expansion_risk_near_big_ma"] = expansion_risk
    return score, details, penalty


def score_futures_context(features: Optional[Dict[str, object]], long_term: bool = False) -> Tuple[int, Dict[str, object]]:
    if not features:
        return 0, {"status": "UNAVAILABLE"}

    score, base = generic_price_structure_score(features, include_long_window=long_term)

    # cap contribution so QQQ + NQ do not double-count too aggressively
    score = max(-2, min(2, score))
    return score, {
        "status": "AVAILABLE",
        "symbol": features.get("symbol"),
        "vs_sma_21": base.get("vs_sma_21"),
        "current_week_zone": base.get("current_week_zone"),
        "range_63_zone": base.get("range_63_zone") if long_term else "N/A",
        "capped_score": score,
    }


def score_macro_short(macro: Dict[str, object]) -> Tuple[int, Dict[str, str]]:
    if macro.get("status") != "OK":
        return 0, {"status": macro.get("status", "UNAVAILABLE")}

    rates = macro["rates"]
    fx = macro["fx"]
    score = 0
    details: Dict[str, str] = {}

    change_5d = rates.get("change_5d_bps")
    if change_5d is None:
        details["curve_signal"] = "NEUTRAL"
    elif change_5d >= 3:
        score += 1
        details["curve_signal"] = "RISK_ON_STEEPENING"
    elif change_5d <= -3:
        score -= 1
        details["curve_signal"] = "RISK_OFF_FLATTENING"
    else:
        details["curve_signal"] = "NEUTRAL"

    daily_fx = fx.get("daily_change_pct")
    fx_5d = fx.get("change_5d_pct")
    if daily_fx is None or fx_5d is None:
        details["yen_usd_signal"] = "NEUTRAL"
    elif daily_fx > 0 and fx_5d > 0.5:
        score += 1
        details["yen_usd_signal"] = "RISK_ON"
    elif daily_fx < 0 and fx_5d < -0.5:
        score -= 1
        details["yen_usd_signal"] = "RISK_OFF"
    else:
        details["yen_usd_signal"] = "NEUTRAL"

    return score, details


def score_macro_long(macro: Dict[str, object]) -> Tuple[int, Dict[str, str]]:
    if macro.get("status") != "OK":
        return 0, {"status": macro.get("status", "UNAVAILABLE")}

    rates = macro["rates"]
    liq = macro["liquidity"]
    score = 0
    details: Dict[str, str] = {}

    spread = rates.get("spread_2y_5y_bps")
    change_5d = rates.get("change_5d_bps")
    if spread is None or change_5d is None:
        details["curve_regime"] = "NEUTRAL"
    elif spread > 0 and change_5d >= 0:
        score += 1
        details["curve_regime"] = "RISK_ON"
    elif spread < 0 and change_5d <= 0:
        score -= 1
        details["curve_regime"] = "RISK_OFF"
    else:
        details["curve_regime"] = "NEUTRAL"

    rrp_20d = liq.get("change_20d_bil")
    if rrp_20d is None:
        details["rrp_liquidity_signal"] = "NEUTRAL"
    elif rrp_20d < 0:
        score += 1
        details["rrp_liquidity_signal"] = "LIQUIDITY_SUPPORTIVE"
    elif rrp_20d > 0:
        score -= 1
        details["rrp_liquidity_signal"] = "LIQUIDITY_DRAINING"
    else:
        details["rrp_liquidity_signal"] = "NEUTRAL"

    return score, details


def compute_curve_metrics(symbols: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    levels = {s: symbols.get(s, {}).get("close") for s in VOL_CURVE_SYMBOLS}
    sma21_levels = {s: symbols.get(s, {}).get("sma_21") for s in VOL_CURVE_SYMBOLS}

    adjacent_pairs = list(zip(VOL_CURVE_SYMBOLS[:-1], VOL_CURVE_SYMBOLS[1:]))
    contango_pairs = 0
    backwardation_pairs = 0
    missing_pairs = 0
    pair_states: Dict[str, str] = {}

    for a, b in adjacent_pairs:
        va = levels.get(a)
        vb = levels.get(b)
        key = f"{SYMBOL_LABELS[a]}->{SYMBOL_LABELS[b]}"
        if va is None or vb is None:
            pair_states[key] = "UNKNOWN"
            missing_pairs += 1
            continue
        if vb > va:
            pair_states[key] = "CONTANGO"
            contango_pairs += 1
        elif va > vb:
            pair_states[key] = "BACKWARDATION"
            backwardation_pairs += 1
        else:
            pair_states[key] = "FLAT"

    above_21 = 0
    below_21 = 0
    for s in VOL_CURVE_SYMBOLS:
        close = levels.get(s)
        sma21 = sma21_levels.get(s)
        if close is None or sma21 is None:
            continue
        if close > sma21:
            above_21 += 1
        elif close < sma21:
            below_21 += 1

    vix9d = levels.get("^VIX9D")
    vix = levels.get("^VIX")
    vix3m = levels.get("^VIX3M")
    vix6m = levels.get("^VIX6M")
    vix1y = levels.get("^VIX1Y")

    return {
        "levels": {SYMBOL_LABELS[k]: v for k, v in levels.items()},
        "pair_states": pair_states,
        "contango_pairs": contango_pairs,
        "backwardation_pairs": backwardation_pairs,
        "missing_pairs": missing_pairs,
        "breadth_above_sma21": above_21,
        "breadth_below_sma21": below_21,
        "front_slope_vix_minus_vix9d": round(vix - vix9d, 4) if vix is not None and vix9d is not None else None,
        "mid_slope_vix3m_minus_vix": round(vix3m - vix, 4) if vix3m is not None and vix is not None else None,
        "back_slope_vix1y_minus_vix3m": round(vix1y - vix3m, 4) if vix1y is not None and vix3m is not None else None,
        "far_slope_vix1y_minus_vix6m": round(vix1y - vix6m, 4) if vix1y is not None and vix6m is not None else None,
    }


def score_vol_curve_short(curve: Dict[str, object]) -> Tuple[int, Dict[str, object]]:
    score = 0
    front_slope = curve.get("front_slope_vix_minus_vix9d")
    mid_slope = curve.get("mid_slope_vix3m_minus_vix")

    if front_slope is not None:
        if front_slope > 0:
            score -= 1
        elif front_slope < 0:
            score += 1

    if mid_slope is not None:
        if mid_slope > 0:
            score -= 1
        elif mid_slope < 0:
            score += 1

    if curve["backwardation_pairs"] > curve["contango_pairs"]:
        score += 1
        state = "BACKWARDATION_STRESS"
    elif curve["contango_pairs"] > curve["backwardation_pairs"]:
        score -= 1
        state = "CONTANGO_CALM"
    else:
        state = "MIXED"

    if curve["breadth_above_sma21"] > curve["breadth_below_sma21"]:
        score += 1
    elif curve["breadth_below_sma21"] > curve["breadth_above_sma21"]:
        score -= 1

    details = {
        "curve_short_state": state,
        "front_slope_vix_minus_vix9d": front_slope,
        "mid_slope_vix3m_minus_vix": mid_slope,
        "breadth_above_sma21": curve["breadth_above_sma21"],
        "breadth_below_sma21": curve["breadth_below_sma21"],
        "contango_pairs": curve["contango_pairs"],
        "backwardation_pairs": curve["backwardation_pairs"],
    }
    return score, details


def score_vol_curve_long(curve: Dict[str, object]) -> Tuple[int, Dict[str, object]]:
    score = 0
    mid_slope = curve.get("mid_slope_vix3m_minus_vix")
    back_slope = curve.get("back_slope_vix1y_minus_vix3m")
    far_slope = curve.get("far_slope_vix1y_minus_vix6m")

    for slope in [mid_slope, back_slope, far_slope]:
        if slope is None:
            continue
        if slope > 0:
            score -= 1
        elif slope < 0:
            score += 1

    if curve["breadth_above_sma21"] > curve["breadth_below_sma21"]:
        score += 1
        state = "VOL_STRESS_BROADENING"
    elif curve["breadth_below_sma21"] > curve["breadth_above_sma21"]:
        score -= 1
        state = "VOL_STRESS_RECEDING"
    else:
        state = "MIXED"

    details = {
        "curve_long_state": state,
        "mid_slope_vix3m_minus_vix": mid_slope,
        "back_slope_vix1y_minus_vix3m": back_slope,
        "far_slope_vix1y_minus_vix6m": far_slope,
        "breadth_above_sma21": curve["breadth_above_sma21"],
        "breadth_below_sma21": curve["breadth_below_sma21"],
    }
    return score, details


def score_vvix_state(features: Optional[Dict[str, object]], long_term: bool = False) -> Tuple[int, Dict[str, object]]:
    if not features:
        return 0, {"status": "UNAVAILABLE"}

    score = 0
    details: Dict[str, object] = {"status": "AVAILABLE"}

    if features.get("vs_sma_21") == "ABOVE":
        score += 1
    elif features.get("vs_sma_21") == "BELOW":
        score -= 1

    pos_key = "range_126_position" if long_term else "range_63_position"
    zone_key = "range_126_zone" if long_term else "range_63_zone"
    pos = features.get(pos_key)
    if pos is not None:
        if pos >= 0.70:
            score += 1
        elif pos <= 0.30:
            score -= 1

    details["vs_sma_21"] = features.get("vs_sma_21")
    details[zone_key] = features.get(zone_key)
    return score, details


def score_move_state(features: Optional[Dict[str, object]], long_term: bool = False) -> Tuple[int, Dict[str, object]]:
    if not features:
        return 0, {"status": "UNAVAILABLE"}

    score = 0
    details: Dict[str, object] = {"status": "AVAILABLE"}

    if features.get("vs_sma_21") == "ABOVE":
        score += 1
    elif features.get("vs_sma_21") == "BELOW":
        score -= 1

    pos_key = "range_126_position" if long_term else "range_63_position"
    zone_key = "range_126_zone" if long_term else "range_63_zone"
    pos = features.get(pos_key)
    if pos is not None:
        if pos >= 0.70:
            score += 1
        elif pos <= 0.30:
            score -= 1

    details["vs_sma_21"] = features.get("vs_sma_21")
    details[zone_key] = features.get(zone_key)
    return score, details


def classify_bias(score: float, threshold: int) -> str:
    if score >= threshold:
        return "BULLISH"
    if score <= -threshold:
        return "BEARISH"
    return "NEUTRAL"


def collapse_regime(price_bias: str, vol_bias: str, price_score: float, vol_score: float, config: RuntimeConfig) -> Tuple[str, str]:
    # Public regime is always 3-state. Internal alignment keeps the nuance.
    if price_bias == "BULLISH" and vol_bias == "BEARISH":
        return "BULLISH_PRICE_BEARISH_VOL", "ALIGNED"
    if price_bias == "BEARISH" and vol_bias == "BULLISH":
        return "BEARISH_PRICE_BULLISH_VOL", "ALIGNED"

    if price_score >= config.strong_alignment_threshold and (price_score - vol_score) >= config.dominance_gap:
        return "BULLISH_PRICE_BEARISH_VOL", "PRICE_DOMINANT"
    if price_score <= -config.strong_alignment_threshold and (vol_score - price_score) >= config.dominance_gap:
        return "BEARISH_PRICE_BULLISH_VOL", "PRICE_DOMINANT"

    return "NEUTRAL", "MIXED_OR_WEAK"


def compute_confidence(price_score: float, vol_score: float, price_bias: str, vol_bias: str, penalty: float = 0.0) -> float:
    price_strength = min(abs(price_score) / 4.0, 1.0)
    vol_strength = min(abs(vol_score) / 4.0, 1.0)
    alignment_bonus = 0.0

    if (price_bias == "BULLISH" and vol_bias == "BEARISH") or (price_bias == "BEARISH" and vol_bias == "BULLISH"):
        alignment_bonus = 0.20
    elif price_bias == "NEUTRAL" and vol_bias == "NEUTRAL":
        alignment_bonus = -0.05
    else:
        alignment_bonus = -0.10

    confidence = 0.45 + 0.22 * price_strength + 0.22 * vol_strength + alignment_bonus - penalty
    return round(clamp(confidence, 0.05, 0.95), 2)


def build_horizon_block(
    *,
    market: Dict[str, object],
    macro: Dict[str, object],
    horizon: str,
    config: RuntimeConfig,
) -> Dict[str, object]:
    symbols = market["symbols"]
    qqq = symbols.get(QQQ_SYMBOL)
    vvix = symbols.get(VVIX_SYMBOL)
    move = symbols.get(MOVE_SYMBOL)
    futures = symbols.get(FUTURES_SYMBOL) if USE_FUTURES_CONTEXT else None

    if not qqq:
        raise ValueError("QQQ data is required for horizon scoring")

    long_term = horizon == "LONG_TERM"

    if long_term:
        qqq_score, qqq_details, qqq_penalty = score_qqq_long(qqq)
        macro_score, macro_details = score_macro_long(macro)
        futures_score, futures_details = score_futures_context(futures, long_term=True)
    else:
        qqq_score, qqq_details, qqq_penalty = score_qqq_short(qqq)
        macro_score, macro_details = score_macro_short(macro)
        futures_score, futures_details = score_futures_context(futures, long_term=False)

    curve_metrics = compute_curve_metrics(symbols)
    if long_term:
        curve_score, curve_details = score_vol_curve_long(curve_metrics)
        vvix_score, vvix_details = score_vvix_state(vvix, long_term=True) if vvix else (0, {"status": "UNAVAILABLE"})
        move_score, move_details = score_move_state(move, long_term=True) if move else (0, {"status": "UNAVAILABLE"})
    else:
        curve_score, curve_details = score_vol_curve_short(curve_metrics)
        vvix_score, vvix_details = score_vvix_state(vvix, long_term=False) if vvix else (0, {"status": "UNAVAILABLE"})
        move_score, move_details = score_move_state(move, long_term=False) if move else (0, {"status": "UNAVAILABLE"})

    price_score = qqq_score + macro_score + futures_score
    vol_score = curve_score + vvix_score + move_score

    price_bias = classify_bias(price_score, config.price_bias_threshold)
    vol_bias = classify_bias(vol_score, config.vol_bias_threshold)
    regime, alignment_state = collapse_regime(price_bias, vol_bias, price_score, vol_score, config)
    confidence = compute_confidence(price_score, vol_score, price_bias, vol_bias, penalty=qqq_penalty)

    return {
        "window_days": "30-90" if long_term else "7-21",
        "price_bias": price_bias,
        "vol_bias": vol_bias,
        "regime": regime,
        "alignment_state": alignment_state,
        "confidence": confidence,
        "scores": {
            "price_score": price_score,
            "vol_score": vol_score,
            "qqq_price_component": qqq_score,
            "macro_price_component": macro_score,
            "futures_price_component": futures_score,
            "curve_vol_component": curve_score,
            "vvix_vol_component": vvix_score,
            "move_vol_component": move_score,
        },
        "factors": {
            "qqq": qqq_details,
            "macro": macro_details,
            "futures": futures_details,
            "vol_curve": curve_details,
            "vvix": vvix_details,
            "move": move_details,
        },
    }


# -------------------- DECISION BUILDER --------------------
def build_market_decision(config: Optional[RuntimeConfig] = None) -> Dict[str, object]:
    config = config or RuntimeConfig()
    cached = load_cache()
    cache_hit = bool(cached)

    if cached:
        market_snapshot = cached["market_snapshot"]
        macro_snapshot = cached["macro_snapshot"]
    else:
        market_snapshot = fetch_market_snapshot()
        macro_snapshot = fetch_macro_snapshot()
        save_cache({
            "market_snapshot": market_snapshot,
            "macro_snapshot": macro_snapshot,
        })

    short_term = build_horizon_block(market=market_snapshot, macro=macro_snapshot, horizon="SHORT_TERM", config=config)
    long_term = build_horizon_block(market=market_snapshot, macro=macro_snapshot, horizon="LONG_TERM", config=config)

    data_quality = {
        "cache_hit": cache_hit,
        "fetched_at_market": market_snapshot.get("fetched_at"),
        "fetched_at_macro": macro_snapshot.get("fetched_at"),
        "macro_status": macro_snapshot.get("status", "UNKNOWN"),
        "missing_symbols": sorted(list(market_snapshot.get("errors", {}).keys())),
        "available_symbols": sorted(list(market_snapshot.get("symbols", {}).keys())),
    }

    decision = {
        "agent": "market_analyst",
        "schema_version": "2.0.0",
        "event_id": now_iso(),
        "data_quality": data_quality,
        "short_term": short_term,
        "long_term": long_term,
        "components": {
            "qqq": market_snapshot["symbols"].get(QQQ_SYMBOL),
            "vol_curve": {SYMBOL_LABELS[s]: market_snapshot["symbols"].get(s) for s in VOL_CURVE_SYMBOLS},
            "vvix": market_snapshot["symbols"].get(VVIX_SYMBOL),
            "move": market_snapshot["symbols"].get(MOVE_SYMBOL),
            "futures": market_snapshot["symbols"].get(FUTURES_SYMBOL) if USE_FUTURES_CONTEXT else None,
            "macro": macro_snapshot,
        },
        "summary": (
            f"Short-term regime is {short_term['regime']} with price bias {short_term['price_bias']} and vol bias {short_term['vol_bias']}. "
            f"Long-term regime is {long_term['regime']} with price bias {long_term['price_bias']} and vol bias {long_term['vol_bias']}."
        ),
    }
    return decision


# -------------------- OPTIONAL LLM SUMMARY --------------------
def add_llm_summary(decision: Dict[str, object]) -> Dict[str, object]:
    llm = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.2,
    )

    prompt_payload = {
        "short_term": decision["short_term"],
        "long_term": decision["long_term"],
        "summary": decision["summary"],
    }

    prompt = (
        "You are a concise market narrator. "
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
        decision = build_market_decision()

        if USE_LLM_SUMMARY:
            decision = add_llm_summary(decision)

        print(json.dumps(decision, indent=2))

        report_path = REPORTS_DIR / f"market_report_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(decision, f, indent=2)

        print(f"\n[INFO] Report written to {report_path}")
    except Exception as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)
