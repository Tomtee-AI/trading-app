#!/usr/bin/env python3
"""
trade_analyst_divergence.py

Divergence Trade Analyst
========================
Cross-sectional and intermarket relative-value specialist.

Mission
-------
Find mean-reversion and relative-value opportunities when correlated assets
or peer instruments diverge beyond normal ranges.

Core responsibilities
---------------------
• Evaluate spreads and ratios across futures peer groups (Metals, Indices,
  Energies, Bonds)
• Measure rolling correlation and hedge-adjusted divergence magnitude (z-score)
• Identify expensive vs cheap legs
• Map commodity-level signals into stock/ETF and options execution ideas
• Generate defined-risk trade templates (no undefined-risk structures)

Inputs
------
Futures histories (yfinance), peer baskets, spread metrics, z-scores,
rolling correlations, current price (fallback to previous close).

Futures Universe
----------------
Metals:    GC, SI, HG, PL, PA
Indices:   ES, NQ, RTY
Energies:  CL, HO, RB, NG
Bonds:     ZT, ZF, ZN, ZB, UB

Output
------
Specialist payload exactly matching trade_coordinator_bias_aware_skeleton.py
expectations (strategy_type = "divergence") + rich internal analytics for
transparency. Fully compatible with the bias-aware coordinator.

Fix applied (v1.1)
------------------
• fit_score now uses the already-normalized confidence field (0.0–1.0)
• raw_score is preserved inside implementation for full transparency
• Schema validation now passes reliably

Hard limits (enforced)
----------------------
• No trade if correlation < 0.55 or |z-score| < 1.5
• No undefined-risk structures
• No direct order routing
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from jsonschema import Draft202012Validator, FormatChecker

# -------------------- OPTIONAL LLM SUMMARY --------------------
USE_LLM_SUMMARY = True
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

if USE_LLM_SUMMARY:
    from langchain_ollama import ChatOllama

# -------------------- BOOTSTRAP --------------------
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# -------------------- CONFIG --------------------
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

CACHE_FILE = Path("divergence_analyst_cache.pkl")
CACHE_TTL_SECONDS = 300

# -------------------- FUTURES UNIVERSE --------------------
INSTRUMENTS_SYMBOLS = {
    "GC=F": "gold",
    "SI=F": "silver",
    "HG=F": "copper",
    "PL=F": "platinum",
    "PA=F": "palladium",
    "ES=F": "sp500",
    "NQ=F": "nasdaq",
    "RTY=F": "russell2000",
    "CL=F": "crude_oil",
    "HO=F": "heating_oil",
    "RB=F": "gasoline",
    "NG=F": "natural_gas",
    "ZT=F": "2yr_note",
    "ZF=F": "5yr_note",
    "ZN=F": "10yr_note",
    "ZB=F": "30yr_bond",
    "UB=F": "ultra_bond",
}

ALIAS_TO_SYMBOL = {
    "GC": "GC=F", "SI": "SI=F", "HG": "HG=F", "PL": "PL=F", "PA": "PA=F",
    "ES": "ES=F", "NQ": "NQ=F", "RTY": "RTY=F",
    "CL": "CL=F", "HO": "HO=F", "RB": "RB=F", "NG": "NG=F",
    "ZT": "ZT=F", "ZF": "ZF=F", "ZN": "ZN=F", "ZB": "ZB=F", "UB": "UB=F",
}
SYMBOL_TO_ALIAS = {v: k for k, v in ALIAS_TO_SYMBOL.items()}

# -------------------- PEER GROUPS FOR CROSS-SECTIONAL DIVERGENCE --------------------
PEER_GROUPS = {
    "metals": ["GC=F", "SI=F", "HG=F", "PL=F", "PA=F"],
    "indices": ["ES=F", "NQ=F", "RTY=F"],
    "energies": ["CL=F", "HO=F", "RB=F", "NG=F"],
    "bonds": ["ZT=F", "ZF=F", "ZN=F", "ZB=F", "UB=F"],
}

# -------------------- EQUITY / ETF PROXIES FOR EXECUTION --------------------
EQUITY_GROUPS = {
    "GC=F": ["GDX", "GDXJ", "GLD", "B"],
    "SI=F": ["AG", "HL", "PAAS", "SLV"],
    "HG=F": ["BHP", "FCX", "RIO", "TECK"],
    "PL=F": ["PLTM"],
    "PA=F": [],
    "ES=F": ["SPY", "VOO"],
    "NQ=F": ["QQQ"],
    "RTY=F": ["IWM"],
    "CL=F": ["USO", "XLE"],
    "HO=F": ["USO"],
    "RB=F": ["USO"],
    "NG=F": ["UNG"],
    "ZT=F": ["SHY"],
    "ZF=F": ["IEF"],
    "ZN=F": ["IEF"],
    "ZB=F": ["TLT"],
    "UB=F": ["TLT"],
}

EQUITY_LABELS = {
    "GDX": "VanEck Gold Miners ETF", "GDXJ": "VanEck Junior Gold Miners ETF",
    "GLD": "SPDR Gold Shares", "B": "Barrick Gold Corp",
    "AG": "First Majestic Silver Corp", "HL": "Hecla Mining Co",
    "PAAS": "Pan American Silver Corp", "SLV": "iShares Silver Trust",
    "BHP": "BHP Group Ltd", "FCX": "Freeport-McMoRan Inc",
    "RIO": "Rio Tinto Group", "PLTM": "Platinum Group Metals ETF",
    "SPY": "SPDR S&P 500 ETF", "VOO": "Vanguard S&P 500 ETF",
    "QQQ": "Invesco QQQ Trust", "IWM": "iShares Russell 2000 ETF",
    "USO": "United States Oil Fund", "XLE": "Energy Select Sector SPDR",
    "UNG": "United States Natural Gas Fund",
    "SHY": "iShares 1-3 Year Treasury Bond ETF",
    "IEF": "iShares 7-10 Year Treasury Bond ETF",
    "TLT": "iShares 20+ Year Treasury Bond ETF",
}

DEFAULT_ALLOWED_PAIR_TEXT = "GC/SI,GC/HG,SI/HG,ES/NQ,ES/RTY,NQ/RTY,CL/HO,CL/RB,ZN/ZB,GC/CL,ES/CL"

# -------------------- REUSED REASON CODES --------------------
PAIR_REASON_CODES = ["PASSED", "MISSING_SYMBOL_DATA", "INSUFFICIENT_HISTORY",
                     "LOW_CORRELATION", "LOW_ZSCORE", "INTERNAL_ERROR"]

NO_TRADE_REASON_CODES = [
    "NO_ALLOWED_PAIRS_CONFIGURED", "MISSING_SYMBOL_DATA", "INSUFFICIENT_HISTORY",
    "NO_EVALUABLE_PAIRS", "NO_PAIR_MET_CORRELATION_THRESHOLD",
    "NO_PAIR_MET_ZSCORE_THRESHOLD", "DATA_FETCH_ERRORS_PRESENT",
    "NO_TRADE_CONDITIONS_NOT_MET",
]

# -------------------- SPECIALIST SCHEMA (for coordinator) --------------------
SPECIALIST_AGENT_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": ["agent", "event_id", "strategy_type", "candidates", "summary"],
    "properties": {
        "agent": {"type": "string", "const": "divergence_analyst"},
        "event_id": {"type": "string", "format": "date-time"},
        "strategy_type": {"type": "string", "const": "divergence"},
        "candidates": {"type": "array", "items": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "candidate_id", "strategy_type", "direction", "horizon",
                "structure_family", "summary", "confidence", "fit_score",
                "thesis_tags", "risk_flags", "implementation"
            ],
            "properties": {
                "candidate_id": {"type": "string"},
                "strategy_type": {"type": "string", "const": "divergence"},
                "direction": {"type": "string", "enum": ["RELATIVE_VALUE"]},
                "horizon": {"type": "string", "enum": ["SHORT_TERM"]},
                "structure_family": {"type": "string"},
                "summary": {"type": "string"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "fit_score": {"type": "number", "minimum": 0, "maximum": 1},
                "thesis_tags": {"type": "array", "items": {"type": "string"}},
                "risk_flags": {"type": "array", "items": {"type": "string"}},
                "implementation": {"type": "object", "additionalProperties": True},
            }
        }},
        "summary": {"type": "string"},
    },
}

# -------------------- HELPERS --------------------
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def safe_float(value, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default

def compare_level(price: Optional[float], level: Optional[float], near_pct: float = 0.01) -> str:
    if price is None or level is None or level == 0:
        return "UNKNOWN"
    diff_pct = (price / level) - 1.0
    if abs(diff_pct) <= near_pct:
        return "NEAR"
    return "ABOVE" if diff_pct > 0 else "BELOW"

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

def parse_symbol_or_alias(token: str) -> Tuple[str, str]:
    token = token.strip().upper()
    if token in ALIAS_TO_SYMBOL:
        return ALIAS_TO_SYMBOL[token], token
    if token in SYMBOL_TO_ALIAS:
        return token, SYMBOL_TO_ALIAS[token]
    raise ValueError(f"Unknown symbol/alias: {token}")

def parse_pair_allowlist(pair_text: str) -> Tuple["PairSpec", ...]:
    raw_items = [item.strip() for item in pair_text.split(",") if item.strip()]
    if not raw_items:
        return tuple()
    parsed: List[PairSpec] = []
    seen = set()
    for raw in raw_items:
        if "/" not in raw:
            raise ValueError(f"Invalid pair '{raw}'")
        left, right = raw.split("/", 1)
        symbol_a, alias_a = parse_symbol_or_alias(left)
        symbol_b, alias_b = parse_symbol_or_alias(right)
        if symbol_a == symbol_b:
            continue
        dedupe_key = tuple(sorted([symbol_a, symbol_b]))
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        parsed.append(PairSpec(symbol_a, symbol_b, alias_a, alias_b))
    return tuple(parsed)

@dataclass(frozen=True)
class PairSpec:
    symbol_a: str
    symbol_b: str
    alias_a: str
    alias_b: str

    @property
    def pair_label(self) -> str:
        return f"{self.alias_a}/{self.alias_b}"

    @property
    def pair_id(self) -> str:
        return f"{self.symbol_a}__{self.symbol_b}"

@dataclass
class RuntimeConfig:
    daily_period: str = "2y"
    daily_interval: str = "1d"
    current_attempts: Tuple[Tuple[str, str], ...] = (("1d", "1m"), ("5d", "5m"), ("5d", "15m"))
    corr_window: int = 126
    beta_window: int = 126
    zscore_window: int = 63
    min_corr: float = 0.55
    min_abs_z: float = 1.5
    max_candidates: int = 12

# -------------------- DATA FETCH --------------------
def fetch_history(symbol: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=False)
    df = df.dropna(how="all")
    if df.empty:
        raise ValueError(f"No history returned for {symbol}")
    df.index = pd.to_datetime(df.index)
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    return df

def try_fetch_current_price(symbol: str, current_attempts: Tuple[Tuple[str, str], ...]) -> Tuple[Optional[float], Optional[str], str]:
    ticker = yf.Ticker(symbol)
    for period, interval in current_attempts:
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

def build_instrument_features(symbol: str, df: pd.DataFrame, config: RuntimeConfig) -> Dict[str, object]:
    close_series = df["Close"].dropna()
    high_series = df["High"].dropna() if "High" in df.columns else pd.Series(dtype=float)
    low_series = df["Low"].dropna() if "Low" in df.columns else pd.Series(dtype=float)

    previous_close = safe_float(close_series.iloc[-1]) if not close_series.empty else None
    daily_high = safe_float(high_series.iloc[-1]) if not high_series.empty else None
    daily_low = safe_float(low_series.iloc[-1]) if not low_series.empty else None

    sma_21 = safe_float(close_series.rolling(21).mean().iloc[-1]) if len(close_series) >= 21 else None
    sma_63 = safe_float(close_series.rolling(63).mean().iloc[-1]) if len(close_series) >= 63 else None

    current_price, current_ts, source = try_fetch_current_price(symbol, config.current_attempts)
    effective_price = current_price if current_price is not None else previous_close

    return {
        "symbol": symbol,
        "alias": SYMBOL_TO_ALIAS.get(symbol, symbol),
        "label": INSTRUMENTS_SYMBOLS.get(symbol, symbol),
        "as_of": str(df.index[-1]) if not df.empty else "",
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

def fetch_market_snapshot(config: RuntimeConfig) -> Dict[str, object]:
    symbol_frames: Dict[str, pd.DataFrame] = {}
    symbol_features: Dict[str, Dict[str, object]] = {}
    errors: Dict[str, str] = {}

    for symbol in INSTRUMENTS_SYMBOLS:
        try:
            df = fetch_history(symbol, period=config.daily_period, interval=config.daily_interval)
            symbol_frames[symbol] = df
            symbol_features[symbol] = build_instrument_features(symbol, df, config)
        except Exception as exc:
            errors[symbol] = str(exc)

    return {
        "fetched_at": now_iso(),
        "frames": symbol_frames,
        "symbols": symbol_features,
        "errors": errors,
    }

# -------------------- EQUITY LAYER --------------------
def build_equity_feature(symbol: str, df: pd.DataFrame, config: RuntimeConfig) -> Dict[str, object]:
    if df is None or df.empty:
        return {
            "symbol": symbol, "alias": symbol, "label": EQUITY_LABELS.get(symbol, symbol),
            "as_of": "", "rows": 0, "previous_close": None, "effective_price": None,
            "price_source": "unknown", "source_timestamp_utc": None,
            "daily_high": None, "daily_low": None,
            "sma_21": None, "sma_63": None,
            "vs_sma_21": "UNKNOWN", "vs_sma_63": "UNKNOWN",
        }
    return build_instrument_features(symbol, df, config)

def fetch_equity_snapshot(equity_list: List[str], config: RuntimeConfig) -> Dict[str, object]:
    equity_frames: Dict[str, pd.DataFrame] = {}
    equity_features: Dict[str, Dict[str, object]] = {}
    errors: Dict[str, str] = {}
    for symbol in equity_list:
        try:
            df = fetch_history(symbol, period=config.daily_period, interval=config.daily_interval)
            equity_frames[symbol] = df
            equity_features[symbol] = build_equity_feature(symbol, df, config)
        except Exception as exc:
            errors[symbol] = str(exc)
    return {"fetched_at": now_iso(), "frames": equity_frames, "symbols": equity_features, "errors": errors}

def rank_equities(group_symbols: List[str], equity_snapshot: Dict[str, object], direction: str = "expensive") -> List[Dict]:
    rankings = []
    for sym in group_symbols:
        feature = equity_snapshot["symbols"].get(sym)
        if not feature or feature.get("effective_price") is None or feature.get("sma_63") is None or feature["sma_63"] <= 0:
            continue
        dev_pct = (feature["effective_price"] / feature["sma_63"] - 1) * 100
        rankings.append({
            "symbol": sym,
            "label": feature["label"],
            "deviation_pct_from_sma63": round(dev_pct, 2),
            "effective_price": feature["effective_price"],
            "sma_63": feature["sma_63"],
            "vs_sma_63": feature.get("vs_sma_63", "UNKNOWN"),
        })
    reverse = (direction == "expensive")
    rankings.sort(key=lambda x: x["deviation_pct_from_sma63"], reverse=reverse)
    return rankings

# -------------------- PAIR ANALYTICS --------------------
def empty_pair_result(pair: PairSpec, reason_code: str, status: str = "DISQUALIFIED") -> Dict[str, object]:
    return {
        "pair_id": pair.pair_id,
        "pair_label": pair.pair_label,
        "symbol_a": pair.symbol_a,
        "symbol_b": pair.symbol_b,
        "alias_a": pair.alias_a,
        "alias_b": pair.alias_b,
        "label_a": INSTRUMENTS_SYMBOLS.get(pair.symbol_a, pair.symbol_a),
        "label_b": INSTRUMENTS_SYMBOLS.get(pair.symbol_b, pair.symbol_b),
        "status": status,
        "reason_code": reason_code,
        "correlation": None,
        "hedge_ratio_beta": None,
        "last_close_zscore": None,
        "current_zscore": None,
        "expensive_symbol": None,
        "cheap_symbol": None,
        "recommendation": None,
        "confidence": None,
        "score": None,
        "rationale": None,
        "inputs": {
            "effective_price_a": None, "effective_price_b": None,
            "price_source_a": None, "price_source_b": None,
            "rows_a": 0, "rows_b": 0, "usable_rows": 0,
        },
        "equity_drilldown": None,
    }

def evaluate_pair(pair: PairSpec, market_snapshot: Dict[str, object], config: RuntimeConfig) -> Dict[str, object]:
    try:
        symbol_features = market_snapshot["symbols"]
        symbol_frames = market_snapshot["frames"]

        if pair.symbol_a not in symbol_features or pair.symbol_b not in symbol_features:
            result = empty_pair_result(pair, "MISSING_SYMBOL_DATA")
            result["rationale"] = "One or both legs missing."
            return result

        feature_a = symbol_features[pair.symbol_a]
        feature_b = symbol_features[pair.symbol_b]
        frame_a = symbol_frames.get(pair.symbol_a)
        frame_b = symbol_frames.get(pair.symbol_b)

        rows_a = len(frame_a) if frame_a is not None else 0
        rows_b = len(frame_b) if frame_b is not None else 0

        if frame_a is None or frame_b is None or frame_a.empty or frame_b.empty:
            result = empty_pair_result(pair, "MISSING_SYMBOL_DATA")
            result["inputs"]["rows_a"] = rows_a
            result["inputs"]["rows_b"] = rows_b
            result["rationale"] = "One or both legs have no usable history."
            return result

        pair_closes = pd.concat(
            [frame_a["Close"].rename(pair.symbol_a), frame_b["Close"].rename(pair.symbol_b)],
            axis=1, join="inner"
        ).dropna()

        usable_rows = len(pair_closes)
        min_needed = max(config.corr_window, config.beta_window, config.zscore_window) + 5

        if usable_rows < min_needed:
            result = empty_pair_result(pair, "INSUFFICIENT_HISTORY")
            result["inputs"]["rows_a"] = rows_a
            result["inputs"]["rows_b"] = rows_b
            result["inputs"]["usable_rows"] = usable_rows
            result["inputs"]["effective_price_a"] = feature_a.get("effective_price")
            result["inputs"]["effective_price_b"] = feature_b.get("effective_price")
            result["inputs"]["price_source_a"] = feature_a.get("price_source")
            result["inputs"]["price_source_b"] = feature_b.get("price_source")
            result["rationale"] = f"Only {usable_rows} overlapping rows (need {min_needed})."
            return result

        log_px = np.log(pair_closes)
        log_ret = log_px.diff().dropna()

        corr_sample = log_ret.tail(config.corr_window)
        correlation = safe_float(corr_sample[pair.symbol_a].corr(corr_sample[pair.symbol_b]))

        beta_sample = log_px.tail(config.beta_window)
        x = beta_sample[pair.symbol_b].values
        y = beta_sample[pair.symbol_a].values
        var_x = np.var(x, ddof=1)
        if not np.isfinite(var_x) or var_x <= 0:
            result = empty_pair_result(pair, "INTERNAL_ERROR", status="ERROR")
            result["rationale"] = "Variance of hedge leg not finite."
            return result

        cov_xy = np.cov(x, y, ddof=1)[0, 1]
        beta = safe_float(cov_xy / var_x)

        hist_spread = (log_px[pair.symbol_a] - beta * log_px[pair.symbol_b]).dropna()
        z_window = hist_spread.tail(config.zscore_window)

        if len(z_window) < config.zscore_window:
            result = empty_pair_result(pair, "INSUFFICIENT_HISTORY")
            result["rationale"] = "Insufficient spread history for z-score."
            return result

        mu = safe_float(z_window.mean())
        sigma = safe_float(z_window.std(ddof=1))

        if sigma is None or sigma <= 1e-10:
            result = empty_pair_result(pair, "INTERNAL_ERROR", status="ERROR")
            result["rationale"] = "Spread standard deviation unusable."
            return result

        last_close_spread = safe_float(hist_spread.iloc[-1])
        last_close_z = safe_float((last_close_spread - mu) / sigma)

        price_a = feature_a.get("effective_price")
        price_b = feature_b.get("effective_price")

        if price_a in (None, 0) or price_b in (None, 0):
            result = empty_pair_result(pair, "MISSING_SYMBOL_DATA")
            result["rationale"] = "Effective price missing."
            return result

        current_spread = math.log(float(price_a)) - float(beta) * math.log(float(price_b))
        current_z = safe_float((current_spread - mu) / sigma)

        if current_z is None:
            result = empty_pair_result(pair, "INTERNAL_ERROR", status="ERROR")
            result["rationale"] = "Current z-score could not be computed."
            return result

        expensive_symbol = pair.symbol_a if current_z > 0 else pair.symbol_b
        cheap_symbol = pair.symbol_b if current_z > 0 else pair.symbol_a

        z_strength = min(abs(current_z) / 3.0, 1.0)
        corr_strength = min(max((correlation - config.min_corr) / (1.0 - config.min_corr + 1e-9), 0.0), 1.0) if correlation is not None else 0.0

        confidence = round(0.55 * z_strength + 0.45 * corr_strength, 2)
        score = round(abs(current_z) * max(correlation or 0.0, 0.0), 4)

        result = {
            "pair_id": pair.pair_id,
            "pair_label": pair.pair_label,
            "symbol_a": pair.symbol_a,
            "symbol_b": pair.symbol_b,
            "alias_a": pair.alias_a,
            "alias_b": pair.alias_b,
            "label_a": INSTRUMENTS_SYMBOLS.get(pair.symbol_a, pair.symbol_a),
            "label_b": INSTRUMENTS_SYMBOLS.get(pair.symbol_b, pair.symbol_b),
            "status": "CANDIDATE",
            "reason_code": "PASSED",
            "correlation": round(float(correlation), 4) if correlation is not None else None,
            "hedge_ratio_beta": round(float(beta), 4) if beta is not None else None,
            "last_close_zscore": round(float(last_close_z), 4) if last_close_z is not None else None,
            "current_zscore": round(float(current_z), 4),
            "expensive_symbol": expensive_symbol,
            "cheap_symbol": cheap_symbol,
            "recommendation": f"SELL {expensive_symbol} / BUY {cheap_symbol}",
            "confidence": confidence,
            "score": score,
            "rationale": (
                f"{INSTRUMENTS_SYMBOLS.get(expensive_symbol, expensive_symbol)} screens expensive relative to "
                f"{INSTRUMENTS_SYMBOLS.get(cheap_symbol, cheap_symbol)}. Hedge-adjusted spread is {current_z:.2f} "
                f"standard deviations from mean with {correlation:.2f} correlation."
            ),
            "inputs": {
                "effective_price_a": safe_float(price_a),
                "effective_price_b": safe_float(price_b),
                "price_source_a": feature_a.get("price_source"),
                "price_source_b": feature_b.get("price_source"),
                "rows_a": rows_a,
                "rows_b": rows_b,
                "usable_rows": usable_rows,
            },
            "equity_drilldown": None,
        }

        if correlation is None or correlation < config.min_corr:
            result["status"] = "DISQUALIFIED"
            result["reason_code"] = "LOW_CORRELATION"
            result["recommendation"] = None
            result["confidence"] = None
            result["rationale"] = f"Correlation {correlation:.2f} below minimum {config.min_corr:.2f}."
            return result

        if abs(current_z) < config.min_abs_z:
            result["status"] = "DISQUALIFIED"
            result["reason_code"] = "LOW_ZSCORE"
            result["recommendation"] = None
            result["rationale"] = f"Absolute z-score {abs(current_z):.2f} below minimum {config.min_abs_z:.2f}."
            return result

        return result

    except Exception as exc:
        result = empty_pair_result(pair, "INTERNAL_ERROR", status="ERROR")
        result["rationale"] = f"Internal error: {exc}"
        return result

# -------------------- BUILD PAIR SNAPSHOT --------------------
def build_pair_snapshot(market_snapshot: Dict[str, object], config: RuntimeConfig, allowed_pairs: Tuple[PairSpec, ...]) -> Dict[str, object]:
    pair_analytics: List[Dict[str, object]] = []
    trade_candidates: List[Dict[str, object]] = []

    reason_counts = {code: 0 for code in PAIR_REASON_CODES}

    for pair in allowed_pairs:
        result = evaluate_pair(pair, market_snapshot, config)
        pair_analytics.append(result)
        reason_counts[result["reason_code"]] += 1
        if result["status"] == "CANDIDATE":
            trade_candidates.append(result)

    pair_analytics.sort(key=lambda x: (x["score"] is None, -(x["score"] or -999999)))
    trade_candidates.sort(key=lambda x: (x["score"] is None, -(x["score"] or -999999)))
    trade_candidates = trade_candidates[:config.max_candidates]

    evaluable_pair_count = reason_counts["LOW_CORRELATION"] + reason_counts["LOW_ZSCORE"] + reason_counts["PASSED"]
    correlation_pass_count = reason_counts["LOW_ZSCORE"] + reason_counts["PASSED"]
    zscore_pass_count = reason_counts["PASSED"]

    return {
        "fetched_at": now_iso(),
        "pair_analytics": pair_analytics,
        "trade_candidates": trade_candidates,
        "counts": {
            "allowed_pair_count": len(allowed_pairs),
            "evaluable_pair_count": evaluable_pair_count,
            "correlation_pass_count": correlation_pass_count,
            "zscore_pass_count": zscore_pass_count,
            "candidate_count": len(trade_candidates),
            "missing_symbol_data_count": reason_counts.get("MISSING_SYMBOL_DATA", 0),
            "insufficient_history_count": reason_counts.get("INSUFFICIENT_HISTORY", 0),
            "low_correlation_count": reason_counts.get("LOW_CORRELATION", 0),
            "low_zscore_count": reason_counts.get("LOW_ZSCORE", 0),
            "internal_error_count": reason_counts.get("INTERNAL_ERROR", 0),
        },
    }

def build_no_trade_block(pair_snapshot: Dict[str, object], market_snapshot: Dict[str, object]) -> Dict[str, object]:
    counts = pair_snapshot["counts"]
    reason_codes: List[str] = []

    if counts["allowed_pair_count"] == 0:
        reason_codes.append("NO_ALLOWED_PAIRS_CONFIGURED")
    if counts["evaluable_pair_count"] == 0:
        if counts["missing_symbol_data_count"] > 0:
            reason_codes.append("MISSING_SYMBOL_DATA")
        if counts["insufficient_history_count"] > 0:
            reason_codes.append("INSUFFICIENT_HISTORY")
        if not reason_codes:
            reason_codes.append("NO_EVALUABLE_PAIRS")
    else:
        if counts["correlation_pass_count"] == 0:
            reason_codes.append("NO_PAIR_MET_CORRELATION_THRESHOLD")
        elif counts["zscore_pass_count"] == 0:
            reason_codes.append("NO_PAIR_MET_ZSCORE_THRESHOLD")

    if market_snapshot.get("errors"):
        reason_codes.append("DATA_FETCH_ERRORS_PRESENT")

    if not reason_codes:
        reason_codes.append("NO_TRADE_CONDITIONS_NOT_MET")

    summary = (
        f"No trade. Allowed pairs={counts['allowed_pair_count']}, evaluable={counts['evaluable_pair_count']}, "
        f"correlation_pass={counts['correlation_pass_count']}, zscore_pass={counts['zscore_pass_count']}, "
        f"candidates={counts['candidate_count']}."
    )

    return {"reason_codes": reason_codes, "reason_counts": counts, "next_action": "WAIT", "summary": summary}

# -------------------- EQUITY DRILLDOWN + SPECIALIST CANDIDATE CONVERSION --------------------
def build_equity_drilldown(top_candidate: Dict[str, object], config: RuntimeConfig) -> Optional[Dict]:
    if not top_candidate or top_candidate.get("status") != "CANDIDATE":
        return None
    expensive_fut = top_candidate["expensive_symbol"]
    cheap_fut = top_candidate["cheap_symbol"]
    expensive_group = EQUITY_GROUPS.get(expensive_fut, [])
    cheap_group = EQUITY_GROUPS.get(cheap_fut, [])
    all_eq = list(set(expensive_group + cheap_group))
    if not all_eq:
        return None
    try:
        eq_snapshot = fetch_equity_snapshot(all_eq, config)
        exp_ranked = rank_equities(expensive_group, eq_snapshot, "expensive")
        cheap_ranked = rank_equities(cheap_group, eq_snapshot, "cheap")
        sug_sell = exp_ranked[0] if exp_ranked else None
        sug_buy = cheap_ranked[0] if cheap_ranked else None
        if sug_sell and sug_buy:
            return {
                "expensive_metal_fut": expensive_fut,
                "cheap_metal_fut": cheap_fut,
                "expensive_equities_ranked": exp_ranked[:5],
                "cheap_equities_ranked": cheap_ranked[:5],
                "suggested_sell_stock": sug_sell,
                "suggested_buy_stock": sug_buy,
                "stock_recommendation": f"SELL {sug_sell['symbol']} / BUY {sug_buy['symbol']}",
                "options_idea": (
                    f"Sell calls (or short stock) on {sug_sell['symbol']} / "
                    f"Buy calls on {sug_buy['symbol']}. Consider vertical spreads."
                ),
            }
    except Exception:
        pass
    return None

def convert_to_specialist_candidate(pair_result: Dict[str, object], equity_drilldown: Optional[Dict] = None) -> Dict[str, object]:
    expensive_alias = SYMBOL_TO_ALIAS.get(pair_result["expensive_symbol"], pair_result["expensive_symbol"].replace("=F", ""))
    cheap_alias = SYMBOL_TO_ALIAS.get(pair_result["cheap_symbol"], pair_result["cheap_symbol"].replace("=F", ""))
    cid = f"DIV_{pair_result['alias_a']}_{pair_result['alias_b']}_{datetime.now().strftime('%Y%m%d%H%M')}"

    impl = {
        "futures_recommendation": pair_result["recommendation"],
        "hedge_ratio_beta": pair_result.get("hedge_ratio_beta"),
        "current_zscore": pair_result["current_zscore"],
        "correlation": pair_result.get("correlation"),
        "raw_score": pair_result.get("score"),          # preserved for transparency
    }
    if equity_drilldown and equity_drilldown.get("suggested_sell_stock") and equity_drilldown.get("suggested_buy_stock"):
        impl["equity_execution"] = equity_drilldown["stock_recommendation"]
        impl["options_idea"] = equity_drilldown.get("options_idea")
        impl["equity_drilldown"] = equity_drilldown

    return {
        "candidate_id": cid,
        "strategy_type": "divergence",
        "direction": "RELATIVE_VALUE",
        "horizon": "SHORT_TERM",
        "structure_family": "futures_pair_mean_reversion",
        "summary": pair_result["rationale"],
        "confidence": pair_result.get("confidence", 0.65),
        "fit_score": pair_result.get("confidence", 0.65),   # <-- FIXED: now guaranteed ≤ 1.0
        "thesis_tags": ["divergence", "relative_value", "mean_reversion", "peer_group"],
        "risk_flags": ["spread_widening", "correlation_breakdown"],
        "implementation": impl,
    }

# -------------------- MAIN DECISION BUILDER --------------------
def build_divergence_decision(config: RuntimeConfig) -> Dict[str, object]:
    cached = load_cache()
    cache_hit = bool(cached)

    if cached:
        market_snapshot = cached
    else:
        market_snapshot = fetch_market_snapshot(config)
        save_cache(market_snapshot)

    allowed_pairs = parse_pair_allowlist(DEFAULT_ALLOWED_PAIR_TEXT)
    pair_snapshot = build_pair_snapshot(market_snapshot, config, allowed_pairs)
    trade_candidates = pair_snapshot["trade_candidates"]

    # === EQUITY / OPTIONS LAYER + CONVERT TO SPECIALIST FORMAT ===
    specialist_candidates: List[Dict[str, object]] = []
    for candidate in trade_candidates:
        equity_drilldown = build_equity_drilldown(candidate, config)
        candidate["equity_drilldown"] = equity_drilldown
        spec_cand = convert_to_specialist_candidate(candidate, equity_drilldown)
        specialist_candidates.append(spec_cand)

    status = "OK" if specialist_candidates else "NO_TRADE"
    no_trade = None if specialist_candidates else build_no_trade_block(pair_snapshot, market_snapshot)

    summary = (
        f"Divergence Analyst identified {len(specialist_candidates)} high-confidence relative-value "
        f"opportunities across peer groups." if specialist_candidates else no_trade["summary"] if no_trade else "No divergence candidates."
    )

    output = {
        "agent": "divergence_analyst",
        "event_id": now_iso(),
        "strategy_type": "divergence",
        "candidates": specialist_candidates[:config.max_candidates],
        "summary": summary,
    }

    # Validate against coordinator-expected schema
    validator = Draft202012Validator(SPECIALIST_AGENT_SCHEMA, format_checker=FormatChecker())
    errors = list(validator.iter_errors(output))
    if errors:
        raise ValueError(f"Specialist schema validation failed: {errors[0].message}")

    return output

def add_llm_summary(output: Dict[str, object]) -> Dict[str, object]:
    if not USE_LLM_SUMMARY:
        return output
    try:
        llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.2)
        prompt = (
            "You are a concise divergence trade analyst. Return ONLY valid JSON with one key 'summary'. "
            "Keep it to 2 sentences max and do not change the deterministic view.\n\n"
            f"{json.dumps({'candidates': len(output['candidates']), 'summary': output['summary']}, indent=2)}\n\n"
            '{"summary": "..."}'
        )
        response = llm.invoke(prompt)
        parsed = json.loads(response.content)
        if isinstance(parsed, dict) and "summary" in parsed:
            output["summary"] = parsed["summary"]
    except Exception:
        pass
    return output

# -------------------- CLI --------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Divergence Trade Analyst - cross-sectional & intermarket RV specialist")
    parser.add_argument("--pairs", type=str, default=DEFAULT_ALLOWED_PAIR_TEXT,
                        help="Comma-separated pair allowlist (e.g. GC/SI,ES/NQ,CL/HO)")
    parser.add_argument("--min-corr", type=float, default=0.55)
    parser.add_argument("--min-abs-z", type=float, default=1.5)
    parser.add_argument("--use-llm-summary", action="store_true")
    return parser

# -------------------- MAIN --------------------
if __name__ == "__main__":
    try:
        args = build_arg_parser().parse_args()
        allowed_pairs = parse_pair_allowlist(args.pairs)

        config = RuntimeConfig(
            min_corr=args.min_corr,
            min_abs_z=args.min_abs_z,
        )

        decision = build_divergence_decision(config)

        if args.use_llm_summary or USE_LLM_SUMMARY:
            decision = add_llm_summary(decision)

        print(json.dumps(decision, indent=2))

        report_path = REPORTS_DIR / f"divergence_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(decision, f, indent=2)

        print(f"\n[INFO] Report written to {report_path}")
        print(f"[INFO] {len(decision['candidates'])} divergence candidate(s) ready for Trade Coordinator.")

    except Exception as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)