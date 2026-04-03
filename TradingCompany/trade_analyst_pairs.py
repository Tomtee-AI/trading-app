#!/usr/bin/env python3
"""
trade_analyst_metals_refactored.py

Standalone Trade Analyst for metals relative-value trading.

Added in this refactor
----------------------
1. Strict JSON schema validation for final output
2. Explicit NO_TRADE reason-code block
3. Configurable pair allowlist, e.g.:
      GC/SI
      PL/PA
      GC/PA

Default behavior
----------------
- Fetches 2 years of daily history for:
    GC=F  Gold
    SI=F  Silver
    HG=F  Copper
    PL=F  Platinum
    PA=F  Palladium
- Prefers current intraday price when available
- Falls back to previous daily close if current price is unavailable
- Only evaluates allowed pairs
- Computes hedge-ratio-adjusted log spread z-scores
- Emits a JSON decision object
- Validates the JSON against an embedded Draft 2020-12 schema
- Writes the validated report to reports/

Examples
--------
python trade_analyst_metals_refactored.py
python trade_analyst_metals_refactored.py --pairs GC/SI,PL/PA,GC/PA
python trade_analyst_metals_refactored.py --pairs GC/SI --min-corr 0.55 --min-abs-z 1.50
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
USE_LLM_SUMMARY = False
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

CACHE_FILE = Path("trade_analyst_metals_cache.pkl")
CACHE_TTL_SECONDS = 300

METALS_SYMBOLS = {
    "GC=F": "gold",
    "SI=F": "silver",
    "HG=F": "copper",
    "PL=F": "platinum",
    "PA=F": "palladium",
}

ALIAS_TO_SYMBOL = {
    "GC": "GC=F",
    "SI": "SI=F",
    "HG": "HG=F",
    "PL": "PL=F",
    "PA": "PA=F",
}

SYMBOL_TO_ALIAS = {v: k for k, v in ALIAS_TO_SYMBOL.items()}

DEFAULT_ALLOWED_PAIR_TEXT = "GC/SI,GC/HG,SI/HG,GC/PL,SI/PL,HG/PL"

NO_TRADE_REASON_CODES = [
    "NO_ALLOWED_PAIRS_CONFIGURED",
    "MISSING_SYMBOL_DATA",
    "INSUFFICIENT_HISTORY",
    "NO_EVALUABLE_PAIRS",
    "NO_PAIR_MET_CORRELATION_THRESHOLD",
    "NO_PAIR_MET_ZSCORE_THRESHOLD",
    "DATA_FETCH_ERRORS_PRESENT",
    "NO_TRADE_CONDITIONS_NOT_MET",
]

PAIR_REASON_CODES = [
    "PASSED",
    "MISSING_SYMBOL_DATA",
    "INSUFFICIENT_HISTORY",
    "LOW_CORRELATION",
    "LOW_ZSCORE",
    "INTERNAL_ERROR",
]


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
    min_corr: float = 0.45
    min_abs_z: float = 1.25
    max_candidates: int = 10
    allowed_pairs: Tuple[PairSpec, ...] = ()


# -------------------- JSON SCHEMA --------------------
SYMBOL_FEATURE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "symbol",
        "alias",
        "label",
        "as_of",
        "rows",
        "previous_close",
        "effective_price",
        "price_source",
        "source_timestamp_utc",
        "daily_high",
        "daily_low",
        "sma_21",
        "sma_63",
        "vs_sma_21",
        "vs_sma_63",
    ],
    "properties": {
        "symbol": {"type": "string"},
        "alias": {"type": "string"},
        "label": {"type": "string"},
        "as_of": {"type": "string"},
        "rows": {"type": "integer", "minimum": 0},
        "previous_close": {"type": ["number", "null"]},
        "effective_price": {"type": ["number", "null"]},
        "price_source": {"type": "string", "enum": ["current", "previous_close"]},
        "source_timestamp_utc": {"type": ["string", "null"], "format": "date-time"},
        "daily_high": {"type": ["number", "null"]},
        "daily_low": {"type": ["number", "null"]},
        "sma_21": {"type": ["number", "null"]},
        "sma_63": {"type": ["number", "null"]},
        "vs_sma_21": {"type": "string", "enum": ["ABOVE", "BELOW", "NEAR", "UNKNOWN"]},
        "vs_sma_63": {"type": "string", "enum": ["ABOVE", "BELOW", "NEAR", "UNKNOWN"]},
    },
}

PAIR_EVAL_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "pair_id",
        "pair_label",
        "symbol_a",
        "symbol_b",
        "alias_a",
        "alias_b",
        "label_a",
        "label_b",
        "status",
        "reason_code",
        "correlation",
        "hedge_ratio_beta",
        "last_close_zscore",
        "current_zscore",
        "expensive_symbol",
        "cheap_symbol",
        "recommendation",
        "confidence",
        "score",
        "rationale",
        "inputs",
    ],
    "properties": {
        "pair_id": {"type": "string"},
        "pair_label": {"type": "string"},
        "symbol_a": {"type": "string"},
        "symbol_b": {"type": "string"},
        "alias_a": {"type": "string"},
        "alias_b": {"type": "string"},
        "label_a": {"type": "string"},
        "label_b": {"type": "string"},
        "status": {"type": "string", "enum": ["CANDIDATE", "DISQUALIFIED", "ERROR"]},
        "reason_code": {"type": "string", "enum": PAIR_REASON_CODES},
        "correlation": {"type": ["number", "null"]},
        "hedge_ratio_beta": {"type": ["number", "null"]},
        "last_close_zscore": {"type": ["number", "null"]},
        "current_zscore": {"type": ["number", "null"]},
        "expensive_symbol": {"type": ["string", "null"]},
        "cheap_symbol": {"type": ["string", "null"]},
        "recommendation": {"type": ["string", "null"]},
        "confidence": {"type": ["number", "null"]},
        "score": {"type": ["number", "null"]},
        "rationale": {"type": ["string", "null"]},
        "inputs": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "effective_price_a",
                "effective_price_b",
                "price_source_a",
                "price_source_b",
                "rows_a",
                "rows_b",
                "usable_rows",
            ],
            "properties": {
                "effective_price_a": {"type": ["number", "null"]},
                "effective_price_b": {"type": ["number", "null"]},
                "price_source_a": {"type": ["string", "null"]},
                "price_source_b": {"type": ["string", "null"]},
                "rows_a": {"type": "integer", "minimum": 0},
                "rows_b": {"type": "integer", "minimum": 0},
                "usable_rows": {"type": "integer", "minimum": 0},
            },
        },
    },
}

NO_TRADE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "reason_codes",
        "reason_counts",
        "next_action",
        "summary",
    ],
    "properties": {
        "reason_codes": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string", "enum": NO_TRADE_REASON_CODES},
            "uniqueItems": True,
        },
        "reason_counts": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "allowed_pair_count",
                "evaluable_pair_count",
                "correlation_pass_count",
                "zscore_pass_count",
                "candidate_count",
                "missing_symbol_data_count",
                "insufficient_history_count",
                "low_correlation_count",
                "low_zscore_count",
                "internal_error_count",
            ],
            "properties": {
                "allowed_pair_count": {"type": "integer", "minimum": 0},
                "evaluable_pair_count": {"type": "integer", "minimum": 0},
                "correlation_pass_count": {"type": "integer", "minimum": 0},
                "zscore_pass_count": {"type": "integer", "minimum": 0},
                "candidate_count": {"type": "integer", "minimum": 0},
                "missing_symbol_data_count": {"type": "integer", "minimum": 0},
                "insufficient_history_count": {"type": "integer", "minimum": 0},
                "low_correlation_count": {"type": "integer", "minimum": 0},
                "low_zscore_count": {"type": "integer", "minimum": 0},
                "internal_error_count": {"type": "integer", "minimum": 0},
            },
        },
        "next_action": {"type": "string", "enum": ["WAIT", "REVIEW_DATA", "WIDEN_UNIVERSE"]},
        "summary": {"type": "string"},
    },
}

OUTPUT_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "agent",
        "schema_version",
        "event_id",
        "data_quality",
        "decision",
        "components",
        "summary",
    ],
    "properties": {
        "agent": {"type": "string", "const": "trade_analyst"},
        "schema_version": {"type": "string", "const": "1.1.0"},
        "event_id": {"type": "string", "format": "date-time"},
        "data_quality": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "cache_hit",
                "fetched_at_market",
                "missing_symbols",
                "available_symbols",
                "price_sources",
            ],
            "properties": {
                "cache_hit": {"type": "boolean"},
                "fetched_at_market": {"type": "string", "format": "date-time"},
                "missing_symbols": {"type": "array", "items": {"type": "string"}},
                "available_symbols": {"type": "array", "items": {"type": "string"}},
                "price_sources": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["symbol", "alias", "price_source"],
                        "properties": {
                            "symbol": {"type": "string"},
                            "alias": {"type": "string"},
                            "price_source": {"type": "string", "enum": ["current", "previous_close"]},
                        },
                    },
                },
            },
        },
        "decision": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "complex",
                "strategy_family",
                "status",
                "candidate_count",
                "top_candidate",
                "allowed_pairs",
                "thresholds",
                "no_trade",
            ],
            "properties": {
                "complex": {"type": "string", "const": "metals"},
                "strategy_family": {"type": "string", "const": "relative_value_mean_reversion"},
                "status": {"type": "string", "enum": ["OK", "NO_TRADE"]},
                "candidate_count": {"type": "integer", "minimum": 0},
                "top_candidate": {
                    "anyOf": [
                        {"type": "null"},
                        PAIR_EVAL_SCHEMA,
                    ]
                },
                "allowed_pairs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "uniqueItems": True,
                },
                "thresholds": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "min_correlation",
                        "min_abs_zscore",
                        "corr_window_days",
                        "beta_window_days",
                        "zscore_window_days",
                    ],
                    "properties": {
                        "min_correlation": {"type": "number"},
                        "min_abs_zscore": {"type": "number"},
                        "corr_window_days": {"type": "integer", "minimum": 1},
                        "beta_window_days": {"type": "integer", "minimum": 1},
                        "zscore_window_days": {"type": "integer", "minimum": 1},
                    },
                },
                "no_trade": {
                    "anyOf": [
                        {"type": "null"},
                        NO_TRADE_SCHEMA,
                    ]
                },
            },
        },
        "components": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "metals",
                "pair_analytics",
                "trade_candidates",
            ],
            "properties": {
                "metals": {
                    "type": "array",
                    "items": SYMBOL_FEATURE_SCHEMA,
                },
                "pair_analytics": {
                    "type": "array",
                    "items": PAIR_EVAL_SCHEMA,
                },
                "trade_candidates": {
                    "type": "array",
                    "items": PAIR_EVAL_SCHEMA,
                },
            },
        },
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


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


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
    raise ValueError(f"Unknown metals symbol/alias: {token}")


def parse_pair_allowlist(pair_text: str) -> Tuple[PairSpec, ...]:
    raw_items = [item.strip() for item in pair_text.split(",") if item.strip()]
    if not raw_items:
        return tuple()

    parsed: List[PairSpec] = []
    seen = set()

    for raw in raw_items:
        if "/" not in raw:
            raise ValueError(f"Invalid pair format '{raw}'. Expected e.g. GC/SI")

        left, right = raw.split("/", 1)
        symbol_a, alias_a = parse_symbol_or_alias(left)
        symbol_b, alias_b = parse_symbol_or_alias(right)

        if symbol_a == symbol_b:
            raise ValueError(f"Pair cannot use the same leg twice: {raw}")

        # Deduplicate economic duplicates while preserving first orientation
        dedupe_key = tuple(sorted([symbol_a, symbol_b]))
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        parsed.append(PairSpec(
            symbol_a=symbol_a,
            symbol_b=symbol_b,
            alias_a=alias_a,
            alias_b=alias_b,
        ))

    return tuple(parsed)


def validate_output_schema(payload: Dict[str, object]) -> None:
    validator = Draft202012Validator(OUTPUT_SCHEMA, format_checker=FormatChecker())
    errors = sorted(validator.iter_errors(payload), key=lambda e: list(e.path))

    if errors:
        lines = []
        for err in errors[:10]:
            path = ".".join(str(p) for p in err.path) if err.path else "<root>"
            lines.append(f"{path}: {err.message}")
        raise ValueError("Output JSON schema validation failed:\n" + "\n".join(lines))


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


def build_symbol_features(symbol: str, df: pd.DataFrame, config: RuntimeConfig) -> Dict[str, object]:
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
        "alias": SYMBOL_TO_ALIAS[symbol],
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


def fetch_market_snapshot(config: RuntimeConfig) -> Dict[str, object]:
    symbol_frames: Dict[str, pd.DataFrame] = {}
    symbol_features: Dict[str, Dict[str, object]] = {}
    errors: Dict[str, str] = {}

    for symbol in METALS_SYMBOLS:
        try:
            df = fetch_history(symbol, period=config.daily_period, interval=config.daily_interval)
            symbol_frames[symbol] = df
            symbol_features[symbol] = build_symbol_features(symbol, df, config)
        except Exception as exc:
            errors[symbol] = str(exc)

    return {
        "fetched_at": now_iso(),
        "frames": symbol_frames,
        "symbols": symbol_features,
        "errors": errors,
    }


# -------------------- PAIR ANALYTICS --------------------
def empty_pair_result(pair: PairSpec, reason_code: str, status: str = "DISQUALIFIED") -> Dict[str, object]:
    return {
        "pair_id": pair.pair_id,
        "pair_label": pair.pair_label,
        "symbol_a": pair.symbol_a,
        "symbol_b": pair.symbol_b,
        "alias_a": pair.alias_a,
        "alias_b": pair.alias_b,
        "label_a": METALS_SYMBOLS[pair.symbol_a],
        "label_b": METALS_SYMBOLS[pair.symbol_b],
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
            "effective_price_a": None,
            "effective_price_b": None,
            "price_source_a": None,
            "price_source_b": None,
            "rows_a": 0,
            "rows_b": 0,
            "usable_rows": 0,
        },
    }


def evaluate_pair(pair: PairSpec, market_snapshot: Dict[str, object], config: RuntimeConfig) -> Dict[str, object]:
    try:
        symbol_features = market_snapshot["symbols"]
        symbol_frames = market_snapshot["frames"]

        if pair.symbol_a not in symbol_features or pair.symbol_b not in symbol_features:
            result = empty_pair_result(pair, "MISSING_SYMBOL_DATA")
            result["rationale"] = "One or both legs are missing from the market snapshot."
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
            result["rationale"] = "One or both legs do not have usable history."
            return result

        pair_closes = pd.concat(
            [frame_a["Close"].rename(pair.symbol_a), frame_b["Close"].rename(pair.symbol_b)],
            axis=1,
            join="inner",
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
            result["rationale"] = f"Pair has only {usable_rows} overlapping rows; {min_needed} required."
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
            result["inputs"]["rows_a"] = rows_a
            result["inputs"]["rows_b"] = rows_b
            result["inputs"]["usable_rows"] = usable_rows
            result["rationale"] = "Variance of hedge leg is not finite."
            return result

        cov_xy = np.cov(x, y, ddof=1)[0, 1]
        beta = safe_float(cov_xy / var_x)

        hist_spread = (log_px[pair.symbol_a] - beta * log_px[pair.symbol_b]).dropna()
        z_window = hist_spread.tail(config.zscore_window)

        if len(z_window) < config.zscore_window:
            result = empty_pair_result(pair, "INSUFFICIENT_HISTORY")
            result["inputs"]["rows_a"] = rows_a
            result["inputs"]["rows_b"] = rows_b
            result["inputs"]["usable_rows"] = usable_rows
            result["rationale"] = "Insufficient spread history for z-score calculation."
            return result

        mu = safe_float(z_window.mean())
        sigma = safe_float(z_window.std(ddof=1))

        if sigma is None or sigma <= 1e-10:
            result = empty_pair_result(pair, "INTERNAL_ERROR", status="ERROR")
            result["inputs"]["rows_a"] = rows_a
            result["inputs"]["rows_b"] = rows_b
            result["inputs"]["usable_rows"] = usable_rows
            result["rationale"] = "Spread standard deviation is not usable."
            return result

        last_close_spread = safe_float(hist_spread.iloc[-1])
        last_close_z = safe_float((last_close_spread - mu) / sigma)

        price_a = feature_a.get("effective_price")
        price_b = feature_b.get("effective_price")

        if price_a in (None, 0) or price_b in (None, 0):
            result = empty_pair_result(pair, "MISSING_SYMBOL_DATA")
            result["inputs"]["rows_a"] = rows_a
            result["inputs"]["rows_b"] = rows_b
            result["inputs"]["usable_rows"] = usable_rows
            result["inputs"]["effective_price_a"] = price_a
            result["inputs"]["effective_price_b"] = price_b
            result["inputs"]["price_source_a"] = feature_a.get("price_source")
            result["inputs"]["price_source_b"] = feature_b.get("price_source")
            result["rationale"] = "Effective price missing for one or both legs."
            return result

        current_spread = math.log(float(price_a)) - float(beta) * math.log(float(price_b))
        current_z = safe_float((current_spread - mu) / sigma)

        if current_z is None:
            result = empty_pair_result(pair, "INTERNAL_ERROR", status="ERROR")
            result["inputs"]["rows_a"] = rows_a
            result["inputs"]["rows_b"] = rows_b
            result["inputs"]["usable_rows"] = usable_rows
            result["rationale"] = "Current z-score could not be computed."
            return result

        expensive_symbol = pair.symbol_a if current_z > 0 else pair.symbol_b
        cheap_symbol = pair.symbol_b if current_z > 0 else pair.symbol_a

        z_strength = min(abs(current_z) / 3.0, 1.0)
        corr_strength = 0.0
        if correlation is not None:
            corr_strength = min(max((correlation - config.min_corr) / (1.0 - config.min_corr + 1e-9), 0.0), 1.0)

        confidence = round(0.55 * z_strength + 0.45 * corr_strength, 2)
        score = round(abs(current_z) * max(correlation or 0.0, 0.0), 4)

        result = {
            "pair_id": pair.pair_id,
            "pair_label": pair.pair_label,
            "symbol_a": pair.symbol_a,
            "symbol_b": pair.symbol_b,
            "alias_a": pair.alias_a,
            "alias_b": pair.alias_b,
            "label_a": METALS_SYMBOLS[pair.symbol_a],
            "label_b": METALS_SYMBOLS[pair.symbol_b],
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
                f"{METALS_SYMBOLS[expensive_symbol]} screens expensive relative to {METALS_SYMBOLS[cheap_symbol]}. "
                f"The hedge-adjusted spread is {current_z:.2f} standard deviations from its recent mean "
                f"with {correlation:.2f} rolling return correlation."
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
        }

        if correlation is None or correlation < config.min_corr:
            result["status"] = "DISQUALIFIED"
            result["reason_code"] = "LOW_CORRELATION"
            result["recommendation"] = None
            result["confidence"] = None
            result["score"] = score
            result["rationale"] = (
                f"Pair failed correlation threshold. Rolling correlation {correlation:.2f if correlation is not None else 'n/a'} "
                f"is below minimum {config.min_corr:.2f}."
            )
            return result

        if abs(current_z) < config.min_abs_z:
            result["status"] = "DISQUALIFIED"
            result["reason_code"] = "LOW_ZSCORE"
            result["recommendation"] = None
            result["rationale"] = (
                f"Pair failed z-score threshold. Absolute current z-score {abs(current_z):.2f} "
                f"is below minimum {config.min_abs_z:.2f}."
            )
            return result

        return result

    except Exception as exc:
        result = empty_pair_result(pair, "INTERNAL_ERROR", status="ERROR")
        result["rationale"] = f"Internal error while evaluating pair: {exc}"
        return result


def build_pair_snapshot(market_snapshot: Dict[str, object], config: RuntimeConfig) -> Dict[str, object]:
    pair_analytics: List[Dict[str, object]] = []
    trade_candidates: List[Dict[str, object]] = []

    reason_counts = {
        "PASSED": 0,
        "MISSING_SYMBOL_DATA": 0,
        "INSUFFICIENT_HISTORY": 0,
        "LOW_CORRELATION": 0,
        "LOW_ZSCORE": 0,
        "INTERNAL_ERROR": 0,
    }

    for pair in config.allowed_pairs:
        result = evaluate_pair(pair, market_snapshot, config)
        pair_analytics.append(result)
        reason_counts[result["reason_code"]] += 1

        if result["status"] == "CANDIDATE":
            trade_candidates.append(result)

    pair_analytics.sort(key=lambda x: (x["score"] is None, -(x["score"] or -999999)))
    trade_candidates.sort(key=lambda x: (x["score"] is None, -(x["score"] or -999999)))
    trade_candidates = trade_candidates[: config.max_candidates]

    evaluable_pair_count = reason_counts["LOW_CORRELATION"] + reason_counts["LOW_ZSCORE"] + reason_counts["PASSED"]
    correlation_pass_count = reason_counts["LOW_ZSCORE"] + reason_counts["PASSED"]
    zscore_pass_count = reason_counts["PASSED"]

    return {
        "fetched_at": now_iso(),
        "pair_analytics": pair_analytics,
        "trade_candidates": trade_candidates,
        "counts": {
            "allowed_pair_count": len(config.allowed_pairs),
            "evaluable_pair_count": evaluable_pair_count,
            "correlation_pass_count": correlation_pass_count,
            "zscore_pass_count": zscore_pass_count,
            "candidate_count": len(trade_candidates),
            "missing_symbol_data_count": reason_counts["MISSING_SYMBOL_DATA"],
            "insufficient_history_count": reason_counts["INSUFFICIENT_HISTORY"],
            "low_correlation_count": reason_counts["LOW_CORRELATION"],
            "low_zscore_count": reason_counts["LOW_ZSCORE"],
            "internal_error_count": reason_counts["INTERNAL_ERROR"],
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
        f"candidates={counts['candidate_count']}. Reason codes: {', '.join(reason_codes)}."
    )

    next_action = "WAIT"
    if "MISSING_SYMBOL_DATA" in reason_codes or "DATA_FETCH_ERRORS_PRESENT" in reason_codes:
        next_action = "REVIEW_DATA"
    elif "NO_ALLOWED_PAIRS_CONFIGURED" in reason_codes:
        next_action = "WIDEN_UNIVERSE"

    return {
        "reason_codes": reason_codes,
        "reason_counts": counts,
        "next_action": next_action,
        "summary": summary,
    }


def build_summary(status: str, top_candidate: Optional[Dict[str, object]], no_trade: Optional[Dict[str, object]]) -> str:
    if status == "OK" and top_candidate:
        return (
            f"Top metals RV opportunity is {top_candidate['recommendation']} in {top_candidate['pair_label']}. "
            f"Current spread z-score is {top_candidate['current_zscore']:.2f} with "
            f"{top_candidate['correlation']:.2f} correlation and confidence {top_candidate['confidence']:.2f}."
        )

    if no_trade:
        return no_trade["summary"]

    return "No trade candidates available."


# -------------------- DECISION BUILDER --------------------
def build_trade_decision(config: RuntimeConfig) -> Dict[str, object]:
    cached_market_snapshot = load_cache()
    cache_hit = bool(cached_market_snapshot)

    if cached_market_snapshot:
        market_snapshot = cached_market_snapshot
    else:
        market_snapshot = fetch_market_snapshot(config)
        save_cache(market_snapshot)

    pair_snapshot = build_pair_snapshot(market_snapshot, config)
    trade_candidates = pair_snapshot["trade_candidates"]
    top_candidate = trade_candidates[0] if trade_candidates else None
    status = "OK" if trade_candidates else "NO_TRADE"
    no_trade = None if trade_candidates else build_no_trade_block(pair_snapshot, market_snapshot)

    metals_list = [
        market_snapshot["symbols"][symbol]
        for symbol in sorted(market_snapshot["symbols"].keys())
    ]

    price_sources = [
        {
            "symbol": feature["symbol"],
            "alias": feature["alias"],
            "price_source": feature["price_source"],
        }
        for feature in metals_list
    ]

    decision = {
        "agent": "trade_analyst",
        "schema_version": "1.1.0",
        "event_id": now_iso(),
        "data_quality": {
            "cache_hit": cache_hit,
            "fetched_at_market": market_snapshot["fetched_at"],
            "missing_symbols": sorted(list(market_snapshot.get("errors", {}).keys())),
            "available_symbols": sorted(list(market_snapshot.get("symbols", {}).keys())),
            "price_sources": price_sources,
        },
        "decision": {
            "complex": "metals",
            "strategy_family": "relative_value_mean_reversion",
            "status": status,
            "candidate_count": len(trade_candidates),
            "top_candidate": top_candidate,
            "allowed_pairs": [pair.pair_label for pair in config.allowed_pairs],
            "thresholds": {
                "min_correlation": config.min_corr,
                "min_abs_zscore": config.min_abs_z,
                "corr_window_days": config.corr_window,
                "beta_window_days": config.beta_window,
                "zscore_window_days": config.zscore_window,
            },
            "no_trade": no_trade,
        },
        "components": {
            "metals": metals_list,
            "pair_analytics": pair_snapshot["pair_analytics"],
            "trade_candidates": trade_candidates,
        },
        "summary": "",
    }

    decision["summary"] = build_summary(status, top_candidate, no_trade)
    validate_output_schema(decision)
    return decision


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
            validate_output_schema(decision)
    except Exception:
        pass

    return decision


# -------------------- CLI --------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Metals relative-value Trade Analyst")
    parser.add_argument(
        "--pairs",
        type=str,
        default=DEFAULT_ALLOWED_PAIR_TEXT,
        help="Comma-separated pair allowlist, e.g. GC/SI,PL/PA,GC/PA",
    )
    parser.add_argument("--min-corr", type=float, default=0.45, help="Minimum rolling correlation")
    parser.add_argument("--min-abs-z", type=float, default=1.25, help="Minimum absolute z-score")
    parser.add_argument("--corr-window", type=int, default=126, help="Correlation window in trading days")
    parser.add_argument("--beta-window", type=int, default=126, help="Beta estimation window in trading days")
    parser.add_argument("--zscore-window", type=int, default=63, help="Spread z-score window in trading days")
    return parser


# -------------------- MAIN --------------------
if __name__ == "__main__":
    try:
        args = build_arg_parser().parse_args()
        allowed_pairs = parse_pair_allowlist(args.pairs)

        config = RuntimeConfig(
            min_corr=args.min_corr,
            min_abs_z=args.min_abs_z,
            corr_window=args.corr_window,
            beta_window=args.beta_window,
            zscore_window=args.zscore_window,
            allowed_pairs=allowed_pairs,
        )

        decision = build_trade_decision(config)

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