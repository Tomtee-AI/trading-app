#!/usr/bin/env python3
"""
divergence_analyst.py

Deterministic Divergence Trade Analyst.

Purpose
-------
Find mean-reversion / relative-value opportunities when correlated futures,
ETFs, or stock proxy groups diverge beyond normal ranges.

Design goals
------------
- Coordinator-compatible payload on stdout
- Strict JSON schema validation
- Deterministic, auditable calculations
- Current price preferred, prior daily close fallback
- No direct order routing
- Defined-risk implementation templates only

Important
---------
The JSON printed to stdout is intentionally limited to the strict top-level
shape expected by the current Trade Coordinator:
    {
      "agent": ...,
      "event_id": ...,
      "strategy_type": "divergence",
      "candidates": [...],
      "summary": ...
    }

Richer internal diagnostics, including NO_TRADE reason codes and counts, are
written separately to reports/ so the coordinator can ingest the payload
without schema breakage.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import pickle
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from jsonschema import Draft202012Validator, FormatChecker

USE_LLM_SUMMARY = False
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

if USE_LLM_SUMMARY:
    from langchain_ollama import ChatOllama

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

CACHE_FILE = Path("divergence_analyst_cache_v2.pkl")
CACHE_TTL_SECONDS = 300
REQUIRED_CACHE_KEYS = {"fetched_at", "futures_frames", "futures_features", "errors", "proxy_frames", "proxy_features", "proxy_errors"}

# ---------- futures universe ----------
# User supplied PLTM in the metals futures list. We map that alias to PL=F.
FUTURES_CATEGORIES: Dict[str, Dict[str, str]] = {
    "metals": {
        "GC": "GC=F",
        "SI": "SI=F",
        "HG": "HG=F",
        "PLTM": "PL=F",
        "PA": "PA=F",
    },
    "indices": {
        "NQ": "NQ=F",
        "ES": "ES=F",
        "RTY": "RTY=F",
    },
    "energies": {
        "CL": "CL=F",
        "HO": "HO=F",
        "RB": "RB=F",
        "NG": "NG=F",
    },
    "bonds": {
        "ZT": "ZT=F",
        "ZF": "ZF=F",
        "ZN": "ZN=F",
        "ZB": "ZB=F",
        "UB": "UB=F",
    },
}

ALIAS_TO_FUTURE: Dict[str, str] = {}
FUTURE_TO_ALIAS: Dict[str, str] = {}
ALIAS_TO_CATEGORY: Dict[str, str] = {}
for category, mapping in FUTURES_CATEGORIES.items():
    for alias, symbol in mapping.items():
        ALIAS_TO_FUTURE[alias] = symbol
        FUTURE_TO_ALIAS[symbol] = alias
        ALIAS_TO_CATEGORY[alias] = category

# ---------- proxy groups for actual trade implementation ----------
# Each alias maps to stock / ETF / ETP implementation candidates.
PROXY_GROUPS: Dict[str, List[str]] = {
    # Metals
    "GC": ["GDX", "GDXJ", "GLD", "B"],
    "SI": ["AG", "HL", "PAAS", "SLV"],
    "HG": ["BHP", "FCX", "RIO", "TECK"],
    "PLTM": ["PLTM"],
    "PA": ["PALL", "SBSW"],
    # Indices
    "NQ": ["QQQ", "TQQQ", "XLK", "SMH"],
    "ES": ["SPY", "IVV", "VOO", "SPLG"],
    "RTY": ["IWM", "VTWO", "IJR", "SCHA"],
    # Energies
    "CL": ["USO", "XLE", "XOM", "CVX"],
    "HO": ["VLO", "MPC", "PSX", "XLE"],
    "RB": ["UGA", "VLO", "MPC", "PSX"],
    "NG": ["UNG", "EQT", "RRC", "AR"],
    # Bonds
    "ZT": ["SHY", "VGSH", "SCHO"],
    "ZF": ["IEI", "VGIT", "SCHR"],
    "ZN": ["IEF", "SCHR", "VGIT"],
    "ZB": ["TLT", "VGLT", "SPTL"],
    "UB": ["EDV", "ZROZ", "TLT"],
}

DEFAULT_ALLOWED_PAIR_TEXT = ",".join([
    # Metals
    "GC/SI", "GC/HG", "GC/PLTM", "SI/HG", "SI/PLTM", "HG/PLTM",
    # Indices
    "NQ/ES", "NQ/RTY", "ES/RTY",
    # Energies
    "CL/HO", "CL/RB", "CL/NG", "HO/RB",
    # Bonds
    "ZT/ZF", "ZF/ZN", "ZN/ZB", "ZB/UB",
])

NO_TRADE_REASON_CODES = [
    "NO_ALLOWED_PAIRS_CONFIGURED",
    "MISSING_SYMBOL_DATA",
    "INSUFFICIENT_HISTORY",
    "NO_EVALUABLE_PAIRS",
    "NO_PAIR_MET_CORRELATION_THRESHOLD",
    "NO_PAIR_MET_ZSCORE_THRESHOLD",
    "NO_PROXY_IMPLEMENTATIONS",
    "DATA_FETCH_ERRORS_PRESENT",
    "NO_TRADE_CONDITIONS_NOT_MET",
]

PAIR_REASON_CODES = [
    "PASSED",
    "MISSING_SYMBOL_DATA",
    "INSUFFICIENT_HISTORY",
    "LOW_CORRELATION",
    "LOW_ZSCORE",
    "CROSS_CATEGORY_NOT_ALLOWED",
    "INTERNAL_ERROR",
]

PROXY_REASON_CODES = [
    "PASSED",
    "NO_PROXY_GROUP",
    "SINGLE_NAME_GROUP",
    "MISSING_SYMBOL_DATA",
    "INSUFFICIENT_HISTORY",
    "INTERNAL_ERROR",
]

COORDINATOR_CANDIDATE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "candidate_id",
        "strategy_type",
        "direction",
        "horizon",
        "structure_family",
        "summary",
        "confidence",
        "fit_score",
        "thesis_tags",
        "risk_flags",
        "implementation",
    ],
    "properties": {
        "candidate_id": {"type": "string", "minLength": 1},
        "strategy_type": {"type": "string", "const": "divergence"},
        "direction": {"type": "string", "enum": ["BULLISH", "BEARISH", "NEUTRAL", "RELATIVE_VALUE"]},
        "horizon": {"type": "string", "enum": ["SHORT_TERM", "LONG_TERM"]},
        "structure_family": {"type": "string"},
        "summary": {"type": "string", "minLength": 1},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "fit_score": {"type": "number", "minimum": 0, "maximum": 1},
        "thesis_tags": {"type": "array", "items": {"type": "string"}},
        "risk_flags": {"type": "array", "items": {"type": "string"}},
        "implementation": {"type": "object", "additionalProperties": True},
    },
}

OUTPUT_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": ["agent", "event_id", "strategy_type", "candidates", "summary"],
    "properties": {
        "agent": {"type": "string", "const": "divergence_analyst"},
        "event_id": {"type": "string", "format": "date-time"},
        "strategy_type": {"type": "string", "const": "divergence"},
        "candidates": {"type": "array", "items": COORDINATOR_CANDIDATE_SCHEMA},
        "summary": {"type": "string"},
    },
}


@dataclass(frozen=True)
class PairSpec:
    alias_a: str
    alias_b: str
    symbol_a: str
    symbol_b: str
    category: str

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
    proxy_beta_window: int = 126
    proxy_zscore_window: int = 63
    min_corr: float = 0.45
    min_abs_z: float = 1.25
    min_fit_score: float = 0.40
    max_candidates: int = 20
    option_target_min_dte: int = 21
    option_target_max_dte: int = 60
    allowed_pairs: Tuple[PairSpec, ...] = ()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
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


def load_cache() -> Optional[Dict[str, Any]]:
    if CACHE_FILE.exists() and time.time() - CACHE_FILE.stat().st_mtime < CACHE_TTL_SECONDS:
        try:
            with open(CACHE_FILE, "rb") as f:
                payload = pickle.load(f)
            if not isinstance(payload, dict):
                return None
            if not REQUIRED_CACHE_KEYS.issubset(payload.keys()):
                return None
            return payload
        except Exception:
            return None
    return None


def save_cache(payload: Dict[str, Any]) -> None:
    try:
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(payload, f)
    except Exception:
        pass


def validate_output_schema(payload: Dict[str, Any]) -> None:
    validator = Draft202012Validator(OUTPUT_SCHEMA, format_checker=FormatChecker())
    errors = sorted(validator.iter_errors(payload), key=lambda e: list(e.path))
    if errors:
        lines = []
        for err in errors[:10]:
            path = ".".join(str(p) for p in err.path) if err.path else "<root>"
            lines.append(f"{path}: {err.message}")
        raise ValueError("Output JSON schema validation failed:\n" + "\n".join(lines))


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


def build_symbol_features(symbol: str, label: str, df: pd.DataFrame, config: RuntimeConfig, alias: Optional[str] = None) -> Dict[str, Any]:
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
        "alias": alias,
        "label": label,
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


def parse_pair_allowlist(pair_text: str) -> Tuple[PairSpec, ...]:
    raw_items = [item.strip().upper() for item in pair_text.split(",") if item.strip()]
    if not raw_items:
        return tuple()

    parsed: List[PairSpec] = []
    seen = set()
    for raw in raw_items:
        if "/" not in raw:
            raise ValueError(f"Invalid pair format '{raw}'. Expected e.g. GC/SI")
        left, right = raw.split("/", 1)
        if left not in ALIAS_TO_FUTURE:
            raise ValueError(f"Unknown left alias '{left}'")
        if right not in ALIAS_TO_FUTURE:
            raise ValueError(f"Unknown right alias '{right}'")
        if ALIAS_TO_CATEGORY[left] != ALIAS_TO_CATEGORY[right]:
            raise ValueError(f"Cross-category pairs are not allowed in allowlist: {raw}")
        if left == right:
            raise ValueError(f"Pair cannot use same leg twice: {raw}")

        key = tuple(sorted([left, right]))
        if key in seen:
            continue
        seen.add(key)
        parsed.append(PairSpec(
            alias_a=left,
            alias_b=right,
            symbol_a=ALIAS_TO_FUTURE[left],
            symbol_b=ALIAS_TO_FUTURE[right],
            category=ALIAS_TO_CATEGORY[left],
        ))
    return tuple(parsed)


def fetch_market_snapshot(config: RuntimeConfig) -> Dict[str, Any]:
    futures_frames: Dict[str, pd.DataFrame] = {}
    futures_features: Dict[str, Dict[str, Any]] = {}
    errors: Dict[str, str] = {}

    for alias, symbol in ALIAS_TO_FUTURE.items():
        try:
            df = fetch_history(symbol, period=config.daily_period, interval=config.daily_interval)
            futures_frames[symbol] = df
            futures_features[symbol] = build_symbol_features(symbol, ALIAS_TO_CATEGORY[alias], df, config, alias=alias)
        except Exception as exc:
            errors[symbol] = str(exc)

    proxy_frames: Dict[str, pd.DataFrame] = {}
    proxy_features: Dict[str, Dict[str, Any]] = {}
    proxy_errors: Dict[str, str] = {}

    proxy_symbols = sorted({sym for symbols in PROXY_GROUPS.values() for sym in symbols})
    for symbol in proxy_symbols:
        try:
            df = fetch_history(symbol, period=config.daily_period, interval=config.daily_interval)
            proxy_frames[symbol] = df
            proxy_features[symbol] = build_symbol_features(symbol, "proxy", df, config, alias=None)
        except Exception as exc:
            proxy_errors[symbol] = str(exc)

    return {
        "fetched_at": now_iso(),
        "futures_frames": futures_frames,
        "futures_features": futures_features,
        "errors": errors,
        "proxy_frames": proxy_frames,
        "proxy_features": proxy_features,
        "proxy_errors": proxy_errors,
    }


def empty_pair_result(pair: PairSpec, reason_code: str, status: str = "DISQUALIFIED") -> Dict[str, Any]:
    return {
        "pair_id": pair.pair_id,
        "pair_label": pair.pair_label,
        "category": pair.category,
        "alias_a": pair.alias_a,
        "alias_b": pair.alias_b,
        "symbol_a": pair.symbol_a,
        "symbol_b": pair.symbol_b,
        "status": status,
        "reason_code": reason_code,
        "correlation": None,
        "hedge_ratio_beta": None,
        "last_close_zscore": None,
        "current_zscore": None,
        "expensive_alias": None,
        "cheap_alias": None,
        "recommendation": None,
        "confidence": None,
        "fit_score": None,
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


def determine_horizon(category: str) -> str:
    if category in {"indices", "energies"}:
        return "SHORT_TERM"
    return "LONG_TERM"


def evaluate_pair(pair: PairSpec, snapshot: Dict[str, Any], config: RuntimeConfig) -> Dict[str, Any]:
    try:
        features = snapshot["futures_features"]
        frames = snapshot["futures_frames"]

        if pair.category != ALIAS_TO_CATEGORY[pair.alias_a] or pair.category != ALIAS_TO_CATEGORY[pair.alias_b]:
            result = empty_pair_result(pair, "CROSS_CATEGORY_NOT_ALLOWED")
            result["rationale"] = "Cross-category divergence pairs are not allowed."
            return result

        if pair.symbol_a not in features or pair.symbol_b not in features:
            result = empty_pair_result(pair, "MISSING_SYMBOL_DATA")
            result["rationale"] = "One or both futures legs are missing from market snapshot."
            return result

        feature_a = features[pair.symbol_a]
        feature_b = features[pair.symbol_b]
        frame_a = frames.get(pair.symbol_a)
        frame_b = frames.get(pair.symbol_b)
        rows_a = len(frame_a) if frame_a is not None else 0
        rows_b = len(frame_b) if frame_b is not None else 0

        if frame_a is None or frame_b is None or frame_a.empty or frame_b.empty:
            result = empty_pair_result(pair, "MISSING_SYMBOL_DATA")
            result["inputs"].update({"rows_a": rows_a, "rows_b": rows_b})
            result["rationale"] = "One or both futures legs do not have usable history."
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
            result["inputs"].update({
                "rows_a": rows_a,
                "rows_b": rows_b,
                "usable_rows": usable_rows,
                "effective_price_a": feature_a.get("effective_price"),
                "effective_price_b": feature_b.get("effective_price"),
                "price_source_a": feature_a.get("price_source"),
                "price_source_b": feature_b.get("price_source"),
            })
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
            result["inputs"].update({"rows_a": rows_a, "rows_b": rows_b, "usable_rows": usable_rows})
            result["rationale"] = "Variance of hedge leg is not finite."
            return result

        beta = safe_float(np.cov(x, y, ddof=1)[0, 1] / var_x)
        hist_spread = (log_px[pair.symbol_a] - float(beta) * log_px[pair.symbol_b]).dropna()
        z_window = hist_spread.tail(config.zscore_window)
        if len(z_window) < config.zscore_window:
            result = empty_pair_result(pair, "INSUFFICIENT_HISTORY")
            result["inputs"].update({"rows_a": rows_a, "rows_b": rows_b, "usable_rows": usable_rows})
            result["rationale"] = "Insufficient spread history for z-score calculation."
            return result

        mu = safe_float(z_window.mean())
        sigma = safe_float(z_window.std(ddof=1))
        if sigma is None or sigma <= 1e-10:
            result = empty_pair_result(pair, "INTERNAL_ERROR", status="ERROR")
            result["inputs"].update({"rows_a": rows_a, "rows_b": rows_b, "usable_rows": usable_rows})
            result["rationale"] = "Spread standard deviation is not usable."
            return result

        last_close_spread = safe_float(hist_spread.iloc[-1])
        last_close_z = safe_float((last_close_spread - mu) / sigma)

        price_a = feature_a.get("effective_price")
        price_b = feature_b.get("effective_price")
        if price_a in (None, 0) or price_b in (None, 0):
            result = empty_pair_result(pair, "MISSING_SYMBOL_DATA")
            result["inputs"].update({
                "rows_a": rows_a,
                "rows_b": rows_b,
                "usable_rows": usable_rows,
                "effective_price_a": price_a,
                "effective_price_b": price_b,
                "price_source_a": feature_a.get("price_source"),
                "price_source_b": feature_b.get("price_source"),
            })
            result["rationale"] = "Effective price missing for one or both legs."
            return result

        current_spread = math.log(float(price_a)) - float(beta) * math.log(float(price_b))
        current_z = safe_float((current_spread - mu) / sigma)
        if current_z is None:
            result = empty_pair_result(pair, "INTERNAL_ERROR", status="ERROR")
            result["rationale"] = "Current z-score could not be computed."
            return result

        expensive_alias = pair.alias_a if current_z > 0 else pair.alias_b
        cheap_alias = pair.alias_b if current_z > 0 else pair.alias_a

        z_strength = min(abs(current_z) / 3.0, 1.0)
        corr_strength = 0.0 if correlation is None else min(max((correlation - config.min_corr) / (1.0 - config.min_corr + 1e-9), 0.0), 1.0)
        confidence = round(0.55 * z_strength + 0.45 * corr_strength, 2)
        fit_score = round(0.60 * z_strength + 0.40 * corr_strength, 2)

        result = {
            "pair_id": pair.pair_id,
            "pair_label": pair.pair_label,
            "category": pair.category,
            "alias_a": pair.alias_a,
            "alias_b": pair.alias_b,
            "symbol_a": pair.symbol_a,
            "symbol_b": pair.symbol_b,
            "status": "CANDIDATE",
            "reason_code": "PASSED",
            "correlation": round(float(correlation), 4) if correlation is not None else None,
            "hedge_ratio_beta": round(float(beta), 4),
            "last_close_zscore": round(float(last_close_z), 4) if last_close_z is not None else None,
            "current_zscore": round(float(current_z), 4),
            "expensive_alias": expensive_alias,
            "cheap_alias": cheap_alias,
            "recommendation": f"SELL {expensive_alias} / BUY {cheap_alias}",
            "confidence": confidence,
            "fit_score": fit_score,
            "rationale": (
                f"{expensive_alias} screens expensive relative to {cheap_alias}. "
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
            result["rationale"] = f"Pair failed correlation threshold. Rolling correlation={correlation}."
            return result

        if abs(current_z) < config.min_abs_z:
            result["status"] = "DISQUALIFIED"
            result["reason_code"] = "LOW_ZSCORE"
            result["rationale"] = (
                f"Pair failed z-score threshold. Absolute current z-score {abs(current_z):.2f} is below {config.min_abs_z:.2f}."
            )
            return result

        return result

    except Exception as exc:
        result = empty_pair_result(pair, "INTERNAL_ERROR", status="ERROR")
        result["rationale"] = f"Internal error while evaluating pair: {exc}"
        return result


def build_pair_snapshot(snapshot: Dict[str, Any], config: RuntimeConfig) -> Dict[str, Any]:
    pair_analytics: List[Dict[str, Any]] = []
    trade_candidates: List[Dict[str, Any]] = []
    reason_counts = {code: 0 for code in PAIR_REASON_CODES}

    for pair in config.allowed_pairs:
        result = evaluate_pair(pair, snapshot, config)
        pair_analytics.append(result)
        reason_counts[result["reason_code"]] += 1
        if result["status"] == "CANDIDATE" and (result["fit_score"] or 0) >= config.min_fit_score:
            trade_candidates.append(result)

    pair_analytics.sort(key=lambda x: (x["fit_score"] is None, -(x["fit_score"] or -999999)))
    trade_candidates.sort(key=lambda x: (x["fit_score"] is None, -(x["fit_score"] or -999999)))
    trade_candidates = trade_candidates[: config.max_candidates]

    evaluable_pair_count = reason_counts["LOW_CORRELATION"] + reason_counts["LOW_ZSCORE"] + reason_counts["PASSED"]
    correlation_pass_count = reason_counts["LOW_ZSCORE"] + reason_counts["PASSED"]
    zscore_pass_count = reason_counts["PASSED"]

    return {
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


def choose_option_expiration(symbol: str, min_dte: int, max_dte: int) -> Tuple[bool, Optional[str]]:
    try:
        expirations = list(yf.Ticker(symbol).options or [])
    except Exception:
        return False, None

    if not expirations:
        return False, None

    now_date = datetime.now(timezone.utc).date()
    target_mid = (min_dte + max_dte) // 2
    best_exp = None
    best_dist = 10**9
    for exp in expirations:
        try:
            exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
        except Exception:
            continue
        dte = (exp_date - now_date).days
        if dte < min_dte or dte > max_dte:
            continue
        dist = abs(dte - target_mid)
        if dist < best_dist:
            best_dist = dist
            best_exp = exp
    if best_exp is not None:
        return True, best_exp
    return True, expirations[0] if expirations else None


def build_options_template(bias: str, config: RuntimeConfig) -> Dict[str, Any]:
    if bias == "BULLISH":
        return {
            "bias": "BULLISH",
            "dte_target_range": f"{config.option_target_min_dte}-{config.option_target_max_dte}",
            "long_leg_delta_target": 0.55,
            "short_leg_delta_target": 0.30,
        }
    return {
        "bias": "BEARISH",
        "dte_target_range": f"{config.option_target_min_dte}-{config.option_target_max_dte}",
        "long_leg_delta_target": -0.55,
        "short_leg_delta_target": -0.30,
    }


def empty_proxy_result(alias: str, symbol: str, reason_code: str, status: str = "DISQUALIFIED") -> Dict[str, Any]:
    return {
        "alias": alias,
        "symbol": symbol,
        "status": status,
        "reason_code": reason_code,
        "effective_price": None,
        "price_source": None,
        "peer_zscore": None,
        "relative_rank": None,
        "rows": 0,
        "options": {
            "options_available": False,
            "selected_expiration": None,
            "preferred_structure": None,
            "template": None,
        },
        "rationale": None,
    }


def score_proxy_group(alias: str, snapshot: Dict[str, Any], config: RuntimeConfig) -> List[Dict[str, Any]]:
    proxies = PROXY_GROUPS.get(alias, [])
    if not proxies:
        return []

    proxy_features = snapshot["proxy_features"]
    proxy_frames = snapshot["proxy_frames"]
    available = [sym for sym in proxies if sym in proxy_features and sym in proxy_frames and not proxy_frames[sym].empty]

    if not available:
        return [
            {**empty_proxy_result(alias, sym, "MISSING_SYMBOL_DATA"), "rationale": f"No usable proxy data for {alias}:{sym}."}
            for sym in proxies
        ]

    if len(available) == 1:
        symbol = available[0]
        options_available, expiration = choose_option_expiration(symbol, config.option_target_min_dte, config.option_target_max_dte)
        result = empty_proxy_result(alias, symbol, "SINGLE_NAME_GROUP", status="CANDIDATE")
        result.update({
            "effective_price": proxy_features[symbol].get("effective_price"),
            "price_source": proxy_features[symbol].get("price_source"),
            "rows": int(proxy_features[symbol].get("rows", 0)),
            "options": {
                "options_available": options_available,
                "selected_expiration": expiration,
                "preferred_structure": "CALL_DEBIT_SPREAD",
                "template": build_options_template("BULLISH", config),
            },
            "rationale": f"{alias} has only one mapped tradable proxy; relative rank versus peers cannot be computed.",
        })
        return [result]

    close_map = {}
    row_map = {}
    for sym in available:
        frame = proxy_frames[sym]
        close_map[sym] = frame["Close"].rename(sym)
        row_map[sym] = len(frame)

    close_df = pd.concat(close_map.values(), axis=1, join="inner")
    close_df.columns = list(close_map.keys())
    close_df = close_df.dropna(how="any")

    min_needed = max(config.proxy_beta_window, config.proxy_zscore_window) + 5
    if len(close_df) < min_needed:
        return [
            {
                **empty_proxy_result(alias, sym, "INSUFFICIENT_HISTORY"),
                "effective_price": proxy_features[sym].get("effective_price") if sym in proxy_features else None,
                "price_source": proxy_features[sym].get("price_source") if sym in proxy_features else None,
                "rows": int(row_map.get(sym, 0)),
                "rationale": f"Proxy group {alias} has only {len(close_df)} overlapping rows; {min_needed} required.",
            }
            for sym in proxies
        ]

    log_df = np.log(close_df)
    zscores: Dict[str, Optional[float]] = {}

    for sym in available:
        others = [c for c in available if c != sym]
        basket = log_df[others].mean(axis=1)
        sample = pd.concat([log_df[sym], basket.rename("basket")], axis=1).dropna()

        x = sample["basket"].tail(config.proxy_beta_window).values
        y = sample[sym].tail(config.proxy_beta_window).values
        var_x = np.var(x, ddof=1)
        if not np.isfinite(var_x) or var_x <= 0:
            zscores[sym] = None
            continue
        beta = np.cov(x, y, ddof=1)[0, 1] / var_x
        spread = sample[sym] - beta * sample["basket"]
        z_window = spread.tail(config.proxy_zscore_window)
        sigma = safe_float(z_window.std(ddof=1))
        if sigma is None or sigma <= 1e-10:
            zscores[sym] = None
            continue
        eff = proxy_features[sym].get("effective_price")
        if eff in (None, 0):
            zscores[sym] = None
            continue
        current_basket_price = np.exp(log_df[others].iloc[-1].mean())
        current_spread = math.log(float(eff)) - float(beta) * math.log(float(current_basket_price))
        zscores[sym] = safe_float((current_spread - safe_float(z_window.mean(), 0.0)) / sigma)

    ranked = [(sym, z) for sym, z in zscores.items() if z is not None]
    ranked.sort(key=lambda t: t[1])
    cheapest = ranked[0][0] if ranked else None
    expensive = ranked[-1][0] if ranked else None

    results: List[Dict[str, Any]] = []
    for sym in proxies:
        if sym not in proxy_features:
            result = empty_proxy_result(alias, sym, "MISSING_SYMBOL_DATA")
            result["rationale"] = f"Proxy {sym} missing from proxy data snapshot."
            results.append(result)
            continue

        feature = proxy_features[sym]
        options_available, expiration = choose_option_expiration(sym, config.option_target_min_dte, config.option_target_max_dte)
        z = zscores.get(sym)
        result = empty_proxy_result(alias, sym, "PASSED", status="CANDIDATE")
        result.update({
            "effective_price": feature.get("effective_price"),
            "price_source": feature.get("price_source"),
            "peer_zscore": round(float(z), 4) if z is not None else None,
            "rows": int(feature.get("rows", 0)),
            "options": {
                "options_available": options_available,
                "selected_expiration": expiration,
                "preferred_structure": None,
                "template": None,
            },
            "rationale": f"{sym} ranked within {alias} proxy basket using hedge-adjusted peer spread z-score.",
        })

        if z is None:
            result["status"] = "DISQUALIFIED"
            result["reason_code"] = "INTERNAL_ERROR"
            result["rationale"] = f"{sym} proxy z-score could not be computed."
        elif sym == expensive:
            result["relative_rank"] = "EXPENSIVE"
            result["options"]["preferred_structure"] = "PUT_DEBIT_SPREAD"
            result["options"]["template"] = build_options_template("BEARISH", config)
        elif sym == cheapest:
            result["relative_rank"] = "CHEAP"
            result["options"]["preferred_structure"] = "CALL_DEBIT_SPREAD"
            result["options"]["template"] = build_options_template("BULLISH", config)
        else:
            result["status"] = "DISQUALIFIED"
            result["reason_code"] = "PASSED"
            result["rationale"] = f"{sym} is in the middle of its {alias} proxy ranking and not the best actual trade vehicle."

        results.append(result)

    return results


def build_proxy_snapshot(snapshot: Dict[str, Any], config: RuntimeConfig) -> Dict[str, Any]:
    analytics: List[Dict[str, Any]] = []
    for alias in sorted(PROXY_GROUPS.keys()):
        analytics.extend(score_proxy_group(alias, snapshot, config))
    return {"proxy_analytics": analytics}


def choose_proxy_leg(alias: str, side: str, proxy_analytics: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    target_rank = "CHEAP" if side == "LONG" else "EXPENSIVE"
    candidates = [
        row for row in proxy_analytics
        if row["alias"] == alias and row["status"] == "CANDIDATE" and row["relative_rank"] == target_rank
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda x: abs(x["peer_zscore"] or 0.0), reverse=True)
    return candidates[0]


def build_candidate_payload(pair_candidate: Dict[str, Any], proxy_analytics: List[Dict[str, Any]], config: RuntimeConfig) -> Optional[Dict[str, Any]]:
    expensive_alias = pair_candidate["expensive_alias"]
    cheap_alias = pair_candidate["cheap_alias"]

    short_leg = choose_proxy_leg(expensive_alias, "SHORT", proxy_analytics)
    long_leg = choose_proxy_leg(cheap_alias, "LONG", proxy_analytics)

    if not short_leg and not long_leg:
        return None

    stock_trade_plan: List[Dict[str, Any]] = []
    options_trade_plan: List[Dict[str, Any]] = []
    risk_flags: List[str] = ["commodity_beta", "relative_value"]

    if short_leg:
        stock_trade_plan.append({
            "symbol": short_leg["symbol"],
            "alias": expensive_alias,
            "side": "SHORT",
            "action": "SELL_STOCK",
            "rationale": f"{short_leg['symbol']} is the richest proxy in the {expensive_alias} basket.",
        })
        options_trade_plan.append({
            "symbol": short_leg["symbol"],
            "alias": expensive_alias,
            "bias": "BEARISH",
            "structure": short_leg["options"]["preferred_structure"] or "NO_OPTIONS_FOUND",
            "expiration": short_leg["options"]["selected_expiration"],
            "template": short_leg["options"]["template"],
            "rationale": f"Use a bearish defined-risk overlay if options are available on {short_leg['symbol']}.",
        })
        if not short_leg["options"]["options_available"]:
            risk_flags.append(f"options_unavailable_{short_leg['symbol']}")

    if long_leg:
        stock_trade_plan.append({
            "symbol": long_leg["symbol"],
            "alias": cheap_alias,
            "side": "LONG",
            "action": "BUY_STOCK",
            "rationale": f"{long_leg['symbol']} is the cheapest proxy in the {cheap_alias} basket.",
        })
        options_trade_plan.append({
            "symbol": long_leg["symbol"],
            "alias": cheap_alias,
            "bias": "BULLISH",
            "structure": long_leg["options"]["preferred_structure"] or "NO_OPTIONS_FOUND",
            "expiration": long_leg["options"]["selected_expiration"],
            "template": long_leg["options"]["template"],
            "rationale": f"Use a bullish defined-risk overlay if options are available on {long_leg['symbol']}.",
        })
        if not long_leg["options"]["options_available"]:
            risk_flags.append(f"options_unavailable_{long_leg['symbol']}")

    horizon = determine_horizon(pair_candidate["category"])
    fit_score = round(float(pair_candidate["fit_score"]), 2)
    confidence = round(float(pair_candidate["confidence"]), 2)

    structure_family = "PAIR_TRADE_STOCK_OPTIONS_OVERLAY"
    if short_leg and long_leg and short_leg["options"]["options_available"] and long_leg["options"]["options_available"]:
        structure_family = "RELATIVE_VALUE_DEFINED_RISK_OVERLAY"
    elif short_leg and long_leg:
        structure_family = "PAIR_TRADE_STOCKS"
    elif short_leg or long_leg:
        structure_family = "PARTIAL_RELATIVE_VALUE"
        risk_flags.append("partial_mapping")

    implementation = {
        "pair_metrics": {
            "pair_id": pair_candidate["pair_id"],
            "pair_label": pair_candidate["pair_label"],
            "category": pair_candidate["category"],
            "correlation": pair_candidate["correlation"],
            "hedge_ratio_beta": pair_candidate["hedge_ratio_beta"],
            "current_zscore": pair_candidate["current_zscore"],
            "expensive_alias": expensive_alias,
            "cheap_alias": cheap_alias,
        },
        "stock_trade_plan": stock_trade_plan,
        "options_trade_plan": options_trade_plan,
        "proxy_selection": {
            "short_leg": short_leg,
            "long_leg": long_leg,
        },
        "hard_limits": {
            "undefined_risk_allowed": False,
            "direct_order_routing_allowed": False,
        },
    }

    summary = (
        f"Divergence candidate {pair_candidate['pair_label']}: short {expensive_alias} proxies and long {cheap_alias} proxies. "
        f"Current spread z-score {pair_candidate['current_zscore']:.2f}, correlation {pair_candidate['correlation']:.2f}."
    )

    thesis_tags = [pair_candidate["category"], "divergence", "relative_value", expensive_alias.lower(), cheap_alias.lower()]

    return {
        "candidate_id": f"div_{pair_candidate['pair_label'].replace('/', '_').lower()}",
        "strategy_type": "divergence",
        "direction": "RELATIVE_VALUE",
        "horizon": horizon,
        "structure_family": structure_family,
        "summary": summary,
        "confidence": confidence,
        "fit_score": fit_score,
        "thesis_tags": thesis_tags,
        "risk_flags": sorted(set(risk_flags)),
        "implementation": implementation,
    }


def build_candidates(pair_snapshot: Dict[str, Any], proxy_snapshot: Dict[str, Any], config: RuntimeConfig) -> List[Dict[str, Any]]:
    payloads: List[Dict[str, Any]] = []
    for pair_candidate in pair_snapshot["trade_candidates"]:
        candidate = build_candidate_payload(pair_candidate, proxy_snapshot["proxy_analytics"], config)
        if candidate is not None:
            payloads.append(candidate)

    payloads.sort(key=lambda x: (x["fit_score"], x["confidence"]), reverse=True)
    return payloads[: config.max_candidates]


def build_no_trade_diagnostics(pair_snapshot: Dict[str, Any], snapshot: Dict[str, Any], candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts = dict(pair_snapshot["counts"])
    counts["candidate_payload_count"] = len(candidates)

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

    if pair_snapshot["counts"]["candidate_count"] > 0 and len(candidates) == 0:
        reason_codes.append("NO_PROXY_IMPLEMENTATIONS")

    if snapshot.get("errors") or snapshot.get("proxy_errors"):
        reason_codes.append("DATA_FETCH_ERRORS_PRESENT")

    if not reason_codes:
        reason_codes.append("NO_TRADE_CONDITIONS_NOT_MET")

    return {
        "reason_codes": sorted(set(reason_codes)),
        "reason_counts": counts,
        "summary": (
            f"No divergence trades. allowed_pairs={counts['allowed_pair_count']}, pair_candidates={pair_snapshot['counts']['candidate_count']}, "
            f"coordinator_candidates={len(candidates)}. Reason codes: {', '.join(sorted(set(reason_codes)))}."
        ),
    }


def build_summary(candidates: List[Dict[str, Any]], diagnostics: Optional[Dict[str, Any]]) -> str:
    if candidates:
        top = candidates[0]
        metrics = top["implementation"]["pair_metrics"]
        return (
            f"Top divergence candidate is {metrics['pair_label']} with current z-score {metrics['current_zscore']:.2f} "
            f"and correlation {metrics['correlation']:.2f}."
        )
    if diagnostics:
        return diagnostics["summary"]
    return "No divergence candidates available."


def build_output(config: RuntimeConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    cached = load_cache()
    cache_hit = bool(cached)
    if cached and REQUIRED_CACHE_KEYS.issubset(cached.keys()):
        snapshot = cached
    else:
        snapshot = fetch_market_snapshot(config)
        save_cache(snapshot)
        cache_hit = False

    pair_snapshot = build_pair_snapshot(snapshot, config)
    proxy_snapshot = build_proxy_snapshot(snapshot, config)
    candidates = build_candidates(pair_snapshot, proxy_snapshot, config)
    diagnostics = None if candidates else build_no_trade_diagnostics(pair_snapshot, snapshot, candidates)

    output = {
        "agent": "divergence_analyst",
        "event_id": now_iso(),
        "strategy_type": "divergence",
        "candidates": candidates,
        "summary": build_summary(candidates, diagnostics),
    }
    validate_output_schema(output)

    internal_report = {
        "agent": "divergence_analyst",
        "event_id": output["event_id"],
        "cache_hit": cache_hit,
        "allowed_pairs": [pair.pair_label for pair in config.allowed_pairs],
        "snapshot_meta": {
            "fetched_at": snapshot["fetched_at"],
            "future_errors": snapshot.get("errors", {}),
            "proxy_errors": snapshot.get("proxy_errors", {}),
        },
        "pair_snapshot": pair_snapshot,
        "proxy_snapshot": proxy_snapshot,
        "candidate_payloads": candidates,
        "no_trade_diagnostics": diagnostics,
        "summary": output["summary"],
    }
    return output, internal_report


def add_llm_summary(output: Dict[str, Any]) -> Dict[str, Any]:
    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.2)
    prompt_payload = {"strategy_type": output["strategy_type"], "summary": output["summary"], "candidate_count": len(output["candidates"])}
    prompt = (
        "You are a concise divergence analyst narrator. Return ONLY valid JSON with one key named summary. "
        "Keep it to 2 sentences max and do not change the deterministic view.\n\n"
        f"{json.dumps(prompt_payload, indent=2)}\n\n"
        'Output format: {"summary": "..."}'
    )
    try:
        response = llm.invoke(prompt)
        parsed = json.loads(response.content)
        if isinstance(parsed, dict) and "summary" in parsed:
            output["summary"] = parsed["summary"]
            validate_output_schema(output)
    except Exception:
        pass
    return output


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deterministic Divergence Trade Analyst")
    parser.add_argument("--pairs", type=str, default=DEFAULT_ALLOWED_PAIR_TEXT, help="Comma-separated same-category pair allowlist, e.g. GC/SI,NQ/ES,CL/RB,ZN/ZB")
    parser.add_argument("--min-corr", type=float, default=0.45, help="Minimum rolling correlation")
    parser.add_argument("--min-abs-z", type=float, default=1.25, help="Minimum absolute z-score")
    parser.add_argument("--corr-window", type=int, default=126, help="Correlation window in trading days")
    parser.add_argument("--beta-window", type=int, default=126, help="Beta estimation window in trading days")
    parser.add_argument("--zscore-window", type=int, default=63, help="Spread z-score window in trading days")
    parser.add_argument("--proxy-beta-window", type=int, default=126, help="Proxy beta estimation window")
    parser.add_argument("--proxy-zscore-window", type=int, default=63, help="Proxy z-score window")
    parser.add_argument("--min-fit-score", type=float, default=0.40, help="Minimum fit score to emit coordinator candidate")
    parser.add_argument("--max-candidates", type=int, default=20, help="Maximum coordinator candidates to emit")
    parser.add_argument("--option-min-dte", type=int, default=21, help="Minimum target DTE for options templates")
    parser.add_argument("--option-max-dte", type=int, default=60, help="Maximum target DTE for options templates")
    parser.add_argument("--use-llm-summary", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    config = RuntimeConfig(
        min_corr=args.min_corr,
        min_abs_z=args.min_abs_z,
        corr_window=args.corr_window,
        beta_window=args.beta_window,
        zscore_window=args.zscore_window,
        proxy_beta_window=args.proxy_beta_window,
        proxy_zscore_window=args.proxy_zscore_window,
        min_fit_score=args.min_fit_score,
        max_candidates=args.max_candidates,
        option_target_min_dte=args.option_min_dte,
        option_target_max_dte=args.option_max_dte,
        allowed_pairs=parse_pair_allowlist(args.pairs),
    )

    output, internal_report = build_output(config)
    if args.use_llm_summary and USE_LLM_SUMMARY:
        output = add_llm_summary(output)
        internal_report["summary"] = output["summary"]

    print(json.dumps(output, indent=2))

    coordinator_path = REPORTS_DIR / f"divergence_payload_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(coordinator_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    diagnostics_path = REPORTS_DIR / f"divergence_diagnostics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(diagnostics_path, "w", encoding="utf-8") as f:
        json.dump(internal_report, f, indent=2)

    print(f"\n[INFO] Coordinator payload written to {coordinator_path}")
    print(f"[INFO] Diagnostics written to {diagnostics_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {exc}")
        raise SystemExit(1)
