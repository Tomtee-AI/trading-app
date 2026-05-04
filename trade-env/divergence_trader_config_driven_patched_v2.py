#!/usr/bin/env python3
"""
config_driven_divergence_trader.py

Config-driven Divergence Trade Analyst that:
- reads futures aliases and proxy baskets from "Futures Symbols.txt"
- resolves aliases to Yahoo Finance symbols
- evaluates same-category futures divergences using rolling correlation,
  hedge ratio beta, and spread z-scores
- selects stock / ETF proxy legs from the config file
- enriches proxy legs with options-analysis context from
  tos_options_agent_functions.py
- emits a coordinator-friendly specialist payload plus a richer diagnostics file

Notes on options analysis
-------------------------
The Thinkorswim-to-Python options helper requires an implied-volatility series.
Yahoo Finance generally does not expose a clean historical IV series for every
stock / ETF proxy, so this script uses a deterministic fallback approximation:
    synthetic_iv = rolling_realized_vol_20d * iv_premium
You can replace that later with a better IV source without changing the rest
of the agent interface.
"""

from __future__ import annotations

import argparse
import importlib.util
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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from jsonschema import Draft202012Validator, FormatChecker

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_FILE = SCRIPT_DIR / "Futures Symbols.txt"
DEFAULT_OPTIONS_MODULE_PATH = SCRIPT_DIR / "tos_options_agent_functions.py"

CACHE_FILE = Path("divergence_trader_config_cache.pkl")
CACHE_TTL_SECONDS = 300

PAIR_REASON_CODES = [
    "PASSED",
    "MISSING_SYMBOL_DATA",
    "INSUFFICIENT_HISTORY",
    "LOW_CORRELATION",
    "LOW_ZSCORE",
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
        "direction": {"type": "string", "enum": ["RELATIVE_VALUE"]},
        "horizon": {"type": "string", "enum": ["SHORT_TERM", "LONG_TERM"]},
        "structure_family": {"type": "string"},
        "summary": {"type": "string", "minLength": 1},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "fit_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
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

# Config file aliases -> Yahoo Finance futures symbols.
# These defaults reflect common Yahoo Finance futures naming.
DEFAULT_ALIAS_TO_YF_SYMBOL: Dict[str, str] = {
    "GC": "GC=F",
    "SI": "SI=F",
    "HG": "HG=F",
    "PL": "PL=F",
    "PA": "PA=F",
    "NQ": "NQ=F",
    "ES": "ES=F",
    "RTY": "RTY=F",
    "CL": "CL=F",
    "HO": "HO=F",
    "RB": "RB=F",
    "NG": "NG=F",
    "ZT": "ZT=F",
    "ZF": "ZF=F",
    "ZN": "ZN=F",
    "ZB": "ZB=F",
    "UB": "UB=F",
    "BTC": "BTC=F",
    "ETH": "ETH=F",
    "SOL": "SOL=F",
    "XRP": "XRP=F",
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
    config_file: str = str(DEFAULT_CONFIG_FILE)
    options_module_path: str = str(DEFAULT_OPTIONS_MODULE_PATH)
    daily_period: str = "2y"
    daily_interval: str = "1d"
    current_attempts: Tuple[Tuple[str, str], ...] = (("1d", "1m"), ("5d", "5m"), ("5d", "15m"))
    corr_window: int = 126
    beta_window: int = 126
    zscore_window: int = 63
    proxy_rank_window: int = 63
    min_corr: float = 0.50
    min_abs_z: float = 1.25
    min_fit_score: float = 0.40
    max_candidates: int = 20
    option_target_min_dte: int = 21
    option_target_max_dte: int = 60
    iv_premium: float = 1.10
    require_vol_up_for_sigma: bool = False
    pair_allowlist_text: Optional[str] = None


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
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


def resolve_runtime_path(user_value: Optional[str], default_path: Path) -> Path:
    """
    Resolve runtime paths robustly so the script works even when launched from a
    different working directory.

    Resolution order:
    1. Explicit absolute path
    2. Explicit relative path from current working directory
    3. Explicit relative path from the script directory
    4. Default path
    """
    if not user_value:
        return default_path

    candidate = Path(user_value).expanduser()
    if candidate.is_absolute() and candidate.exists():
        return candidate

    if candidate.exists():
        return candidate.resolve()

    script_relative = (SCRIPT_DIR / candidate).resolve()
    if script_relative.exists():
        return script_relative

    return default_path


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


def validate_output_schema(payload: Dict[str, Any]) -> None:
    validator = Draft202012Validator(OUTPUT_SCHEMA, format_checker=FormatChecker())
    errors = sorted(validator.iter_errors(payload), key=lambda e: list(e.path))
    if errors:
        lines = []
        for err in errors[:10]:
            path = ".".join(str(p) for p in err.path) if err.path else "<root>"
            lines.append(f"{path}: {err.message}")
        raise ValueError("Output JSON schema validation failed:\n" + "\n".join(lines))


def import_options_module(module_path: str):
    module_file = Path(module_path)
    if not module_file.exists():
        raise FileNotFoundError(f"Options module not found: {module_path}")

    spec = importlib.util.spec_from_file_location("tos_options_agent_functions", module_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    required = ["build_options_analysis_packet"]
    for attr in required:
        if not hasattr(module, attr):
            raise AttributeError(f"Missing required function '{attr}' in {module_path}")
    return module


def parse_config_file(config_file: str) -> Dict[str, Any]:
    path = Path(config_file)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    category_aliases: Dict[str, List[str]] = {}
    proxy_groups: Dict[str, List[str]] = {}
    additional_baskets: Dict[str, List[str]] = {}

    mode = "futures"
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line.lower().startswith("futures symbols"):
                continue
            if line.lower().startswith("stock future proxies"):
                mode = "proxies"
                continue
            if "=" not in line:
                continue

            left, right = [part.strip() for part in line.split("=", 1)]
            values = [token.strip() for token in right.split(",") if token.strip()]

            if mode == "futures":
                category = left.replace("Futures Symbols", "").strip().lower()
                category_aliases[category] = [v.upper() for v in values]
            else:
                alias = left.strip().upper()
                if alias in DEFAULT_ALIAS_TO_YF_SYMBOL:
                    proxy_groups[alias] = [v.upper() for v in values]
                else:
                    additional_baskets[alias] = [v.upper() for v in values]

    alias_to_category: Dict[str, str] = {}
    for category, aliases in category_aliases.items():
        for alias in aliases:
            alias_to_category[alias] = category

    alias_to_symbol = {
        alias: DEFAULT_ALIAS_TO_YF_SYMBOL[alias]
        for alias in alias_to_category
        if alias in DEFAULT_ALIAS_TO_YF_SYMBOL
    }

    missing_symbol_aliases = sorted([alias for alias in alias_to_category if alias not in alias_to_symbol])

    return {
        "category_aliases": category_aliases,
        "alias_to_category": alias_to_category,
        "alias_to_symbol": alias_to_symbol,
        "proxy_groups": proxy_groups,
        "additional_baskets": additional_baskets,
        "missing_symbol_aliases": missing_symbol_aliases,
    }


def build_allowed_pairs(parsed_config: Dict[str, Any], pair_allowlist_text: Optional[str] = None) -> Tuple[PairSpec, ...]:
    alias_to_category = parsed_config["alias_to_category"]
    alias_to_symbol = parsed_config["alias_to_symbol"]
    category_aliases = parsed_config["category_aliases"]

    if pair_allowlist_text:
        raw_items = [item.strip().upper() for item in pair_allowlist_text.split(",") if item.strip()]
        pairs: List[PairSpec] = []
        seen = set()
        for raw in raw_items:
            if "/" not in raw:
                raise ValueError(f"Invalid pair format '{raw}'. Expected e.g. GC/SI")
            left, right = raw.split("/", 1)
            left = left.strip().upper()
            right = right.strip().upper()
            if left not in alias_to_symbol:
                raise ValueError(f"Unknown or unresolved left alias '{left}'")
            if right not in alias_to_symbol:
                raise ValueError(f"Unknown or unresolved right alias '{right}'")
            if alias_to_category[left] != alias_to_category[right]:
                raise ValueError(f"Cross-category pair not allowed: {raw}")
            key = tuple(sorted([left, right]))
            if key in seen:
                continue
            seen.add(key)
            pairs.append(PairSpec(left, right, alias_to_symbol[left], alias_to_symbol[right], alias_to_category[left]))
        return tuple(pairs)

    pairs = []
    for category, aliases in category_aliases.items():
        resolved = [alias for alias in aliases if alias in alias_to_symbol]
        for left, right in itertools.combinations(resolved, 2):
            pairs.append(PairSpec(left, right, alias_to_symbol[left], alias_to_symbol[right], category))
    return tuple(pairs)


def fetch_history(symbol: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=False)
    df = df.dropna(how="all")
    if df.empty:
        raise ValueError(f"No history returned for {symbol}")
    df.index = pd.to_datetime(df.index)
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    return df


def try_fetch_current_price(symbol: str, current_attempts: Sequence[Tuple[str, str]]) -> Tuple[Optional[float], Optional[str], str]:
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


def build_symbol_features(symbol: str, df: pd.DataFrame, current_attempts: Sequence[Tuple[str, str]], alias: Optional[str] = None) -> Dict[str, Any]:
    close_series = df["Close"].dropna()
    high_series = df["High"].dropna() if "High" in df.columns else pd.Series(dtype=float)
    low_series = df["Low"].dropna() if "Low" in df.columns else pd.Series(dtype=float)

    previous_close = safe_float(close_series.iloc[-1]) if not close_series.empty else None
    daily_high = safe_float(high_series.iloc[-1]) if not high_series.empty else None
    daily_low = safe_float(low_series.iloc[-1]) if not low_series.empty else None
    sma_21 = safe_float(close_series.rolling(21).mean().iloc[-1]) if len(close_series) >= 21 else None
    sma_63 = safe_float(close_series.rolling(63).mean().iloc[-1]) if len(close_series) >= 63 else None

    current_price, current_ts, source = try_fetch_current_price(symbol, current_attempts)
    effective_price = current_price if current_price is not None else previous_close

    return {
        "symbol": symbol,
        "alias": alias,
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


def fetch_market_snapshot(parsed_config: Dict[str, Any], runtime: RuntimeConfig) -> Dict[str, Any]:
    futures_frames: Dict[str, pd.DataFrame] = {}
    futures_features: Dict[str, Dict[str, Any]] = {}
    future_errors: Dict[str, str] = {}

    for alias, symbol in parsed_config["alias_to_symbol"].items():
        try:
            df = fetch_history(symbol, runtime.daily_period, runtime.daily_interval)
            futures_frames[symbol] = df
            futures_features[symbol] = build_symbol_features(symbol, df, runtime.current_attempts, alias=alias)
        except Exception as exc:
            future_errors[symbol] = str(exc)

    proxy_symbols = sorted({sym for values in parsed_config["proxy_groups"].values() for sym in values})
    proxy_frames: Dict[str, pd.DataFrame] = {}
    proxy_features: Dict[str, Dict[str, Any]] = {}
    proxy_errors: Dict[str, str] = {}

    for symbol in proxy_symbols:
        try:
            df = fetch_history(symbol, runtime.daily_period, runtime.daily_interval)
            proxy_frames[symbol] = df
            proxy_features[symbol] = build_symbol_features(symbol, df, runtime.current_attempts, alias=None)
        except Exception as exc:
            proxy_errors[symbol] = str(exc)

    return {
        "fetched_at": now_iso(),
        "futures_frames": futures_frames,
        "futures_features": futures_features,
        "future_errors": future_errors,
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
    if category in {"indices", "energies", "crypto"}:
        return "SHORT_TERM"
    return "LONG_TERM"


def evaluate_pair(pair: PairSpec, snapshot: Dict[str, Any], runtime: RuntimeConfig) -> Dict[str, Any]:
    try:
        features = snapshot["futures_features"]
        frames = snapshot["futures_frames"]

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
        min_needed = max(runtime.corr_window, runtime.beta_window, runtime.zscore_window) + 5
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
        corr_sample = log_ret.tail(runtime.corr_window)
        correlation = safe_float(corr_sample[pair.symbol_a].corr(corr_sample[pair.symbol_b]))

        beta_sample = log_px.tail(runtime.beta_window)
        x = beta_sample[pair.symbol_b].values
        y = beta_sample[pair.symbol_a].values
        var_x = np.var(x, ddof=1)
        if not np.isfinite(var_x) or var_x <= 0:
            result = empty_pair_result(pair, "INTERNAL_ERROR", status="ERROR")
            result["rationale"] = "Variance of hedge leg is not finite."
            return result

        beta = safe_float(np.cov(x, y, ddof=1)[0, 1] / var_x)
        hist_spread = (log_px[pair.symbol_a] - float(beta) * log_px[pair.symbol_b]).dropna()
        z_window = hist_spread.tail(runtime.zscore_window)
        if len(z_window) < runtime.zscore_window:
            result = empty_pair_result(pair, "INSUFFICIENT_HISTORY")
            result["rationale"] = "Insufficient spread history for z-score calculation."
            return result

        mu = safe_float(z_window.mean())
        sigma = safe_float(z_window.std(ddof=1))
        if sigma is None or sigma <= 1e-10:
            result = empty_pair_result(pair, "INTERNAL_ERROR", status="ERROR")
            result["rationale"] = "Spread standard deviation is not usable."
            return result

        last_close_spread = safe_float(hist_spread.iloc[-1])
        last_close_z = safe_float((last_close_spread - mu) / sigma)

        price_a = feature_a.get("effective_price")
        price_b = feature_b.get("effective_price")
        if price_a in (None, 0) or price_b in (None, 0):
            result = empty_pair_result(pair, "MISSING_SYMBOL_DATA")
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
        corr_strength = 0.0 if correlation is None else min(max((correlation - runtime.min_corr) / (1.0 - runtime.min_corr + 1e-9), 0.0), 1.0)
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

        if correlation is None or correlation < runtime.min_corr:
            result["status"] = "DISQUALIFIED"
            result["reason_code"] = "LOW_CORRELATION"
            result["rationale"] = f"Pair failed correlation threshold. Rolling correlation={correlation}."
            return result

        if abs(current_z) < runtime.min_abs_z:
            result["status"] = "DISQUALIFIED"
            result["reason_code"] = "LOW_ZSCORE"
            result["rationale"] = (
                f"Pair failed z-score threshold. Absolute current z-score {abs(current_z):.2f} is below {runtime.min_abs_z:.2f}."
            )
            return result

        return result

    except Exception as exc:
        result = empty_pair_result(pair, "INTERNAL_ERROR", status="ERROR")
        result["rationale"] = f"Internal error while evaluating pair: {exc}"
        return result


def build_pair_snapshot(snapshot: Dict[str, Any], runtime: RuntimeConfig, pairs: Sequence[PairSpec]) -> Dict[str, Any]:
    pair_analytics: List[Dict[str, Any]] = []
    trade_candidates: List[Dict[str, Any]] = []
    reason_counts = {code: 0 for code in PAIR_REASON_CODES}

    for pair in pairs:
        result = evaluate_pair(pair, snapshot, runtime)
        pair_analytics.append(result)
        reason_counts[result["reason_code"]] += 1
        if result["status"] == "CANDIDATE" and (result["fit_score"] or 0.0) >= runtime.min_fit_score:
            trade_candidates.append(result)

    pair_analytics.sort(key=lambda x: (x["fit_score"] is None, -(x["fit_score"] or -999999)))
    trade_candidates.sort(key=lambda x: (x["fit_score"] is None, -(x["fit_score"] or -999999)))
    trade_candidates = trade_candidates[: runtime.max_candidates]

    return {
        "pair_analytics": pair_analytics,
        "trade_candidates": trade_candidates,
        "counts": {
            "allowed_pair_count": len(pairs),
            "candidate_count": len(trade_candidates),
            "missing_symbol_data_count": reason_counts["MISSING_SYMBOL_DATA"],
            "insufficient_history_count": reason_counts["INSUFFICIENT_HISTORY"],
            "low_correlation_count": reason_counts["LOW_CORRELATION"],
            "low_zscore_count": reason_counts["LOW_ZSCORE"],
            "internal_error_count": reason_counts["INTERNAL_ERROR"],
        },
    }


def build_synthetic_implied_volatility(df: pd.DataFrame, iv_premium: float = 1.10) -> pd.Series:
    close = df["Close"].astype(float)
    log_ret = np.log(close / close.shift(1))
    realized = log_ret.rolling(20).std(ddof=1) * np.sqrt(252.0)
    synthetic_iv = (realized * iv_premium).clip(lower=0.05, upper=3.0)
    synthetic_iv = synthetic_iv.bfill().ffill()
    return synthetic_iv


def get_option_expirations(symbol: str) -> List[str]:
    try:
        expirations = list(yf.Ticker(symbol).options or [])
        return expirations
    except Exception:
        return []


def choose_option_expiration(symbol: str, min_dte: int, max_dte: int) -> Tuple[bool, Optional[str]]:
    expirations = get_option_expirations(symbol)
    if not expirations:
        return False, None

    now_date = datetime.now(timezone.utc).date()
    target_mid = (min_dte + max_dte) // 2
    best_exp: Optional[str] = None
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
    return True, expirations[0]


def infer_expected_move_side(expected_move: Dict[str, Any]) -> str:
    if expected_move.get("monthly_breach"):
        up = expected_move.get("distance_to_month_upper_pct")
        down = expected_move.get("distance_to_month_lower_pct")
        if up is not None and up > 0:
            return "ABOVE_MONTH_UPPER"
        if down is not None and down < 0:
            return "BELOW_MONTH_LOWER"
    if expected_move.get("weekly_breach"):
        up = expected_move.get("distance_to_week_upper_pct")
        down = expected_move.get("distance_to_week_lower_pct")
        if up is not None and up > 0:
            return "ABOVE_WEEK_UPPER"
        if down is not None and down < 0:
            return "BELOW_WEEK_LOWER"
    return "INSIDE_EXPECTED_MOVES"


def build_proxy_option_template(side: str, runtime: RuntimeConfig) -> Dict[str, Any]:
    if side == "LONG":
        return {
            "bias": "BULLISH",
            "preferred_structure": "CALL_DEBIT_SPREAD",
            "dte_target_range": f"{runtime.option_target_min_dte}-{runtime.option_target_max_dte}",
            "long_leg_delta_target": 0.55,
            "short_leg_delta_target": 0.30,
        }
    return {
        "bias": "BEARISH",
        "preferred_structure": "PUT_DEBIT_SPREAD",
        "dte_target_range": f"{runtime.option_target_min_dte}-{runtime.option_target_max_dte}",
        "long_leg_delta_target": -0.55,
        "short_leg_delta_target": -0.30,
    }


def analyze_proxy_symbol(
    alias: str,
    symbol: str,
    side: str,
    snapshot: Dict[str, Any],
    options_module,
    runtime: RuntimeConfig,
) -> Dict[str, Any]:
    feature = snapshot["proxy_features"].get(symbol)
    frame = snapshot["proxy_frames"].get(symbol)
    if feature is None or frame is None or frame.empty:
        return {
            "alias": alias,
            "symbol": symbol,
            "status": "UNAVAILABLE",
            "reason": "MISSING_HISTORY",
            "composite_score": -999.0,
        }

    effective_price = feature.get("effective_price")
    sma_63 = feature.get("sma_63")
    deviation_pct = None
    directional_dev_pct = None
    if effective_price not in (None, 0) and sma_63 not in (None, 0):
        deviation_pct = ((effective_price / sma_63) - 1.0) * 100.0
        directional_dev_pct = deviation_pct if side == "SHORT" else -deviation_pct

    synthetic_iv = build_synthetic_implied_volatility(frame, iv_premium=runtime.iv_premium)
    px = frame[["Open", "High", "Low", "Close"]].rename(columns=str.lower)
    packet = options_module.build_options_analysis_packet(
        px,
        implied_volatility=synthetic_iv,
        require_vol_up_for_sigma=runtime.require_vol_up_for_sigma,
    )

    sigma_state = packet["sigma_reentry"]["state"]
    expected_move_side = infer_expected_move_side(packet["expected_move"])
    vol_up = bool(packet["vol_filter"].get("vol_up", False))

    directional_signal_score = 0.0
    if side == "LONG":
        if sigma_state == "BUY_REENTRY":
            directional_signal_score += 1.0
        if expected_move_side in {"BELOW_WEEK_LOWER", "BELOW_MONTH_LOWER"}:
            directional_signal_score += 1.0
    else:
        if sigma_state == "SELL_REENTRY":
            directional_signal_score += 1.0
        if expected_move_side in {"ABOVE_WEEK_UPPER", "ABOVE_MONTH_UPPER"}:
            directional_signal_score += 1.0

    options_available, selected_expiration = choose_option_expiration(
        symbol,
        runtime.option_target_min_dte,
        runtime.option_target_max_dte,
    )

    deviation_component = 0.0
    if directional_dev_pct is not None:
        deviation_component = clamp(directional_dev_pct / 6.0, -1.0, 1.0)

    composite_score = (
        0.45 * max(deviation_component, 0.0)
        + 0.30 * (directional_signal_score / 2.0)
        + 0.15 * (1.0 if vol_up else 0.0)
        + 0.10 * (1.0 if options_available else 0.0)
    )

    return {
        "alias": alias,
        "symbol": symbol,
        "status": "CANDIDATE",
        "side": side,
        "effective_price": effective_price,
        "sma_63": sma_63,
        "deviation_pct_from_sma63": round(deviation_pct, 4) if deviation_pct is not None else None,
        "directional_deviation_pct": round(directional_dev_pct, 4) if directional_dev_pct is not None else None,
        "composite_score": round(composite_score, 4),
        "price_source": feature.get("price_source"),
        "options_available": options_available,
        "selected_expiration": selected_expiration,
        "options_template": build_proxy_option_template(side, runtime),
        "options_analysis_packet": packet,
    }


def rank_proxy_candidates(
    alias: str,
    side: str,
    parsed_config: Dict[str, Any],
    snapshot: Dict[str, Any],
    options_module,
    runtime: RuntimeConfig,
) -> List[Dict[str, Any]]:
    proxy_symbols = parsed_config["proxy_groups"].get(alias, [])
    rows: List[Dict[str, Any]] = []
    for symbol in proxy_symbols:
        rows.append(analyze_proxy_symbol(alias, symbol, side, snapshot, options_module, runtime))
    rows.sort(key=lambda x: x.get("composite_score", -999.0), reverse=True)
    return rows


def describe_leg_trade(leg: Optional[Dict[str, Any]]) -> Optional[str]:
    if not leg or leg.get("status") != "CANDIDATE":
        return None

    side = str(leg.get("side", "")).upper()
    symbol = leg.get("symbol")
    options_template = leg.get("options_template") or {}
    preferred_structure = options_template.get("preferred_structure")

    if leg.get("options_available") and preferred_structure:
        structure_label_map = {
            "PUT_DEBIT_SPREAD": "put debit spread",
            "CALL_DEBIT_SPREAD": "call debit spread",
        }
        structure_label = structure_label_map.get(preferred_structure, preferred_structure.replace("_", " ").lower())
        action = "Short" if side == "SHORT" else "Long"
        return f"{action} {symbol} via {structure_label}"

    fallback_action = "SELL" if side == "SHORT" else "BUY"
    return f"{fallback_action} {symbol} stock"


def build_proxy_first_candidate_summary(
    pair_candidate: Dict[str, Any],
    short_leg: Optional[Dict[str, Any]],
    long_leg: Optional[Dict[str, Any]],
) -> str:
    leg_descriptions = [desc for desc in [describe_leg_trade(short_leg), describe_leg_trade(long_leg)] if desc]
    proxy_trade = " / ".join(leg_descriptions) if leg_descriptions else "No proxy trade available"
    return (
        f"Proxy/options trade: {proxy_trade}. "
        f"Signal source: {pair_candidate['pair_label']} futures divergence "
        f"(z-score {pair_candidate['current_zscore']:.2f}, correlation {pair_candidate['correlation']:.2f})."
    )


def build_candidate_payload(
    pair_candidate: Dict[str, Any],
    parsed_config: Dict[str, Any],
    snapshot: Dict[str, Any],
    options_module,
    runtime: RuntimeConfig,
) -> Optional[Dict[str, Any]]:
    expensive_alias = pair_candidate["expensive_alias"]
    cheap_alias = pair_candidate["cheap_alias"]

    short_ranked = rank_proxy_candidates(expensive_alias, "SHORT", parsed_config, snapshot, options_module, runtime)
    long_ranked = rank_proxy_candidates(cheap_alias, "LONG", parsed_config, snapshot, options_module, runtime)

    short_leg = short_ranked[0] if short_ranked else None
    long_leg = long_ranked[0] if long_ranked else None
    if not short_leg and not long_leg:
        return None

    risk_flags: List[str] = ["relative_value", "defined_risk_preferred", "synthetic_iv_fallback"]
    if short_leg and not short_leg.get("options_available"):
        risk_flags.append(f"no_listed_options_{short_leg['symbol']}")
    if long_leg and not long_leg.get("options_available"):
        risk_flags.append(f"no_listed_options_{long_leg['symbol']}")
    if short_leg and short_leg.get("status") != "CANDIDATE":
        risk_flags.append("short_leg_unavailable")
    if long_leg and long_leg.get("status") != "CANDIDATE":
        risk_flags.append("long_leg_unavailable")

    short_composite = short_leg.get("composite_score", 0.0) if short_leg else 0.0
    long_composite = long_leg.get("composite_score", 0.0) if long_leg else 0.0
    proxy_alignment = clamp((short_composite + long_composite) / 2.0, 0.0, 1.0)

    confidence = round(clamp(0.70 * float(pair_candidate["confidence"]) + 0.30 * proxy_alignment, 0.0, 1.0), 2)
    fit_score = round(clamp(0.65 * float(pair_candidate["fit_score"]) + 0.35 * proxy_alignment, 0.0, 1.0), 2)
    if fit_score < runtime.min_fit_score:
        return None

    structure_family = "PAIR_TRADE_WITH_DEFINED_RISK_OPTIONS"
    if (short_leg and not short_leg.get("options_available")) or (long_leg and not long_leg.get("options_available")):
        structure_family = "PAIR_TRADE_STOCK_PLUS_OPTION_OVERLAY"

    primary_proxy_trade = build_proxy_first_candidate_summary(pair_candidate, short_leg, long_leg)

    implementation = {
        "signal_source": {
            "pair_id": pair_candidate["pair_id"],
            "pair_label": pair_candidate["pair_label"],
            "category": pair_candidate["category"],
            "correlation": pair_candidate["correlation"],
            "hedge_ratio_beta": pair_candidate["hedge_ratio_beta"],
            "current_zscore": pair_candidate["current_zscore"],
            "expensive_alias": expensive_alias,
            "cheap_alias": cheap_alias,
        },
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
        "selected_proxy_legs": {
            "short_leg": short_leg,
            "long_leg": long_leg,
        },
        "proxy_rankings": {
            "short_leg_candidates": short_ranked[:5],
            "long_leg_candidates": long_ranked[:5],
        },
        "primary_recommendation": primary_proxy_trade,
        "trade_templates": {
            "short_leg_stock_action": (
                f"SELL {short_leg['symbol']}" if short_leg and short_leg.get("status") == "CANDIDATE" else None
            ),
            "long_leg_stock_action": (
                f"BUY {long_leg['symbol']}" if long_leg and long_leg.get("status") == "CANDIDATE" else None
            ),
            "short_leg_options": short_leg.get("options_template") if short_leg else None,
            "long_leg_options": long_leg.get("options_template") if long_leg else None,
        },
        "hard_limits": {
            "undefined_risk_allowed": False,
            "direct_order_routing_allowed": False,
        },
    }

    summary = primary_proxy_trade

    return {
        "candidate_id": f"div_{pair_candidate['pair_label'].replace('/', '_').lower()}",
        "strategy_type": "divergence",
        "direction": "RELATIVE_VALUE",
        "horizon": determine_horizon(pair_candidate["category"]),
        "structure_family": structure_family,
        "summary": summary,
        "confidence": confidence,
        "fit_score": fit_score,
        "thesis_tags": [
            "divergence",
            "relative_value",
            pair_candidate["category"],
            expensive_alias.lower(),
            cheap_alias.lower(),
            "options_analysis",
        ],
        "risk_flags": sorted(set(risk_flags)),
        "implementation": implementation,
    }


def build_candidates(
    pair_snapshot: Dict[str, Any],
    parsed_config: Dict[str, Any],
    snapshot: Dict[str, Any],
    options_module,
    runtime: RuntimeConfig,
) -> List[Dict[str, Any]]:
    payloads: List[Dict[str, Any]] = []
    for pair_candidate in pair_snapshot["trade_candidates"]:
        candidate = build_candidate_payload(pair_candidate, parsed_config, snapshot, options_module, runtime)
        if candidate is not None:
            payloads.append(candidate)
    payloads.sort(key=lambda x: (x["fit_score"], x["confidence"]), reverse=True)
    return payloads[: runtime.max_candidates]


def build_summary(candidates: List[Dict[str, Any]], pair_snapshot: Dict[str, Any]) -> str:
    if candidates:
        top = candidates[0]
        top_summary = top.get("summary")
        if top_summary:
            return f"Top divergence candidate: {top_summary}"

        pair_metrics = top["implementation"]["pair_metrics"]
        return (
            f"Top divergence candidate is {pair_metrics['pair_label']} with z-score {pair_metrics['current_zscore']:.2f} "
            f"and correlation {pair_metrics['correlation']:.2f}."
        )

    counts = pair_snapshot["counts"]
    return (
        f"No divergence candidates emitted. allowed_pairs={counts['allowed_pair_count']}, "
        f"pair_candidates={counts['candidate_count']}."
    )


def build_output(runtime: RuntimeConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    resolved_config_file = resolve_runtime_path(runtime.config_file, DEFAULT_CONFIG_FILE)
    resolved_options_module_path = resolve_runtime_path(runtime.options_module_path, DEFAULT_OPTIONS_MODULE_PATH)

    parsed_config = parse_config_file(str(resolved_config_file))
    pairs = build_allowed_pairs(parsed_config, runtime.pair_allowlist_text)
    options_module = import_options_module(str(resolved_options_module_path))

    cached = load_cache()
    use_cache = False
    if cached:
        if cached.get("config_file") == str(resolved_config_file.resolve()) and cached.get("options_module_path") == str(resolved_options_module_path.resolve()):
            snapshot = cached["snapshot"]
            use_cache = True
        else:
            snapshot = fetch_market_snapshot(parsed_config, runtime)
            save_cache({
                "config_file": str(resolved_config_file.resolve()),
                "options_module_path": str(resolved_options_module_path.resolve()),
                "snapshot": snapshot,
            })
    else:
        snapshot = fetch_market_snapshot(parsed_config, runtime)
        save_cache({
            "config_file": str(resolved_config_file.resolve()),
            "options_module_path": str(resolved_options_module_path.resolve()),
            "snapshot": snapshot,
        })

    pair_snapshot = build_pair_snapshot(snapshot, runtime, pairs)
    candidates = build_candidates(pair_snapshot, parsed_config, snapshot, options_module, runtime)

    output = {
        "agent": "divergence_analyst",
        "event_id": now_iso(),
        "strategy_type": "divergence",
        "candidates": candidates,
        "summary": build_summary(candidates, pair_snapshot),
    }
    validate_output_schema(output)

    internal_report = {
        "agent": "divergence_analyst",
        "event_id": output["event_id"],
        "cache_hit": use_cache,
        "config": {
            "config_file": str(resolved_config_file),
            "options_module_path": str(resolved_options_module_path),
            "missing_symbol_aliases": parsed_config["missing_symbol_aliases"],
            "additional_baskets": parsed_config["additional_baskets"],
            "resolved_alias_to_symbol": parsed_config["alias_to_symbol"],
        },
        "snapshot_meta": {
            "fetched_at": snapshot["fetched_at"],
            "future_errors": snapshot.get("future_errors", {}),
            "proxy_errors": snapshot.get("proxy_errors", {}),
        },
        "pair_snapshot": pair_snapshot,
        "candidate_payloads": candidates,
        "summary": output["summary"],
    }
    return output, internal_report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Config-driven Divergence Trade Analyst")
    parser.add_argument("--config-file", default=str(DEFAULT_CONFIG_FILE), help="Path to Futures Symbols.txt (defaults to the script directory copy)")
    parser.add_argument("--options-module-path", default=str(DEFAULT_OPTIONS_MODULE_PATH), help="Path to tos_options_agent_functions.py (defaults to the script directory copy)")
    parser.add_argument("--pairs", default=None, help="Optional comma-separated pair allowlist, e.g. GC/SI,CL/HO")
    parser.add_argument("--min-corr", type=float, default=0.50)
    parser.add_argument("--min-abs-z", type=float, default=1.25)
    parser.add_argument("--min-fit-score", type=float, default=0.40)
    parser.add_argument("--corr-window", type=int, default=126)
    parser.add_argument("--beta-window", type=int, default=126)
    parser.add_argument("--zscore-window", type=int, default=63)
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument("--option-min-dte", type=int, default=21)
    parser.add_argument("--option-max-dte", type=int, default=60)
    parser.add_argument("--iv-premium", type=float, default=1.10, help="Multiplier applied to 20d realized vol to synthesize IV series")
    parser.add_argument("--require-vol-up-for-sigma", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    runtime = RuntimeConfig(
        config_file=args.config_file,
        options_module_path=args.options_module_path,
        corr_window=args.corr_window,
        beta_window=args.beta_window,
        zscore_window=args.zscore_window,
        min_corr=args.min_corr,
        min_abs_z=args.min_abs_z,
        min_fit_score=args.min_fit_score,
        max_candidates=args.max_candidates,
        option_target_min_dte=args.option_min_dte,
        option_target_max_dte=args.option_max_dte,
        iv_premium=args.iv_premium,
        require_vol_up_for_sigma=args.require_vol_up_for_sigma,
        pair_allowlist_text=args.pairs,
    )

    output, internal_report = build_output(runtime)

    print(json.dumps(output, indent=2))

    payload_path = REPORTS_DIR / f"divergence_payload_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(payload_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    diagnostics_path = REPORTS_DIR / f"divergence_diagnostics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(diagnostics_path, "w", encoding="utf-8") as f:
        json.dump(internal_report, f, indent=2)

    print(f"\n[INFO] Coordinator payload written to {payload_path}")
    print(f"[INFO] Diagnostics written to {diagnostics_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {exc}")
        raise SystemExit(1)
