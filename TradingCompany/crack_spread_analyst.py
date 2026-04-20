#!/usr/bin/env python3
"""
crack_spread_analyst.py

Config-driven Crack Spread Analyst that:
- reads futures aliases and refiner basket from "Futures Symbols.txt"
- translates the Thinkorswim crack spread tracker into deterministic Python
- calculates raw crack spread, normalized crack spread, and crack-vs-refiner relative ranks
- uses tos_options_agent_functions.py to analyze sigma re-entry, expected moves, and HV/IV state
- emits a Trade Coordinator-compatible specialist payload on stdout
- writes a richer diagnostics report to reports/

Notes
-----
- The crack-spread math mirrors the Thinkorswim script:
    CrackSpread = ((2/3) * RB * 42 + (1/3) * HO * 42) - CL
- Refiners are normalized from the chart start / first aligned observation.
- Refiner ranks mimic the Thinkorswim global rank idea by comparing each
  crack-vs-refiner spread against the historical min/max across all refiners.
- This is analysis-only. It never routes orders.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import pickle
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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

CACHE_FILE = Path("crack_spread_analyst_cache.pkl")
CACHE_TTL_SECONDS = 300

NO_TRADE_REASON_CODES = [
    "MISSING_CRACK_FUTURES",
    "MISSING_REFINER_BASKET",
    "NO_REFINER_HISTORY",
    "NO_REFINER_SIGNAL",
    "LOW_CRACK_ZSCORE",
    "DATA_FETCH_ERRORS_PRESENT",
    "NO_TRADE_CONDITIONS_NOT_MET",
]

OUTPUT_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": ["agent", "event_id", "strategy_type", "candidates", "summary"],
    "properties": {
        "agent": {"type": "string", "const": "crack_spread_analyst"},
        "event_id": {"type": "string", "format": "date-time"},
        "strategy_type": {"type": "string", "const": "crack_spread"},
        "candidates": {
            "type": "array",
            "items": {
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
                    "strategy_type": {"type": "string", "const": "crack_spread"},
                    "direction": {"type": "string", "enum": ["BULLISH", "BEARISH", "RELATIVE_VALUE"]},
                    "horizon": {"type": "string", "enum": ["SHORT_TERM", "LONG_TERM"]},
                    "structure_family": {"type": "string"},
                    "summary": {"type": "string", "minLength": 1},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "fit_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "thesis_tags": {"type": "array", "items": {"type": "string"}},
                    "risk_flags": {"type": "array", "items": {"type": "string"}},
                    "implementation": {"type": "object", "additionalProperties": True},
                },
            },
        },
        "summary": {"type": "string"},
    },
}

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


@dataclass
class RuntimeConfig:
    config_file: str = "Futures Symbols.txt"
    options_module_path: str = "tos_options_agent_functions.py"
    daily_period: str = "2y"
    daily_interval: str = "1d"
    current_attempts: Tuple[Tuple[str, str], ...] = (("1d", "1m"), ("5d", "5m"), ("5d", "15m"))
    zscore_window: int = 63
    min_abs_z: float = 1.25
    min_fit_score: float = 0.40
    max_candidates: int = 10
    option_target_min_dte: int = 21
    option_target_max_dte: int = 60
    iv_premium: float = 1.10
    require_vol_up_for_sigma: bool = False


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

    if not hasattr(module, "build_options_analysis_packet"):
        raise AttributeError(f"Missing required function 'build_options_analysis_packet' in {module_path}")
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
            values = [token.strip().upper() for token in right.split(",") if token.strip()]

            if mode == "futures":
                category = left.replace("Futures Symbols", "").strip().lower()
                category_aliases[category] = values
            else:
                alias = left.strip().upper()
                if alias in DEFAULT_ALIAS_TO_YF_SYMBOL:
                    proxy_groups[alias] = values
                else:
                    additional_baskets[alias] = values

    alias_to_category: Dict[str, str] = {}
    for category, aliases in category_aliases.items():
        for alias in aliases:
            alias_to_category[alias] = category

    alias_to_symbol = {
        alias: DEFAULT_ALIAS_TO_YF_SYMBOL[alias]
        for alias in alias_to_category
        if alias in DEFAULT_ALIAS_TO_YF_SYMBOL
    }

    return {
        "category_aliases": category_aliases,
        "alias_to_category": alias_to_category,
        "alias_to_symbol": alias_to_symbol,
        "proxy_groups": proxy_groups,
        "additional_baskets": additional_baskets,
        "missing_symbol_aliases": sorted([alias for alias in alias_to_category if alias not in alias_to_symbol]),
    }


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


def build_synthetic_implied_volatility(df: pd.DataFrame, iv_premium: float = 1.10) -> pd.Series:
    close = df["Close"].astype(float)
    log_ret = np.log(close / close.shift(1))
    realized = log_ret.rolling(20).std(ddof=1) * np.sqrt(252.0)
    synthetic_iv = (realized * iv_premium).clip(lower=0.05, upper=3.0)
    return synthetic_iv.bfill().ffill()


def get_option_expirations(symbol: str) -> List[str]:
    try:
        return list(yf.Ticker(symbol).options or [])
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


def build_refiner_option_template(side: str, runtime: RuntimeConfig) -> Dict[str, Any]:
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


def prepare_price_like_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    min_low = safe_float(out["Low"].min(), 1.0)
    shift_value = 0.0
    if min_low is not None and min_low <= 0:
        shift_value = abs(min_low) + 1.0
        for col in ["Open", "High", "Low", "Close"]:
            out[col] = out[col].astype(float) + shift_value
    out.attrs["price_shift"] = shift_value
    return out


def build_crack_ohlc(cl_df: pd.DataFrame, rb_df: pd.DataFrame, ho_df: pd.DataFrame) -> pd.DataFrame:
    joined = pd.concat(
        [
            cl_df[["Open", "High", "Low", "Close"]].add_prefix("CL_"),
            rb_df[["Open", "High", "Low", "Close"]].add_prefix("RB_"),
            ho_df[["Open", "High", "Low", "Close"]].add_prefix("HO_"),
        ],
        axis=1,
        join="inner",
    ).dropna()

    crack = pd.DataFrame(index=joined.index)
    crack["Open"] = ((2.0 / 3.0) * joined["RB_Open"] * 42.0 + (1.0 / 3.0) * joined["HO_Open"] * 42.0) - joined["CL_Open"]
    crack["Close"] = ((2.0 / 3.0) * joined["RB_Close"] * 42.0 + (1.0 / 3.0) * joined["HO_Close"] * 42.0) - joined["CL_Close"]
    crack["High"] = ((2.0 / 3.0) * joined["RB_High"] * 42.0 + (1.0 / 3.0) * joined["HO_High"] * 42.0) - joined["CL_Low"]
    crack["Low"] = ((2.0 / 3.0) * joined["RB_Low"] * 42.0 + (1.0 / 3.0) * joined["HO_Low"] * 42.0) - joined["CL_High"]
    return crack.dropna(how="any")


def fetch_market_snapshot(parsed_config: Dict[str, Any], runtime: RuntimeConfig) -> Dict[str, Any]:
    crack_aliases = ["CL", "RB", "HO"]
    futures_frames: Dict[str, pd.DataFrame] = {}
    futures_features: Dict[str, Dict[str, Any]] = {}
    future_errors: Dict[str, str] = {}

    for alias in crack_aliases:
        symbol = parsed_config["alias_to_symbol"].get(alias)
        if not symbol:
            future_errors[alias] = "Missing alias to symbol mapping"
            continue
        try:
            df = fetch_history(symbol, runtime.daily_period, runtime.daily_interval)
            futures_frames[symbol] = df
            futures_features[symbol] = build_symbol_features(symbol, df, runtime.current_attempts, alias=alias)
        except Exception as exc:
            future_errors[symbol] = str(exc)

    refiners = parsed_config["additional_baskets"].get("REFINERS", [])
    refiner_frames: Dict[str, pd.DataFrame] = {}
    refiner_features: Dict[str, Dict[str, Any]] = {}
    refiner_errors: Dict[str, str] = {}

    for symbol in refiners:
        try:
            df = fetch_history(symbol, runtime.daily_period, runtime.daily_interval)
            refiner_frames[symbol] = df
            refiner_features[symbol] = build_symbol_features(symbol, df, runtime.current_attempts, alias=None)
        except Exception as exc:
            refiner_errors[symbol] = str(exc)

    return {
        "fetched_at": now_iso(),
        "futures_frames": futures_frames,
        "futures_features": futures_features,
        "future_errors": future_errors,
        "refiner_frames": refiner_frames,
        "refiner_features": refiner_features,
        "refiner_errors": refiner_errors,
    }


def compute_crack_tracker(snapshot: Dict[str, Any], parsed_config: Dict[str, Any], runtime: RuntimeConfig) -> Dict[str, Any]:
    cl_symbol = parsed_config["alias_to_symbol"].get("CL")
    rb_symbol = parsed_config["alias_to_symbol"].get("RB")
    ho_symbol = parsed_config["alias_to_symbol"].get("HO")

    if not cl_symbol or not rb_symbol or not ho_symbol:
        raise ValueError("CL, RB, and HO aliases must resolve to futures symbols.")

    cl_df = snapshot["futures_frames"].get(cl_symbol)
    rb_df = snapshot["futures_frames"].get(rb_symbol)
    ho_df = snapshot["futures_frames"].get(ho_symbol)
    if cl_df is None or rb_df is None or ho_df is None or cl_df.empty or rb_df.empty or ho_df.empty:
        raise ValueError("Missing CL/RB/HO history required for crack spread calculation.")

    crack_df = build_crack_ohlc(cl_df, rb_df, ho_df)
    if crack_df.empty:
        raise ValueError("Crack OHLC frame is empty after alignment.")

    base_crack = safe_float(crack_df["Close"].iloc[0])
    if base_crack in (None, 0):
        raise ValueError("Base crack spread is missing or zero.")

    crack_df["RawCrackSpread"] = crack_df["Close"]
    crack_df["NormCrack"] = (crack_df["Close"] / float(base_crack) * 100.0) - 100.0

    crack_price_like = prepare_price_like_ohlc(crack_df[["Open", "High", "Low", "Close"]].copy())
    crack_iv = build_synthetic_implied_volatility(crack_price_like, iv_premium=runtime.iv_premium)

    cl_eff = snapshot["futures_features"].get(cl_symbol, {}).get("effective_price")
    rb_eff = snapshot["futures_features"].get(rb_symbol, {}).get("effective_price")
    ho_eff = snapshot["futures_features"].get(ho_symbol, {}).get("effective_price")
    current_raw_crack = None
    if cl_eff not in (None, 0) and rb_eff not in (None, 0) and ho_eff not in (None, 0):
        current_raw_crack = ((2.0 / 3.0) * float(rb_eff) * 42.0 + (1.0 / 3.0) * float(ho_eff) * 42.0) - float(cl_eff)
    else:
        current_raw_crack = safe_float(crack_df["RawCrackSpread"].iloc[-1])
    current_norm_crack = (float(current_raw_crack) / float(base_crack) * 100.0) - 100.0 if current_raw_crack not in (None, 0) else safe_float(crack_df["NormCrack"].iloc[-1])

    return {
        "crack_df": crack_df,
        "base_crack": float(base_crack),
        "crack_price_like": crack_price_like,
        "crack_iv": crack_iv,
        "current_raw_crack": safe_float(current_raw_crack),
        "current_norm_crack": safe_float(current_norm_crack),
    }


def build_current_refiner_price(frame: pd.DataFrame, feature: Dict[str, Any]) -> float:
    eff = safe_float(feature.get("effective_price"))
    return float(eff) if eff not in (None, 0) else float(frame["Close"].iloc[-1])


def analyze_refiner(
    symbol: str,
    crack_tracker: Dict[str, Any],
    snapshot: Dict[str, Any],
    options_module,
    runtime: RuntimeConfig,
) -> Dict[str, Any]:
    feature = snapshot["refiner_features"].get(symbol)
    frame = snapshot["refiner_frames"].get(symbol)
    if feature is None or frame is None or frame.empty:
        return {
            "symbol": symbol,
            "status": "UNAVAILABLE",
            "reason": "MISSING_HISTORY",
            "fit_score": -999.0,
        }

    crack_df = crack_tracker["crack_df"]
    aligned = pd.concat(
        [crack_df[["NormCrack"]], frame[["Open", "High", "Low", "Close"]]],
        axis=1,
        join="inner",
    ).dropna()

    min_needed = runtime.zscore_window + 5
    if len(aligned) < min_needed:
        return {
            "symbol": symbol,
            "status": "UNAVAILABLE",
            "reason": f"INSUFFICIENT_HISTORY_{len(aligned)}",
            "fit_score": -999.0,
        }

    base_ref = safe_float(aligned["Close"].iloc[0])
    if base_ref in (None, 0):
        return {
            "symbol": symbol,
            "status": "UNAVAILABLE",
            "reason": "ZERO_BASE_REFINER",
            "fit_score": -999.0,
        }

    aligned["RefinerNorm"] = (aligned["Close"] / float(base_ref) * 100.0) - 100.0
    aligned["CrackVsRefiner"] = aligned["NormCrack"] - aligned["RefinerNorm"]

    spread_series = aligned["CrackVsRefiner"].dropna()
    hist_window = spread_series.tail(runtime.zscore_window)
    mu = safe_float(hist_window.mean())
    sigma = safe_float(hist_window.std(ddof=1))
    if sigma is None or sigma <= 1e-10:
        return {
            "symbol": symbol,
            "status": "UNAVAILABLE",
            "reason": "BAD_SIGMA",
            "fit_score": -999.0,
        }

    current_refiner_price = build_current_refiner_price(frame, feature)
    current_refiner_norm = (current_refiner_price / float(base_ref) * 100.0) - 100.0
    current_norm_crack = float(crack_tracker["current_norm_crack"])
    current_spread = current_norm_crack - current_refiner_norm
    current_z = (current_spread - float(mu)) / float(sigma)

    # positive spread => crack outperforming refiner => refiner cheap vs margins => LONG refiner
    side = "LONG" if current_spread > 0 else "SHORT"
    direction = "BULLISH" if side == "LONG" else "BEARISH"

    global_min = safe_float(spread_series.min(), current_spread)
    global_max = safe_float(spread_series.max(), current_spread)
    if global_min is None or global_max is None or global_max == global_min:
        global_rank = 0.5
    else:
        global_rank = (current_spread - global_min) / (global_max - global_min)

    price_like = frame[["Open", "High", "Low", "Close"]].copy()
    synthetic_iv = build_synthetic_implied_volatility(price_like, iv_premium=runtime.iv_premium)
    packet = options_module.build_options_analysis_packet(
        price_like.rename(columns=str.lower),
        implied_volatility=synthetic_iv,
        require_vol_up_for_sigma=runtime.require_vol_up_for_sigma,
    )

    crack_packet = options_module.build_options_analysis_packet(
        crack_tracker["crack_price_like"].rename(columns=str.lower),
        implied_volatility=crack_tracker["crack_iv"],
        require_vol_up_for_sigma=runtime.require_vol_up_for_sigma,
    )

    sigma_state = packet["sigma_reentry"]["state"]
    expected_move_side = infer_expected_move_side(packet["expected_move"])
    crack_sigma_state = crack_packet["sigma_reentry"]["state"]
    crack_em_side = infer_expected_move_side(crack_packet["expected_move"])
    vol_up = bool(packet["vol_filter"].get("vol_up", False))

    signal_score = 0.0
    if side == "LONG":
        if sigma_state == "BUY_REENTRY":
            signal_score += 1.0
        if expected_move_side in {"BELOW_WEEK_LOWER", "BELOW_MONTH_LOWER"}:
            signal_score += 1.0
        if crack_sigma_state == "SELL_REENTRY":
            signal_score += 1.0
        if crack_em_side in {"ABOVE_WEEK_UPPER", "ABOVE_MONTH_UPPER"}:
            signal_score += 1.0
        rank_component = global_rank
    else:
        if sigma_state == "SELL_REENTRY":
            signal_score += 1.0
        if expected_move_side in {"ABOVE_WEEK_UPPER", "ABOVE_MONTH_UPPER"}:
            signal_score += 1.0
        if crack_sigma_state == "BUY_REENTRY":
            signal_score += 1.0
        if crack_em_side in {"BELOW_WEEK_LOWER", "BELOW_MONTH_LOWER"}:
            signal_score += 1.0
        rank_component = 1.0 - global_rank

    z_component = clamp(abs(current_z) / 3.0, 0.0, 1.0)
    signal_component = signal_score / 4.0
    vol_component = 1.0 if vol_up else 0.0

    options_available, selected_expiration = choose_option_expiration(
        symbol,
        runtime.option_target_min_dte,
        runtime.option_target_max_dte,
    )
    options_component = 1.0 if options_available else 0.0

    fit_score = clamp(
        0.40 * z_component +
        0.25 * clamp(rank_component, 0.0, 1.0) +
        0.20 * signal_component +
        0.10 * vol_component +
        0.05 * options_component,
        0.0,
        1.0,
    )
    confidence = clamp(0.65 * fit_score + 0.35 * z_component, 0.0, 1.0)

    return {
        "symbol": symbol,
        "status": "CANDIDATE",
        "side": side,
        "direction": direction,
        "effective_price": safe_float(current_refiner_price),
        "price_source": feature.get("price_source"),
        "normalized_refiner_return": round(float(current_refiner_norm), 4),
        "current_spread": round(float(current_spread), 4),
        "current_zscore": round(float(current_z), 4),
        "global_rank": round(float(global_rank), 4),
        "fit_score": round(float(fit_score), 4),
        "confidence": round(float(confidence), 4),
        "options_available": options_available,
        "selected_expiration": selected_expiration,
        "options_template": build_refiner_option_template(side, runtime),
        "refiner_options_packet": packet,
        "crack_options_packet": crack_packet,
    }


def build_refiner_rankings(
    parsed_config: Dict[str, Any],
    crack_tracker: Dict[str, Any],
    snapshot: Dict[str, Any],
    options_module,
    runtime: RuntimeConfig,
) -> List[Dict[str, Any]]:
    refiners = parsed_config["additional_baskets"].get("REFINERS", [])
    analytics: List[Dict[str, Any]] = []
    for symbol in refiners:
        analytics.append(analyze_refiner(symbol, crack_tracker, snapshot, options_module, runtime))
    analytics.sort(key=lambda x: x.get("fit_score", -999.0), reverse=True)
    return analytics


def build_candidate_payload(row: Dict[str, Any], crack_tracker: Dict[str, Any], runtime: RuntimeConfig) -> Optional[Dict[str, Any]]:
    if row.get("status") != "CANDIDATE":
        return None
    if abs(float(row.get("current_zscore", 0.0))) < runtime.min_abs_z:
        return None
    if float(row.get("fit_score", 0.0)) < runtime.min_fit_score:
        return None

    side = row["side"]
    direction = row["direction"]
    symbol = row["symbol"]
    template = row["options_template"]
    structure_family = "CRACK_SPREAD_REFINER_DEFINED_RISK"

    stock_action = f"BUY {symbol}" if side == "LONG" else f"SELL {symbol}"
    option_action = template.get("preferred_structure") if template else None

    risk_flags = ["crack_spread", "synthetic_iv_fallback", "defined_risk_preferred"]
    if not row.get("options_available"):
        risk_flags.append(f"no_listed_options_{symbol}")

    summary = (
        f"{stock_action} because crack-vs-{symbol} spread z-score is {row['current_zscore']:.2f} "
        f"with rank {row['global_rank']:.2f}; use {option_action.replace('_', ' ').lower()} when options are available."
        if option_action else
        f"{stock_action} because crack-vs-{symbol} spread z-score is {row['current_zscore']:.2f} with rank {row['global_rank']:.2f}."
    )

    implementation = {
        "signal_source": {
            "raw_crack_spread": crack_tracker["current_raw_crack"],
            "normalized_crack_spread": crack_tracker["current_norm_crack"],
            "refiner_symbol": symbol,
            "refiner_normalized_return": row["normalized_refiner_return"],
            "current_spread": row["current_spread"],
            "current_zscore": row["current_zscore"],
            "global_rank": row["global_rank"],
        },
        "primary_recommendation": stock_action,
        "trade_templates": {
            "stock_action": stock_action,
            "options": template,
            "selected_expiration": row.get("selected_expiration"),
        },
        "signal_context": {
            "refiner_options_packet": row.get("refiner_options_packet"),
            "crack_options_packet": row.get("crack_options_packet"),
        },
        "hard_limits": {
            "undefined_risk_allowed": False,
            "direct_order_routing_allowed": False,
        },
    }

    return {
        "candidate_id": f"crack_{symbol.lower()}_{side.lower()}",
        "strategy_type": "crack_spread",
        "direction": direction,
        "horizon": "SHORT_TERM",
        "structure_family": structure_family,
        "summary": summary,
        "confidence": round(float(row["confidence"]), 2),
        "fit_score": round(float(row["fit_score"]), 2),
        "thesis_tags": ["crack_spread", "refiners", side.lower(), symbol.lower(), "options_analysis"],
        "risk_flags": sorted(set(risk_flags)),
        "implementation": implementation,
    }


def build_summary(candidates: List[Dict[str, Any]], refiner_rankings: List[Dict[str, Any]]) -> str:
    if candidates:
        return f"Top crack spread candidate: {candidates[0]['summary']}"
    ranked = [r for r in refiner_rankings if r.get("status") == "CANDIDATE"]
    return f"No crack spread candidates emitted. refiners_analyzed={len(ranked)}."


def build_output(runtime: RuntimeConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    parsed_config = parse_config_file(runtime.config_file)
    options_module = import_options_module(runtime.options_module_path)

    cached = load_cache()
    use_cache = False
    if cached:
        if cached.get("config_file") == str(Path(runtime.config_file).resolve()) and cached.get("options_module_path") == str(Path(runtime.options_module_path).resolve()):
            snapshot = cached["snapshot"]
            use_cache = True
        else:
            snapshot = fetch_market_snapshot(parsed_config, runtime)
            save_cache({
                "config_file": str(Path(runtime.config_file).resolve()),
                "options_module_path": str(Path(runtime.options_module_path).resolve()),
                "snapshot": snapshot,
            })
    else:
        snapshot = fetch_market_snapshot(parsed_config, runtime)
        save_cache({
            "config_file": str(Path(runtime.config_file).resolve()),
            "options_module_path": str(Path(runtime.options_module_path).resolve()),
            "snapshot": snapshot,
        })

    crack_tracker = compute_crack_tracker(snapshot, parsed_config, runtime)
    refiner_rankings = build_refiner_rankings(parsed_config, crack_tracker, snapshot, options_module, runtime)

    candidates: List[Dict[str, Any]] = []
    for row in refiner_rankings:
        candidate = build_candidate_payload(row, crack_tracker, runtime)
        if candidate is not None:
            candidates.append(candidate)
    candidates.sort(key=lambda x: (x["fit_score"], x["confidence"]), reverse=True)
    candidates = candidates[: runtime.max_candidates]

    output = {
        "agent": "crack_spread_analyst",
        "event_id": now_iso(),
        "strategy_type": "crack_spread",
        "candidates": candidates,
        "summary": build_summary(candidates, refiner_rankings),
    }
    validate_output_schema(output)

    no_trade_codes: List[str] = []
    if parsed_config["alias_to_symbol"].get("CL") is None or parsed_config["alias_to_symbol"].get("RB") is None or parsed_config["alias_to_symbol"].get("HO") is None:
        no_trade_codes.append("MISSING_CRACK_FUTURES")
    if not parsed_config["additional_baskets"].get("REFINERS"):
        no_trade_codes.append("MISSING_REFINER_BASKET")
    if snapshot.get("future_errors") or snapshot.get("refiner_errors"):
        no_trade_codes.append("DATA_FETCH_ERRORS_PRESENT")
    if not candidates:
        if not refiner_rankings:
            no_trade_codes.append("NO_REFINER_HISTORY")
        else:
            all_z_low = all(abs(float(r.get("current_zscore", 0.0))) < runtime.min_abs_z for r in refiner_rankings if r.get("status") == "CANDIDATE")
            if all_z_low:
                no_trade_codes.append("LOW_CRACK_ZSCORE")
            else:
                no_trade_codes.append("NO_REFINER_SIGNAL")
    if not no_trade_codes and not candidates:
        no_trade_codes.append("NO_TRADE_CONDITIONS_NOT_MET")

    diagnostics = {
        "agent": "crack_spread_analyst",
        "event_id": output["event_id"],
        "cache_hit": use_cache,
        "config": {
            "config_file": runtime.config_file,
            "options_module_path": runtime.options_module_path,
            "resolved_alias_to_symbol": parsed_config["alias_to_symbol"],
            "refiner_basket": parsed_config["additional_baskets"].get("REFINERS", []),
        },
        "snapshot_meta": {
            "fetched_at": snapshot["fetched_at"],
            "future_errors": snapshot.get("future_errors", {}),
            "refiner_errors": snapshot.get("refiner_errors", {}),
        },
        "crack_tracker": {
            "base_crack": crack_tracker["base_crack"],
            "current_raw_crack": crack_tracker["current_raw_crack"],
            "current_norm_crack": crack_tracker["current_norm_crack"],
        },
        "refiner_rankings": refiner_rankings,
        "candidate_payloads": candidates,
        "no_trade": {
            "reason_codes": sorted(set(no_trade_codes)) if no_trade_codes else None,
            "summary": output["summary"] if no_trade_codes else None,
        },
        "summary": output["summary"],
    }
    return output, diagnostics


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Config-driven Crack Spread Analyst")
    parser.add_argument("--config-file", default="Futures Symbols.txt")
    parser.add_argument("--options-module-path", default="tos_options_agent_functions.py")
    parser.add_argument("--min-abs-z", type=float, default=1.25)
    parser.add_argument("--min-fit-score", type=float, default=0.40)
    parser.add_argument("--zscore-window", type=int, default=63)
    parser.add_argument("--max-candidates", type=int, default=10)
    parser.add_argument("--option-min-dte", type=int, default=21)
    parser.add_argument("--option-max-dte", type=int, default=60)
    parser.add_argument("--iv-premium", type=float, default=1.10)
    parser.add_argument("--require-vol-up-for-sigma", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    runtime = RuntimeConfig(
        config_file=args.config_file,
        options_module_path=args.options_module_path,
        zscore_window=args.zscore_window,
        min_abs_z=args.min_abs_z,
        min_fit_score=args.min_fit_score,
        max_candidates=args.max_candidates,
        option_target_min_dte=args.option_min_dte,
        option_target_max_dte=args.option_max_dte,
        iv_premium=args.iv_premium,
        require_vol_up_for_sigma=args.require_vol_up_for_sigma,
    )

    output, diagnostics = build_output(runtime)

    print(json.dumps(output, indent=2))

    payload_path = REPORTS_DIR / f"crack_spread_payload_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(payload_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    diagnostics_path = REPORTS_DIR / f"crack_spread_diagnostics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(diagnostics_path, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)

    print(f"\n[INFO] Coordinator payload written to {payload_path}")
    print(f"[INFO] Diagnostics written to {diagnostics_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {exc}")
        raise SystemExit(1)
