#!/usr/bin/env python3
"""
crack_spread_analyst_refactored_v2.py

Refactored Crack Spread Analyst
===============================
Built to preserve the behavior of crack_spread_analyst_verbose.py while adding
peer-relative stock proxy ranking on:
- 3-year weekly charts
- 1-year daily charts

New in v2
---------
- Refiners are now ranked by peer-relative price performance using:
    * 3y weekly percentage change
    * 1y daily percentage change
- A composite peer-expensiveness score is computed from both horizons
- In verbose mode, the full ranking table is printed
- The actual stock proxy selected now depends on crack direction:
    * crack cheap / bullish refiners  -> choose the CHEAPEST peer
    * crack rich / bearish refiners   -> choose the MOST EXPENSIVE peer
- Crack-deviation analysis is preserved and shown alongside the new rankings

Core behavior preserved
-----------------------
- Loads energies futures and refiners from Futures Symbols.txt
- Computes crack spread exactly as in Crack Tracker with Normalizat.txt
- Runs build_options_analysis_packet(...) from tos_options_agent_functions.py
- Applies safety-first z-score and fit-score filters
- Emits coordinator-compatible JSON on stdout
- Writes a timestamped full diagnostics report to reports/

This script is analysis only.
It does not place trades.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from jsonschema import Draft202012Validator, FormatChecker

from tos_options_agent_functions import build_options_analysis_packet

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_FILE = Path("Futures Symbols.txt")

USE_LLM_SUMMARY = False
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# -------------------- SCHEMAS --------------------
COORDINATOR_CANDIDATE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "candidate_id", "strategy_type", "direction", "horizon", "structure_family",
        "summary", "confidence", "fit_score", "thesis_tags", "risk_flags", "implementation"
    ],
    "properties": {
        "candidate_id": {"type": "string"},
        "strategy_type": {"type": "string", "const": "crack_spread"},
        "direction": {"type": "string", "const": "RELATIVE_VALUE"},
        "horizon": {"type": "string", "const": "SHORT_TERM"},
        "structure_family": {"type": "string"},
        "summary": {"type": "string"},
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
        "agent": {"type": "string", "const": "crack_spread_analyst"},
        "event_id": {"type": "string", "format": "date-time"},
        "strategy_type": {"type": "string", "const": "crack_spread"},
        "candidates": {"type": "array", "items": COORDINATOR_CANDIDATE_SCHEMA},
        "summary": {"type": "string"},
    },
}


@dataclass
class RuntimeConfig:
    daily_period: str = "2y"
    daily_interval: str = "1d"
    zscore_window: int = 63
    min_abs_z: float = 1.0
    min_fit_score: float = 0.55
    max_candidates: int = 6
    require_vol_up: bool = True
    sigma_length: int = 21
    verbose: bool = True
    proxy_weekly_period: str = "3y"
    proxy_weekly_interval: str = "1wk"
    proxy_daily_period: str = "1y"
    proxy_daily_interval: str = "1d"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def log(msg: str, enabled: bool = True) -> None:
    if enabled:
        print(msg)


def validate_output(payload: Dict[str, Any]) -> None:
    validator = Draft202012Validator(OUTPUT_SCHEMA, format_checker=FormatChecker())
    errors = sorted(validator.iter_errors(payload), key=lambda e: list(e.path))
    if errors:
        lines = []
        for err in errors[:10]:
            path = ".".join(str(p) for p in err.path) if err.path else "<root>"
            lines.append(f"{path}: {err.message}")
        raise ValueError("Schema validation failed:\n" + "\n".join(lines))


def load_futures_config(config_path: Path = CONFIG_FILE) -> Tuple[Dict[str, List[str]], List[str]]:
    log(f"\n📄 LOADING CONFIG → {config_path}")
    text = config_path.read_text(encoding="utf-8")
    energies_futures: List[str] = []
    refiners: List[str] = []

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "Energies Futures Symbols" in line:
            energies_futures = [s.strip() for s in line.split("=", 1)[1].split(",") if s.strip()]
        elif line.startswith("Refiners ="):
            refiners = [s.strip() for s in line.split("=", 1)[1].split(",") if s.strip()]

    log(f"   → Energies futures: {energies_futures}")
    log(f"   → Refiners: {refiners}")
    return {"energies": energies_futures}, refiners


def fetch_history(symbol: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=False)
    df = df.dropna(how="all")
    if df.empty:
        raise ValueError(f"No history returned for {symbol}")
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


def align_frames(*frames: pd.DataFrame) -> List[pd.DataFrame]:
    common_idx = frames[0].index
    for df in frames[1:]:
        common_idx = common_idx.intersection(df.index)
    return [df.loc[common_idx].copy() for df in frames]


def compute_crack_spread_and_normalized(df_cl: pd.DataFrame, df_rb: pd.DataFrame, df_ho: pd.DataFrame) -> pd.DataFrame:
    cl = df_cl["Close"].astype(float)
    rb = df_rb["Close"].astype(float)
    ho = df_ho["Close"].astype(float)

    rb_per_barrel = rb * 42.0
    ho_per_barrel = ho * 42.0
    crack_spread = ((2.0 / 3.0) * rb_per_barrel + (1.0 / 3.0) * ho_per_barrel) - cl

    base_crack = ((2.0 / 3.0) * rb.iloc[0] * 42.0) + ((1.0 / 3.0) * ho.iloc[0] * 42.0) - cl.iloc[0]
    if base_crack == 0:
        raise ValueError("Base crack spread is zero; cannot normalize.")

    norm_crack = (crack_spread / base_crack * 100.0) - 100.0

    return pd.DataFrame({
        "crack_spread": crack_spread,
        "norm_crack_pct": norm_crack,
    }, index=cl.index)


def build_synthetic_crack_ohlc(crack_df: pd.DataFrame, sigma_length: int) -> pd.DataFrame:
    close = crack_df["crack_spread"].astype(float)
    daily_range = close.rolling(sigma_length).std(ddof=1) * 1.5
    out = pd.DataFrame({
        "Close": close,
        "High": close + daily_range,
        "Low": close - daily_range,
    }, index=crack_df.index)
    return out.dropna(how="any")


def compute_period_return_pct(df: pd.DataFrame) -> Optional[float]:
    if df is None or df.empty or len(df) < 2:
        return None
    close = df["Close"].astype(float).dropna()
    if len(close) < 2:
        return None
    start = safe_float(close.iloc[0])
    end = safe_float(close.iloc[-1])
    if start in (None, 0) or end is None:
        return None
    return round(((end / start) - 1.0) * 100.0, 4)


def percentile_rank(values: List[float], target: float) -> Optional[float]:
    if not values:
        return None
    arr = sorted(values)
    n = len(arr)
    if n == 1:
        return 0.5
    count_less = sum(1 for x in arr if x < target)
    count_equal = sum(1 for x in arr if x == target)
    return round((count_less + 0.5 * count_equal) / n, 4)


def rank_refiners_by_peer_performance(refiners_list: List[str], config: RuntimeConfig) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for sym in refiners_list:
        try:
            weekly = fetch_history(sym, period=config.proxy_weekly_period, interval=config.proxy_weekly_interval)
            daily = fetch_history(sym, period=config.proxy_daily_period, interval=config.proxy_daily_interval)
            weekly_ret = compute_period_return_pct(weekly)
            daily_ret = compute_period_return_pct(daily)
            rows.append({
                "symbol": sym,
                "weekly_3y_return_pct": weekly_ret,
                "daily_1y_return_pct": daily_ret,
                "weekly_rows": int(len(weekly)) if weekly is not None else 0,
                "daily_rows": int(len(daily)) if daily is not None else 0,
            })
        except Exception as exc:
            rows.append({
                "symbol": sym,
                "weekly_3y_return_pct": None,
                "daily_1y_return_pct": None,
                "weekly_rows": 0,
                "daily_rows": 0,
                "error": str(exc),
            })

    valid_weekly = [r["weekly_3y_return_pct"] for r in rows if r.get("weekly_3y_return_pct") is not None]
    valid_daily = [r["daily_1y_return_pct"] for r in rows if r.get("daily_1y_return_pct") is not None]

    for row in rows:
        wr = row.get("weekly_3y_return_pct")
        dr = row.get("daily_1y_return_pct")
        row["weekly_3y_percentile"] = percentile_rank(valid_weekly, wr) if wr is not None else None
        row["daily_1y_percentile"] = percentile_rank(valid_daily, dr) if dr is not None else None

        percentiles = [x for x in [row.get("weekly_3y_percentile"), row.get("daily_1y_percentile")] if x is not None]
        if percentiles:
            composite = float(np.mean(percentiles))
            row["peer_expensiveness_score"] = round((composite - 0.5) * 2.0, 4)  # -1 cheap, +1 expensive
            row["peer_rank"] = "EXPENSIVE" if composite >= 0.67 else "CHEAP" if composite <= 0.33 else "MID"
        else:
            row["peer_expensiveness_score"] = None
            row["peer_rank"] = "UNKNOWN"

    rows.sort(key=lambda x: (x.get("peer_expensiveness_score") is None, -(x.get("peer_expensiveness_score") or -999.0)))
    return rows


def rank_refiners_by_deviation_from_crack(
    refiners_list: List[str],
    crack_norm_pct: float,
    config: RuntimeConfig,
) -> List[Dict[str, Any]]:
    rankings: List[Dict[str, Any]] = []
    for sym in refiners_list:
        try:
            df = fetch_history(sym, period=config.daily_period, interval=config.daily_interval)
            if len(df) < 5:
                continue
            close = df["Close"].astype(float)
            base = float(close.iloc[0])
            if base == 0:
                continue
            norm_pct = (float(close.iloc[-1]) / base * 100.0) - 100.0
            deviation = norm_pct - crack_norm_pct
            rankings.append({
                "symbol": sym,
                "norm_pct": round(norm_pct, 2),
                "deviation_from_crack": round(deviation, 2),
                "abs_deviation": round(abs(deviation), 2),
            })
        except Exception:
            continue

    rankings.sort(key=lambda x: x["abs_deviation"], reverse=True)
    return rankings


def choose_refiner_from_rankings(peer_rankings: List[Dict[str, Any]], side: str) -> str:
    valid = [r for r in peer_rankings if r.get("peer_expensiveness_score") is not None]
    if not valid:
        return peer_rankings[0]["symbol"] if peer_rankings else "VLO"
    if side == "LONG":
        # Cheapest peer based on combined 3y weekly + 1y daily performance
        valid.sort(key=lambda x: x["peer_expensiveness_score"])
    else:
        # Most expensive peer based on combined 3y weekly + 1y daily performance
        valid.sort(key=lambda x: x["peer_expensiveness_score"], reverse=True)
    return valid[0]["symbol"]


def build_candidate(
    zscore: float,
    sigma_state: str,
    em_state: str,
    vol_up: bool,
    fit_score: float,
    selected_refiner: str,
    refiner_deviation_rankings: List[Dict[str, Any]],
    refiner_peer_rankings: List[Dict[str, Any]],
) -> Dict[str, Any]:
    bullish_refiners = zscore < 0
    futures_recommendation = "BUY HO=F + RB=F / SELL CL=F" if bullish_refiners else "SELL HO=F + RB=F / BUY CL=F"
    refiner_side = "LONG" if bullish_refiners else "SHORT"
    options_structure = "CALL_DEBIT_SPREAD" if bullish_refiners else "PUT_DEBIT_SPREAD"

    return {
        "candidate_id": f"CRACK_{datetime.now().strftime('%Y%m%d%H%M')}",
        "strategy_type": "crack_spread",
        "direction": "RELATIVE_VALUE",
        "horizon": "SHORT_TERM",
        "structure_family": "crack_spread_options_overlay",
        "summary": (
            f"Crack spread at {zscore:.2f}σ normalized | sigma={sigma_state} | EM={em_state} | "
            f"proxy={selected_refiner} {refiner_side}"
        ),
        "confidence": round(min(0.95, fit_score + 0.15), 2),
        "fit_score": round(fit_score, 2),
        "thesis_tags": [
            "crack_spread", "relative_value", "sigma_reentry", "expected_move",
            "peer_ranking", "3y_weekly", "1y_daily"
        ],
        "risk_flags": [
            "spread_widening",
            "vol_up_confirmed" if vol_up else "vol_neutral",
            "proxy_selected_by_peer_ranking",
        ],
        "implementation": {
            "futures_recommendation": futures_recommendation,
            "current_zscore": round(float(zscore), 4),
            "refiner_side": refiner_side,
            "selected_refiner": selected_refiner,
            "vol_packet_summary": {
                "sigma_state": sigma_state,
                "em_state": em_state,
                "vol_up": bool(vol_up),
            },
            "equity_execution": f"{refiner_side} {selected_refiner} (selected by 3y weekly + 1y daily peer ranking)",
            "options_idea": f"{options_structure} on {selected_refiner} anchored to crack EM",
            "proxy_deduplicated": True,
            "refiner_rankings_by_crack_deviation": refiner_deviation_rankings,
            "refiner_rankings_by_peer_performance": refiner_peer_rankings,
        },
    }


def evaluate_crack_with_options_context(config: RuntimeConfig, refiners_list: List[str]) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    diagnostics: Dict[str, Any] = {
        "current_crack_spread": None,
        "normalized_crack_pct": None,
        "sigma_state": None,
        "expected_move_state": None,
        "vol_up": None,
        "normalized_crack_zscore": None,
        "refiner_rankings_by_crack_deviation": [],
        "refiner_rankings_by_peer_performance": [],
        "selected_refiner": None,
        "selected_refiner_side": None,
        "filter_reason": None,
    }

    try:
        df_cl = fetch_history("CL=F", period=config.daily_period, interval=config.daily_interval)
        df_rb = fetch_history("RB=F", period=config.daily_period, interval=config.daily_interval)
        df_ho = fetch_history("HO=F", period=config.daily_period, interval=config.daily_interval)

        df_cl, df_rb, df_ho = align_frames(df_cl, df_rb, df_ho)

        if len(df_cl) < config.zscore_window + 20:
            diagnostics["filter_reason"] = "INSUFFICIENT_OVERLAPPING_HISTORY"
            log("   ❌ Insufficient overlapping history", config.verbose)
            return None, diagnostics

        crack_df = compute_crack_spread_and_normalized(df_cl, df_rb, df_ho)
        current_crack = float(crack_df["crack_spread"].iloc[-1])
        norm_crack_pct = float(crack_df["norm_crack_pct"].iloc[-1])

        diagnostics["current_crack_spread"] = round(current_crack, 4)
        diagnostics["normalized_crack_pct"] = round(norm_crack_pct, 4)

        log(f"   • Current Crack Spread: ${current_crack:.2f}", config.verbose)
        log(f"   • Normalized Crack: {norm_crack_pct:.2f}%", config.verbose)

        crack_ohlc = build_synthetic_crack_ohlc(crack_df, sigma_length=config.sigma_length)
        if crack_ohlc.empty:
            diagnostics["filter_reason"] = "EMPTY_SYNTHETIC_CRACK_OHLC"
            return None, diagnostics

        iv_proxy = pd.Series([0.0] * len(crack_ohlc), index=crack_ohlc.index)
        vol_packet = build_options_analysis_packet(
            crack_ohlc,
            implied_volatility=iv_proxy,
            sigma_length=config.sigma_length,
            require_vol_up_for_sigma=config.require_vol_up,
        )

        sigma_state = vol_packet["sigma_reentry"]["state"]
        em_state = vol_packet["expected_move"]["state"]
        vol_up = bool(vol_packet["vol_filter"]["vol_up"])

        diagnostics["sigma_state"] = sigma_state
        diagnostics["expected_move_state"] = em_state
        diagnostics["vol_up"] = vol_up

        log(f"   • Sigma Reentry : {sigma_state}", config.verbose)
        log(f"   • Expected Move : {em_state}", config.verbose)
        log(f"   • Vol Up (HVIV) : {vol_up}", config.verbose)

        norm_crack = crack_df["norm_crack_pct"].astype(float)
        z_window = norm_crack.tail(config.zscore_window)
        zscore = float((norm_crack.iloc[-1] - z_window.mean()) / z_window.std(ddof=1))
        diagnostics["normalized_crack_zscore"] = round(zscore, 4)

        log(f"   • Normalized Crack Z-score: {zscore:.2f}", config.verbose)
        log("   📊 Ranking refiners by deviation from crack spread...", config.verbose)
        refiner_deviation_rankings = rank_refiners_by_deviation_from_crack(refiners_list, norm_crack_pct, config)
        diagnostics["refiner_rankings_by_crack_deviation"] = refiner_deviation_rankings

        if refiner_deviation_rankings and config.verbose:
            log("   Top 3 refiners by deviation:", True)
            for r in refiner_deviation_rankings[:3]:
                log(f"      {r['symbol']:6} | norm={r['norm_pct']:+6.2f}% | dev={r['deviation_from_crack']:+6.2f}%", True)

        log("   📈 Ranking refiners by peer-relative price performance (3y weekly + 1y daily)...", config.verbose)
        refiner_peer_rankings = rank_refiners_by_peer_performance(refiners_list, config)
        diagnostics["refiner_rankings_by_peer_performance"] = refiner_peer_rankings

        if config.verbose and refiner_peer_rankings:
            log("   Full refiner peer ranking:", True)
            for r in refiner_peer_rankings:
                wk = "n/a" if r.get("weekly_3y_return_pct") is None else f"{r['weekly_3y_return_pct']:+7.2f}%"
                dy = "n/a" if r.get("daily_1y_return_pct") is None else f"{r['daily_1y_return_pct']:+7.2f}%"
                sc = "n/a" if r.get("peer_expensiveness_score") is None else f"{r['peer_expensiveness_score']:+.3f}"
                log(
                    f"      {r['symbol']:6} | 3y wk={wk:>9} | 1y d={dy:>9} | "
                    f"peer_score={sc:>6} | rank={r['peer_rank']}",
                    True,
                )

        selected_refiner_side = "LONG" if zscore < 0 else "SHORT"
        selected_refiner = choose_refiner_from_rankings(refiner_peer_rankings, selected_refiner_side)
        diagnostics["selected_refiner"] = selected_refiner
        diagnostics["selected_refiner_side"] = selected_refiner_side

        log(
            f"   • Selected refiner by peer ranking: {selected_refiner} ({selected_refiner_side})",
            config.verbose,
        )

        if abs(zscore) < config.min_abs_z:
            diagnostics["filter_reason"] = f"LOW_ZSCORE_{abs(zscore):.4f}"
            log("   ❌ Filtered: |z-score| too small", config.verbose)
            return None, diagnostics

        fit_score = round(0.45 * (abs(zscore) / 3.0) + 0.35 * (1.0 if vol_up else 0.0) + 0.20, 2)
        diagnostics["fit_score"] = fit_score
        if fit_score < config.min_fit_score:
            diagnostics["filter_reason"] = f"LOW_FIT_SCORE_{fit_score:.4f}"
            log(f"   ❌ Filtered: fit_score {fit_score:.2f} too low", config.verbose)
            return None, diagnostics

        candidate = build_candidate(
            zscore=zscore,
            sigma_state=sigma_state,
            em_state=em_state,
            vol_up=vol_up,
            fit_score=fit_score,
            selected_refiner=selected_refiner,
            refiner_deviation_rankings=refiner_deviation_rankings[:5],
            refiner_peer_rankings=refiner_peer_rankings,
        )

        log(
            f"   ✅ CRACK CANDIDATE ACCEPTED (fit_score={fit_score:.2f}, selected refiner={selected_refiner})\n",
            config.verbose,
        )
        return candidate, diagnostics

    except Exception as exc:
        diagnostics["filter_reason"] = f"EXCEPTION_{exc}"
        log(f"   ❌ Exception in crack analysis: {exc}", config.verbose)
        return None, diagnostics


def build_crack_spread_decision(config: RuntimeConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    _, refiners = load_futures_config()

    log("🚀 STARTING CRACK SPREAD ANALYSIS", config.verbose)

    candidate, diagnostics = evaluate_crack_with_options_context(config, refiners)
    candidates = [candidate] if candidate else []

    log("\n" + "=" * 90, config.verbose)
    log("📊 FINAL CRACK SPREAD CANDIDATE SUMMARY", config.verbose)
    log("=" * 90, config.verbose)
    for c in candidates:
        log(f"{c['candidate_id']:40} | fit={c['fit_score']:.2f} | conf={c['confidence']:.2f} | {c['summary'][:70]}...", config.verbose)
    log("=" * 90, config.verbose)

    output = {
        "agent": "crack_spread_analyst",
        "event_id": now_iso(),
        "strategy_type": "crack_spread",
        "candidates": candidates,
        "summary": (
            f"Crack Spread Analyst identified {len(candidates)} high-confidence crack-spread opportunities "
            f"using normalized crack, sigma reentry, expected-move filters, and peer-relative refiner ranking."
        ),
    }

    validate_output(output)

    report = {
        "output": output,
        "diagnostics": diagnostics,
        "config": {
            "daily_period": config.daily_period,
            "daily_interval": config.daily_interval,
            "zscore_window": config.zscore_window,
            "min_abs_z": config.min_abs_z,
            "min_fit_score": config.min_fit_score,
            "max_candidates": config.max_candidates,
            "require_vol_up": config.require_vol_up,
            "sigma_length": config.sigma_length,
            "proxy_weekly_period": config.proxy_weekly_period,
            "proxy_weekly_interval": config.proxy_weekly_interval,
            "proxy_daily_period": config.proxy_daily_period,
            "proxy_daily_interval": config.proxy_daily_interval,
        },
        "refiners": refiners,
    }
    return output, report


def maybe_add_llm_summary(output: Dict[str, Any], enabled: bool) -> Dict[str, Any]:
    if not enabled:
        return output
    try:
        from langchain_ollama import ChatOllama  # lazy import
        llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.2)
        prompt = (
            "You are a concise crack spread analyst narrator. Return ONLY valid JSON with one key named summary. "
            "Keep it to 2 sentences max and do not change the deterministic view.\n\n"
            f"{json.dumps({'candidates': output['candidates'], 'summary': output['summary']}, indent=2)}\n\n"
            '{"summary": "..."}'
        )
        response = llm.invoke(prompt)
        parsed = json.loads(response.content)
        if isinstance(parsed, dict) and isinstance(parsed.get("summary"), str):
            output["summary"] = parsed["summary"]
            validate_output(output)
    except Exception:
        pass
    return output


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Refactored Crack Spread Analyst v2")
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--use-llm-summary", action="store_true")
    parser.add_argument("--daily-period", type=str, default="2y")
    parser.add_argument("--daily-interval", type=str, default="1d")
    parser.add_argument("--zscore-window", type=int, default=63)
    parser.add_argument("--min-abs-z", type=float, default=1.0)
    parser.add_argument("--min-fit-score", type=float, default=0.55)
    parser.add_argument("--max-candidates", type=int, default=6)
    parser.add_argument("--require-vol-up", action="store_true", default=True)
    parser.add_argument("--sigma-length", type=int, default=21)
    parser.add_argument("--proxy-weekly-period", type=str, default="3y")
    parser.add_argument("--proxy-weekly-interval", type=str, default="1wk")
    parser.add_argument("--proxy-daily-period", type=str, default="1y")
    parser.add_argument("--proxy-daily-interval", type=str, default="1d")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    config = RuntimeConfig(
        daily_period=args.daily_period,
        daily_interval=args.daily_interval,
        zscore_window=args.zscore_window,
        min_abs_z=args.min_abs_z,
        min_fit_score=args.min_fit_score,
        max_candidates=args.max_candidates,
        require_vol_up=args.require_vol_up,
        sigma_length=args.sigma_length,
        verbose=args.verbose,
        proxy_weekly_period=args.proxy_weekly_period,
        proxy_weekly_interval=args.proxy_weekly_interval,
        proxy_daily_period=args.proxy_daily_period,
        proxy_daily_interval=args.proxy_daily_interval,
    )

    output, report = build_crack_spread_decision(config)
    output = maybe_add_llm_summary(output, args.use_llm_summary)

    log("\n📤 FINAL COORDINATOR PAYLOAD (clean JSON):", config.verbose)
    print(json.dumps(output, indent=2))

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = REPORTS_DIR / f"crack_spread_report_refactored_v2_{ts}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    log(f"\n✅ Analysis complete! Full report saved to {report_path}", config.verbose)
    log(f"   {len(output['candidates'])} candidate(s) ready for Trade Coordinator.", config.verbose)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
