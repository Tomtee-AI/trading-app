#!/usr/bin/env python3
"""
crack_spread_analyst_verbose.py

VERBOSE Crack Spread Analyst v1.4
==============================================================
• Loads Futures Symbols.txt
• Exact port of Crack Tracker script
• Full tos_options_agent_functions.py integration
• Smarter refiner ranking (always shown in verbose mode)
• Safety-first z-score filter (configurable)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from jsonschema import Draft202012Validator, FormatChecker

from tos_options_agent_functions import build_options_analysis_packet

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_FILE = Path("Futures Symbols.txt")

USE_LLM_SUMMARY = True
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

if USE_LLM_SUMMARY:
    from langchain_ollama import ChatOllama

sys.stdout.reconfigure(encoding="utf-8")


# ── SCHEMAS (coordinator-compatible) ──
COORDINATOR_CANDIDATE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["candidate_id", "strategy_type", "direction", "horizon", "structure_family",
                 "summary", "confidence", "fit_score", "thesis_tags", "risk_flags", "implementation"],
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
    }
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


def load_futures_config() -> Tuple[Dict[str, List[str]], List[str]]:
    print("\n📄 LOADING CONFIG → Futures Symbols.txt")
    text = CONFIG_FILE.read_text(encoding="utf-8")
    energies_futures = []
    refiners = []

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "Energies Futures Symbols" in line:
            energies_futures = [s.strip() for s in line.split("=")[1].split(",")]
        elif line.startswith("Refiners ="):
            refiners = [s.strip() for s in line.split("=")[1].split(",")]

    print(f"   → Energies futures: {energies_futures}")
    print(f"   → Refiners: {refiners}")
    return {"energies": energies_futures}, refiners


@dataclass
class RuntimeConfig:
    daily_period: str = "2y"
    daily_interval: str = "1d"
    zscore_window: int = 63
    min_abs_z: float = 1.0          # ← Lowered for more realistic crack signals (was 1.2)
    min_fit_score: float = 0.55
    max_candidates: int = 6
    require_vol_up: bool = True
    sigma_length: int = 21
    verbose: bool = True


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def compute_crack_spread_and_normalized(df_cl: pd.DataFrame, df_rb: pd.DataFrame, df_ho: pd.DataFrame) -> pd.DataFrame:
    cl = df_cl["Close"]
    rb = df_rb["Close"]
    ho = df_ho["Close"]

    rb_per_barrel = rb * 42
    ho_per_barrel = ho * 42
    crack_spread = ((2/3) * rb_per_barrel + (1/3) * ho_per_barrel) - cl

    base_crack = ((2/3) * rb.iloc[0] * 42 + (1/3) * ho.iloc[0] * 42) - cl.iloc[0]
    norm_crack = (crack_spread / base_crack * 100) - 100

    return pd.DataFrame({
        "crack_spread": crack_spread,
        "norm_crack_pct": norm_crack
    }, index=cl.index)


def rank_refiners_by_deviation_from_crack(
    refiners_list: List[str],
    crack_norm_pct: float,
    config: RuntimeConfig
) -> List[Dict]:
    """Smarter refiner ranking by deviation from crack spread"""
    rankings = []
    for sym in refiners_list:
        try:
            df = yf.Ticker(sym).history(period=config.daily_period, interval=config.daily_interval)
            if df.empty or len(df) < 5:
                continue
            # Force tz-naive
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            close = df["Close"]
            base = close.iloc[0]
            norm_pct = (close.iloc[-1] / base * 100) - 100
            deviation = norm_pct - crack_norm_pct
            rankings.append({
                "symbol": sym,
                "norm_pct": round(norm_pct, 2),
                "deviation_from_crack": round(deviation, 2),
                "abs_deviation": abs(deviation)
            })
        except Exception:
            continue

    rankings.sort(key=lambda x: x["abs_deviation"], reverse=True)
    return rankings


def evaluate_crack_with_options_context(
    config: RuntimeConfig,
    refiners_list: List[str]
) -> Optional[Dict]:
    try:
        df_cl = yf.Ticker("CL=F").history(period=config.daily_period, interval=config.daily_interval)
        df_rb = yf.Ticker("RB=F").history(period=config.daily_period, interval=config.daily_interval)
        df_ho = yf.Ticker("HO=F").history(period=config.daily_period, interval=config.daily_interval)

        for df in [df_cl, df_rb, df_ho]:
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)

        common_idx = df_cl.index.intersection(df_rb.index).intersection(df_ho.index)
        df_cl = df_cl.loc[common_idx]
        df_rb = df_rb.loc[common_idx]
        df_ho = df_ho.loc[common_idx]

        if len(common_idx) < config.zscore_window + 20:
            if config.verbose:
                print("   ❌ Insufficient overlapping history")
            return None

        crack_df = compute_crack_spread_and_normalized(df_cl, df_rb, df_ho)
        current_crack = crack_df['crack_spread'].iloc[-1]
        norm_crack_pct = crack_df['norm_crack_pct'].iloc[-1]

        if config.verbose:
            print(f"   • Current Crack Spread: ${current_crack:.2f}")
            print(f"   • Normalized Crack: {norm_crack_pct:.2f}%")

        # Synthetic OHLC
        close = crack_df["crack_spread"]
        daily_range = close.rolling(21).std() * 1.5
        df_crack_ohlc = pd.DataFrame({
            "Close": close,
            "High": close + daily_range,
            "Low": close - daily_range
        }, index=crack_df.index)

        iv_proxy = pd.Series([0.0] * len(df_crack_ohlc), index=df_crack_ohlc.index)

        vol_packet = build_options_analysis_packet(
            df_crack_ohlc,
            implied_volatility=iv_proxy,
            sigma_length=config.sigma_length,
            require_vol_up_for_sigma=config.require_vol_up
        )

        sigma_state = vol_packet["sigma_reentry"]["state"]
        em_state = vol_packet["expected_move"]["state"]
        vol_up = vol_packet["vol_filter"]["vol_up"]

        if config.verbose:
            print(f"   • Sigma Reentry : {sigma_state}")
            print(f"   • Expected Move : {em_state}")
            print(f"   • Vol Up (HVIV) : {vol_up}")

        # Z-score
        norm_crack = crack_df["norm_crack_pct"]
        zscore = (norm_crack.iloc[-1] - norm_crack.tail(config.zscore_window).mean()) / \
                 norm_crack.tail(config.zscore_window).std(ddof=1)

        if config.verbose:
            print(f"   • Normalized Crack Z-score: {zscore:.2f}")

        # ── ALWAYS SHOW REFINER RANKING (even if filtered) ──
        if config.verbose:
            print("   📊 Ranking refiners by deviation from crack spread...")
        refiner_rankings = rank_refiners_by_deviation_from_crack(refiners_list, norm_crack_pct, config)

        if refiner_rankings and config.verbose:
            print("   Top 3 refiners by deviation:")
            for r in refiner_rankings[:3]:
                print(f"      {r['symbol']:6} | norm={r['norm_pct']:+6.2f}% | dev={r['deviation_from_crack']:+6.2f}%")

        top_refiner = refiner_rankings[0]["symbol"] if refiner_rankings else "VLO"

        if abs(zscore) < config.min_abs_z:
            if config.verbose:
                print("   ❌ Filtered: |z-score| too small")
            return None   # still return None for candidate, but ranking was shown

        fit_score = round(0.55 * (abs(zscore) / 3.0) + 0.45 * (1 if vol_up else 0), 2)
        if fit_score < config.min_fit_score:
            if config.verbose:
                print(f"   ❌ Filtered: fit_score {fit_score:.2f} too low")
            return None

        candidate = {
            "candidate_id": f"CRACK_{datetime.now().strftime('%Y%m%d%H%M')}",
            "strategy_type": "crack_spread",
            "direction": "RELATIVE_VALUE",
            "horizon": "SHORT_TERM",
            "structure_family": "crack_spread_options_overlay",
            "summary": f"Crack spread at {zscore:.2f}σ normalized | sigma={sigma_state} | EM={em_state}",
            "confidence": round(min(0.95, fit_score + 0.15), 2),
            "fit_score": fit_score,
            "thesis_tags": ["crack_spread", "relative_value", "sigma_reentry", "expected_move"],
            "risk_flags": ["spread_widening", "vol_up_confirmed" if vol_up else "vol_neutral"],
            "implementation": {
                "futures_recommendation": "BUY HO=F + RB=F / SELL CL=F" if zscore < 0 else "SELL HO=F + RB=F / BUY CL=F",
                "current_zscore": round(float(zscore), 4),
                "vol_packet_summary": {"sigma_state": sigma_state, "em_state": em_state, "vol_up": vol_up},
                "equity_execution": f"Long {top_refiner} (refiner proxy)",
                "options_idea": f"Vertical spread on {top_refiner} anchored to crack EM",
                "proxy_deduplicated": True,
                "refiner_rankings": refiner_rankings[:3]
            }
        }

        if config.verbose:
            print(f"   ✅ CRACK CANDIDATE ACCEPTED (fit_score={fit_score:.2f}, top refiner={top_refiner})\n")
        return candidate

    except Exception as e:
        if config.verbose:
            print(f"   ❌ Exception in crack analysis: {e}")
        return None


def build_crack_spread_decision(config: RuntimeConfig) -> Dict:
    _, refiners = load_futures_config()

    print("🚀 STARTING CRACK SPREAD ANALYSIS")

    candidate = evaluate_crack_with_options_context(config, refiners)
    candidates = [candidate] if candidate else []

    print("\n" + "="*90)
    print("📊 FINAL CRACK SPREAD CANDIDATE SUMMARY")
    print("="*90)
    for c in candidates:
        print(f"{c['candidate_id']:40} | fit={c['fit_score']:.2f} | conf={c['confidence']:.2f} | {c['summary'][:70]}...")
    print("="*90)

    output = {
        "agent": "crack_spread_analyst",
        "event_id": now_iso(),
        "strategy_type": "crack_spread",
        "candidates": candidates,
        "summary": f"Crack Spread Analyst identified {len(candidates)} high-confidence crack-spread options opportunities "
                   f"using normalized crack + sigma reentry + expected-move filters."
    }

    validator = Draft202012Validator(OUTPUT_SCHEMA, format_checker=FormatChecker())
    if list(validator.iter_errors(output)):
        raise ValueError("Schema validation failed")

    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--use-llm-summary", action="store_true")
    args = parser.parse_args()

    config = RuntimeConfig(verbose=args.verbose)
    decision = build_crack_spread_decision(config)

    print("\n📤 FINAL COORDINATOR PAYLOAD (clean JSON):")
    print(json.dumps(decision, indent=2))

    report_path = REPORTS_DIR / f"crack_spread_report_verbose_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(decision, f, indent=2)

    print(f"\n✅ Analysis complete! Full report saved to {report_path}")
    print(f"   {len(decision['candidates'])} candidate(s) ready for Trade Coordinator.")


if __name__ == "__main__":
    main()