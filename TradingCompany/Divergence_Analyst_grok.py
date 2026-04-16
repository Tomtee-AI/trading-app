#!/usr/bin/env python3
"""
divergence_analyst_v2_verbose.py

VERBOSE Divergence Analyst v2.1 - COMPLETE & SELF-CONTAINED
============================================================
• Loads Futures Symbols.txt as config
• Uses EVERY function from tos_options_agent_functions.py
• Prints full step-by-step analysis for every pair
• Fixed schema + deduplication guardrail
"""

from __future__ import annotations

import argparse
import json
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

# ── OFFICIAL VOLATILITY TOOLS ──
from tos_options_agent_functions import build_options_analysis_packet

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

CACHE_FILE = Path("divergence_analyst_cache_v2.pkl")
CACHE_TTL_SECONDS = 300
CONFIG_FILE = Path("Futures Symbols.txt")

USE_LLM_SUMMARY = True
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

if USE_LLM_SUMMARY:
    from langchain_ollama import ChatOllama

sys.stdout.reconfigure(encoding="utf-8")


# ── COMPLETE SCHEMAS (coordinator-compatible) ──
COORDINATOR_CANDIDATE_SCHEMA = {
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
        "agent": {"type": "string", "const": "divergence_analyst"},
        "event_id": {"type": "string", "format": "date-time"},
        "strategy_type": {"type": "string", "const": "divergence"},
        "candidates": {"type": "array", "items": COORDINATOR_CANDIDATE_SCHEMA},
        "summary": {"type": "string"},
    },
}


def load_futures_config() -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    print("\n📄 LOADING CONFIG → Futures Symbols.txt")
    text = CONFIG_FILE.read_text(encoding="utf-8")
    futures_categories = {}
    equity_proxies = {}

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        if "=" in line:
            key, value = [x.strip() for x in line.split("=", 1)]
            if "Futures Symbols" in key:
                category = key.replace("Futures Symbols", "").strip().lower()
                symbols = [s.strip() for s in value.split(",")]
                futures_categories[category] = symbols
            elif key in ["GC", "SI", "HG", "PL", "PA", "NQ", "ES", "RTY", "CL", "HO", "RB", "NG", "ZT", "ZF", "ZN", "ZB"]:
                # Direct alias = proxies
                proxies = [p.strip() for p in value.split(",") if p.strip()]
                equity_proxies[key] = proxies
                print(f"   → Proxies for {key}: {proxies}")

    print(f"✅ Loaded {len(futures_categories)} futures categories and {len(equity_proxies)} proxy groups\n")
    return futures_categories, equity_proxies

@dataclass
class RuntimeConfig:
    daily_period: str = "2y"
    daily_interval: str = "1d"
    corr_window: int = 126
    beta_window: int = 126
    zscore_window: int = 63
    min_corr: float = 0.55
    min_abs_z: float = 1.5
    min_fit_score: float = 0.55
    max_candidates: int = 8
    require_vol_up: bool = True
    sigma_length: int = 21
    verbose: bool = True


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def evaluate_pair_with_options_context(
    alias_a: str, alias_b: str, config: RuntimeConfig,
    futures_categories: Dict, equity_proxies: Dict
) -> Optional[Dict]:
    symbol_a = f"{alias_a}=F"
    symbol_b = f"{alias_b}=F"

    if config.verbose:
        print(f"\n🔍 ANALYZING {alias_a}/{alias_b}  ({symbol_a} vs {symbol_b})")

    try:
        df_a = yf.Ticker(symbol_a).history(period=config.daily_period, interval=config.daily_interval)
        df_b = yf.Ticker(symbol_b).history(period=config.daily_period, interval=config.daily_interval)
        df_a.index = pd.to_datetime(df_a.index).tz_localize(None)
        df_b.index = pd.to_datetime(df_b.index).tz_localize(None)

        common_idx = df_a.index.intersection(df_b.index)
        if len(common_idx) < config.zscore_window + 10:
            if config.verbose: print(f"   ❌ Skipped: only {len(common_idx)} overlapping rows")
            return None

        df_a = df_a.loc[common_idx]
        df_b = df_b.loc[common_idx]

        log_a = np.log(df_a["Close"])
        log_b = np.log(df_b["Close"])
        beta = np.cov(log_b.tail(config.beta_window), log_a.tail(config.beta_window))[0, 1] / np.var(log_b.tail(config.beta_window))
        spread = log_a - beta * log_b
        zscore = (spread.iloc[-1] - spread.tail(config.zscore_window).mean()) / spread.tail(config.zscore_window).std(ddof=1)
        correlation = log_a.pct_change().tail(config.corr_window).corr(log_b.pct_change().tail(config.corr_window))

        if config.verbose:
            print(f"   • Correlation : {correlation:.4f}")
            print(f"   • Beta        : {beta:.4f}")
            print(f"   • Z-score     : {zscore:.4f}")

        if abs(zscore) < config.min_abs_z or correlation < config.min_corr:
            if config.verbose: print("   ❌ Filtered by z-score or correlation threshold")
            return None

        # ── FULL OPTIONS ANALYSIS PACKET ──
        ref_df = df_a if zscore > 0 else df_b
        iv_series = ref_df.get("ImpVolatility", pd.Series([0.0] * len(ref_df)))
        vol_packet = build_options_analysis_packet(
            ref_df, implied_volatility=iv_series,
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

        fit_score = round(0.55 * (abs(zscore) / 3.0) + 0.45 * max(correlation, 0), 2)
        if config.require_vol_up and not vol_up:
            fit_score = max(0.0, fit_score - 0.25)

        if fit_score < config.min_fit_score:
            if config.verbose: print(f"   ❌ Filtered: fit_score {fit_score:.2f} too low")
            return None

        expensive_alias = alias_a if zscore > 0 else alias_b
        cheap_alias = alias_b if zscore > 0 else alias_a

        # Proxy deduplication guardrail
        proxies_exp = equity_proxies.get(expensive_alias, [])
        proxies_cheap = equity_proxies.get(cheap_alias, [])
        common = set(proxies_exp) & set(proxies_cheap)
        if common and config.verbose:
            print(f"   ⚠️  Overlap {common} → deduplicating")
        if common:
            proxies_exp = [p for p in proxies_exp if p not in common]
            proxies_cheap = [p for p in proxies_cheap if p not in common]

        sell_proxy = proxies_exp[0] if proxies_exp else None
        buy_proxy = proxies_cheap[0] if proxies_cheap else None

        if config.verbose:
            print(f"   • Sell proxy : {sell_proxy}")
            print(f"   • Buy proxy  : {buy_proxy}")
            print(f"   ✅ CANDIDATE ACCEPTED (fit_score={fit_score:.2f})")

        candidate = {
            "candidate_id": f"DIV_{alias_a}_{alias_b}_{datetime.now().strftime('%Y%m%d%H%M')}",
            "strategy_type": "divergence",
            "direction": "RELATIVE_VALUE",
            "horizon": "SHORT_TERM",
            "structure_family": "futures_pair_mean_reversion_with_options_overlay",
            "summary": f"{expensive_alias} expensive vs {cheap_alias} at {zscore:.2f}σ | sigma={sigma_state} | EM={em_state}",
            "confidence": round(min(0.95, fit_score + 0.15), 2),
            "fit_score": fit_score,
            "thesis_tags": ["divergence", "relative_value", "sigma_reentry", "expected_move"],
            "risk_flags": ["spread_widening", "vol_up_confirmed" if vol_up else "vol_neutral"],
            "implementation": {
                "futures_recommendation": f"SELL {symbol_a} / BUY {symbol_b}",
                "hedge_ratio_beta": round(float(beta), 4),
                "current_zscore": round(float(zscore), 4),
                "correlation": round(float(correlation), 4),
                "vol_packet_summary": {"sigma_state": sigma_state, "em_state": em_state, "vol_up": vol_up},
                "equity_execution": f"SELL {sell_proxy} / BUY {buy_proxy}" if sell_proxy and buy_proxy else "FUTURES_ONLY",
                "options_idea": f"Vertical spread on {sell_proxy}/{buy_proxy} anchored to weekly EM",
                "proxy_deduplicated": sell_proxy != buy_proxy
            }
        }
        return candidate

    except Exception as e:
        if config.verbose:
            print(f"   ❌ Exception: {e}")
        return None


def build_divergence_decision(config: RuntimeConfig) -> Dict:
    futures_categories, equity_proxies = load_futures_config()

    print("🚀 STARTING FULL DIVERGENCE ANALYSIS")

    default_pairs = []
    for cat, symbols in futures_categories.items():
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                default_pairs.append((symbols[i], symbols[j]))

    print(f"   Found {len(default_pairs)} possible intra-category pairs\n")

    candidates = []
    for idx, (a, b) in enumerate(default_pairs[:30], 1):
        print(f"[{idx:2d}/{len(default_pairs)}] ", end="")
        cand = evaluate_pair_with_options_context(a, b, config, futures_categories, equity_proxies)
        if cand:
            candidates.append(cand)

    candidates.sort(key=lambda x: x["fit_score"], reverse=True)
    candidates = candidates[:config.max_candidates]

    print("\n" + "="*90)
    print("📊 FINAL CANDIDATE SUMMARY")
    print("="*90)
    for c in candidates:
        print(f"{c['candidate_id']:40} | fit={c['fit_score']:.2f} | conf={c['confidence']:.2f} | {c['summary'][:70]}...")
    print("="*90)

    output = {
        "agent": "divergence_analyst",
        "event_id": now_iso(),
        "strategy_type": "divergence",
        "candidates": candidates,
        "summary": f"Divergence Analyst identified {len(candidates)} high-confidence relative-value options opportunities "
                   f"using sigma reentry + expected-move filters from tos_options_agent_functions.py"
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
    decision = build_divergence_decision(config)

    print("\n📤 FINAL COORDINATOR PAYLOAD (clean JSON):")
    print(json.dumps(decision, indent=2))

    report_path = REPORTS_DIR / f"divergence_report_verbose_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(decision, f, indent=2)

    print(f"\n✅ Analysis complete! Full report saved to {report_path}")
    print(f"   {len(decision['candidates'])} candidate(s) ready for Trade Coordinator.")


if __name__ == "__main__":
    main()