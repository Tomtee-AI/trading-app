#!/usr/bin/env python3
"""
agents/trade_coordinator.py
Step 3 - Bias-aware Trade Coordinator (FIXED)
Consumes MarketAnalystOutput + specialist candidates
"""

import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any

# === ROBUST PROJECT ROOT DETECTION ===
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.schemas import TradeCoordinatorOutput, MarketAnalystOutput
from services.validation import validate_trade_coordinator_output

# Import your original skeleton logic + RuntimeConfig
from trade_coordinator_bias_aware_skeleton import (
    derive_coordinator_bias,
    build_strategy_permissions,
    build_ranking_weights,
    normalize_candidates,
    RuntimeConfig   # ← This was missing
)


def build_trade_coordinator_decision(
    market: MarketAnalystOutput,
    specialist_payloads: List[Dict[str, Any]] = None
) -> TradeCoordinatorOutput:
    if specialist_payloads is None:
        specialist_payloads = []

    # Create default config (your skeleton uses sensible defaults)
    config = RuntimeConfig(
        min_specialist_confidence=0.55,
        min_market_alignment=-0.10,
        max_queue_size=20
    )

    # Reuse your battle-tested logic with correct arguments
    candidate_queue, no_trade = normalize_candidates(
        market.model_dump(), 
        specialist_payloads,
        config
    )

    coordinator_bias = derive_coordinator_bias(market.model_dump())

    output = TradeCoordinatorOutput(
        agent="trade_coordinator",
        event_id=f"coord_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        market_bias=market,
        strategy_permissions=build_strategy_permissions(coordinator_bias),
        ranking_weights=build_ranking_weights(coordinator_bias),
        candidate_queue=candidate_queue,
        no_trade=no_trade,
        summary=f"Coordinator bias: {coordinator_bias}. "
                f"Forwarded {len([c for c in candidate_queue if c.get('status') == 'FORWARD'])} candidates."
    )

    validated = validate_trade_coordinator_output(output)
    return validated


if __name__ == "__main__":
    # Demo: Market Analyst → Trade Coordinator
    from agents.market_analyst import build_market_analyst_decision
    
    print("Running Market Analyst → Trade Coordinator pipeline...\n")
    market_output = build_market_analyst_decision()
    decision = build_trade_coordinator_decision(market_output)
    
    print(decision.model_dump_json(indent=2))
    print(f"\n✅ Trade Coordinator completed successfully")
    print(f"   Event ID: {decision.event_id}")
    print(f"   Coordinator Bias: {decision.market_bias.short_term.get('regime')}")