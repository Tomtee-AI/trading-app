#!/usr/bin/env python3
"""
workflows/daily_pipeline.py
COMPLETE governed daily loop for ZeroHumanCompany
"""

import sys
from pathlib import Path
from datetime import datetime, timezone
import json

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.market_analyst import build_market_analyst_decision
from agents.divergence_analyst import build_divergence_analyst_decision
from agents.crack_spread_analyst import build_crack_spread_decision
from agents.unusual_options_analyst import build_unusual_options_decision
from agents.trade_coordinator import build_trade_coordinator_decision
from agents.portfolio_risk_manager import PortfolioRiskManager
from agents.conscience import ConscienceAgent
from agents.ceo import CEOAgent
from services.execution_service import ExecutionService


def run_daily_pipeline():
    print("🚀 ZERO HUMAN COMPANY DAILY PIPELINE STARTED\n")

    # === CORE INTELLIGENCE LAYER ===
    market = build_market_analyst_decision()

    # === STRATEGY SPECIALISTS (parallel) ===
    divergence = build_divergence_analyst_decision()
    crack_spread = build_crack_spread_decision()
    uoa = build_unusual_options_decision()

    # === COORDINATION & GOVERNANCE LAYERS ===
    coordinator = build_trade_coordinator_decision(
        market, [divergence, crack_spread, uoa]
    )
    portfolio = PortfolioRiskManager.build_portfolio_decision(coordinator)
    conscience = ConscienceAgent.review_decision(portfolio.model_dump())
    ceo = CEOAgent.synthesize(
        market.model_dump(),
        coordinator.model_dump(),
        portfolio.model_dump(),
        conscience.model_dump()
    )

    # === DETERMINISTIC EXECUTION (only layer allowed to place trades) ===
    if ceo.strategic_directive == "CONTINUE_TACTICAL_ALPHA_GENERATION" and conscience.approved:
        execution = ExecutionService.execute(portfolio.model_dump(), ceo.model_dump())
    else:
        execution = {"status": "PAUSED", "reason": "CEO or Conscience veto"}

    # === HUMAN-READABLE SUMMARY ===
    print("\n" + "="*90)
    print("🎯 FULL PIPELINE COMPLETE")
    print("="*90)
    print(f"Market Regime : {market.short_term['regime']} (conf {market.short_term['confidence']})")
    print(f"Divergence     : {len(divergence.get('candidates', []))} candidates")
    print(f"Crack Spread   : {len(crack_spread.get('candidates', []))} candidates")
    print(f"Unusual Options: {len(uoa.get('candidates', []))} candidates")
    print(f"Coordinator    : {len([c for c in coordinator.candidate_queue if c.get('status') == 'FORWARD'])} forwarded")
    print(f"Portfolio      : Tactical {portfolio.tactical_exposure:.1%} | Sweep ${portfolio.profit_transfer_amount:,.0f}")
    print(f"Conscience     : {conscience.review_status}")
    print(f"CEO            : {ceo.strategic_directive}")
    print(f"Execution      : {execution.get('status')}")
    print("="*90)

    # Save full pipeline (audit-ready)
    with open(ROOT / "reports" / f"full_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
        json.dump({
            "pipeline_event_id": f"pipeline_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
            "market": market.model_dump(),
            "divergence": divergence,
            "crack_spread": crack_spread,
            "uoa": uoa,
            "coordinator": coordinator.model_dump(),
            "portfolio": portfolio.model_dump(),
            "conscience": conscience.model_dump(),
            "ceo": ceo.model_dump(),
            "execution": execution
        }, f, indent=2)

    print("Full pipeline report saved to reports/")


if __name__ == "__main__":
    run_daily_pipeline()