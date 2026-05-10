#!/usr/bin/env python3
"""
agents/portfolio_risk_manager.py
Step 4 - Enforces your exact business objective:
Tactical Short-Term Options Book (leverage) funds Long-Term Retirement Portfolio
"""

import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any

# Robust root detection
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.schemas import PortfolioDecision, TradeCoordinatorOutput
from services.validation import validate_portfolio_decision


class PortfolioRiskManager:
    TACTICAL_MAX_EXPOSURE_PCT = 0.15   # 15% of total capital max for leverage book
    MIN_PROFIT_SWEEP_PCT = 0.60        # Sweep 60% of tactical profits to retirement book

    @staticmethod
    def build_portfolio_decision(
        coordinator: TradeCoordinatorOutput,
        current_tactical_exposure: float = 0.0,
        current_retirement_balance: float = 100_000.0,
        total_capital: float = 500_000.0
    ) -> PortfolioDecision:
        approved = []
        veto_reasons = []
        profit_transfer = 0.0

        if coordinator.no_trade:
            veto_reasons.append("Coordinator issued no_trade decision")
        else:
            # Enforce tactical book size limit
            proposed_tactical = min(len(coordinator.candidate_queue) * 0.03, 
                                  PortfolioRiskManager.TACTICAL_MAX_EXPOSURE_PCT)
            
            if current_tactical_exposure + proposed_tactical > PortfolioRiskManager.TACTICAL_MAX_EXPOSURE_PCT:
                veto_reasons.append(f"Tactical exposure would exceed {PortfolioRiskManager.TACTICAL_MAX_EXPOSURE_PCT*100}% limit")

            # Profit sweep rule (core business objective)
            if current_tactical_exposure > 0:
                profit_transfer = current_tactical_exposure * PortfolioRiskManager.MIN_PROFIT_SWEEP_PCT
                approved.append({
                    "action": "PROFIT_TRANSFER",
                    "amount": round(profit_transfer, 2),
                    "from_book": "tactical_short_term_options",
                    "to_book": "long_term_retirement"
                })

        decision = PortfolioDecision(
            agent="portfolio_risk_manager",
            event_id=f"portfolio_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
            approved_allocations=approved,
            veto_reasons=veto_reasons,
            profit_transfer_amount=round(profit_transfer, 2),
            tactical_exposure=round(current_tactical_exposure, 4),
            retirement_exposure=round(current_retirement_balance + profit_transfer, 2),
            summary="Portfolio & Risk Manager decision issued."
        )

        validated = validate_portfolio_decision(decision)
        return validated


if __name__ == "__main__":
    # Demo chain: Market Analyst → Coordinator → Portfolio Manager
    from agents.market_analyst import build_market_analyst_decision
    from agents.trade_coordinator import build_trade_coordinator_decision

    market = build_market_analyst_decision()
    coordinator = build_trade_coordinator_decision(market)
    portfolio = PortfolioRiskManager.build_portfolio_decision(coordinator)

    print(portfolio.model_dump_json(indent=2))
    print(f"\n✅ Portfolio & Risk Manager completed - Event ID: {portfolio.event_id}")