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


def _as_float(value, default=None):
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _round_or_none(value, digits=2):
    return round(value, digits) if value is not None else None


def _spread_economics(spread: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(spread, dict):
        return {
            "entry_debit": None,
            "max_loss": None,
            "max_reward": None,
            "reward_to_risk": None,
            "liquidity_pass": False,
            "premium_in_preferred_range": False,
            "symbol": None,
            "structure": None,
        }

    return {
        "entry_debit": _as_float(spread.get("estimated_entry_debit")),
        "max_loss": _as_float(spread.get("estimated_max_loss")),
        "max_reward": _as_float(spread.get("estimated_max_reward")),
        "reward_to_risk": _as_float(spread.get("estimated_reward_to_risk")),
        "liquidity_pass": bool(spread.get("liquidity_pass")),
        "premium_in_preferred_range": bool(spread.get("premium_in_preferred_range")),
        "symbol": spread.get("symbol"),
        "structure": spread.get("structure"),
        "expiration": spread.get("expiration"),
    }


def _candidate_economics(candidate: Dict[str, Any]) -> Dict[str, Any]:
    implementation = candidate.get("implementation", {}) or {}
    templates = implementation.get("trade_templates", {}) or {}

    spread_keys = [
        "short_leg_exact_spread",
        "long_leg_exact_spread",
    ]
    spreads = [
        _spread_economics(templates.get(key))
        for key in spread_keys
        if templates.get(key) is not None
    ]

    if not spreads:
        selected_trade = implementation.get("selected_trade", {}) or {}
        reward_risk = implementation.get("reward_risk", {}) or {}
        return {
            "legs": [],
            "total_max_loss": _as_float(selected_trade.get("maximum_loss_per_contract")),
            "total_max_reward": _as_float(reward_risk.get("estimated_reward")),
            "combined_reward_to_risk": _as_float(reward_risk.get("estimated_reward_to_risk")),
            "all_liquid": True,
            "all_premium_in_range": True,
        }

    total_loss = sum(spread.get("max_loss") or 0.0 for spread in spreads)
    total_reward = sum(spread.get("max_reward") or 0.0 for spread in spreads)
    combined_rr = total_reward / total_loss if total_loss > 0 and total_reward > 0 else None

    return {
        "legs": spreads,
        "total_max_loss": total_loss,
        "total_max_reward": total_reward,
        "combined_reward_to_risk": combined_rr,
        "all_liquid": all(spread.get("liquidity_pass") for spread in spreads),
        "all_premium_in_range": all(spread.get("premium_in_preferred_range") for spread in spreads),
    }


class PortfolioRiskManager:
    TACTICAL_MAX_EXPOSURE_PCT = 0.15   # 15% of total capital max for leverage book
    MIN_PROFIT_SWEEP_PCT = 0.60        # Sweep 60% of tactical profits to retirement book
    MAX_APPROVED_TRADE_CANDIDATES = 3
    MIN_REWARD_TO_RISK_FOR_APPROVAL = 3.0

    @staticmethod
    def build_portfolio_decision(
        coordinator: TradeCoordinatorOutput,
        current_tactical_exposure: float = 0.0,
        current_retirement_balance: float = 100_000.0,
        total_capital: float = 500_000.0
    ) -> PortfolioDecision:
        approved_allocations = []
        candidate_reviews = []
        veto_reasons = []
        profit_transfer = 0.0
        proposed_tactical = 0.0
        portfolio_approved = True

        if coordinator.no_trade:
            veto_reasons.append("Coordinator issued no_trade decision")
            portfolio_approved = False
        else:
            forwarded_candidates = [
                candidate
                for candidate in coordinator.candidate_queue
                if candidate.get("status") == "FORWARD"
            ]

            for candidate in forwarded_candidates:
                economics = _candidate_economics(candidate)
                rr = economics.get("combined_reward_to_risk")
                candidate_reasons = []

                if not economics.get("all_liquid"):
                    candidate_reasons.append("OPTION_LIQUIDITY_NOT_VERIFIED")
                if not economics.get("all_premium_in_range"):
                    candidate_reasons.append("PREMIUM_OUTSIDE_PREFERRED_RANGE")
                if economics.get("total_max_loss") in (None, 0):
                    candidate_reasons.append("MISSING_MAX_LOSS")
                if rr is None:
                    candidate_reasons.append("MISSING_REWARD_TO_RISK")
                elif rr < PortfolioRiskManager.MIN_REWARD_TO_RISK_FOR_APPROVAL:
                    candidate_reasons.append("REWARD_TO_RISK_BELOW_PORTFOLIO_MINIMUM")

                review_status = "APPROVED_FOR_REVIEW" if not candidate_reasons else "VETOED"
                review = {
                    "candidate_id": candidate.get("candidate_id"),
                    "strategy_type": candidate.get("strategy_type"),
                    "status": review_status,
                    "reason_codes": candidate_reasons or ["PASSED_PORTFOLIO_REVIEW"],
                    "ranking_score": candidate.get("ranking_score"),
                    "market_alignment_score": candidate.get("market_alignment_score"),
                    "estimated_max_loss_dollars": _round_or_none((economics.get("total_max_loss") or 0.0) * 100.0),
                    "estimated_max_reward_dollars": _round_or_none((economics.get("total_max_reward") or 0.0) * 100.0),
                    "estimated_reward_to_risk": _round_or_none(rr, 4),
                    "legs": economics.get("legs", []),
                }
                candidate_reviews.append(review)

            trade_reviews = [
                review
                for review in candidate_reviews
                if review["status"] == "APPROVED_FOR_REVIEW"
            ][:PortfolioRiskManager.MAX_APPROVED_TRADE_CANDIDATES]

            for review in trade_reviews:
                approved_allocations.append({
                    "action": "REVIEW_TRADE_CANDIDATE",
                    "candidate_id": review["candidate_id"],
                    "strategy_type": review["strategy_type"],
                    "estimated_max_loss_dollars": review["estimated_max_loss_dollars"],
                    "estimated_max_reward_dollars": review["estimated_max_reward_dollars"],
                    "estimated_reward_to_risk": review["estimated_reward_to_risk"],
                    "allocation_note": "Research approval only; no broker order is placed.",
                })

            proposed_tactical = min(
                len(approved_allocations) * 0.03,
                PortfolioRiskManager.TACTICAL_MAX_EXPOSURE_PCT
            )

            if current_tactical_exposure + proposed_tactical > PortfolioRiskManager.TACTICAL_MAX_EXPOSURE_PCT:
                veto_reasons.append(f"Tactical exposure would exceed {PortfolioRiskManager.TACTICAL_MAX_EXPOSURE_PCT*100}% limit")
                portfolio_approved = False

            # Profit sweep rule (core business objective)
            if current_tactical_exposure > 0:
                profit_transfer = current_tactical_exposure * PortfolioRiskManager.MIN_PROFIT_SWEEP_PCT
                approved_allocations.append({
                    "action": "PROFIT_TRANSFER",
                    "amount": round(profit_transfer, 2),
                    "from_book": "tactical_short_term_options",
                    "to_book": "long_term_retirement"
                })

        decision = PortfolioDecision(
            agent="portfolio_risk_manager",
            event_id=f"portfolio_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
            approved=portfolio_approved,
            approved_allocations=approved_allocations,
            candidate_reviews=candidate_reviews,
            veto_reasons=veto_reasons,
            profit_transfer_amount=round(profit_transfer, 2),
            tactical_exposure=round(current_tactical_exposure + proposed_tactical, 4),
            retirement_exposure=round(current_retirement_balance + profit_transfer, 2),
            summary=(
                f"Portfolio & Risk Manager reviewed {len(candidate_reviews)} forwarded candidates; "
                f"{len([r for r in candidate_reviews if r['status'] == 'APPROVED_FOR_REVIEW'])} passed candidate review."
            )
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
