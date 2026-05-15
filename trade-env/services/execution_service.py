#!/usr/bin/env python3
"""
services/execution_service.py
DETERMINISTIC execution layer — the ONLY place where actual trades can be placed.
All agents are advisory only.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)


class ExecutionService:
    """Deterministic, auditable, idempotent execution layer."""

    @staticmethod
    def execute(portfolio_decision: Dict[str, Any], ceo_directive: Dict[str, Any]) -> Dict[str, Any]:
        event_id = f"exec_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        timestamp = datetime.now(timezone.utc).isoformat()

        if not ceo_directive.get("strategic_directive") == "CONTINUE_TACTICAL_ALPHA_GENERATION":
            return {
                "execution_event_id": event_id,
                "status": "PAUSED_BY_CEO",
                "reason": "Strategic pause issued",
                "approved_allocations": [],
                "profit_transfer": 0.0,
                "timestamp": timestamp,
                "execution_note": "CEO strategic directive paused tactical execution.",
            }

        if not portfolio_decision.get("approved", True):
            return {
                "execution_event_id": event_id,
                "status": "VETOED",
                "reason": "Portfolio or Conscience veto",
                "approved_allocations": portfolio_decision.get("approved_allocations", []),
                "profit_transfer": portfolio_decision.get("profit_transfer_amount", 0.0),
                "timestamp": timestamp,
                "execution_note": "Portfolio or conscience veto prevented execution.",
            }

        approved_allocations = portfolio_decision.get("approved_allocations", [])
        profit_transfer = portfolio_decision.get("profit_transfer_amount", 0.0)
        if not approved_allocations and not profit_transfer:
            return {
                "execution_event_id": event_id,
                "status": "NO_ACTION",
                "reason": "No approved allocations or profit transfer to execute",
                "approved_allocations": [],
                "profit_transfer": 0.0,
                "timestamp": timestamp,
                "execution_note": "Run completed without deterministic side effects.",
            }

        research_only = (
            approved_allocations
            and not profit_transfer
            and all(
                allocation.get("action") == "REVIEW_TRADE_CANDIDATE"
                for allocation in approved_allocations
            )
        )
        status = "RESEARCH_RECORDED" if research_only else "EXECUTED"
        execution_note = (
            "Research-only candidate approvals were recorded; no broker order was placed."
            if research_only
            else "All agents were advisory only. Deterministic execution recorded approved actions."
        )

        execution_report = {
            "execution_event_id": event_id,
            "status": status,
            "approved_allocations": approved_allocations,
            "profit_transfer": profit_transfer,
            "timestamp": timestamp,
            "execution_note": execution_note
        }

        # Save audit log
        with open(REPORTS_DIR / f"execution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
            json.dump(execution_report, f, indent=2)

        print(f"{status}: {len(execution_report['approved_allocations'])} allocations | Profit sweep: ${execution_report['profit_transfer']:,.0f}")
        return execution_report
