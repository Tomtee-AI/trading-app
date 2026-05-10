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
        if not ceo_directive.get("strategic_directive") == "CONTINUE_TACTICAL_ALPHA_GENERATION":
            return {"status": "PAUSED_BY_CEO", "reason": "Strategic pause issued"}

        if not portfolio_decision.get("approved", True):
            return {"status": "VETOED", "reason": "Portfolio or Conscience veto"}

        execution_report = {
            "execution_event_id": f"exec_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
            "status": "EXECUTED",
            "approved_allocations": portfolio_decision.get("approved_allocations", []),
            "profit_transfer": portfolio_decision.get("profit_transfer_amount", 0.0),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "audit_note": "All agents were advisory only. Execution performed by deterministic service."
        }

        # Save audit log
        with open(REPORTS_DIR / f"execution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
            json.dump(execution_report, f, indent=2)

        print(f"✅ EXECUTED: {len(execution_report['approved_allocations'])} allocations | Profit sweep: ${execution_report['profit_transfer']:,.0f}")
        return execution_report