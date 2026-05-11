#!/usr/bin/env python3
"""
agents/conscience.py
Step 6 - Conscience / Governance Agent
Final ethical, systemic risk, and capital-preservation veto layer.
Never overridden except by explicit CEO escalation.
"""

import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any

# Robust root detection
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.schemas import ConscienceOutput
from services.validation import validate_conscience_output


class ConscienceAgent:
    """High-bar veto agent — protects capital preservation mandate."""

    @staticmethod
    def review_decision(portfolio_decision: Dict[str, Any]) -> ConscienceOutput:
        veto_reasons = []
        approved = True

        # Hard capital preservation rules
        if portfolio_decision.get("tactical_exposure", 0.0) > 0.20:
            veto_reasons.append("Tactical exposure exceeds 20% hard limit")
            approved = False

        if len(portfolio_decision.get("veto_reasons", [])) > 0:
            veto_reasons.extend(portfolio_decision["veto_reasons"])
            approved = False

        # High-conviction concentration check (example)
        tactical_allocations = [
            allocation
            for allocation in portfolio_decision.get("approved_allocations", [])
            if allocation.get("action") != "PROFIT_TRANSFER"
        ]
        if len(tactical_allocations) > 3:
            veto_reasons.append("Too many simultaneous tactical positions")
            approved = False

        status = "APPROVED" if approved else "VETOED"

        output = ConscienceOutput(
            agent="conscience",
            event_id=f"conscience_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
            review_status=status,
            veto_reasons=veto_reasons,
            approved=approved,
            summary=f"Conscience review: {status}. {'No issues detected.' if approved else 'Veto triggered.'}",
            metadata={
                "schema_version": "1.0",
                "validated_at": datetime.now(timezone.utc).isoformat()
            }
        )

        validated = validate_conscience_output(output)
        return validated


if __name__ == "__main__":
    # Demo — assumes portfolio decision exists
    print("Conscience Agent ready for review.")
    # In production this receives output from Portfolio & Risk Manager
