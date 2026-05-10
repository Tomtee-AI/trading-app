#!/usr/bin/env python3
"""
agents/ceo.py
Final strategic orchestrator — maintains coherence between tactical leverage and long-term retirement mandate.
"""

import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.schemas import CEOOutput
from services.validation import validate_ceo_output


class CEOAgent:
    """Highest-level strategic synthesizer."""

    @staticmethod
    def synthesize(
        market: Dict[str, Any],
        coordinator: Dict[str, Any],
        portfolio: Dict[str, Any],
        conscience: Dict[str, Any]
    ) -> CEOOutput:
        strategic_directive = "CONTINUE_TACTICAL_ALPHA_GENERATION" if conscience.get("approved", True) else "PAUSE_ALL_TACTICAL"

        output = CEOOutput(
            agent="ceo",
            event_id=f"ceo_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
            strategic_directive=strategic_directive,
            summary=f"CEO directive: {strategic_directive}. Market regime is {market.get('short_term', {}).get('regime')}. "
                    f"Conscience status: {conscience.get('review_status')}.",
            metadata={
                "schema_version": "1.0",
                "validated_at": datetime.now(timezone.utc).isoformat(),
                "tactical_vs_retirement_alignment": "STRONG" if portfolio.get("profit_transfer_amount", 0) > 0 else "NEUTRAL"
            }
        )

        validated = validate_ceo_output(output)
        return validated


if __name__ == "__main__":
    print("CEO Agent ready for strategic synthesis.")