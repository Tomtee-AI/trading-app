#!/usr/bin/env python3
"""
agents/market_analyst.py
Robust dual-horizon Market Analyst using your existing market_analyst_v2.py
"""

import sys
from pathlib import Path
from datetime import datetime, timezone

# === ROBUST PROJECT ROOT DETECTION ===
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Import your original strong deterministic engine from the root
try:
    from market_analyst_v2 import build_market_decision
except ImportError as e:
    print("❌ ERROR: Could not find market_analyst_dual_horizon.py")
    print("Make sure the file exists in the project root folder:")
    print(f"   {ROOT}")
    print("Current files in root:", list(ROOT.glob("*.py")))
    sys.exit(1)

from config.schemas import MarketAnalystOutput
from services.validation import validate_market_analyst_output


def build_market_analyst_decision() -> MarketAnalystOutput:
    """Run deterministic regime analysis and return validated output"""
    raw = build_market_decision()  # Your original strong function

    output = MarketAnalystOutput(
        agent="market_analyst",
        event_id=f"market_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        short_term=raw["short_term"],
        long_term=raw["long_term"],
        components=raw.get("components", {}),
        summary=raw.get("summary", "Market regime analysis completed."),
        metadata={
            "schema_version": "1.0",
            "validated_at": datetime.now(timezone.utc).isoformat(),
            "cache_hit": raw.get("data_quality", {}).get("cache_hit", False),
            "source": "market_analyst_dual_horizon"
        }
    )

    validated = validate_market_analyst_output(output)
    return validated


if __name__ == "__main__":
    try:
        decision = build_market_analyst_decision()
        print(decision.model_dump_json(indent=2))
        print(f"\n✅ Market Analyst completed successfully")
        print(f"   Event ID: {decision.event_id}")
        print(f"   Short-term Regime: {decision.short_term.get('regime')}")
        print(f"   Long-term Regime:  {decision.long_term.get('regime')}")
    except Exception as e:
        print(f"❌ Error running Market Analyst: {e}")
        import traceback
        traceback.print_exc()