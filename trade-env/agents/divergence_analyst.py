#!/usr/bin/env python3
"""
agents/divergence_analyst.py
Step 5 - Clean, schema-compliant Divergence Analyst Agent
Fully integrates the rich divergence_analyst.py you provided
"""

import sys
from pathlib import Path
from datetime import datetime, timezone
import json

# === ROBUST PROJECT ROOT DETECTION ===
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Import the full rich divergence engine (the long file you attached)
import divergence_analyst   # ← your attached file must be in project root


def build_divergence_analyst_decision() -> dict:
    """Run the full divergence analysis and return coordinator-ready payload"""
    # Call the original build_output function from your rich script
    output, internal_report = divergence_analyst.build_output(
        divergence_analyst.RuntimeConfig()
    )

    # Add pipeline metadata for consistency with other agents
    output["agent"] = "divergence_analyst"
    if "event_id" not in output:
        output["event_id"] = f"div_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    # The output is already coordinator-compatible (has "candidates" list)
    print(f"✅ Divergence Analyst generated {len(output.get('candidates', []))} candidate(s)")
    return output


if __name__ == "__main__":
    try:
        decision = build_divergence_analyst_decision()
        print(json.dumps(decision, indent=2))
        print(f"\n✅ Divergence Analyst completed successfully")
        print(f"   Event ID: {decision.get('event_id')}")
        print(f"   Candidates generated: {len(decision.get('candidates', []))}")
    except Exception as e:
        print(f"❌ Error running Divergence Analyst: {e}")
        import traceback
        traceback.print_exc()