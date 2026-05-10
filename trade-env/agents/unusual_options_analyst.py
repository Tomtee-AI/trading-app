#!/usr/bin/env python3
"""
agents/unusual_options_analyst.py
Strategy Specialist wrapper — Unusual Options Activity
Uses the full rich implementation you attached.
"""

import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

def build_unusual_options_decision() -> Dict[str, Any]:
    """UOA Strategy Specialist — produces CoordinatorCandidate payloads."""
    # Import the rich production script (the one you just attached)
    try:
        from unusual_options_analyst import build_payload, RuntimeConfig
    except ImportError:
        # Dynamic fallback (works even if run from agents/ subdir)
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "unusual_options_analyst", ROOT / "unusual_options_analyst.py"
        )
        uoa_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(uoa_module)
        build_payload = uoa_module.build_payload
        RuntimeConfig = uoa_module.RuntimeConfig

    # Run the full rich analysis with conservative defaults
    config = RuntimeConfig()                    # all your guardrails applied
    payload, full_report = build_payload(config)

    # Light wrapper metadata for pipeline traceability
    output = {
        "agent": "unusual_options_analyst",
        "event_id": f"uoa_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        "strategy_type": "uoa",
        "candidates": payload.get("candidates", []),
        "no_trade": None,
        "summary": payload.get("summary", f"Unusual Options Analyst generated {len(payload.get('candidates', []))} candidates."),
        "metadata": {
            "schema_version": "2.2.0",
            "source": "unusual_options_analyst",
            "input_rows": full_report.get("diagnostics", {}).get("input_rows", 0),
            "report_path": str(full_report.get("report_path", "N/A"))
        }
    }
    return output


if __name__ == "__main__":
    result = build_unusual_options_decision()
    print(f"✅ Unusual Options Analyst completed — {len(result.get('candidates', []))} candidates")
    print(f"   Event ID: {result.get('event_id')}")