#!/usr/bin/env python3
"""
agents/crack_spread_analyst.py
Strategy Specialist wrapper — Crack Spread
Uses the full rich implementation in the root crack_spread_analyst.py
"""

import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

def build_crack_spread_decision() -> Dict[str, Any]:
    """Crack Spread Strategy Specialist — produces CoordinatorCandidate payloads."""
    try:
        # Direct import of rich function + its config class
        from crack_spread_analyst import build_crack_spread_decision as rich_build, RuntimeConfig
        config = RuntimeConfig()                     # uses all your conservative defaults
        result = rich_build(config)

        # Handle tuple return from rich function (output, internal_report)
        if isinstance(result, tuple):
            output = result[0] if len(result) > 0 else {}
        else:
            output = result

    except (ImportError, AttributeError):
        # Dynamic fallback (works from any subdir)
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "crack_spread_analyst", ROOT / "crack_spread_analyst.py"
        )
        crack_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(crack_module)
        rich_build = crack_module.build_crack_spread_decision
        RuntimeConfig = getattr(crack_module, "RuntimeConfig", None)

        if RuntimeConfig:
            config = RuntimeConfig()
            result = rich_build(config)
            if isinstance(result, tuple):
                output = result[0] if len(result) > 0 else {}
            else:
                output = result
        else:
            output = rich_build()   # last-resort no-arg call

    # Ensure consistent wrapper metadata for the pipeline
    if "agent" not in output:
        output["agent"] = "crack_spread_analyst"
    if "event_id" not in output:
        output["event_id"] = f"crack_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    if "metadata" not in output:
        output["metadata"] = {
            "schema_version": "2.2.0",
            "source": "crack_spread_analyst"
        }

    return output


if __name__ == "__main__":
    result = build_crack_spread_decision()
    print(f"✅ Crack Spread Analyst completed — {len(result.get('candidates', []))} candidates")
    print(f"   Event ID: {result.get('event_id')}")