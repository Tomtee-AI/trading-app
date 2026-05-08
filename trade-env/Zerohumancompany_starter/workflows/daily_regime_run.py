"""
workflows/daily_regime_run.py

Example daily workflow that runs the Market Analyst and validates the output.
This demonstrates the full governance flow: Agent → Validation → (future) Scoring/Execution.
"""

from __future__ import annotations

from agents.market_analyst import analyze_market_regime
from services.validation import validate_market_analyst


def run_daily_market_regime_analysis() -> None:
    print("=== ZeroHumanTradingCompany - Daily Regime Analysis ===\n")

    # 1. Run the agent
    raw_or_validated_output = analyze_market_regime(use_llm_summary=False)

    # In real usage the agent would return a dict/JSON.
    # Here we already return a Pydantic model, so convert for demo.
    agent_json = raw_or_validated_output.model_dump(mode="json")

    # 2. Validate through governance layer
    validation_result = validate_market_analyst(agent_json)

    if validation_result.is_valid:
        print("✅ Validation PASSED")
        print(f"Event ID: {validation_result.output.event_id}")
        print(f"Regime: {validation_result.output.regime}")
        print(f"Confidence: {validation_result.output.confidence}")
        print(f"Summary: {validation_result.output.summary}")
        print("\nEnriched metadata:")
        for k, v in validation_result.enriched_metadata.items():
            print(f"  {k}: {v}")

        # TODO: Next steps in full system
        # - Pass to scoring.py
        # - Run Trade Analyst if regime warrants
        # - Portfolio Manager review
        # - Conscience gate for any proposed actions

    else:
        print("❌ Validation FAILED")
        for err in validation_result.errors:
            print(f"  - {err}")


if __name__ == "__main__":
    run_daily_market_regime_analysis()