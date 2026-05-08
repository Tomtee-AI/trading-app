"""
services/validation.py

Centralized validation, enrichment, and governance layer for agent outputs.
All agent JSON must pass through here before any downstream use.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Type, TypeVar

from pydantic import ValidationError

from config.schemas import (
    BaseAgentOutput,
    MarketAnalystOutput,
    TradeAnalystOutput,
    PortfolioManagerOutput,
    ConscienceOutput,
    CEOOutput,
)

T = TypeVar("T", bound=BaseAgentOutput)


class ValidationResult:
    """Result of validation + enrichment."""

    def __init__(
        self,
        is_valid: bool,
        output: Optional[BaseAgentOutput] = None,
        errors: Optional[list[str]] = None,
        enriched_metadata: Optional[Dict[str, Any]] = None,
    ):
        self.is_valid = is_valid
        self.output = output
        self.errors = errors or []
        self.enriched_metadata = enriched_metadata or {}


def generate_event_id(prefix: str = "evt") -> str:
    """Generate a unique, sortable event ID."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    rand = hashlib.sha256(str(datetime.now().timestamp()).encode()).hexdigest()[:8]
    return f"{prefix}_{ts}_{rand}"


def validate_agent_output(
    raw_output: Dict[str, Any],
    expected_model: Type[T],
    min_confidence: float = 0.6,
) -> ValidationResult:
    """
    Validate raw dict from an agent against its Pydantic schema.
    Adds standard metadata and performs basic governance checks.
    """
    errors: list[str] = []
    enriched: Dict[str, Any] = {}

    try:
        # Parse and validate
        validated: T = expected_model.model_validate(raw_output)

        # Confidence gate
        if validated.confidence < min_confidence:
            errors.append(
                f"Confidence {validated.confidence:.2f} below minimum threshold {min_confidence}"
            )

        # Add standard enrichment
        enriched = {
            "validated_at": datetime.now(timezone.utc).isoformat(),
            "schema_version": "1.0.0",
            "validation_passed": len(errors) == 0,
            "original_agent": validated.agent,
        }

        # Optional: Add hash of the output for audit
        output_json = validated.model_dump_json()
        enriched["output_hash"] = hashlib.sha256(output_json.encode()).hexdigest()[:16]

        if errors:
            return ValidationResult(
                is_valid=False, output=validated, errors=errors, enriched_metadata=enriched
            )

        return ValidationResult(
            is_valid=True, output=validated, errors=[], enriched_metadata=enriched
        )

    except ValidationError as ve:
        errors.append(f"Pydantic validation failed: {ve}")
        return ValidationResult(is_valid=False, errors=errors)
    except Exception as e:
        errors.append(f"Unexpected validation error: {str(e)}")
        return ValidationResult(is_valid=False, errors=errors)


def validate_market_analyst(raw: Dict[str, Any]) -> ValidationResult:
    """Convenience wrapper for Market Analyst."""
    return validate_agent_output(raw, MarketAnalystOutput, min_confidence=0.65)


def validate_trade_analyst(raw: Dict[str, Any]) -> ValidationResult:
    return validate_agent_output(raw, TradeAnalystOutput, min_confidence=0.55)


def validate_portfolio_manager(raw: Dict[str, Any]) -> ValidationResult:
    return validate_agent_output(raw, PortfolioManagerOutput, min_confidence=0.70)


def validate_conscience(raw: Dict[str, Any]) -> ValidationResult:
    """Conscience has higher bar because it is a governance layer."""
    return validate_agent_output(raw, ConscienceOutput, min_confidence=0.75)


# Example usage in a workflow:
# result = validate_market_analyst(agent_json)
# if result.is_valid:
#     enriched_output = result.output
#     # proceed to scoring / execution gating
# else:
#     log_and_escalate(result.errors)