# services/validation.py
from datetime import datetime, timezone
from pydantic import ValidationError
from config.schemas import (
    MarketAnalystOutput,
    TradeCoordinatorOutput,
    PortfolioDecision,
    ConscienceOutput,
    CEOOutput
)

def validate_market_analyst_output(data) -> MarketAnalystOutput:
    if isinstance(data, dict):
        data = MarketAnalystOutput.model_validate(data)
    if data.metadata.get("validated_at") is None:
        data.metadata["validated_at"] = datetime.now(timezone.utc).isoformat()
    return data

def validate_trade_coordinator_output(data) -> TradeCoordinatorOutput:
    if isinstance(data, dict):
        data = TradeCoordinatorOutput.model_validate(data)
    return data

def validate_portfolio_decision(data) -> PortfolioDecision:
    if isinstance(data, dict):
        data = PortfolioDecision.model_validate(data)
    return data

def validate_conscience_output(data) -> ConscienceOutput:
    if isinstance(data, dict):
        data = ConscienceOutput.model_validate(data)
    return data

def validate_ceo_output(data: dict) -> CEOOutput:
    """Validate CEO output against strict schema."""
    output = CEOOutput.model_validate(data)
    print("✅ CEO output validated")
    return output