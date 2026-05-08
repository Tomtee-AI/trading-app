"""
config/schemas.py
Pydantic v2 schemas for all agent outputs in ZeroHumanTradingCompany.

All agents MUST return instances of these models (or subclasses).
Extra fields are forbidden for strict contract enforcement.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# =============================================================================
# Enums
# =============================================================================

class RegimeType(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    MIXED = "MIXED"
    SIDEWAYS = "SIDEWAYS"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    UNKNOWN = "UNKNOWN"


class BiasType(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


class ActionType(str, Enum):
    HOLD = "HOLD"
    REDUCE_RISK = "REDUCE_RISK"
    INCREASE_EXPOSURE = "INCREASE_EXPOSURE"
    REBALANCE = "REBALANCE"
    ALERT_ONLY = "ALERT_ONLY"


class ReviewStatus(str, Enum):
    AUTO_APPROVED = "AUTO_APPROVED"
    PENDING_REVIEW = "PENDING_REVIEW"
    REJECTED = "REJECTED"
    ESCALATED = "ESCALATED"


# =============================================================================
# Base Models
# =============================================================================

class BaseAgentOutput(BaseModel):
    """Base class for all agent outputs. Enforces strict structured data."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        validate_default=True,
        str_strip_whitespace=True,
    )

    agent: str = Field(..., description="Name of the agent that produced this output")
    event_id: str = Field(
        ...,
        description="Unique identifier for this decision/event (recommend ISO8601 + short hash)",
        min_length=10,
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp when the analysis was generated",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Agent's self-assessed confidence in the output (0.0-1.0)",
    )
    summary: str = Field(
        ...,
        min_length=20,
        description="Concise natural language summary of the analysis and recommendation",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional structured metadata (model version, data freshness, etc.)",
    )

    @field_validator("event_id")
    @classmethod
    def validate_event_id(cls, v: str) -> str:
        if not v or len(v) < 10:
            raise ValueError("event_id must be at least 10 characters")
        return v


class FactorSignal(BaseModel):
    """Individual factor used in analysis."""

    name: str
    value: Any
    signal: str = Field(..., description="e.g. 'bullish', 'bearish', 'neutral', 'supportive'")
    weight: float = Field(0.1, ge=0.0, le=1.0)
    comment: Optional[str] = None


# =============================================================================
# Specific Agent Outputs
# =============================================================================

class MarketAnalystOutput(BaseAgentOutput):
    """Output contract for the Market Analyst agent."""

    agent: str = "market_analyst"
    regime: RegimeType
    price_bias: BiasType
    vol_bias: BiasType
    factors: Dict[str, FactorSignal] = Field(
        default_factory=dict,
        description="Key market regime factors with signals and weights"
    )
    key_levels: Dict[str, float] = Field(
        default_factory=dict,
        description="Important price levels (support, resistance, pivots)"
    )
    macro_context: Optional[str] = Field(
        None, description="High-level macro environment summary"
    )


class TradeAnalystOutput(BaseAgentOutput):
    """Output contract for the Trade Analyst agent."""

    agent: str = "trade_analyst"
    proposed_action: ActionType
    instruments: List[str] = Field(default_factory=list)
    rationale: str
    risk_reward_ratio: Optional[float] = Field(None, ge=0.0)
    suggested_size_pct: Optional[float] = Field(None, ge=0.0, le=0.2)
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    time_horizon: str = Field("swing", description="e.g. 'intraday', 'swing', 'position'")


class PortfolioManagerOutput(BaseAgentOutput):
    """Output contract for the Portfolio Manager agent."""

    agent: str = "portfolio_manager"
    current_allocation: Dict[str, float] = Field(default_factory=dict)
    recommended_changes: Dict[str, float] = Field(default_factory=dict)
    overall_risk_score: float = Field(..., ge=0.0, le=10.0)
    max_drawdown_estimate: Optional[float] = None
    veto_reasons: List[str] = Field(default_factory=list)
    review_status: ReviewStatus = ReviewStatus.PENDING_REVIEW


class ConscienceOutput(BaseAgentOutput):
    """Output from the governance / ethics layer."""

    agent: str = "conscience"
    decision: str = Field(..., description="'APPROVE', 'REJECT', 'MODIFY', or 'ESCALATE'")
    reasons: List[str] = Field(default_factory=list)
    ethical_flags: List[str] = Field(default_factory=list)
    suggested_modifications: Optional[Dict[str, Any]] = None


class CEOOutput(BaseAgentOutput):
    """High-level synthesis from the CEO agent."""

    agent: str = "ceo"
    strategic_priority: str
    key_insights: List[str] = Field(default_factory=list)
    action_items: List[Dict[str, Any]] = Field(default_factory=list)
    overall_market_view: RegimeType


# Add more agent schemas as needed (Intern, IT Pro, Security, CPA, HR)
# They can inherit from BaseAgentOutput and add domain-specific fields.


# =============================================================================
# Utility / Request Models
# =============================================================================

class MarketDataRequest(BaseModel):
    symbols: List[str] = Field(default=["QQQ", "^VIX"])
    period: str = "1mo"
    interval: str = "1d"


class RegimeAnalysisRequest(BaseModel):
    event_id: str
    include_fred: bool = True
    model_name: str = "llama3.1"