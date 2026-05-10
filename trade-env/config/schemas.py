# config/schemas.py
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, ConfigDict

class BiasType(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"

class RegimeType(str, Enum):
    BULLISH_PRICE_BEARISH_VOL = "BULLISH_PRICE_BEARISH_VOL"
    BEARISH_PRICE_BULLISH_VOL = "BEARISH_PRICE_BULLISH_VOL"
    NEUTRAL = "NEUTRAL"
    MIXED = "MIXED"

class FactorSignal(BaseModel):
    name: str
    value: Any
    score: int = Field(..., ge=-10, le=10)

class MarketAnalystOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    agent: str = "market_analyst"
    event_id: str
    short_term: Dict[str, Any]
    long_term: Dict[str, Any]
    components: Dict[str, Any]
    summary: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class TradeCoordinatorOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    agent: str = "trade_coordinator"
    event_id: str
    market_bias: MarketAnalystOutput
    strategy_permissions: Dict[str, str]
    ranking_weights: Dict[str, float]
    candidate_queue: List[Dict[str, Any]]
    no_trade: Optional[Dict[str, Any]] = None
    summary: str

class PortfolioDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    agent: str = "portfolio_risk_manager"
    event_id: str
    approved_allocations: List[Dict[str, Any]]
    veto_reasons: List[str]
    profit_transfer_amount: float
    tactical_exposure: float
    retirement_exposure: float
    summary: str

class ConscienceOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    agent: str = "conscience"
    event_id: str
    review_status: str  # APPROVED or VETOED
    veto_reasons: List[str]
    approved: bool
    summary: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class CEOOutput(BaseModel):
    """CEO / Strategic Synthesizer output — single source of truth for final strategic directive."""
    model_config = ConfigDict(extra="forbid")
    agent: str = "ceo"
    event_id: str
    strategic_directive: str   # "CONTINUE_TACTICAL_ALPHA_GENERATION" or "PAUSE_ALL_TACTICAL"
    summary: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

# Add this if you want a generic base for all agents later
class BaseAgentOutput(BaseModel):
    agent: str
    event_id: str
    metadata: Dict[str, Any] = Field(default_factory=dict)