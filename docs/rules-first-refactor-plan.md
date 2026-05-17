# Rules-First Refactor Plan

Date: 2026-05-11

## Purpose

Refactor the trading app into a deterministic, rules-first trading research engine. The system should convert `rules.docx` into executable constraints, produce shared `TradeCandidate` objects across all strategy analysts, and emit a coordinator output that can format the exact trading report the rules require without inventing prices, catalysts, premiums, or liquidity.

This plan starts from the current codebase:

- `trade-env/config/schemas.py` defines agent-level Pydantic models, but trade candidates are still mostly `Dict[str, Any]`.
- `trade-env/unusual_options_analyst.py` already enforces several rules from `rules.docx`, including long-premium structures, stock price range, option cost range, flow/liquidity checks, and reward/risk estimates.
- `trade-env/option_liquidity.py` centralizes option-chain liquidity checks and exact vertical spread selection.
- `trade-env/earnings_analyst.py` exists, but `trade-env/workflows/daily_pipeline.py` does not include it.
- `trade-env/trade_coordinator_bias_aware_skeleton.py` has a richer coordinator output with `trade_analysis`, but `trade-env/agents/trade_coordinator.py` wraps only part of it.
- `trade-env/agents/portfolio_risk_manager.py` does not yet approve or size actual trade candidates.

## Target Behavior

The daily pipeline should do this in order:

1. State and persist the actual run date and data freshness.
2. Fetch or load market context, price action, volatility, earnings calendar, economic calendar, and relevant news/catalyst inputs.
3. Generate strategy candidates only from deterministic data fields.
4. Convert every specialist output into the same `TradeCandidate` schema.
5. Apply an executable rulebook before ranking.
6. Rank valid candidates by setup quality, catalyst strength, liquidity, affordability, asymmetry, and payoff realism.
7. Return exactly five trade ideas when at least five valid candidates exist.
8. Return fewer ideas plus clear `no_trade` / `insufficient_valid_candidates` reasons when the rulebook cannot honestly support five.
9. Include the final required sections: best overall, cheapest, aggressive moonshot, and one tempting trade to avoid.

LLMs may summarize already-validated facts, but Python remains authoritative for data, validation, ranking, and report fields.

## Executable Rulebook

Add a rulebook file and loader:

- `trade-env/config/rulebook.yaml`
- `trade-env/services/rulebook.py`
- `trade-env/services/rule_evaluator.py`
- `trade-env/tests/test_rulebook.py`

Initial rulebook sections:

```yaml
version: "1.0.0"
as_of_policy:
  require_actual_date: true
  require_data_timestamp: true
  max_market_data_age_hours: 24
  fail_if_live_data_unavailable: true

underlying:
  min_price: 5.00
  max_price: 100.00
  prefer_market_cap_max: 20000000000
  hard_exclude_missing_price: true

structures:
  allowed:
    - LONG_CALL
    - LONG_PUT
    - CALL_DEBIT_SPREAD
    - PUT_DEBIT_SPREAD
    - STRADDLE
    - STRANGLE
  disallowed:
    - NAKED_SHORT_OPTION
    - CREDIT_SPREAD
    - IRON_CONDOR
    - BUTTERFLY

premium:
  min_debit: 0.50
  max_debit: 5.00
  allow_exceptional_expensive: false

liquidity:
  min_open_interest: 20
  min_volume: 1
  max_bid_ask_spread_pct: 0.35
  hard_exclude_missing_bid_ask: true

reward_risk:
  target: 5.0
  minimum_for_main_report: 3.0
  allow_below_target_with_label: true
  hard_exclude_unestimated: true

report:
  target_idea_count: 5
  require_best_overall: true
  require_best_cheapest: true
  require_best_moonshot: true
  require_trade_to_avoid: true
```

The evaluator should return both a boolean decision and auditable reason codes:

- `PASS`
- `BLOCK_STALE_DATA`
- `BLOCK_DISALLOWED_STRUCTURE`
- `BLOCK_UNDERLYING_PRICE_OUT_OF_RANGE`
- `BLOCK_PREMIUM_OUT_OF_RANGE`
- `BLOCK_POOR_LIQUIDITY`
- `BLOCK_REWARD_RISK_UNESTIMATED`
- `WARN_REWARD_RISK_BELOW_5_TO_1`
- `WARN_REWARD_RISK_3_TO_1_TO_5_TO_1`
- `WARN_CATALYST_WEAK`

## Shared TradeCandidate Schema

Add first-class Pydantic models in `trade-env/config/schemas.py` or a new `trade-env/config/trade_schemas.py`.

Core model sketch:

```python
class TradeDirection(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"

class TradeStructure(str, Enum):
    LONG_CALL = "LONG_CALL"
    LONG_PUT = "LONG_PUT"
    CALL_DEBIT_SPREAD = "CALL_DEBIT_SPREAD"
    PUT_DEBIT_SPREAD = "PUT_DEBIT_SPREAD"
    STRADDLE = "STRADDLE"
    STRANGLE = "STRANGLE"

class OptionLeg(BaseModel):
    action: Literal["BUY", "SELL"]
    option_type: Literal["CALL", "PUT"]
    strike: float
    expiration: date
    bid: float | None = None
    ask: float | None = None
    mid: float | None = None
    open_interest: int | None = None
    volume: int | None = None
    bid_ask_spread_pct: float | None = None

class TradeCandidate(BaseModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    candidate_id: str
    source_agent: str
    strategy_type: str
    ticker: str
    company_name: str | None = None
    direction: TradeDirection
    structure: TradeStructure
    expiration: date
    legs: list[OptionLeg]
    underlying_price: float
    estimated_entry_debit: float
    max_loss: float
    estimated_reward: float | None
    reward_to_risk: float | None
    catalyst: str | None
    catalyst_date: date | None = None
    thesis: str
    main_risk: str
    liquidity: dict[str, Any]
    data_quality: dict[str, Any]
    confidence_level: Literal["LOW", "MEDIUM", "HIGH"]
    confidence_score: int = Field(ge=1, le=10)
    rule_results: list[dict[str, Any]] = Field(default_factory=list)
    ranking_scores: dict[str, float] = Field(default_factory=dict)
```

Every analyst should emit or adapt into this schema before the coordinator sees the candidate. Specialist-specific details can remain in `metadata`, but the coordinator must not depend on untyped nested dictionaries for core trade fields.

## Coordinator Output Contract

Replace the current loose `candidate_queue: List[Dict[str, Any]]` coordinator model with a strict output:

```python
class CoordinatorCandidate(BaseModel):
    rank: int
    candidate: TradeCandidate
    status: Literal["FORWARD", "BLOCK", "WATCHLIST"]
    reason_codes: list[str]
    score: float

class FinalTradeIdea(BaseModel):
    rank: int
    ticker: str
    company_name: str | None
    why_interesting_now: str
    direction: TradeDirection
    trade_structure: TradeStructure
    expiration: date
    strikes: list[float]
    approximate_premium_or_cost: float
    maximum_loss: float
    estimated_reward: float | None
    reward_to_risk: float | None
    catalyst: str | None
    attractive_vs_alternatives: str
    main_risk: str
    conviction_level: Literal["LOW", "MEDIUM", "HIGH"]
    confidence_score: int
    five_hundred_to_one_thousand_structure: dict[str, Any]

class TradeCoordinatorOutput(BaseModel):
    agent: Literal["trade_coordinator"] = "trade_coordinator"
    schema_version: Literal["2.0.0"] = "2.0.0"
    event_id: str
    as_of: datetime
    data_freshness: dict[str, Any]
    market_context: dict[str, Any]
    rulebook_version: str
    candidate_queue: list[CoordinatorCandidate]
    final_ideas: list[FinalTradeIdea]
    best_overall_trade: str | None
    best_cheapest_trade: str | None
    best_aggressive_moonshot_trade: str | None
    trade_to_avoid: dict[str, Any] | None
    no_trade: dict[str, Any] | None
    summary: str
```

The final report layer should format `final_ideas`, not re-interpret raw analyst payloads.

## Ranking Model

Use deterministic weighted scoring. Start simple and testable:

- Setup quality: 25%
- Catalyst strength: 20%
- Liquidity: 20%
- Affordability: 15%
- Reward/risk realism: 15%
- Market regime alignment: 5%

Hard gates run before ranking. Warnings reduce score but do not necessarily block. A candidate below the minimum reward/risk can still be used for `trade_to_avoid`, but not as a final idea unless the rulebook explicitly allows it.

## Implementation Phases

### Phase 1: Schemas and Rulebook

- Add `TradeCandidate`, `OptionLeg`, `RuleResult`, `CoordinatorCandidate`, and `FinalTradeIdea` models.
- Add `rulebook.yaml` with rules from `rules.docx`.
- Add a `Rulebook` loader with Pydantic validation.
- Add unit tests for allowed structures, premium limits, underlying price limits, liquidity limits, and reward/risk labels.

Acceptance criteria:

- Invalid structures and missing core trade fields fail validation.
- The rulebook can be loaded without analysts or live market data.
- Rule evaluation returns stable reason codes.

### Phase 2: Candidate Adapters

- Add adapter functions that convert current UOA, earnings, divergence, and crack-spread payloads into `TradeCandidate`.
- Keep existing analyst internals intact at first.
- Make missing price, premium, expiration, liquidity, or reward/risk visible as rule failures.

Acceptance criteria:

- Existing analyst outputs can be adapted or blocked with explicit reasons.
- No coordinator path consumes untyped core trade fields.

### Phase 3: Coordinator Contract

- Update `TradeCoordinatorOutput` to schema version `2.0.0`.
- Route `agents/trade_coordinator.py` through the richer skeleton-style `build_output` behavior, then replace skeleton dicts with typed models.
- Require `final_ideas` to be generated from rule-passing candidates only.
- Preserve `no_trade` when there are not enough valid ideas.

Acceptance criteria:

- Coordinator output includes `as_of`, `data_freshness`, `rulebook_version`, typed `candidate_queue`, typed `final_ideas`, and the final four summary picks.
- The pipeline no longer drops `trade_analysis`.

### Phase 4: Daily Pipeline Integration

- Add `earnings_analyst` to `trade-env/workflows/daily_pipeline.py`.
- Ensure `reports/` is created before writing reports.
- Add a final report formatter that maps `FinalTradeIdea` into the exact `rules.docx` output fields.
- Keep execution disabled for trade orders; output remains research and review only.

Acceptance criteria:

- The daily pipeline includes market, divergence, crack spread, UOA, and earnings inputs.
- A dry run produces a structured JSON report even when some data sources are missing.
- Missing live/recent data is disclosed and blocks fabricated trade ideas.

### Phase 5: Portfolio and Governance

- Update portfolio risk manager to approve, size, watchlist, or veto individual `TradeCandidate` objects.
- Add per-trade sizing guidance for $500-$1000 risk.
- Add portfolio-level exposure checks by ticker, sector, direction, expiration week, and strategy type.

Acceptance criteria:

- Portfolio decisions reference candidate IDs and explicit size/risk.
- Conscience and CEO layers receive typed candidate decisions, not opaque dictionaries.

### Phase 6: Tests and Hygiene

- Add a minimal dependency file.
- Remove committed cache artifacts in a separate cleanup commit.
- Add tests for rulebook loading, candidate validation, coordinator report count behavior, and report formatting.

Acceptance criteria:

- `pytest` can validate schema and rulebook behavior without network calls.
- Live-data tests are isolated or marked integration.

## First Concrete Code Changes

Recommended first implementation commit:

1. Add `trade-env/config/rulebook.yaml`.
2. Add `trade-env/config/trade_schemas.py`.
3. Add `trade-env/services/rulebook.py`.
4. Add `trade-env/services/rule_evaluator.py`.
5. Add `trade-env/tests/test_rule_evaluator.py`.

After that, adapt the UOA payload first because it already carries the most complete rule-relevant fields. Then adapt earnings and add it to the daily pipeline.

## Open Decisions

- Whether reward/risk below 3:1 should be a hard block for the final five ideas or only a heavy penalty.
- Whether relative-value divergence trades belong in the same final report, since `rules.docx` is oriented around single-name long options ideas.
- Whether `rules.docx` should be copied into the repo as Markdown for versioned rule changes, or kept as an external source document.
- Whether expensive premium exceptions should ever be allowed, and who approves them.
