# Strategy Research Factory Implementation Guide

Version: 0.1 - May 2026

Status: Practical companion to `Strategy_Research_Factory_Design.md`

## 1. Executive Summary

The strategy research factory should be built as a staged promotion pipeline. Each stage produces typed artifacts, and each gate either advances the strategy or archives a failure with enough detail to replay the decision.

Build order:

1. schemas and artifact storage
2. research dossier workflow
3. parameter synthesis and benchmark pre-registration
4. VectorBT Pro backtest planning
5. backtest code generation and test harness
6. validation reports
7. IB paper-trading plan
8. IB paper implementation

Live execution remains out of scope until the paper workflow has durable audit logs, replay tests, reconciliation, and human approval.

## 2. Proposed Package Layout

Add a strategy factory module under the trading app codebase:

```text
trade-env/
  strategy_factory/
    __init__.py
    schemas.py
    states.py
    orchestrator.py
    policy.py
    storage.py
    research_agents.py
    synthesis_agent.py
    vectorbt_planner.py
    backtest_developer.py
    backtest_runner.py
    validation.py
    ib_planner.py
    ib_implementer.py
    registry.py
  tests/
    strategy_factory/
      test_schemas.py
      test_policy_gates.py
      test_backtest_validation.py
      fixtures/
```

If the current repo layout prefers agents under `trade-env/agents`, keep agent wrappers there and put shared service logic in `strategy_factory`.

## 3. Minimum Schemas

### StrategyRequest

```python
class StrategyRequest(BaseModel):
    strategy_id: str
    requested_at: datetime
    requested_by: str
    strategy_type: str
    asset_universe_hint: list[str] = []
    constraints: dict[str, Any] = {}
    target_metrics: dict[str, float] = {}
    broker_execution_allowed: bool = False
```

`broker_execution_allowed` must default to `False`.

### EvidenceSource

```python
class EvidenceSource(BaseModel):
    source_id: str
    strategy_id: str
    source_type: Literal["academic", "practitioner", "code", "dataset"]
    title: str
    url: str | None = None
    citation: str | None = None
    retrieved_at: datetime
    source_date: date | None = None
    quality_score: float
    relevance_score: float
    reproducibility_score: float
    extracted_claims: list[str]
    parameter_claims: dict[str, Any] = {}
    caveats: list[str] = []
```

### ParameterDecision

```python
class ParameterDecision(BaseModel):
    strategy_id: str
    decision_id: str
    created_at: datetime
    selected_parameters: dict[str, Any]
    allowed_ranges: dict[str, Any]
    benchmark_expectations: dict[str, Any]
    citations_by_parameter: dict[str, list[str]]
    rejected_alternatives: list[dict[str, Any]]
    decision_rationale: str
```

Benchmark expectations must be written before the validation run.

### BacktestSpec

```python
class BacktestSpec(BaseModel):
    strategy_id: str
    spec_id: str
    engine: Literal["vectorbt_pro"]
    data_requirements: dict[str, Any]
    signal_spec: dict[str, Any]
    portfolio_spec: dict[str, Any]
    cost_model: dict[str, Any]
    parameter_grid: dict[str, list[Any]]
    benchmark_specs: list[dict[str, Any]]
    validation_windows: list[dict[str, Any]]
    expected_outputs: list[str]
```

### ValidationReport

```python
class ValidationReport(BaseModel):
    strategy_id: str
    report_id: str
    created_at: datetime
    status: Literal["PASS", "FAIL", "NEEDS_REVIEW"]
    tested_run_ids: list[str]
    benchmark_results: dict[str, Any]
    selected_metrics: dict[str, float]
    sensitivity_results: dict[str, Any]
    leakage_checks: dict[str, Any]
    failure_reasons: list[str] = []
    promotion_allowed: bool
```

## 4. Gate Policy

Implement gates as deterministic functions first. Agents may recommend a decision, but policy code makes it.

```text
can_complete_research(dossier) -> GateDecision
can_accept_parameters(parameter_decision) -> GateDecision
can_run_backtest(backtest_spec) -> GateDecision
can_promote_after_backtest(validation_report) -> GateDecision
can_generate_ib_plan(validation_report) -> GateDecision
can_mark_paper_ready(execution_plan, code_review, tests) -> GateDecision
```

Each `GateDecision` should include:

```text
status: PASS | FAIL | NEEDS_REVIEW
reasons: list[str]
required_actions: list[str]
policy_version: str
```

## 5. Backtest Validation Rules

Start with a conservative default policy:

| Check | Default Rule |
|---|---|
| Syntax/import | Must pass |
| Smoke data | Must complete |
| Full run | Must complete |
| Benchmark | Must beat relevant benchmark after costs |
| Sharpe | Must be inside or above pre-registered expectation |
| Drawdown | Must be inside configured maximum |
| Sensitivity | Nearby parameters must not collapse the edge |
| Costs | Base and stressed cost models must be reported |
| Leakage | No unresolved lookahead/leakage flags |
| Reproducibility | Same config and data fingerprint reproduces metrics within tolerance |

The system should record impressive results as suspicious until the checks pass. A very high Sharpe with very low drawdown should trigger extra review rather than automatic promotion.

## 6. Six-Run Harness

Use the six-run idea as a minimum harness:

```text
Run 1: import and syntax
Run 2: fixture data smoke test
Run 3: full sample smoke test
Run 4: baseline benchmark parameters
Run 5: selected research-derived parameters
Run 6: sensitivity grid around selected parameters
```

Do not let the tester edit strategy rules during the validation run. If the test fails, send the workflow back to planning or archive it.

## 7. VectorBT Pro Planning Contract

The VectorBT plan should answer:

- What input arrays or frames are required?
- What signals are entries and exits?
- Is sizing fixed, volatility-targeted, equal-weight, or rank-based?
- Is rebalancing daily, weekly, monthly, event-driven, or hybrid?
- How are fees and slippage represented?
- What benchmarks are computed in the same data window?
- What metrics are required for promotion?
- What charts or tearsheets are generated?

The first implementation should support long-only and market-neutral research strategies before adding options, futures, or multi-leg execution.

## 8. IB Paper Implementation Contract

The IB implementation must be paper-only at first.

Required modules:

```text
ib_connection.py
ib_contracts.py
ib_orders.py
ib_fills.py
ib_reconciliation.py
ib_scheduler.py
ib_risk.py
```

Required tests:

- connection failure handling
- order object construction
- duplicate order prevention
- partial fill reconciliation
- cancel/replace behavior
- market-closed behavior
- paper/live mode enforcement

## 9. Observability

Every workflow should emit:

- `strategy_id`
- `workflow_id`
- `stage`
- `agent_name`
- `tool_name`
- `artifact_id`
- `gate_decision`
- `policy_version`
- `data_fingerprint`
- `code_version`
- `cost_usd_estimate`
- `duration_ms`

Backtest metrics should include:

- total return
- CAGR
- Sharpe
- Sortino
- max drawdown
- Calmar
- volatility
- turnover
- average holding period
- win rate
- profit factor
- exposure
- benchmark-relative return

## 10. First Milestone

The first milestone should not include IB code. It should prove the factory can produce a replayable, governed backtest package.

Definition of done:

- `StrategyRequest`, `EvidenceSource`, `ParameterDecision`, `BacktestSpec`, and `ValidationReport` schemas exist
- research dossier can be saved and loaded
- parameter decisions require citations
- benchmark expectations are pre-registered
- a sample strategy can generate a VectorBT Pro backtest spec
- validation report can produce `PASS`, `FAIL`, or `NEEDS_REVIEW`
- failed strategies are archived with reasons

## 11. Second Milestone

Add IB paper planning only after the backtest milestone is stable.

Definition of done:

- validated strategy can create an `ExecutionPlan`
- paper mode is mandatory
- risk limits are enforced outside agent code
- order lifecycle is modeled before implementation
- no live order function is reachable from the generated workflow

## 12. Updated 30/60/90-Day Plan

### First 30 Days

- Add schemas and storage for strategy factory artifacts.
- Add deterministic policy gates.
- Add manual research dossier ingestion.
- Add parameter decision validation.
- Add a first VectorBT Pro `BacktestSpec`.

### Days 31-60

- Add research fan-out agents.
- Add synthesis agent.
- Add backtest developer workflow.
- Add six-run harness.
- Add validation reports and strategy registry.

### Days 61-90

- Add IB paper planning.
- Add paper-only IB implementation.
- Add fill reconciliation and audit logging.
- Add replay tests for promotion decisions.
- Add dashboard/reporting for strategy status.

## 13. Open Decisions

- Whether VectorBT Pro is available in the local environment or needs a separate licensed runtime.
- Whether the first strategy factory example should be RSI sector rotation, uranium stat arb, or a smaller synthetic fixture strategy.
- Whether source retrieval should be live internet search, curated local source packs, or both.
- Whether generated code should be committed as strategy modules or stored as artifacts pending human review.

## 14. Bottom Line

Build the research factory as a promotion system, not a hype loop. Its value is not that agents can produce impressive backtests. Its value is that every strategy idea leaves a trail of evidence, assumptions, code, tests, failures, and policy decisions.

