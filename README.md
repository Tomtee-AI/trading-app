# Zero Human Company

Local-first multi-agent wealth management and trading research system built with **Python**, **CrewAI**, and **Ollama**.

The project is designed around a strict separation of concerns:

* **LLMs produce structured analysis**
* **Python enforces rules and executes side effects**

Agents do not directly place trades, send emails, or modify external systems. They return validated JSON. Deterministic Python services handle execution, logging, idempotency, and governance.

## Status

Early-stage project. The architecture and contracts are being defined first, with execution and broker integration intentionally delayed until governance and reliability layers are stable.

## Goals

* Run locally on a home server
* Use local LLMs through Ollama
* Keep agent outputs structured and machine-readable
* Make execution deterministic and auditable
* Add layered risk controls and veto logic
* Prevent duplicate actions with event IDs and state tracking
* Support iterative expansion from market analysis into broader wealth management workflows

## Non-Goals

* Unchecked autonomous trade execution
* Direct tool access from agents to live systems
* Opaque decision-making without logs or schemas
* “AutoGPT-style” autonomy without governance

## Core Design

### Agents are advisory

Each agent returns structured JSON only. No direct side effects are allowed inside the agent layer.

### Python is authoritative

Python services handle:

* schema validation
* scoring and ranking
* idempotency checks
* notification delivery
* execution gating
* database logging
* audit trail generation

### Governance is required

High-impact actions should pass through one or more oversight layers before they are allowed to execute.

## Planned Agent Roles

* **CEO** — coordinates high-level priorities and synthesizes outputs
* **Conscience** — governance, ethics, and veto authority
* **Market Analyst** — classifies market regime and macro backdrop
* **Trade Analyst** — proposes trade ideas within system constraints
* **Portfolio Manager** — allocation, exposure, and risk veto layer
* **Intern** — low-risk research and background gathering
* **IT Pro** — infrastructure and runtime health
* **Security** — secrets, hardening, and audit concerns
* **CPA** — bookkeeping and reporting support
* **HR** — process, policy, and workflow support

## Architecture

```text
Market / Macro Data
        |
        v
  Python Ingestion Layer
        |
        v
 Deterministic Signal Logic
        |
        v
      CrewAI Agents
        |
        v
 Structured JSON Outputs
        |
        v
 Python Validation + Scoring + Governance
        |
        +--> Reject
        +--> Log Only
        +--> Request Review
        |
        v
 Deterministic Execution Layer
        |
        +--> Notifications
        +--> Broker Adapter
        +--> Database Writes
        +--> Audit Trail
```

## Why this design

A common failure mode in loop-based agent systems is repeated side effects: duplicate emails, repeated alerts, or multiple execution attempts for the same condition.

This project avoids that by keeping side effects outside the agent layer and enforcing:

* unique event IDs
* persisted action state
* idempotency checks
* transition-based triggers
* schema validation before execution

## Market Analyst approach

The Market Analyst is intended to use deterministic Python logic for core regime classification. LLMs may be used for explanation, commentary, or summarization, but not as the source of truth for regime state.

### Current regime inputs

* QQQ daily and weekly highs/lows
* VIX term structure
* VVIX behavior
* 2Y vs 5Y steepening/flattening
* Yen vs dollar carry proxy
* Fed reverse repo liquidity

### Prototype data sources

* `yfinance` for QQQ and VIX-related data
* `fredapi` / FRED for macro series such as:

  * `DGS2`
  * `DGS5`
  * `DEXJPUS`
  * `RRPONTSYD`

## Tech Stack

### Core

* Python 3.11+
* CrewAI
* Ollama
* Pydantic
* Pydantic Settings
* SQLAlchemy
* Pandas
* NumPy

### Data and utilities

* yfinance
* fredapi
* requests
* python-dotenv
* tenacity
* rich
* ipykernel

## Repository Layout

```text
zero-human-company/
├── README.md
├── requirements.txt
├── .env.example
├── config/
│   ├── settings.py
│   └── schemas.py
├── agents/
│   ├── ceo.py
│   ├── conscience.py
│   ├── market_analyst.py
│   ├── trade_analyst.py
│   ├── portfolio_manager.py
│   └── intern.py
├── services/
│   ├── market_data.py
│   ├── fred_data.py
│   ├── validation.py
│   ├── scoring.py
│   ├── notifications.py
│   ├── execution.py
│   └── idempotency.py
├── storage/
│   ├── db.py
│   └── models.py
├── workflows/
│   ├── daily_regime_run.py
│   └── trade_review_run.py
├── notebooks/
├── logs/
├── data/
└── tests/
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or .venv\Scripts\activate on Windows
python -m pip install --upgrade pip
pip install crewai ollama pandas numpy yfinance fredapi pydantic pydantic-settings python-dotenv tenacity rich SQLAlchemy requests ipykernel
```

## Configuration

Create a `.env` file for local configuration.

```env
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.1
FRED_API_KEY=your_fred_api_key_here
DATABASE_URL=sqlite:///zero_human_company.db
LOG_LEVEL=INFO
```

## Output Contract

All agents should return structured JSON.

```json
{
  "agent": "market_analyst",
  "event_id": "2026-03-20T08:30:00Z",
  "regime": "MIXED",
  "price_bias": "BEARISH",
  "vol_bias": "BULLISH",
  "confidence": 0.85,
  "factors": {
    "qqq_structure": "weak",
    "vix_term_structure": "contango",
    "vvix_signal": "falling",
    "curve_signal": "neutral",
    "yen_usd_signal": "risk_on",
    "rrp_signal": "liquidity_supportive"
  },
  "summary": "Market structure is mixed with weak price action but improving volatility conditions."
}
```

Python should validate every agent response before it is used downstream.

## Development workflow

1. Ingest market and macro data
2. Compute deterministic signals in Python
3. Pass structured context into agents
4. Receive JSON outputs
5. Validate outputs against schemas
6. Apply scoring and governance rules
7. Log all decisions and state changes
8. Allow side effects only after all checks pass

## Roadmap

* [ ] Define Pydantic schemas for every agent
* [ ] Build persistent event store
* [ ] Implement idempotent notification service
* [ ] Implement deterministic regime classification in Python
* [ ] Add governance scoring and veto logic
* [ ] Build trade recommendation review workflow
* [ ] Add portfolio-level exposure controls
* [ ] Add dashboarding for logs and system state
* [ ] Add broker integration after governance is stable

## Contributing

Contributions are welcome, especially around:

* architecture review
* schema design
* deterministic signal pipelines
* testing and observability
* risk controls and governance layers
* local-first deployment patterns

For larger changes, open an issue first to discuss the proposed direction.

## Disclaimer

This repository is for research and engineering purposes. It is not financial advice, and it should not be used for live trading without independent review, risk controls, and thorough testing.

## License

MIT
