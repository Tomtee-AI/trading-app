#!/usr/bin/env python3
"""
trade_coordinator_bias_aware_skeleton.py

Bias-aware Trade Coordinator skeleton.

What it does
------------
- Consumes a market analyst JSON payload with:
    - short_term
    - long_term
    - components
    - summary
- Consumes one or more specialist trade agent payloads
- Applies deterministic market-bias gating and ranking
- Emits a normalized coordinator decision payload
- Validates both input and output with JSON Schema
- Writes a timestamped JSON report to reports/

This is a coordinator only.
It does not generate trades and does not place trades.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from jsonschema import Draft202012Validator, FormatChecker

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_STRATEGIES = [
    "divergence",
    "earnings",
    "uoa",
    "crack_spread",
]

BIAS_LABELS = [
    "SHORT_TERM_BULLISH_RISK_ON",
    "SHORT_TERM_BEARISH_RISK_OFF",
    "LONG_TERM_BULLISH",
    "LONG_TERM_BEARISH",
    "MIXED",
    "NEUTRAL",
    "OBSERVATION_ONLY",
]

COORDINATOR_REASON_CODES = [
    "PASSED",
    "MARKET_BIAS_BLOCKED",
    "LOW_MARKET_ALIGNMENT",
    "LOW_SPECIALIST_CONFIDENCE",
    "MISSING_REQUIRED_FIELDS",
    "UNKNOWN_STRATEGY",
    "DUPLICATE_CANDIDATE_ID",
]

NO_TRADE_REASON_CODES = [
    "NO_SPECIALIST_INPUTS",
    "NO_CANDIDATES_SUBMITTED",
    "ALL_CANDIDATES_BLOCKED_BY_MARKET_BIAS",
    "ALL_CANDIDATES_FAILED_ALIGNMENT",
    "ALL_CANDIDATES_FAILED_CONFIDENCE",
    "ALL_CANDIDATES_INVALID",
    "MARKET_ANALYST_INVALID",
]

MARKET_ANALYST_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": ["agent", "event_id", "short_term", "long_term", "components", "summary"],
    "properties": {
        "agent": {"type": "string", "const": "market_analyst"},
        "event_id": {"type": "string"},
        "schema_version": {"type": "string"},
        "data_quality": {
            "type": "object",
            "additionalProperties": True,
            "properties": {
                "cache_hit": {"type": "boolean"},
                "fetched_at_market": {"type": "string"},
                "fetched_at_macro": {"type": "string"},
                "missing_symbols": {"type": "array", "items": {"type": "string"}},
                "available_symbols": {"type": "array", "items": {"type": "string"}},
            },
        },
        "summary": {"type": "string"},
        "short_term": {
            "type": "object",
            "additionalProperties": True,
            "required": ["window_days", "price_bias", "vol_bias", "regime", "confidence", "scores", "factors"],
            "properties": {
                "window_days": {"type": "string"},
                "price_bias": {"type": "string", "enum": ["BULLISH", "BEARISH", "NEUTRAL"]},
                "vol_bias": {"type": "string", "enum": ["BULLISH", "BEARISH", "NEUTRAL"]},
                "regime": {"type": "string", "enum": ["BULLISH_PRICE_BEARISH_VOL", "BEARISH_PRICE_BULLISH_VOL", "NEUTRAL", "MIXED"]},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "scores": {"type": "object"},
                "factors": {"type": "object"},
            },
        },
        "long_term": {
            "type": "object",
            "additionalProperties": True,
            "required": ["window_days", "price_bias", "vol_bias", "regime", "confidence", "scores", "factors"],
            "properties": {
                "window_days": {"type": "string"},
                "price_bias": {"type": "string", "enum": ["BULLISH", "BEARISH", "NEUTRAL"]},
                "vol_bias": {"type": "string", "enum": ["BULLISH", "BEARISH", "NEUTRAL"]},
                "regime": {"type": "string", "enum": ["BULLISH_PRICE_BEARISH_VOL", "BEARISH_PRICE_BULLISH_VOL", "NEUTRAL", "MIXED"]},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "scores": {"type": "object"},
                "factors": {"type": "object"},
            },
        },
        "components": {"type": "object"},
    },
}

SPECIALIST_CANDIDATE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "candidate_id",
        "strategy_type",
        "direction",
        "horizon",
        "structure_family",
        "summary",
        "confidence",
        "fit_score",
        "thesis_tags",
        "risk_flags",
        "implementation",
    ],
    "properties": {
        "candidate_id": {"type": "string", "minLength": 1},
        "strategy_type": {"type": "string", "enum": ALLOWED_STRATEGIES},
        "direction": {"type": "string", "enum": ["BULLISH", "BEARISH", "NEUTRAL", "RELATIVE_VALUE"]},
        "horizon": {"type": "string", "enum": ["SHORT_TERM", "LONG_TERM"]},
        "structure_family": {"type": "string"},
        "summary": {"type": "string", "minLength": 1},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "fit_score": {"type": "number", "minimum": 0, "maximum": 1},
        "thesis_tags": {"type": "array", "items": {"type": "string"}},
        "risk_flags": {"type": "array", "items": {"type": "string"}},
        "implementation": {"type": "object", "additionalProperties": True},
    },
}

SPECIALIST_AGENT_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": ["agent", "event_id", "strategy_type", "candidates"],
    "properties": {
        "agent": {"type": "string"},
        "event_id": {"type": "string"},
        "strategy_type": {"type": "string", "enum": ALLOWED_STRATEGIES},
        "candidates": {"type": "array", "items": SPECIALIST_CANDIDATE_SCHEMA},
        "summary": {"type": "string"},
    },
}

COORDINATOR_OUTPUT_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "agent",
        "schema_version",
        "event_id",
        "market_bias",
        "strategy_permissions",
        "ranking_weights",
        "candidate_queue",
        "no_trade",
        "summary",
    ],
    "properties": {
        "agent": {"type": "string", "const": "trade_coordinator"},
        "schema_version": {"type": "string", "const": "1.0.0"},
        "event_id": {"type": "string", "format": "date-time"},
        "market_bias": {
            "type": "object",
            "additionalProperties": False,
            "required": ["short_term", "long_term", "coordinator_bias"],
            "properties": {
                "short_term": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["price_bias", "vol_bias", "regime", "confidence"],
                    "properties": {
                        "price_bias": {"type": "string"},
                        "vol_bias": {"type": "string"},
                        "regime": {"type": "string"},
                        "confidence": {"type": "number"},
                    },
                },
                "long_term": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["price_bias", "vol_bias", "regime", "confidence"],
                    "properties": {
                        "price_bias": {"type": "string"},
                        "vol_bias": {"type": "string"},
                        "regime": {"type": "string"},
                        "confidence": {"type": "number"},
                    },
                },
                "coordinator_bias": {"type": "string", "enum": BIAS_LABELS},
            },
        },
        "strategy_permissions": {
            "type": "object",
            "additionalProperties": False,
            "required": ALLOWED_STRATEGIES,
            "properties": {k: {"type": "string", "enum": ["ALLOW", "DISFAVOR", "BLOCK"]} for k in ALLOWED_STRATEGIES},
        },
        "ranking_weights": {
            "type": "object",
            "additionalProperties": False,
            "required": ALLOWED_STRATEGIES,
            "properties": {k: {"type": "number", "minimum": 0.0, "maximum": 2.0} for k in ALLOWED_STRATEGIES},
        },
        "candidate_queue": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "candidate_id",
                    "strategy_type",
                    "direction",
                    "horizon",
                    "market_alignment_score",
                    "ranking_score",
                    "status",
                    "reason_code",
                    "summary",
                    "implementation",
                ],
                "properties": {
                    "candidate_id": {"type": "string"},
                    "strategy_type": {"type": "string", "enum": ALLOWED_STRATEGIES},
                    "direction": {"type": "string"},
                    "horizon": {"type": "string"},
                    "market_alignment_score": {"type": "number", "minimum": -1.0, "maximum": 1.0},
                    "ranking_score": {"type": "number"},
                    "status": {"type": "string", "enum": ["FORWARD", "SUPPRESS", "BLOCK"]},
                    "reason_code": {"type": "string", "enum": COORDINATOR_REASON_CODES},
                    "summary": {"type": "string"},
                    "implementation": {"type": "object", "additionalProperties": True},
                },
            },
        },
        "no_trade": {
            "anyOf": [
                {"type": "null"},
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["reason_codes", "summary"],
                    "properties": {
                        "reason_codes": {
                            "type": "array",
                            "minItems": 1,
                            "items": {"type": "string", "enum": NO_TRADE_REASON_CODES},
                            "uniqueItems": True,
                        },
                        "summary": {"type": "string"},
                    },
                },
            ]
        },
        "summary": {"type": "string"},
    },
}


@dataclass
class RuntimeConfig:
    min_specialist_confidence: float = 0.55
    min_market_alignment: float = -0.10
    max_queue_size: int = 20


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_payload(payload: Dict[str, Any], schema: Dict[str, Any], label: str) -> None:
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors = sorted(validator.iter_errors(payload), key=lambda e: list(e.path))
    if errors:
        lines = []
        for err in errors[:10]:
            path = ".".join(str(p) for p in err.path) if err.path else "<root>"
            lines.append(f"{label}.{path}: {err.message}")
        raise ValueError("Schema validation failed:\n" + "\n".join(lines))


def derive_coordinator_bias(market: Dict[str, Any]) -> str:
    st = market["short_term"]
    lt = market["long_term"]

    if st["regime"] == "BULLISH_PRICE_BEARISH_VOL" and st["confidence"] >= 0.65:
        return "SHORT_TERM_BULLISH_RISK_ON"
    if st["regime"] == "BEARISH_PRICE_BULLISH_VOL" and st["confidence"] >= 0.65:
        return "SHORT_TERM_BEARISH_RISK_OFF"
    if lt["price_bias"] == "BULLISH" and lt["confidence"] >= 0.65:
        return "LONG_TERM_BULLISH"
    if lt["price_bias"] == "BEARISH" and lt["confidence"] >= 0.65:
        return "LONG_TERM_BEARISH"
    if st["regime"] == "MIXED" and lt["regime"] == "MIXED":
        return "OBSERVATION_ONLY"
    if st["regime"] == "NEUTRAL" and lt["regime"] == "NEUTRAL":
        return "NEUTRAL"
    return "MIXED"


def build_strategy_permissions(coordinator_bias: str) -> Dict[str, str]:
    base = {strategy: "ALLOW" for strategy in ALLOWED_STRATEGIES}

    if coordinator_bias == "SHORT_TERM_BULLISH_RISK_ON":
        base["earnings"] = "ALLOW"
        base["uoa"] = "ALLOW"
        base["divergence"] = "ALLOW"
        base["crack_spread"] = "DISFAVOR"
    elif coordinator_bias == "SHORT_TERM_BEARISH_RISK_OFF":
        base["earnings"] = "DISFAVOR"
        base["uoa"] = "DISFAVOR"
        base["divergence"] = "ALLOW"
        base["crack_spread"] = "ALLOW"
    elif coordinator_bias == "LONG_TERM_BULLISH":
        base["divergence"] = "ALLOW"
        base["crack_spread"] = "ALLOW"
        base["earnings"] = "DISFAVOR"
        base["uoa"] = "DISFAVOR"
    elif coordinator_bias == "LONG_TERM_BEARISH":
        base["divergence"] = "ALLOW"
        base["crack_spread"] = "ALLOW"
        base["earnings"] = "BLOCK"
        base["uoa"] = "DISFAVOR"
    elif coordinator_bias == "OBSERVATION_ONLY":
        return {strategy: "BLOCK" for strategy in ALLOWED_STRATEGIES}

    return base


def build_ranking_weights(coordinator_bias: str) -> Dict[str, float]:
    weights = {strategy: 1.0 for strategy in ALLOWED_STRATEGIES}

    if coordinator_bias == "SHORT_TERM_BULLISH_RISK_ON":
        weights.update({"earnings": 1.15, "uoa": 1.10, "divergence": 1.00, "crack_spread": 0.90})
    elif coordinator_bias == "SHORT_TERM_BEARISH_RISK_OFF":
        weights.update({"earnings": 0.85, "uoa": 0.90, "divergence": 1.10, "crack_spread": 1.05})
    elif coordinator_bias == "LONG_TERM_BULLISH":
        weights.update({"earnings": 0.85, "uoa": 0.90, "divergence": 1.15, "crack_spread": 1.05})
    elif coordinator_bias == "LONG_TERM_BEARISH":
        weights.update({"earnings": 0.75, "uoa": 0.85, "divergence": 1.10, "crack_spread": 1.10})
    elif coordinator_bias == "OBSERVATION_ONLY":
        weights.update({s: 0.0 for s in ALLOWED_STRATEGIES})

    return weights


def compute_market_alignment(candidate: Dict[str, Any], market: Dict[str, Any]) -> float:
    horizon_block = market["short_term"] if candidate["horizon"] == "SHORT_TERM" else market["long_term"]
    price_bias = horizon_block["price_bias"]
    confidence = float(horizon_block["confidence"])

    if candidate["direction"] == "RELATIVE_VALUE":
        return round(0.25 * confidence, 4)
    if candidate["direction"] == "NEUTRAL":
        return round(0.10 * confidence, 4)
    if candidate["direction"] == price_bias:
        return round(1.0 * confidence, 4)
    if price_bias == "NEUTRAL":
        return round(-0.05 * confidence, 4)
    return round(-1.0 * confidence, 4)


def normalize_candidates(
    market: Dict[str, Any],
    specialist_payloads: List[Dict[str, Any]],
    config: RuntimeConfig,
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    coordinator_bias = derive_coordinator_bias(market)
    permissions = build_strategy_permissions(coordinator_bias)
    weights = build_ranking_weights(coordinator_bias)

    if not specialist_payloads:
        return [], {
            "reason_codes": ["NO_SPECIALIST_INPUTS"],
            "summary": "No specialist agent payloads were supplied to the coordinator.",
        }

    seen_ids = set()
    queue: List[Dict[str, Any]] = []
    total_submitted = 0
    blocked = 0
    bad_alignment = 0
    low_conf = 0
    invalid = 0

    for payload in specialist_payloads:
        strategy_type = payload["strategy_type"]
        for candidate in payload.get("candidates", []):
            total_submitted += 1
            cid = candidate.get("candidate_id")
            if cid in seen_ids:
                invalid += 1
                queue.append({
                    "candidate_id": cid or "UNKNOWN",
                    "strategy_type": strategy_type,
                    "direction": candidate.get("direction", "UNKNOWN"),
                    "horizon": candidate.get("horizon", "UNKNOWN"),
                    "market_alignment_score": 0.0,
                    "ranking_score": 0.0,
                    "status": "BLOCK",
                    "reason_code": "DUPLICATE_CANDIDATE_ID",
                    "summary": candidate.get("summary", "Duplicate candidate id."),
                    "implementation": candidate.get("implementation", {}),
                })
                continue
            seen_ids.add(cid)

            if strategy_type not in ALLOWED_STRATEGIES:
                invalid += 1
                queue.append({
                    "candidate_id": cid,
                    "strategy_type": strategy_type,
                    "direction": candidate["direction"],
                    "horizon": candidate["horizon"],
                    "market_alignment_score": 0.0,
                    "ranking_score": 0.0,
                    "status": "BLOCK",
                    "reason_code": "UNKNOWN_STRATEGY",
                    "summary": candidate["summary"],
                    "implementation": candidate["implementation"],
                })
                continue

            permission = permissions[strategy_type]
            alignment = compute_market_alignment(candidate, market)
            confidence = float(candidate["confidence"])
            ranking_score = round((0.60 * float(candidate["fit_score"]) + 0.40 * alignment) * weights[strategy_type], 4)

            status = "FORWARD"
            reason_code = "PASSED"

            if permission == "BLOCK":
                status = "BLOCK"
                reason_code = "MARKET_BIAS_BLOCKED"
                blocked += 1
            elif confidence < config.min_specialist_confidence:
                status = "SUPPRESS"
                reason_code = "LOW_SPECIALIST_CONFIDENCE"
                low_conf += 1
            elif alignment < config.min_market_alignment:
                status = "SUPPRESS"
                reason_code = "LOW_MARKET_ALIGNMENT"
                bad_alignment += 1

            queue.append({
                "candidate_id": cid,
                "strategy_type": strategy_type,
                "direction": candidate["direction"],
                "horizon": candidate["horizon"],
                "market_alignment_score": alignment,
                "ranking_score": ranking_score,
                "status": status,
                "reason_code": reason_code,
                "summary": candidate["summary"],
                "implementation": candidate["implementation"],
            })

    forwarded = [row for row in queue if row["status"] == "FORWARD"]
    forwarded.sort(key=lambda row: row["ranking_score"], reverse=True)
    forwarded = forwarded[: config.max_queue_size]

    if forwarded:
        return forwarded, None

    reason_codes: List[str] = []
    if total_submitted == 0:
        reason_codes.append("NO_CANDIDATES_SUBMITTED")
    if blocked == total_submitted and total_submitted > 0:
        reason_codes.append("ALL_CANDIDATES_BLOCKED_BY_MARKET_BIAS")
    if bad_alignment > 0 and blocked + bad_alignment + low_conf + invalid == total_submitted:
        reason_codes.append("ALL_CANDIDATES_FAILED_ALIGNMENT")
    if low_conf > 0 and blocked + bad_alignment + low_conf + invalid == total_submitted:
        reason_codes.append("ALL_CANDIDATES_FAILED_CONFIDENCE")
    if invalid > 0 and invalid == total_submitted:
        reason_codes.append("ALL_CANDIDATES_INVALID")
    if not reason_codes:
        reason_codes.append("ALL_CANDIDATES_INVALID")

    return queue, {
        "reason_codes": sorted(set(reason_codes)),
        "summary": (
            f"No trades forwarded. submitted={total_submitted}, blocked={blocked}, "
            f"low_alignment={bad_alignment}, low_confidence={low_conf}, invalid={invalid}."
        ),
    }


def build_output(
    market: Dict[str, Any],
    specialist_payloads: List[Dict[str, Any]],
    config: RuntimeConfig,
) -> Dict[str, Any]:
    validate_payload(market, MARKET_ANALYST_SCHEMA, "market_analyst")
    for i, payload in enumerate(specialist_payloads):
        validate_payload(payload, SPECIALIST_AGENT_SCHEMA, f"specialist[{i}]")

    coordinator_bias = derive_coordinator_bias(market)
    permissions = build_strategy_permissions(coordinator_bias)
    weights = build_ranking_weights(coordinator_bias)

    candidate_queue, no_trade = normalize_candidates(market, specialist_payloads, config)

    output = {
        "agent": "trade_coordinator",
        "schema_version": "1.0.0",
        "event_id": now_iso(),
        "market_bias": {
            "short_term": {
                "price_bias": market["short_term"]["price_bias"],
                "vol_bias": market["short_term"]["vol_bias"],
                "regime": market["short_term"]["regime"],
                "confidence": market["short_term"]["confidence"],
            },
            "long_term": {
                "price_bias": market["long_term"]["price_bias"],
                "vol_bias": market["long_term"]["vol_bias"],
                "regime": market["long_term"]["regime"],
                "confidence": market["long_term"]["confidence"],
            },
            "coordinator_bias": coordinator_bias,
        },
        "strategy_permissions": permissions,
        "ranking_weights": weights,
        "candidate_queue": candidate_queue,
        "no_trade": no_trade,
        "summary": "",
    }

    if no_trade is None:
        top = candidate_queue[0] if candidate_queue else None
        if top:
            output["summary"] = (
                f"Coordinator bias is {coordinator_bias}. Top forwarded candidate is {top['candidate_id']} "
                f"from {top['strategy_type']} with ranking score {top['ranking_score']:.2f}."
            )
        else:
            output["summary"] = f"Coordinator bias is {coordinator_bias}. No candidates were forwarded."
    else:
        output["summary"] = no_trade["summary"]

    validate_payload(output, COORDINATOR_OUTPUT_SCHEMA, "trade_coordinator")
    return output


def add_llm_summary(output: Dict[str, Any]) -> Dict[str, Any]:
    llm = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.2,
    )
    prompt = (
        "You are a concise trade coordinator narrator. Return ONLY valid JSON with one key named summary. "
        "Keep it to 2 sentences max and do not change the deterministic view.\n\n"
        f"{json.dumps({'market_bias': output['market_bias'], 'summary': output['summary']}, indent=2)}\n\n"
        'Output format: {"summary": "..."}'
    )
    try:
        response = llm.invoke(prompt)
        parsed = json.loads(response.content)
        if isinstance(parsed, dict) and "summary" in parsed:
            output["summary"] = parsed["summary"]
            validate_payload(output, COORDINATOR_OUTPUT_SCHEMA, "trade_coordinator")
    except Exception:
        pass
    return output


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bias-aware Trade Coordinator skeleton")
    parser.add_argument("--market-analyst", required=True, help="Path to market analyst JSON")
    parser.add_argument(
        "--specialists",
        nargs="*",
        default=[],
        help="Paths to specialist agent JSON payloads",
    )
    parser.add_argument("--min-specialist-confidence", type=float, default=0.55)
    parser.add_argument("--min-market-alignment", type=float, default=-0.10)
    parser.add_argument("--max-queue-size", type=int, default=20)
    parser.add_argument("--use-llm-summary", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    config = RuntimeConfig(
        min_specialist_confidence=args.min_specialist_confidence,
        min_market_alignment=args.min_market_alignment,
        max_queue_size=args.max_queue_size,
    )

    market = load_json(Path(args.market_analyst))
    specialist_payloads = [load_json(Path(path)) for path in args.specialists]

    output = build_output(market, specialist_payloads, config)

    if args.use_llm_summary and USE_LLM_SUMMARY:
        output = add_llm_summary(output)

    print(json.dumps(output, indent=2))

    report_path = REPORTS_DIR / f"trade_coordinator_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\n[INFO] Report written to {report_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {exc}")
        raise SystemExit(1)
