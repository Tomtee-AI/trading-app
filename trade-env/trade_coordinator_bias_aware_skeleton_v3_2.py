#!/usr/bin/env python3
"""
trade_coordinator_bias_aware_skeleton_v2.py

Bias-aware Trade Coordinator skeleton.

What changed in v2
------------------
- Explicitly prefers the 3-state market_analyst_v2 regime model:
    * BULLISH_PRICE_BEARISH_VOL
    * NEUTRAL
    * BEARISH_PRICE_BULLISH_VOL
- Still accepts older market analyst payloads that may emit MIXED
- Converts older MIXED states into a coordinator-friendly normalized form
- Uses normalized regime state, not just raw price_bias, for alignment and ranking
- Makes neutral regimes more selective on outright directional strategies
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

REGIME_ENUM_INPUT = [
    "BULLISH_PRICE_BEARISH_VOL",
    "BEARISH_PRICE_BULLISH_VOL",
    "NEUTRAL",
    "MIXED",  # backward compatibility with pre-v2 payloads
]

ALIGNMENT_STATE_ENUM = [
    "ALIGNED",
    "PRICE_DOMINANT",
    "MIXED_OR_WEAK",
    "UNKNOWN",
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
                "macro_status": {"type": "string"},
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
                "regime": {"type": "string", "enum": REGIME_ENUM_INPUT},
                "alignment_state": {"type": "string", "enum": ALIGNMENT_STATE_ENUM},
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
                "regime": {"type": "string", "enum": REGIME_ENUM_INPUT},
                "alignment_state": {"type": "string", "enum": ALIGNMENT_STATE_ENUM},
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
        "schema_version": {"type": "string"},
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
        "schema_version": {"type": "string", "const": "1.1.0"},
        "event_id": {"type": "string", "format": "date-time"},
        "market_bias": {
            "type": "object",
            "additionalProperties": False,
            "required": ["model_preference", "short_term", "long_term", "coordinator_bias"],
            "properties": {
                "model_preference": {"type": "string", "const": "3_STATE_V2"},
                "short_term": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["price_bias", "vol_bias", "regime", "normalized_regime", "alignment_state", "confidence"],
                    "properties": {
                        "price_bias": {"type": "string"},
                        "vol_bias": {"type": "string"},
                        "regime": {"type": "string"},
                        "normalized_regime": {"type": "string", "enum": ["BULLISH_PRICE_BEARISH_VOL", "BEARISH_PRICE_BULLISH_VOL", "NEUTRAL"]},
                        "alignment_state": {"type": "string"},
                        "confidence": {"type": "number"},
                    },
                },
                "long_term": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["price_bias", "vol_bias", "regime", "normalized_regime", "alignment_state", "confidence"],
                    "properties": {
                        "price_bias": {"type": "string"},
                        "vol_bias": {"type": "string"},
                        "regime": {"type": "string"},
                        "normalized_regime": {"type": "string", "enum": ["BULLISH_PRICE_BEARISH_VOL", "BEARISH_PRICE_BULLISH_VOL", "NEUTRAL"]},
                        "alignment_state": {"type": "string"},
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
    prefer_v2_neutral_for_directional: bool = True


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)




def normalize_specialist_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Accept either a raw specialist payload or a wrapped report file whose
    coordinator-ready payload lives under top-level key "output".
    """
    required = {"agent", "event_id", "strategy_type", "candidates"}
    if isinstance(payload, dict) and required.issubset(payload.keys()):
        return payload

    nested = payload.get("output") if isinstance(payload, dict) else None
    if isinstance(nested, dict) and required.issubset(nested.keys()):
        return nested

    return payload

def validate_payload(payload: Dict[str, Any], schema: Dict[str, Any], label: str) -> None:
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors = sorted(validator.iter_errors(payload), key=lambda e: list(e.path))
    if errors:
        lines = []
        for err in errors[:10]:
            path = ".".join(str(p) for p in err.path) if err.path else "<root>"
            lines.append(f"{label}.{path}: {err.message}")
        raise ValueError("Schema validation failed:\n" + "\n".join(lines))


def normalize_regime(horizon_block: Dict[str, Any]) -> str:
    """Prefer the v2 three-state model, but collapse older MIXED inputs safely."""
    regime = horizon_block.get("regime", "NEUTRAL")
    price_bias = horizon_block.get("price_bias", "NEUTRAL")
    vol_bias = horizon_block.get("vol_bias", "NEUTRAL")

    if regime in {"BULLISH_PRICE_BEARISH_VOL", "BEARISH_PRICE_BULLISH_VOL", "NEUTRAL"}:
        return regime

    # backward compatibility with older payloads
    if regime == "MIXED":
        if price_bias == "BULLISH" and vol_bias == "BEARISH":
            return "BULLISH_PRICE_BEARISH_VOL"
        if price_bias == "BEARISH" and vol_bias == "BULLISH":
            return "BEARISH_PRICE_BULLISH_VOL"
        return "NEUTRAL"

    return "NEUTRAL"


def normalize_horizon_block(horizon_block: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "price_bias": horizon_block["price_bias"],
        "vol_bias": horizon_block["vol_bias"],
        "regime": horizon_block["regime"],
        "normalized_regime": normalize_regime(horizon_block),
        "alignment_state": horizon_block.get("alignment_state", "UNKNOWN"),
        "confidence": float(horizon_block["confidence"]),
    }


def is_v2_market_payload(market: Dict[str, Any]) -> bool:
    version = str(market.get("schema_version", ""))
    return version.startswith("2.")


def derive_coordinator_bias(market: Dict[str, Any]) -> str:
    st = normalize_horizon_block(market["short_term"])
    lt = normalize_horizon_block(market["long_term"])

    if st["normalized_regime"] == "BULLISH_PRICE_BEARISH_VOL" and st["confidence"] >= 0.60:
        return "SHORT_TERM_BULLISH_RISK_ON"
    if st["normalized_regime"] == "BEARISH_PRICE_BULLISH_VOL" and st["confidence"] >= 0.60:
        return "SHORT_TERM_BEARISH_RISK_OFF"
    if lt["normalized_regime"] == "BULLISH_PRICE_BEARISH_VOL" and lt["confidence"] >= 0.67:
        return "LONG_TERM_BULLISH"
    if lt["normalized_regime"] == "BEARISH_PRICE_BULLISH_VOL" and lt["confidence"] >= 0.67:
        return "LONG_TERM_BEARISH"

    # v2 preference: neutral is explicit, not mixed
    if st["normalized_regime"] == "NEUTRAL" and lt["normalized_regime"] == "NEUTRAL":
        if st["confidence"] < 0.58 and lt["confidence"] < 0.58:
            return "OBSERVATION_ONLY"
        return "NEUTRAL"

    return "MIXED"


def build_strategy_permissions(coordinator_bias: str) -> Dict[str, str]:
    base = {strategy: "ALLOW" for strategy in ALLOWED_STRATEGIES}

    if coordinator_bias == "SHORT_TERM_BULLISH_RISK_ON":
        base["earnings"] = "ALLOW"
        base["uoa"] = "ALLOW"
        base["divergence"] = "ALLOW"
        base["crack_spread"] = "ALLOW"
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
    elif coordinator_bias == "NEUTRAL":
        # Explicit v2 neutral preference: favor relative-value, de-emphasize outright event chasing.
        base["divergence"] = "ALLOW"
        base["crack_spread"] = "ALLOW"
        base["earnings"] = "DISFAVOR"
        base["uoa"] = "DISFAVOR"
    elif coordinator_bias == "OBSERVATION_ONLY":
        return {strategy: "BLOCK" for strategy in ALLOWED_STRATEGIES}

    return base


def build_ranking_weights(coordinator_bias: str) -> Dict[str, float]:
    weights = {strategy: 1.0 for strategy in ALLOWED_STRATEGIES}

    if coordinator_bias == "SHORT_TERM_BULLISH_RISK_ON":
        weights.update({"earnings": 1.15, "uoa": 1.10, "divergence": 1.00, "crack_spread": 1.00})
    elif coordinator_bias == "SHORT_TERM_BEARISH_RISK_OFF":
        weights.update({"earnings": 0.85, "uoa": 0.90, "divergence": 1.10, "crack_spread": 1.05})
    elif coordinator_bias == "LONG_TERM_BULLISH":
        weights.update({"earnings": 0.85, "uoa": 0.90, "divergence": 1.15, "crack_spread": 1.05})
    elif coordinator_bias == "LONG_TERM_BEARISH":
        weights.update({"earnings": 0.75, "uoa": 0.85, "divergence": 1.10, "crack_spread": 1.10})
    elif coordinator_bias == "NEUTRAL":
        weights.update({"earnings": 0.80, "uoa": 0.80, "divergence": 1.15, "crack_spread": 1.05})
    elif coordinator_bias == "OBSERVATION_ONLY":
        weights.update({s: 0.0 for s in ALLOWED_STRATEGIES})

    return weights


def compute_market_alignment(candidate: Dict[str, Any], market: Dict[str, Any], config: RuntimeConfig) -> float:
    horizon_block = normalize_horizon_block(market["short_term"] if candidate["horizon"] == "SHORT_TERM" else market["long_term"])
    regime = horizon_block["normalized_regime"]
    confidence = float(horizon_block["confidence"])
    direction = candidate["direction"]

    # Crack spread should be informed by analyst output as a refiner-margin signal.
    if candidate["strategy_type"] == "crack_spread":
        implementation = candidate.get("implementation", {})
        refiner_bias = implementation.get("refiner_bias") or (direction if direction in {"BULLISH", "BEARISH"} else None)
        if regime == "NEUTRAL":
            if refiner_bias in {"BULLISH", "BEARISH"}:
                return round(0.20 * confidence, 4)
            return round(0.10 * confidence, 4)
        if refiner_bias is None:
            return round(0.15 * confidence, 4)
        if regime == "BULLISH_PRICE_BEARISH_VOL":
            return round(0.60 * confidence, 4) if refiner_bias == "BULLISH" else round(-0.20 * confidence, 4)
        if regime == "BEARISH_PRICE_BULLISH_VOL":
            return round(0.60 * confidence, 4) if refiner_bias == "BEARISH" else round(-0.20 * confidence, 4)
        return round(0.10 * confidence, 4)

    # Explicit preference for the v2 3-state regime model.
    if regime == "NEUTRAL":
        if direction == "RELATIVE_VALUE":
            return round(0.35 * confidence, 4)
        if direction == "NEUTRAL":
            return round(0.15 * confidence, 4)
        if config.prefer_v2_neutral_for_directional:
            return round(-0.20 * confidence, 4)
        return round(-0.05 * confidence, 4)

    if direction == "RELATIVE_VALUE":
        return round(0.25 * confidence, 4)
    if direction == "NEUTRAL":
        return round(0.10 * confidence, 4)

    if regime == "BULLISH_PRICE_BEARISH_VOL":
        return round(1.0 * confidence, 4) if direction == "BULLISH" else round(-1.0 * confidence, 4)

    if regime == "BEARISH_PRICE_BULLISH_VOL":
        return round(1.0 * confidence, 4) if direction == "BEARISH" else round(-1.0 * confidence, 4)

    return 0.0


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
            alignment = compute_market_alignment(candidate, market, config)
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

    st = normalize_horizon_block(market["short_term"])
    lt = normalize_horizon_block(market["long_term"])
    coordinator_bias = derive_coordinator_bias(market)
    permissions = build_strategy_permissions(coordinator_bias)
    weights = build_ranking_weights(coordinator_bias)
    candidate_queue, no_trade = normalize_candidates(market, specialist_payloads, config)

    output = {
        "agent": "trade_coordinator",
        "schema_version": "1.1.0",
        "event_id": now_iso(),
        "market_bias": {
            "model_preference": "3_STATE_V2",
            "short_term": st,
            "long_term": lt,
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
                f"Coordinator is using the 3-state v2 regime model. Bias is {coordinator_bias}. "
                f"Top forwarded candidate is {top['candidate_id']} from {top['strategy_type']} with ranking score {top['ranking_score']:.2f}."
            )
        else:
            output["summary"] = f"Coordinator is using the 3-state v2 regime model. Bias is {coordinator_bias}. No candidates were forwarded."
    else:
        output["summary"] = no_trade["summary"]

    validate_payload(output, COORDINATOR_OUTPUT_SCHEMA, "trade_coordinator")
    return output


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bias-aware Trade Coordinator skeleton (v2 regime preferred)")
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
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    config = RuntimeConfig(
        min_specialist_confidence=args.min_specialist_confidence,
        min_market_alignment=args.min_market_alignment,
        max_queue_size=args.max_queue_size,
    )

    market = load_json(Path(args.market_analyst))
    specialist_payloads = [normalize_specialist_payload(load_json(Path(path))) for path in args.specialists]

    output = build_output(market, specialist_payloads, config)

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
