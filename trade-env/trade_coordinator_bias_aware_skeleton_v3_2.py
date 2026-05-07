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
        "trade_analysis",
        "summary",
    ],
    "properties": {
        "agent": {"type": "string", "const": "trade_coordinator"},
        "schema_version": {"type": "string", "const": "1.2.0"},
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
        "trade_analysis": {"type": "object", "additionalProperties": True},
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



# -------------------- HUMAN-READABLE TRADE ANALYSIS --------------------
def _as_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    """Safely coerce analyst-provided numeric fields without throwing."""
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _nested_get(obj: Dict[str, Any], *path: str, default: Any = None) -> Any:
    """Small helper for reading deeply nested optional analyst fields."""
    cur: Any = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _money(value: Optional[float]) -> Optional[str]:
    if value is None:
        return None
    return f"${value:,.2f}"


def _pct(value: Optional[float]) -> Optional[str]:
    if value is None:
        return None
    return f"{value:.1%}"


def _round_or_none(value: Optional[float], digits: int = 2) -> Optional[float]:
    return round(float(value), digits) if value is not None else None


def _candidate_symbol(candidate: Dict[str, Any]) -> Optional[str]:
    impl = candidate.get("implementation", {})
    if candidate.get("strategy_type") == "uoa":
        return _nested_get(impl, "symbol")
    if candidate.get("strategy_type") == "earnings":
        return _nested_get(impl, "symbol")
    if candidate.get("strategy_type") == "crack_spread":
        return _nested_get(impl, "selected_refiner")
    if candidate.get("strategy_type") == "divergence":
        return _nested_get(impl, "signal_source", "pair_label")
    return None


def _extract_uoa_review(candidate: Dict[str, Any]) -> Dict[str, Any]:
    impl = candidate.get("implementation", {})
    selected = impl.get("selected_trade", {})
    contract = impl.get("contract", {})
    quote = impl.get("quote", {})
    flow = impl.get("observed_flow", {})
    rr = impl.get("reward_risk", {})
    catalyst = impl.get("catalyst", {})

    return {
        "symbol": impl.get("symbol"),
        "company_name": impl.get("company_name"),
        "trade_structure": selected.get("preferred_structure"),
        "contract": contract.get("contract_string"),
        "expiration": selected.get("expiration"),
        "entry_price": _as_float(selected.get("estimated_entry_price")),
        "max_loss_dollars": _as_float(selected.get("maximum_loss_per_contract")),
        "estimated_reward_to_risk": _as_float(rr.get("estimated_reward_to_risk")),
        "reward_risk_meets_target": bool(rr.get("reward_risk_meets_target", False)),
        "premium": _as_float(flow.get("premium")),
        "volume": _as_float(flow.get("volume")),
        "open_interest": _as_float(flow.get("open_interest")),
        "volume_open_interest_ratio": _as_float(flow.get("volume_open_interest_ratio")),
        "ask_side_ratio": _as_float(_nested_get(flow, "side_ratios", "ask_side_ratio")),
        "bid_ask_spread_pct": _as_float(quote.get("bid_ask_spread_pct")),
        "dte": _as_float(contract.get("dte")),
        "catalyst_tags": catalyst.get("catalyst_tags", []),
        "next_earnings_date": catalyst.get("next_earnings_date"),
        "model_note": rr.get("move_model"),
    }


def _extract_earnings_review(candidate: Dict[str, Any]) -> Dict[str, Any]:
    impl = candidate.get("implementation", {})
    preferred = impl.get("preferred_trade", {})
    econ = preferred.get("economics", {})
    vol_value = impl.get("vol_value", {})
    debit = _as_float(econ.get("estimated_max_loss"))
    max_loss_dollars = debit * 100.0 if debit is not None else None

    return {
        "symbol": impl.get("symbol"),
        "trade_structure": preferred.get("preferred_structure"),
        "contract": f"{impl.get('symbol')} {preferred.get('expiration')} {preferred.get('preferred_structure')}",
        "expiration": preferred.get("expiration"),
        "entry_price": _as_float(econ.get("estimated_debit")),
        "max_loss_dollars": max_loss_dollars,
        "estimated_reward_to_risk": _as_float(econ.get("estimated_reward_to_risk")),
        "reward_risk_meets_target": bool(econ.get("reward_risk_meets_target", False)),
        "earnings_date": impl.get("earnings_date"),
        "earnings_time": impl.get("earnings_time"),
        "historical_to_implied_ratio": _as_float(vol_value.get("historical_to_implied_ratio")),
        "historical_avg_abs_move": _as_float(vol_value.get("historical_avg_abs_move")),
        "selected_implied_move": _as_float(vol_value.get("selected_implied_move")),
        "vol_value_state": vol_value.get("vol_value_state"),
        "date_validation_status": impl.get("date_validation_status"),
    }


def _spread_economics(spread: Optional[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    if not isinstance(spread, dict):
        return {"debit": None, "max_loss": None, "max_reward": None, "reward_to_risk": None}
    return {
        "debit": _as_float(spread.get("estimated_entry_debit")),
        "max_loss": _as_float(spread.get("estimated_max_loss")),
        "max_reward": _as_float(spread.get("estimated_max_reward")),
        "reward_to_risk": _as_float(spread.get("estimated_reward_to_risk")),
    }


def _extract_divergence_review(candidate: Dict[str, Any]) -> Dict[str, Any]:
    impl = candidate.get("implementation", {})
    signal = impl.get("signal_source", {}) or impl.get("pair_metrics", {})
    templates = impl.get("trade_templates", {})
    short_spread = templates.get("short_leg_exact_spread")
    long_spread = templates.get("long_leg_exact_spread")
    short_econ = _spread_economics(short_spread)
    long_econ = _spread_economics(long_spread)

    total_debit = None
    total_reward = None
    if short_econ["max_loss"] is not None or long_econ["max_loss"] is not None:
        total_debit = (short_econ["max_loss"] or 0.0) + (long_econ["max_loss"] or 0.0)
    if short_econ["max_reward"] is not None or long_econ["max_reward"] is not None:
        total_reward = (short_econ["max_reward"] or 0.0) + (long_econ["max_reward"] or 0.0)
    total_rr = (total_reward / total_debit) if total_debit and total_debit > 0 and total_reward is not None else None

    short_label = None
    if isinstance(short_spread, dict):
        short_label = f"{short_spread.get('symbol')} {short_spread.get('structure')}"
    long_label = None
    if isinstance(long_spread, dict):
        long_label = f"{long_spread.get('symbol')} {long_spread.get('structure')}"

    return {
        "symbol": signal.get("pair_label"),
        "trade_structure": "RELATIVE_VALUE_OPTION_PAIR",
        "contract": " / ".join([x for x in [short_label, long_label] if x]),
        "expiration": short_spread.get("expiration") if isinstance(short_spread, dict) else None,
        "entry_price": total_debit,
        "max_loss_dollars": total_debit * 100.0 if total_debit is not None else None,
        "estimated_reward_to_risk": total_rr,
        "reward_risk_meets_target": bool(total_rr is not None and total_rr >= 5.0),
        "pair_label": signal.get("pair_label"),
        "correlation": _as_float(signal.get("correlation")),
        "current_zscore": _as_float(signal.get("current_zscore")),
        "expensive_alias": signal.get("expensive_alias"),
        "cheap_alias": signal.get("cheap_alias"),
        "short_leg_reward_to_risk": short_econ["reward_to_risk"],
        "long_leg_reward_to_risk": long_econ["reward_to_risk"],
    }


def _extract_generic_review(candidate: Dict[str, Any]) -> Dict[str, Any]:
    impl = candidate.get("implementation", {})
    return {
        "symbol": _candidate_symbol(candidate),
        "trade_structure": candidate.get("structure_family"),
        "contract": candidate.get("summary"),
        "expiration": None,
        "entry_price": None,
        "max_loss_dollars": None,
        "estimated_reward_to_risk": None,
        "reward_risk_meets_target": False,
        "implementation_keys": sorted(impl.keys()) if isinstance(impl, dict) else [],
    }


def extract_trade_review_fields(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize specialist-specific implementation details into common review fields."""
    strategy = candidate.get("strategy_type")
    if strategy == "uoa":
        return _extract_uoa_review(candidate)
    if strategy == "earnings":
        return _extract_earnings_review(candidate)
    if strategy == "divergence":
        return _extract_divergence_review(candidate)
    return _extract_generic_review(candidate)


def build_candidate_review(candidate: Dict[str, Any], rank: int, duplicate_symbol_count: int = 1) -> Dict[str, Any]:
    """
    Create a concise, deterministic trade-review object for one coordinator candidate.

    This is intentionally not an LLM summary. It uses only fields already emitted by
    the specialist analysts and coordinator, making the review auditable and stable.
    """
    fields = extract_trade_review_fields(candidate)
    strategy = candidate.get("strategy_type")
    symbol = fields.get("symbol") or "UNKNOWN"
    rr = _as_float(fields.get("estimated_reward_to_risk"))
    max_loss = _as_float(fields.get("max_loss_dollars"))
    entry = _as_float(fields.get("entry_price"))
    ranking_score = _as_float(candidate.get("ranking_score"), 0.0) or 0.0
    alignment = _as_float(candidate.get("market_alignment_score"), 0.0) or 0.0

    strengths: List[str] = []
    cautions: List[str] = []

    if alignment >= 0.50:
        strengths.append("Strongly aligned with current market regime.")
    elif alignment >= 0.10:
        strengths.append("Acceptably aligned with current market regime.")
    elif alignment < 0:
        cautions.append("Fights the current market regime.")

    if rr is not None:
        if rr >= 5.0:
            strengths.append("Meets the 5:1 style reward/risk target.")
        elif rr >= 3.0:
            strengths.append("Has attractive defined-risk asymmetry above 3:1.")
        elif rr < 1.5:
            cautions.append("Reward/risk is modest and below the preferred asymmetric target.")
        else:
            cautions.append("Reward/risk is acceptable but below the preferred 5:1 target.")
    else:
        cautions.append("Reward/risk could not be estimated from analyst output.")

    if strategy == "uoa":
        premium = _as_float(fields.get("premium"))
        ask_ratio = _as_float(fields.get("ask_side_ratio"))
        vol_oi = _as_float(fields.get("volume_open_interest_ratio"))
        spread = _as_float(fields.get("bid_ask_spread_pct"))
        dte = _as_float(fields.get("dte"))
        catalyst_tags = fields.get("catalyst_tags") or []
        if premium is not None and premium >= 250_000:
            strengths.append(f"Large observed premium: {_money(premium)}.")
        if ask_ratio is not None and ask_ratio >= 0.80:
            strengths.append("Flow appears heavily ask-side / buyer initiated.")
        if vol_oi is not None and vol_oi >= 2.0:
            strengths.append("Volume is meaningfully larger than open interest.")
        if catalyst_tags:
            strengths.append("Has near-term catalyst tags: " + ", ".join(str(x) for x in catalyst_tags) + ".")
        if spread is not None and spread >= 0.25:
            cautions.append("Bid/ask spread is wide; use strict limit orders.")
        if dte is not None and dte <= 5:
            cautions.append("Very short-dated option; timing risk is high.")
        if str(fields.get("model_note", "")).startswith("fallback"):
            cautions.append("Reward/risk uses fallback move model because IV/Greeks were missing.")

    elif strategy == "earnings":
        ratio = _as_float(fields.get("historical_to_implied_ratio"))
        hist = _as_float(fields.get("historical_avg_abs_move"))
        implied = _as_float(fields.get("selected_implied_move"))
        if ratio is not None and ratio >= 1.25:
            strengths.append("Historical earnings move is larger than the current implied move.")
        if fields.get("date_validation_status") == "CONFIRMED_3_SOURCE":
            strengths.append("Earnings date is confirmed by multiple sources.")
        if hist is not None and implied is not None:
            strengths.append(f"Historical avg move {_pct(hist)} vs implied move {_pct(implied)}.")
        if rr is not None and rr <= 1.1:
            cautions.append("ATM straddle economics are roughly 1:1 unless the move exceeds implied range.")

    elif strategy == "divergence":
        corr = _as_float(fields.get("correlation"))
        z = _as_float(fields.get("current_zscore"))
        short_rr = _as_float(fields.get("short_leg_reward_to_risk"))
        long_rr = _as_float(fields.get("long_leg_reward_to_risk"))
        if corr is not None and corr >= 0.80:
            strengths.append("Pair correlation is high enough for a cleaner relative-value signal.")
        if z is not None and abs(z) >= 2.0:
            strengths.append("Spread z-score is meaningfully extended.")
        if short_rr is not None and long_rr is not None and min(short_rr, long_rr) < 1.25:
            cautions.append("One leg has weak spread economics and may drag down the pair trade.")

    if duplicate_symbol_count > 1:
        cautions.append(f"Duplicate exposure: {symbol} appears {duplicate_symbol_count} times in the forwarded queue.")

    if max_loss is not None and max_loss > 500:
        cautions.append("One-lot max loss is above $500; position sizing needs extra care.")

    actionability_score = round(
        ranking_score
        + (0.08 if rr is not None and rr >= 3.0 else 0.0)
        + (0.06 if max_loss is not None and max_loss <= 150 else 0.0)
        - (0.05 * min(len(cautions), 4)),
        4,
    )

    thesis = candidate.get("summary", "")
    plain_english = (
        f"#{rank}: {symbol} from {strategy}. {thesis} "
        f"Estimated R/R={rr:.2f}:1." if rr is not None else f"#{rank}: {symbol} from {strategy}. {thesis}"
    )

    return {
        "rank": rank,
        "candidate_id": candidate.get("candidate_id"),
        "strategy_type": strategy,
        "symbol": symbol,
        "direction": candidate.get("direction"),
        "ranking_score": ranking_score,
        "market_alignment_score": alignment,
        "trade_structure": fields.get("trade_structure"),
        "contract": fields.get("contract"),
        "expiration": fields.get("expiration"),
        "entry_price": _round_or_none(entry, 4),
        "max_loss_dollars": _round_or_none(max_loss, 2),
        "estimated_reward_to_risk": _round_or_none(rr, 4),
        "reward_risk_meets_target": bool(fields.get("reward_risk_meets_target", False)),
        "actionability_score": actionability_score,
        "strengths": strengths[:6],
        "cautions": cautions[:6],
        "plain_english": plain_english,
    }


def _choose_review(reviews: List[Dict[str, Any]], key: str) -> Optional[Dict[str, Any]]:
    if not reviews:
        return None
    if key == "best_overall":
        return max(reviews, key=lambda r: r.get("actionability_score", -999.0))
    if key == "best_cheapest":
        priced = [r for r in reviews if r.get("max_loss_dollars") is not None]
        return min(priced, key=lambda r: (r["max_loss_dollars"], -r.get("ranking_score", 0.0))) if priced else None
    if key == "best_moonshot":
        candidates = [r for r in reviews if r.get("estimated_reward_to_risk") is not None]
        if not candidates:
            candidates = reviews
        return max(candidates, key=lambda r: (r.get("estimated_reward_to_risk") or 0.0, -r.get("max_loss_dollars") if r.get("max_loss_dollars") is not None else 0.0))
    if key == "most_caution":
        return max(reviews, key=lambda r: (len(r.get("cautions", [])), r.get("ranking_score", 0.0)))
    return None


def _summary_stub(review: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if review is None:
        return None
    return {
        "candidate_id": review.get("candidate_id"),
        "symbol": review.get("symbol"),
        "strategy_type": review.get("strategy_type"),
        "contract": review.get("contract"),
        "max_loss_dollars": review.get("max_loss_dollars"),
        "estimated_reward_to_risk": review.get("estimated_reward_to_risk"),
        "why": review.get("plain_english"),
        "main_cautions": review.get("cautions", [])[:3],
    }


def build_trade_analysis(
    market: Dict[str, Any],
    candidate_queue: List[Dict[str, Any]],
    no_trade: Optional[Dict[str, Any]],
    coordinator_bias: str,
) -> Dict[str, Any]:
    """
    Build a trader-readable analysis block from the final coordinator queue.

    The coordinator still does not place trades. This section explains why the
    queue looks the way it does and highlights which forwarded candidates deserve
    the most attention or caution.
    """
    forwarded = [row for row in candidate_queue if row.get("status") == "FORWARD"]
    strategy_counts: Dict[str, int] = {strategy: 0 for strategy in ALLOWED_STRATEGIES}
    symbol_counts: Dict[str, int] = {}
    for row in forwarded:
        strategy_counts[row.get("strategy_type", "unknown")] = strategy_counts.get(row.get("strategy_type", "unknown"), 0) + 1
        sym = _candidate_symbol(row) or "UNKNOWN"
        symbol_counts[sym] = symbol_counts.get(sym, 0) + 1

    if no_trade is not None:
        return {
            "analysis_version": "1.0.0",
            "market_takeaway": f"Coordinator bias is {coordinator_bias}. No trade was forwarded.",
            "strategy_mix": strategy_counts,
            "top_recommendations": {},
            "reviews": [],
            "plain_english_summary": [no_trade.get("summary", "No trade was forwarded.")],
        }

    reviews = [
        build_candidate_review(row, rank=i + 1, duplicate_symbol_count=symbol_counts.get(_candidate_symbol(row) or "UNKNOWN", 1))
        for i, row in enumerate(forwarded)
    ]

    best_overall = _choose_review(reviews, "best_overall")
    best_cheapest = _choose_review(reviews, "best_cheapest")
    best_moonshot = _choose_review(reviews, "best_moonshot")
    most_caution = _choose_review(reviews, "most_caution")

    st = normalize_horizon_block(market["short_term"])
    lt = normalize_horizon_block(market["long_term"])
    market_takeaway = (
        f"Market regime is {coordinator_bias}. Short-term is {st['normalized_regime']} "
        f"with {st['confidence']:.2f} confidence; long-term is {lt['normalized_regime']} "
        f"with {lt['confidence']:.2f} confidence."
    )

    plain_summary: List[str] = [market_takeaway]
    if best_overall:
        plain_summary.append(f"Best overall candidate: {best_overall['symbol']} ({best_overall['candidate_id']}).")
    if best_cheapest:
        plain_summary.append(f"Best cheapest candidate by one-lot max loss: {best_cheapest['symbol']} ({_money(best_cheapest.get('max_loss_dollars'))}).")
    if best_moonshot:
        rr = best_moonshot.get("estimated_reward_to_risk")
        rr_text = f"{rr:.2f}:1" if rr is not None else "unknown R/R"
        plain_summary.append(f"Best aggressive/moonshot profile: {best_moonshot['symbol']} with estimated R/R {rr_text}.")
    if most_caution:
        plain_summary.append(f"Highest-caution forwarded idea: {most_caution['symbol']} — {'; '.join(most_caution.get('cautions', [])[:2])}.")

    return {
        "analysis_version": "1.0.0",
        "market_takeaway": market_takeaway,
        "strategy_mix": strategy_counts,
        "duplicate_forwarded_symbols": {k: v for k, v in symbol_counts.items() if v > 1},
        "top_recommendations": {
            "best_overall": _summary_stub(best_overall),
            "best_cheapest": _summary_stub(best_cheapest),
            "best_aggressive_moonshot": _summary_stub(best_moonshot),
            "one_to_review_or_avoid": _summary_stub(most_caution),
        },
        "reviews": reviews,
        "plain_english_summary": plain_summary,
        "execution_reminder": (
            "FORWARD means the idea passed coordinator filters; it is not an order instruction. "
            "Validate live bid/ask, liquidity, updated news, and position size before entry."
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
        "schema_version": "1.2.0",
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
        "trade_analysis": build_trade_analysis(market, candidate_queue, no_trade, coordinator_bias),
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
