#!/usr/bin/env python3
"""
option_liquidity.py

Shared option-chain liquidity and exact-leg selection helpers for the trading
analysts. These functions do analysis only; they never place orders.

Purpose
-------
Before an analyst promotes an options idea, it should verify that the actual
option chain has usable bid/ask quotes, open interest, volume, and concrete
strikes. This module centralizes those checks so every analyst uses the same
risk-aware gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import math

import pandas as pd
import yfinance as yf


@dataclass(frozen=True)
class OptionLiquidityConfig:
    """Simple, auditable option liquidity thresholds.

    The defaults are intentionally conservative enough to reject obviously bad
    chains while still allowing small/mid-cap names to pass. Tighten these for
    live trading.
    """

    min_open_interest: int = 20
    min_volume: int = 1
    max_bid_ask_spread_pct: float = 0.35
    min_trade_debit: float = 0.50
    max_trade_debit: float = 5.00
    min_dte: int = 21
    max_dte: int = 60
    target_otm_pct: float = 0.08


def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or pd.isna(value):
            return default
        return int(float(value))
    except Exception:
        return default


def jsonable(value: Any) -> Any:
    """Convert pandas/numpy-ish values into JSON-safe scalars."""
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def row_to_dict(row: Optional[pd.Series | Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if row is None:
        return None
    raw = row.to_dict() if hasattr(row, "to_dict") else dict(row)
    return {str(k): jsonable(v) for k, v in raw.items()}


def option_mid(row: Optional[Dict[str, Any]]) -> Optional[float]:
    """Return option mid from bid/ask, falling back to lastPrice only if needed."""
    if not row:
        return None
    bid = safe_float(row.get("bid"))
    ask = safe_float(row.get("ask"))
    last = safe_float(row.get("lastPrice"))
    if bid is not None and ask is not None and bid >= 0 and ask > 0 and ask >= bid:
        return (bid + ask) / 2.0
    return last


def bid_ask_spread_pct(row: Optional[Dict[str, Any]]) -> Optional[float]:
    """Bid/ask width as a percentage of mid price."""
    if not row:
        return None
    bid = safe_float(row.get("bid"))
    ask = safe_float(row.get("ask"))
    mid = option_mid(row)
    if bid is None or ask is None or mid in (None, 0) or ask < bid:
        return None
    return abs(ask - bid) / mid


def evaluate_option_row_liquidity(row: Optional[Dict[str, Any]], config: OptionLiquidityConfig) -> Dict[str, Any]:
    """Evaluate one option contract against the shared liquidity rules."""
    if not row:
        return {"ok": False, "reasons": ["missing_option_row"]}

    bid = safe_float(row.get("bid"))
    ask = safe_float(row.get("ask"))
    mid = option_mid(row)
    oi = safe_int(row.get("openInterest"), 0)
    vol = safe_int(row.get("volume"), 0)
    spread_pct = bid_ask_spread_pct(row)

    reasons: List[str] = []
    if bid is None or ask is None or bid <= 0 or ask <= 0 or ask < bid:
        reasons.append("bad_bid_ask_quote")
    if mid is None or mid <= 0:
        reasons.append("missing_mid_price")
    if oi < config.min_open_interest:
        reasons.append("low_open_interest")
    if vol < config.min_volume:
        reasons.append("low_option_volume")
    if spread_pct is None:
        reasons.append("missing_bid_ask_spread")
    elif spread_pct > config.max_bid_ask_spread_pct:
        reasons.append("wide_bid_ask_spread")

    return {
        "ok": not reasons,
        "bid": round(bid, 4) if bid is not None else None,
        "ask": round(ask, 4) if ask is not None else None,
        "mid": round(mid, 4) if mid is not None else None,
        "open_interest": oi,
        "volume": vol,
        "bid_ask_spread_pct": round(spread_pct, 4) if spread_pct is not None else None,
        "reasons": reasons,
    }


def get_option_expirations(symbol: str) -> List[str]:
    try:
        return list(yf.Ticker(symbol).options or [])
    except Exception:
        return []


def choose_expiration(symbol: str, min_dte: int, max_dte: int) -> Tuple[bool, Optional[str], Optional[int]]:
    """Choose the expiration nearest the middle of the configured DTE window."""
    expirations = get_option_expirations(symbol)
    if not expirations:
        return False, None, None

    today = datetime.now(timezone.utc).date()
    target_mid = (min_dte + max_dte) // 2
    best: Tuple[Optional[str], Optional[int], int] = (None, None, 10**9)
    parsed_future: List[Tuple[str, int]] = []

    for exp in expirations:
        try:
            exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
        except Exception:
            continue
        dte = (exp_date - today).days
        if dte >= 0:
            parsed_future.append((exp, dte))
        if min_dte <= dte <= max_dte:
            dist = abs(dte - target_mid)
            if dist < best[2]:
                best = (exp, dte, dist)

    if best[0] is not None:
        return True, best[0], best[1]

    if parsed_future:
        # Fallback to nearest future expiration, but report the actual DTE.
        exp, dte = min(parsed_future, key=lambda x: abs(x[1] - target_mid))
        return True, exp, dte

    return True, expirations[0], None


def _nearest_rows(df: pd.DataFrame, target: float, limit: int = 5) -> List[Dict[str, Any]]:
    if df is None or df.empty or "strike" not in df.columns:
        return []
    work = df.copy()
    work["_distance"] = (work["strike"].astype(float) - float(target)).abs()
    rows = work.sort_values("_distance").head(limit)
    return [row_to_dict(row) for _, row in rows.iterrows() if row_to_dict(row) is not None]


def select_vertical_debit_spread(
    symbol: str,
    side: str,
    underlying_price: float,
    config: OptionLiquidityConfig,
    expiration: Optional[str] = None,
) -> Dict[str, Any]:
    """Select exact strikes for a defined-risk debit spread and validate liquidity.

    side:
      - BULLISH -> call debit spread
      - BEARISH -> put debit spread

    Uses conservative entry debit = long_leg_ask - short_leg_bid because that is
    closer to a realistic executable debit than pure midpoint pricing.
    """
    side = side.upper()
    if side not in {"BULLISH", "BEARISH"}:
        return {"status": "ERROR", "reason": f"unsupported_side_{side}"}
    if underlying_price is None or underlying_price <= 0:
        return {"status": "ERROR", "reason": "missing_underlying_price"}

    has_options = True
    dte = None
    if expiration is None:
        has_options, expiration, dte = choose_expiration(symbol, config.min_dte, config.max_dte)
    else:
        try:
            dte = (datetime.strptime(expiration, "%Y-%m-%d").date() - datetime.now(timezone.utc).date()).days
        except Exception:
            dte = None

    if not has_options or not expiration:
        return {"status": "FAILED_LIQUIDITY", "reason": "no_listed_options", "symbol": symbol}

    try:
        chain = yf.Ticker(symbol).option_chain(expiration)
    except Exception as exc:
        return {"status": "ERROR", "reason": f"option_chain_fetch_failed: {exc}", "symbol": symbol, "expiration": expiration}

    option_type = "CALL" if side == "BULLISH" else "PUT"
    chain_df = chain.calls if side == "BULLISH" else chain.puts
    if chain_df is None or chain_df.empty:
        return {"status": "FAILED_LIQUIDITY", "reason": "empty_option_chain", "symbol": symbol, "expiration": expiration, "option_type": option_type}

    long_target = float(underlying_price)
    short_target = float(underlying_price) * (1.0 + config.target_otm_pct if side == "BULLISH" else 1.0 - config.target_otm_pct)
    long_candidates = _nearest_rows(chain_df, long_target, limit=8)
    short_candidates = _nearest_rows(chain_df, short_target, limit=12)

    evaluated: List[Dict[str, Any]] = []
    for long_row in long_candidates:
        long_strike = safe_float(long_row.get("strike"))
        long_ask = safe_float(long_row.get("ask"))
        if long_strike is None or long_ask is None:
            continue
        for short_row in short_candidates:
            short_strike = safe_float(short_row.get("strike"))
            short_bid = safe_float(short_row.get("bid"))
            if short_strike is None or short_bid is None:
                continue
            if side == "BULLISH" and short_strike <= long_strike:
                continue
            if side == "BEARISH" and short_strike >= long_strike:
                continue

            width = abs(short_strike - long_strike)
            entry_debit = long_ask - short_bid
            long_mid = option_mid(long_row)
            short_mid = option_mid(short_row)
            mid_debit = long_mid - short_mid if long_mid is not None and short_mid is not None else None
            max_reward = width - entry_debit if entry_debit is not None else None

            long_liq = evaluate_option_row_liquidity(long_row, config)
            short_liq = evaluate_option_row_liquidity(short_row, config)
            reasons: List[str] = []
            if not long_liq["ok"]:
                reasons.extend([f"long_{r}" for r in long_liq["reasons"]])
            if not short_liq["ok"]:
                reasons.extend([f"short_{r}" for r in short_liq["reasons"]])
            if entry_debit is None or entry_debit <= 0:
                reasons.append("non_positive_entry_debit")
            if max_reward is None or max_reward <= 0:
                reasons.append("no_positive_max_reward")

            premium_in_range = bool(
                entry_debit is not None
                and config.min_trade_debit <= entry_debit <= config.max_trade_debit
            )
            if not premium_in_range:
                reasons.append("debit_outside_preferred_range")

            evaluated.append({
                "liquidity_pass": not [r for r in reasons if r != "debit_outside_preferred_range"],
                "premium_in_preferred_range": premium_in_range,
                "reasons": reasons,
                "symbol": symbol,
                "expiration": expiration,
                "dte": dte,
                "side": side,
                "structure": "CALL_DEBIT_SPREAD" if side == "BULLISH" else "PUT_DEBIT_SPREAD",
                "option_type": option_type,
                "underlying_price": round(float(underlying_price), 4),
                "long_leg": {
                    "action": "BUY",
                    "type": option_type,
                    "strike": long_strike,
                    "contract_symbol": long_row.get("contractSymbol"),
                    "liquidity": long_liq,
                },
                "short_leg": {
                    "action": "SELL",
                    "type": option_type,
                    "strike": short_strike,
                    "contract_symbol": short_row.get("contractSymbol"),
                    "liquidity": short_liq,
                },
                "width": round(width, 4),
                "estimated_entry_debit": round(entry_debit, 4) if entry_debit is not None else None,
                "estimated_mid_debit": round(mid_debit, 4) if mid_debit is not None else None,
                "estimated_max_loss": round(entry_debit, 4) if entry_debit is not None and entry_debit > 0 else None,
                "estimated_max_reward": round(max_reward, 4) if max_reward is not None and max_reward > 0 else None,
                "estimated_reward_to_risk": round(max_reward / entry_debit, 4) if entry_debit and entry_debit > 0 and max_reward and max_reward > 0 else None,
            })

    if not evaluated:
        return {
            "status": "FAILED_LIQUIDITY",
            "reason": "no_valid_vertical_candidates",
            "symbol": symbol,
            "expiration": expiration,
            "side": side,
            "option_type": option_type,
        }

    def sort_key(row: Dict[str, Any]) -> Tuple[int, int, float, float]:
        # Prefer liquidity pass, then preferred debit range, then better reward/risk, then lower debit.
        rr = safe_float(row.get("estimated_reward_to_risk"), 0.0) or 0.0
        debit = safe_float(row.get("estimated_entry_debit"), 999.0) or 999.0
        return (
            0 if row.get("liquidity_pass") else 1,
            0 if row.get("premium_in_preferred_range") else 1,
            -rr,
            debit,
        )

    best = sorted(evaluated, key=sort_key)[0]
    if best.get("liquidity_pass"):
        best["status"] = "OK"
        best["reason"] = "PASSED_OPTION_LIQUIDITY"
    else:
        best["status"] = "FAILED_LIQUIDITY"
        best["reason"] = "OPTION_LIQUIDITY_FAILED"

    # Keep a short audit trail without exploding the JSON output.
    best["evaluated_candidate_count"] = len(evaluated)
    return best
