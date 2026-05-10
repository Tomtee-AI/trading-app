#!/usr/bin/env python3
"""
unusual_options_analyst.py

Deterministic Unusual Options Activity Analyst
==============================================

Mission
-------
Read an unusual-options-flow CSV, identify high-quality long-premium trade
candidates, and emit a coordinator-compatible specialist payload for the
ZeroHumanCompany trading-agent roster.

This script is analysis-only. It never places orders.

Designed to be consistent with the existing analyst roster:
- Emits strict JSON to stdout for the Trade Coordinator.
- Writes a richer diagnostics report to reports/.
- Uses strategy_type="uoa", which is already recognized by the coordinator.
- Uses only long-premium / defined-risk structures.
- Fails closed when required data is missing or liquidity is poor.

Input
-----
Default input file is the user's sample unusual-flow CSV:
    flow_true_bid_side,mid_side_false_ADR,Common Stock_250_20000000000_-.1_10000_5_0.6_Prem_true_true_true_true_true_60_0.5_marketcaplesstwentb.csv

Expected useful columns include, but are not limited to:
    date, time, underlying_symbol, side, strike, type, expiry, DTE,
    option_chain_id, ewma_nbbo_bid, ewma_nbbo_ask, underlying_price,
    size, premium, volume, open_interest, bid_vol, mid_vol, ask_vol,
    implied_volatility, delta, theta, gamma, next_earnings_date,
    bearish_or_bullish, tags, sector, industry_type, nbbo_bid, nbbo_ask,
    canceled, er_time, full_name, marketcap, option_type, price, string

Install
-------
pip install pandas numpy jsonschema yfinance

Run
---
python unusual_options_analyst.py
python unusual_options_analyst.py --input-file flow.csv
python unusual_options_analyst.py --max-candidates 10 --min-total-premium 250000
python unusual_options_analyst.py --min-option-volume 100 --min-open-interest 50 --max-bid-ask-spread-pct 0.25
python unusual_options_analyst.py --post-flow-window-minutes 10 --min-post-flow-confirm-move-pct 0.001
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from jsonschema import Draft202012Validator, FormatChecker

try:
    import yfinance as yf  # type: ignore
except Exception:  # pragma: no cover - optional runtime dependency
    yf = None

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

SCRIPT_DIR = Path(__file__).resolve().parent
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_INPUT_FILE = (
    "flow.csv"
)

AGENT_NAME = "unusual_options_analyst"
SCHEMA_VERSION = "1.2.0"
STRATEGY_TYPE = "uoa"

ALLOWED_STRUCTURES = [
    "LONG_CALL",
    "LONG_PUT",
    "CALL_DEBIT_SPREAD",
    "PUT_DEBIT_SPREAD",
    "STRADDLE",
    "STRANGLE",
]

COORDINATOR_CANDIDATE_SCHEMA: Dict[str, Any] = {
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
        "strategy_type": {"type": "string", "const": STRATEGY_TYPE},
        "direction": {"type": "string", "enum": ["BULLISH", "BEARISH", "NEUTRAL"]},
        "horizon": {"type": "string", "enum": ["SHORT_TERM", "LONG_TERM"]},
        "structure_family": {"type": "string", "enum": ALLOWED_STRUCTURES},
        "summary": {"type": "string", "minLength": 1},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "fit_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "thesis_tags": {"type": "array", "items": {"type": "string"}},
        "risk_flags": {"type": "array", "items": {"type": "string"}},
        "implementation": {"type": "object", "additionalProperties": True},
    },
}

OUTPUT_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": ["agent", "schema_version", "event_id", "strategy_type", "candidates", "summary"],
    "properties": {
        "agent": {"type": "string", "const": AGENT_NAME},
        "schema_version": {"type": "string", "const": SCHEMA_VERSION},
        "event_id": {"type": "string", "format": "date-time"},
        "strategy_type": {"type": "string", "const": STRATEGY_TYPE},
        "candidates": {"type": "array", "items": COORDINATOR_CANDIDATE_SCHEMA},
        "summary": {"type": "string"},
    },
}


@dataclass
class RuntimeConfig:
    """Auditable knobs for the unusual-options analyst.

    Defaults are intentionally conservative and aligned with the rules file:
    smaller/mid-cap preference, underlying price between $5 and $100, option
    premium between $0.50 and $5.00, DTE no more than 60, and no poor-liquidity
    option chains.
    """

    input_file: str = DEFAULT_INPUT_FILE
    max_candidates: int = 10
    max_watchlist: int = 25

    # Core rules-file guardrails.
    min_underlying_price: float = 5.00
    max_underlying_price: float = 100.00
    min_option_price: float = 0.50
    max_option_price: float = 5.00
    min_dte: int = 0
    max_dte: int = 60
    max_market_cap: float = 20_000_000_000.0

    # Flow/liquidity thresholds.
    min_total_premium: float = 250_000.0
    min_option_volume: int = 100
    min_open_interest: int = 20
    min_size: int = 1
    max_bid_ask_spread_pct: float = 0.35
    min_ask_side_ratio: float = 0.60
    min_volume_oi_ratio_for_bonus: float = 1.00

    # Scoring thresholds.
    min_fit_score: float = 0.55
    watchlist_min_fit_score: float = 0.40
    min_reward_to_risk: float = 1.00
    target_reward_to_risk: float = 5.00

    # Behavior switches.
    require_buyer_initiated: bool = True
    allow_expensive_exceptional: bool = False

    # Anti-synthetic / flow-confirmation controls.
    # A single options print can be misleading when it is part of a synthetic,
    # conversion/reversal, delta hedge, spread, or multi-leg package.  To reduce
    # false positives, require the underlying stock/ETF to confirm the inferred
    # direction over the next 10 one-minute bars after the flow print.
    require_post_flow_confirmation: bool = True
    allow_missing_post_flow_confirmation: bool = False
    post_flow_window_minutes: int = 10
    min_post_flow_confirm_move_pct: float = 0.0010  # 0.10% in thesis direction
    trade_timestamp_timezone: str = "America/New_York"

    verbose: bool = False


# -------------------- generic helpers --------------------
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str, enabled: bool = False) -> None:
    if enabled:
        print(message, file=sys.stderr)


def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None or pd.isna(value):
            return default
        if isinstance(value, str):
            text = value.strip().replace(",", "").replace("$", "").replace("%", "")
            if text.lower() in {"", "nan", "none", "null", "#value!"}:
                return default
            return float(text)
        return float(value)
    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    f = safe_float(value)
    if f is None or not math.isfinite(f):
        return default
    return int(f)


def safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    return str(value).strip()


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def jsonable(value: Any) -> Any:
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def resolve_runtime_path(raw: Optional[str], default_filename: str = DEFAULT_INPUT_FILE) -> Path:
    """Resolve paths like the other analysts do: cwd first, then script folder."""
    text = raw or default_filename
    p = Path(text).expanduser()
    if p.is_absolute() and p.exists():
        return p

    cwd_candidate = (Path.cwd() / p).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    script_candidate = (SCRIPT_DIR / p).resolve()
    if script_candidate.exists():
        return script_candidate

    # Return cwd candidate for useful error messages.
    return cwd_candidate


def parse_date(value: Any) -> Optional[date]:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value

    text = safe_str(value)
    if not text:
        return None
    for fmt in ["%m/%d/%Y", "%m/%d/%y", "%Y-%m-%d", "%b %d %Y", "%b %d, %Y", "%B %d %Y", "%B %d, %Y"]:
        try:
            return datetime.strptime(text, fmt).date()
        except Exception:
            continue
    try:
        parsed = pd.to_datetime(text, errors="coerce")
        if pd.notna(parsed):
            return parsed.date()
    except Exception:
        pass
    return None


def parse_datetime_from_row(row: Dict[str, Any]) -> Optional[datetime]:
    d = safe_str(row.get("date"))
    t = safe_str(row.get("time"))
    if not d:
        return None
    text = f"{d} {t}".strip()
    for fmt in ["%m/%d/%Y %I:%M:%S %p", "%m/%d/%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S"]:
        try:
            return datetime.strptime(text, fmt)
        except Exception:
            continue
    try:
        parsed = pd.to_datetime(text, errors="coerce")
        if pd.notna(parsed):
            return parsed.to_pydatetime()
    except Exception:
        pass
    return None




# -------------------- post-flow underlying confirmation --------------------
def localize_trade_datetime(trade_dt: Optional[datetime], timezone_name: str) -> Optional[datetime]:
    """Return a timezone-aware trade timestamp.

    Most UOA exports timestamp prints in US/Eastern market time.  Treat naive
    timestamps as the configured market timezone so the 1-minute chart window is
    aligned with the actual option print time.
    """
    if trade_dt is None:
        return None
    try:
        tz = ZoneInfo(timezone_name)
    except Exception:
        tz = ZoneInfo("America/New_York")
    if trade_dt.tzinfo is None:
        return trade_dt.replace(tzinfo=tz)
    return trade_dt.astimezone(tz)


def _flatten_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize yfinance output to simple OHLCV column names."""
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        # yfinance may return either ('Close', 'AAPL') or ('AAPL', 'Close').
        candidates = []
        for level in range(out.columns.nlevels):
            values = [str(x).lower() for x in out.columns.get_level_values(level)]
            if any(v in {"open", "high", "low", "close", "volume"} for v in values):
                candidates.append(level)
        if candidates:
            level = candidates[0]
            out.columns = [str(col[level]).title() for col in out.columns]
        else:
            out.columns = ["_".join(str(part) for part in col if part) for col in out.columns]
    else:
        out.columns = [str(c).title() for c in out.columns]

    rename = {
        "Adj Close": "Adj Close",
        "Open": "Open",
        "High": "High",
        "Low": "Low",
        "Close": "Close",
        "Volume": "Volume",
    }
    out = out.rename(columns=rename)
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in out.columns]
    return out[keep].copy() if keep else pd.DataFrame()


def fetch_one_minute_bars_for_trade_day(
    symbol: str,
    trade_dt: datetime,
    config: RuntimeConfig,
    intraday_cache: Dict[str, Any],
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Fetch/cache 1-minute bars for the underlying on the UOA print date.

    The analyst only needs a tiny 10-minute window, but fetching the full trading
    day once per symbol/date avoids repeated API calls for multiple prints in the
    same ticker.
    """
    symbol = safe_str(symbol).upper()
    if not symbol:
        return None, "missing_symbol"
    if yf is None:
        return None, "yfinance_not_installed"

    local_dt = localize_trade_datetime(trade_dt, config.trade_timestamp_timezone)
    if local_dt is None:
        return None, "missing_trade_timestamp"

    cache_key = f"{symbol}|{local_dt.date().isoformat()}"
    if cache_key in intraday_cache:
        cached = intraday_cache[cache_key]
        return cached.get("bars"), cached.get("error")

    start_day = local_dt.date()
    end_day = start_day + timedelta(days=1)
    try:
        raw = yf.download(
            symbol,
            start=start_day.isoformat(),
            end=end_day.isoformat(),
            interval="1m",
            prepost=True,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        bars = _flatten_yfinance_columns(raw)
        if bars.empty or "Close" not in bars.columns:
            intraday_cache[cache_key] = {"bars": None, "error": "no_1m_bars_returned"}
            return None, "no_1m_bars_returned"

        # Make the index timezone-aware and comparable to the localized trade timestamp.
        if bars.index.tz is None:
            bars.index = bars.index.tz_localize(config.trade_timestamp_timezone, nonexistent="shift_forward", ambiguous="NaT")
        else:
            bars.index = bars.index.tz_convert(config.trade_timestamp_timezone)
        bars = bars[~bars.index.isna()].sort_index()

        intraday_cache[cache_key] = {"bars": bars, "error": None}
        return bars, None
    except Exception as exc:
        err = f"intraday_fetch_error:{type(exc).__name__}:{exc}"
        intraday_cache[cache_key] = {"bars": None, "error": err}
        return None, err


def infer_synthetic_risk(row: Dict[str, Any], guardrails: Dict[str, Any]) -> Dict[str, Any]:
    """Flag flow patterns that can hide synthetic long/short activity.

    This is intentionally conservative.  It does not automatically reject a row;
    it explains why post-flow underlying confirmation is required before the row
    can become a trade candidate.
    """
    flow = guardrails.get("flow", {})
    ratios = flow.get("side_ratios", {}) or {}
    volume = safe_int(flow.get("volume"), 0)
    size = safe_int(flow.get("size"), 0)
    multi_vol = safe_int(ratios.get("multi_vol"), 0)
    ask_ratio = safe_float(ratios.get("ask_side_ratio"), 0.0) or 0.0
    bid_ratio = safe_float(ratios.get("bid_side_ratio"), 0.0) or 0.0
    mid_ratio = safe_float(ratios.get("mid_side_ratio"), 0.0) or 0.0
    tags = safe_str(row.get("tags")).lower()
    report_flags = safe_str(row.get("report_flags")).lower()

    reasons: List[str] = []
    if volume > 0 and multi_vol / volume >= 0.50:
        reasons.append("large_multi_leg_volume")
    if size > 0 and volume > 0 and size / volume >= 0.80 and multi_vol > 0:
        reasons.append("block_size_overlaps_multi_leg_print")
    if "spread" in tags or "multi" in tags or "combo" in tags:
        reasons.append("vendor_tags_suggest_multi_leg_or_combo")
    if "floor" in report_flags:
        reasons.append("floor_print_can_be_complex_or_negotiated")
    if ask_ratio >= 0.60 and bid_ratio >= 0.10:
        reasons.append("mixed_bid_ask_participation")
    if mid_ratio >= 0.20:
        reasons.append("large_mid_market_component")

    return {
        "synthetic_or_complex_flow_possible": bool(reasons),
        "reasons": sorted(set(reasons)),
        "multi_leg_volume_ratio": round(multi_vol / volume, 4) if volume > 0 else None,
        "buyer_side_ratio": round(ask_ratio, 4),
        "seller_side_ratio": round(bid_ratio, 4),
    }


def evaluate_post_flow_confirmation(
    row: Dict[str, Any],
    guardrails: Dict[str, Any],
    config: RuntimeConfig,
    intraday_cache: Dict[str, Any],
) -> Dict[str, Any]:
    """Validate that the underlying confirms the inferred UOA direction.

    For bullish flow, the underlying should rise over the next N one-minute bars.
    For bearish flow, the underlying should fall.  This helps reject prints that
    look bullish/bearish on the tape but are actually part of a synthetic, hedge,
    or market-maker inventory transfer.
    """
    direction = guardrails.get("flow", {}).get("direction_info", {}).get("direction")
    symbol = safe_str(row.get("underlying_symbol")).upper()
    trade_dt_raw = parse_datetime_from_row(row)
    trade_dt = localize_trade_datetime(trade_dt_raw, config.trade_timestamp_timezone)
    synthetic_review = infer_synthetic_risk(row, guardrails)

    base = {
        "enabled": bool(config.require_post_flow_confirmation),
        "status": "NOT_EVALUATED",
        "symbol": symbol or None,
        "trade_timestamp": trade_dt.isoformat() if trade_dt else None,
        "window_minutes": config.post_flow_window_minutes,
        "min_confirm_move_pct": config.min_post_flow_confirm_move_pct,
        "direction_expected": direction,
        "synthetic_flow_review": synthetic_review,
    }

    if not config.require_post_flow_confirmation:
        return {**base, "status": "DISABLED", "passed": True, "reason": "post_flow_confirmation_disabled"}

    if direction not in {"BULLISH", "BEARISH"}:
        return {**base, "status": "UNAVAILABLE", "passed": False, "reason": "direction_not_directional"}
    if trade_dt is None:
        return {**base, "status": "UNAVAILABLE", "passed": False, "reason": "missing_trade_timestamp"}

    bars, error = fetch_one_minute_bars_for_trade_day(symbol, trade_dt, config, intraday_cache)
    if bars is None or bars.empty:
        passed = bool(config.allow_missing_post_flow_confirmation)
        return {
            **base,
            "status": "UNAVAILABLE",
            "passed": passed,
            "reason": error or "no_intraday_bars",
            "fail_closed": not passed,
        }

    window_end = trade_dt + timedelta(minutes=max(config.post_flow_window_minutes, 1))
    window = bars[(bars.index > trade_dt) & (bars.index <= window_end)].copy()
    if len(window) < 3:
        passed = bool(config.allow_missing_post_flow_confirmation)
        return {
            **base,
            "status": "UNAVAILABLE",
            "passed": passed,
            "reason": "insufficient_1m_bars_after_flow",
            "bar_count": int(len(window)),
            "fail_closed": not passed,
        }

    row_underlying = safe_float(row.get("underlying_price"))
    first_open = safe_float(window["Open"].iloc[0]) if "Open" in window.columns else None
    start_price = row_underlying if row_underlying is not None and row_underlying > 0 else first_open
    end_close = safe_float(window["Close"].iloc[-1])
    high = safe_float(window["High"].max()) if "High" in window.columns else None
    low = safe_float(window["Low"].min()) if "Low" in window.columns else None

    if start_price is None or start_price <= 0 or end_close is None or end_close <= 0:
        passed = bool(config.allow_missing_post_flow_confirmation)
        return {
            **base,
            "status": "UNAVAILABLE",
            "passed": passed,
            "reason": "invalid_1m_price_window",
            "bar_count": int(len(window)),
            "fail_closed": not passed,
        }

    raw_return = (end_close - start_price) / start_price
    favorable_excursion = ((high - start_price) / start_price) if high is not None and direction == "BULLISH" else None
    adverse_excursion = ((low - start_price) / start_price) if low is not None and direction == "BULLISH" else None
    if direction == "BEARISH":
        favorable_excursion = ((start_price - low) / start_price) if low is not None else None
        adverse_excursion = ((start_price - high) / start_price) if high is not None else None

    directional_return = raw_return if direction == "BULLISH" else -raw_return
    min_move = max(float(config.min_post_flow_confirm_move_pct), 0.0)

    if directional_return >= min_move:
        status = "CONFIRMED"
        passed = True
        reason = "underlying_confirmed_flow_direction"
    elif directional_return <= -min_move:
        status = "CONTRADICTED"
        passed = False
        reason = "underlying_moved_against_flow_direction"
    else:
        status = "AMBIGUOUS"
        passed = False
        reason = "underlying_did_not_confirm_enough"

    return {
        **base,
        "status": status,
        "passed": passed,
        "reason": reason,
        "bar_count": int(len(window)),
        "window_start": window.index[0].isoformat(),
        "window_end": window.index[-1].isoformat(),
        "start_price": round(start_price, 4),
        "end_close": round(end_close, 4),
        "window_high": round(high, 4) if high is not None else None,
        "window_low": round(low, 4) if low is not None else None,
        "underlying_return_pct": round(raw_return, 5),
        "directional_confirmation_return_pct": round(directional_return, 5),
        "favorable_excursion_pct": round(favorable_excursion, 5) if favorable_excursion is not None else None,
        "adverse_excursion_pct": round(adverse_excursion, 5) if adverse_excursion is not None else None,
        "fail_closed": not passed,
    }

def validate_output_schema(payload: Dict[str, Any]) -> None:
    validator = Draft202012Validator(OUTPUT_SCHEMA, format_checker=FormatChecker())
    errors = sorted(validator.iter_errors(payload), key=lambda e: list(e.path))
    if errors:
        lines: List[str] = []
        for err in errors[:12]:
            path = ".".join(str(p) for p in err.path) if err.path else "<root>"
            lines.append(f"{path}: {err.message}")
        raise ValueError("Output JSON schema validation failed:\n" + "\n".join(lines))


# -------------------- CSV normalization --------------------
def load_flow_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    if df.empty:
        raise ValueError(f"Input CSV is empty: {path}")
    return df


def row_dict(row: pd.Series) -> Dict[str, Any]:
    return {str(k): jsonable(v) for k, v in row.to_dict().items()}


def option_type_from_row(row: Dict[str, Any]) -> Optional[str]:
    raw = safe_str(row.get("option_type") or row.get("type")).upper()
    if raw in {"C", "CALL"}:
        return "CALL"
    if raw in {"P", "PUT"}:
        return "PUT"
    return None


def is_canceled(row: Dict[str, Any]) -> bool:
    raw = safe_str(row.get("canceled")).lower()
    return raw in {"true", "1", "yes", "y"}


def nbbo_bid_ask(row: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], str]:
    """Prefer live NBBO fields, then EWMA NBBO fields."""
    bid = safe_float(row.get("nbbo_bid"))
    ask = safe_float(row.get("nbbo_ask"))
    if bid is not None and ask is not None and ask > 0:
        return bid, ask, "nbbo"

    bid = safe_float(row.get("ewma_nbbo_bid"))
    ask = safe_float(row.get("ewma_nbbo_ask"))
    if bid is not None and ask is not None and ask > 0:
        return bid, ask, "ewma_nbbo"

    return None, None, "missing"


def option_price(row: Dict[str, Any], bid: Optional[float], ask: Optional[float]) -> Optional[float]:
    px = safe_float(row.get("price"))
    if px is not None and px > 0:
        return px
    if bid is not None and ask is not None and ask >= bid and ask > 0:
        return (bid + ask) / 2.0
    return None


def bid_ask_spread_pct(bid: Optional[float], ask: Optional[float], px: Optional[float]) -> Optional[float]:
    if bid is None or ask is None or ask <= 0 or bid < 0 or ask < bid:
        return None
    mid = (bid + ask) / 2.0
    basis = mid if mid > 0 else px
    if basis is None or basis <= 0:
        return None
    return (ask - bid) / basis


def flow_side_ratios(row: Dict[str, Any]) -> Dict[str, Any]:
    bid_vol = safe_int(row.get("bid_vol"), 0)
    mid_vol = safe_int(row.get("mid_vol"), 0)
    ask_vol = safe_int(row.get("ask_vol"), 0)
    no_side_vol = safe_int(row.get("no_side_vol"), 0)
    multi_vol = safe_int(row.get("multi_vol"), 0)
    total_side_vol = bid_vol + mid_vol + ask_vol + no_side_vol
    basis = total_side_vol if total_side_vol > 0 else safe_int(row.get("volume"), 0)

    def ratio(x: int) -> Optional[float]:
        if basis <= 0:
            return None
        return x / basis

    return {
        "bid_vol": bid_vol,
        "mid_vol": mid_vol,
        "ask_vol": ask_vol,
        "no_side_vol": no_side_vol,
        "multi_vol": multi_vol,
        "side_volume_basis": basis,
        "ask_side_ratio": round(ratio(ask_vol), 4) if ratio(ask_vol) is not None else None,
        "bid_side_ratio": round(ratio(bid_vol), 4) if ratio(bid_vol) is not None else None,
        "mid_side_ratio": round(ratio(mid_vol), 4) if ratio(mid_vol) is not None else None,
    }


# -------------------- trade inference and risk checks --------------------
def infer_direction(row: Dict[str, Any], option_type: Optional[str], ratios: Dict[str, Any]) -> Dict[str, Any]:
    """Infer directional intent from vendor label first, then flow mechanics.

    Conservative interpretation:
    - ASK-side call buying is bullish.
    - ASK-side put buying is bearish.
    - BID-side option prints are not treated as long-premium signals unless the
      vendor already labeled them directionally.
    """
    vendor_label = safe_str(row.get("bearish_or_bullish")).lower()
    tags = safe_str(row.get("tags")).lower()
    side = safe_str(row.get("side")).upper()
    ask_ratio = safe_float(ratios.get("ask_side_ratio"), 0.0) or 0.0
    bid_ratio = safe_float(ratios.get("bid_side_ratio"), 0.0) or 0.0

    evidence: List[str] = []
    if vendor_label in {"bullish", "bearish"}:
        direction = "BULLISH" if vendor_label == "bullish" else "BEARISH"
        evidence.append(f"vendor_label_{vendor_label}")
    elif "bullish" in tags:
        direction = "BULLISH"
        evidence.append("tag_bullish")
    elif "bearish" in tags:
        direction = "BEARISH"
        evidence.append("tag_bearish")
    elif side == "ASK" and option_type == "CALL":
        direction = "BULLISH"
        evidence.append("ask_side_call_buying")
    elif side == "ASK" and option_type == "PUT":
        direction = "BEARISH"
        evidence.append("ask_side_put_buying")
    else:
        direction = "NEUTRAL"
        evidence.append("direction_not_clear")

    buyer_initiated = side == "ASK" or ask_ratio >= 0.60 or "ask_side" in tags
    seller_initiated = side == "BID" or bid_ratio >= 0.60 or "bid_side" in tags

    return {
        "direction": direction,
        "buyer_initiated": bool(buyer_initiated),
        "seller_initiated": bool(seller_initiated),
        "evidence": evidence,
        "raw_side": side,
    }


def choose_structure(direction: str, option_type: Optional[str], buyer_initiated: bool) -> Optional[str]:
    """Choose only structures allowed by the rules and compatible with observed flow."""
    if not buyer_initiated:
        return None
    if direction == "BULLISH" and option_type == "CALL":
        return "LONG_CALL"
    if direction == "BEARISH" and option_type == "PUT":
        return "LONG_PUT"
    # A bullish put print or bearish call print can be a sale, spread leg, or hedge.
    # Do not promote it as a long-premium candidate without spread context.
    return None


def evaluate_row_guardrails(row: Dict[str, Any], config: RuntimeConfig) -> Dict[str, Any]:
    bid, ask, quote_source = nbbo_bid_ask(row)
    px = option_price(row, bid, ask)
    spread_pct = bid_ask_spread_pct(bid, ask, px)
    option_type = option_type_from_row(row)
    ratios = flow_side_ratios(row)
    direction_info = infer_direction(row, option_type, ratios)
    structure = choose_structure(direction_info["direction"], option_type, bool(direction_info["buyer_initiated"]))

    underlying = safe_float(row.get("underlying_price"))
    strike = safe_float(row.get("strike"))
    dte = safe_int(row.get("DTE"), -999)
    premium = safe_float(row.get("premium"), 0.0) or 0.0
    size = safe_int(row.get("size"), 0)
    volume = safe_int(row.get("volume"), 0)
    open_interest = safe_int(row.get("open_interest"), 0)
    market_cap = safe_float(row.get("marketcap"))

    reasons: List[str] = []
    warnings: List[str] = []

    if is_canceled(row):
        reasons.append("canceled_flow")
    if not safe_str(row.get("underlying_symbol")):
        reasons.append("missing_symbol")
    if option_type not in {"CALL", "PUT"}:
        reasons.append("missing_option_type")
    if structure is None:
        reasons.append("not_clean_long_premium_flow")
    if config.require_buyer_initiated and not direction_info["buyer_initiated"]:
        reasons.append("not_buyer_initiated")
    ask_ratio_for_gate = safe_float(ratios.get("ask_side_ratio"), 0.0) or 0.0
    if config.require_buyer_initiated and ask_ratio_for_gate < config.min_ask_side_ratio:
        reasons.append("ask_side_ratio_below_threshold")

    if underlying is None or underlying <= 0:
        reasons.append("missing_underlying_price")
    else:
        if underlying < config.min_underlying_price:
            reasons.append("underlying_below_rule_min")
        if underlying > config.max_underlying_price:
            reasons.append("underlying_above_rule_max")

    if strike is None or strike <= 0:
        reasons.append("missing_strike")

    if dte < config.min_dte or dte > config.max_dte:
        reasons.append("dte_outside_rule_window")

    if px is None or px <= 0:
        reasons.append("missing_option_price")
    else:
        if px < config.min_option_price:
            reasons.append("option_price_below_preferred_range")
        if px > config.max_option_price:
            if config.allow_expensive_exceptional and premium >= 1_000_000:
                warnings.append("expensive_premium_exception_used")
            else:
                reasons.append("option_price_above_preferred_range")

    if bid is None or ask is None or ask <= 0 or bid <= 0 or ask < bid:
        reasons.append("bad_bid_ask_quote")
    elif spread_pct is None:
        reasons.append("missing_bid_ask_spread")
    elif spread_pct > config.max_bid_ask_spread_pct:
        reasons.append("wide_bid_ask_spread")

    if premium < config.min_total_premium:
        reasons.append("total_premium_too_small")
    if size < config.min_size:
        reasons.append("size_too_small")
    if volume < config.min_option_volume:
        reasons.append("low_option_volume")
    if open_interest < config.min_open_interest:
        reasons.append("low_open_interest")

    if market_cap is not None and market_cap > config.max_market_cap:
        reasons.append("market_cap_above_preferred_range")

    return {
        "passed": not reasons,
        "reasons": reasons,
        "warnings": warnings,
        "quote": {
            "quote_source": quote_source,
            "bid": round(bid, 4) if bid is not None else None,
            "ask": round(ask, 4) if ask is not None else None,
            "price": round(px, 4) if px is not None else None,
            "bid_ask_spread_pct": round(spread_pct, 4) if spread_pct is not None else None,
        },
        "contract": {
            "option_type": option_type,
            "strike": strike,
            "expiry": safe_str(row.get("expiry")),
            "dte": dte,
            "option_chain_id": safe_str(row.get("option_chain_id")),
            "contract_string": safe_str(row.get("string")),
        },
        "flow": {
            "premium": round(premium, 2),
            "size": size,
            "volume": volume,
            "open_interest": open_interest,
            "volume_open_interest_ratio": round(volume / open_interest, 4) if open_interest > 0 else None,
            "side_ratios": ratios,
            "direction_info": direction_info,
        },
        "structure": structure,
        "underlying_price": underlying,
        "market_cap": market_cap,
    }


def estimate_reward_risk(
    direction: str,
    option_type: Optional[str],
    underlying: Optional[float],
    strike: Optional[float],
    option_price_value: Optional[float],
    implied_volatility: Optional[float],
    dte: Optional[int],
) -> Dict[str, Any]:
    """Simple, auditable payoff estimate for a long option.

    This is not a pricing model. It estimates the payoff if the underlying moves
    one implied-volatility move in the thesis direction by expiration. If IV is
    unavailable, it uses a conservative 10% move proxy.
    """
    if (
        direction not in {"BULLISH", "BEARISH"}
        or option_type not in {"CALL", "PUT"}
        or underlying is None
        or strike is None
        or option_price_value is None
        or option_price_value <= 0
    ):
        return {
            "status": "UNAVAILABLE",
            "reason": "missing_inputs",
            "estimated_reward_to_risk": None,
            "reward_risk_meets_target": False,
        }

    dte = int(dte or 30)
    iv = implied_volatility if implied_volatility is not None and implied_volatility > 0 else None
    move_pct = iv * math.sqrt(max(dte, 1) / 365.0) if iv is not None else 0.10
    move_pct = clamp(move_pct, 0.03, 0.75)

    if direction == "BULLISH":
        target_underlying = underlying * (1.0 + move_pct)
        intrinsic = max(target_underlying - strike, 0.0) if option_type == "CALL" else max(strike - target_underlying, 0.0)
        breakeven = strike + option_price_value if option_type == "CALL" else strike - option_price_value
    else:
        target_underlying = underlying * (1.0 - move_pct)
        intrinsic = max(strike - target_underlying, 0.0) if option_type == "PUT" else max(target_underlying - strike, 0.0)
        breakeven = strike - option_price_value if option_type == "PUT" else strike + option_price_value

    estimated_profit = max(intrinsic - option_price_value, 0.0)
    rr = estimated_profit / option_price_value if option_price_value > 0 else None

    return {
        "status": "OK",
        "move_model": "one_iv_move_to_expiry" if iv is not None else "fallback_10pct_underlying_move",
        "implied_volatility_used": round(iv, 6) if iv is not None else None,
        "estimated_move_pct": round(move_pct, 4),
        "target_underlying_price": round(target_underlying, 4),
        "breakeven_price": round(breakeven, 4),
        "max_loss_per_contract": round(option_price_value * 100.0, 2),
        "estimated_profit_per_contract": round(estimated_profit * 100.0, 2),
        "estimated_reward_to_risk": round(rr, 4) if rr is not None else None,
        "reward_risk_meets_minimum": bool(rr is not None and rr >= 1.0),
        "reward_risk_meets_target": bool(rr is not None and rr >= 5.0),
    }


# -------------------- scoring --------------------
def score_row(row: Dict[str, Any], guardrails: Dict[str, Any], config: RuntimeConfig) -> Dict[str, Any]:
    flow = guardrails["flow"]
    quote = guardrails["quote"]
    contract = guardrails["contract"]
    premium = safe_float(flow.get("premium"), 0.0) or 0.0
    volume = safe_int(flow.get("volume"), 0)
    open_interest = safe_int(flow.get("open_interest"), 0)
    size = safe_int(flow.get("size"), 0)
    ask_ratio = safe_float(flow.get("side_ratios", {}).get("ask_side_ratio"), 0.0) or 0.0
    spread_pct = safe_float(quote.get("bid_ask_spread_pct"))
    px = safe_float(quote.get("price"))
    dte = safe_int(contract.get("dte"), 0)
    delta = safe_float(row.get("delta"))
    tags = safe_str(row.get("tags")).lower()
    next_earnings = parse_date(row.get("next_earnings_date"))
    trade_dt = parse_datetime_from_row(row)
    trade_date = trade_dt.date() if trade_dt else datetime.now(timezone.utc).date()

    # Premium component: log-scaled, because $5M is not 20x better than $250k.
    premium_component = 0.0
    if premium > 0:
        premium_component = clamp(math.log10(max(premium, 1.0) / config.min_total_premium) / math.log10(10.0), 0.0, 1.0)

    side_component = clamp((ask_ratio - 0.40) / 0.60, 0.0, 1.0)

    vol_oi_ratio = volume / open_interest if open_interest > 0 else None
    new_position_component = 0.0
    if vol_oi_ratio is not None:
        new_position_component = clamp(vol_oi_ratio / 3.0, 0.0, 1.0)
    elif volume >= config.min_option_volume:
        new_position_component = 0.30

    liquidity_component = 0.0
    if guardrails["passed"] or not any(r in guardrails["reasons"] for r in ["bad_bid_ask_quote", "wide_bid_ask_spread", "low_option_volume"]):
        liquidity_component += 0.35
    if spread_pct is not None:
        liquidity_component += 0.35 * clamp(1.0 - (spread_pct / max(config.max_bid_ask_spread_pct, 0.01)), 0.0, 1.0)
    if open_interest >= config.min_open_interest:
        liquidity_component += 0.15
    if volume >= config.min_option_volume:
        liquidity_component += 0.15
    liquidity_component = clamp(liquidity_component, 0.0, 1.0)

    affordability_component = 1.0 if px is not None and config.min_option_price <= px <= config.max_option_price else 0.0

    catalyst_component = 0.0
    catalyst_tags: List[str] = []
    if "earnings_this_week" in tags:
        catalyst_component = max(catalyst_component, 1.0)
        catalyst_tags.append("earnings_this_week")
    elif "earnings_next_week" in tags:
        catalyst_component = max(catalyst_component, 0.80)
        catalyst_tags.append("earnings_next_week")
    if next_earnings is not None:
        days_to_earnings = (next_earnings - trade_date).days
        if 0 <= days_to_earnings <= 7:
            catalyst_component = max(catalyst_component, 1.0)
            catalyst_tags.append("earnings_within_7_days")
        elif 8 <= days_to_earnings <= 21:
            catalyst_component = max(catalyst_component, 0.65)
            catalyst_tags.append("earnings_within_21_days")

    delta_component = 0.30
    if delta is not None:
        abs_delta = abs(delta)
        # Prefer options with enough delta to respond, but not so deep ITM that convexity is poor.
        if 0.25 <= abs_delta <= 0.65:
            delta_component = 1.0
        elif 0.15 <= abs_delta < 0.25 or 0.65 < abs_delta <= 0.80:
            delta_component = 0.60
        else:
            delta_component = 0.20

    score = (
        0.22 * premium_component
        + 0.18 * side_component
        + 0.15 * new_position_component
        + 0.18 * liquidity_component
        + 0.12 * affordability_component
        + 0.10 * catalyst_component
        + 0.05 * delta_component
    )

    # Penalize rows that are not actually tradeable by the hard rules.
    if not guardrails["passed"]:
        score *= 0.70

    return {
        "fit_score": round(clamp(score, 0.0, 1.0), 4),
        "score_components": {
            "premium_component": round(premium_component, 4),
            "side_component": round(side_component, 4),
            "new_position_component": round(new_position_component, 4),
            "liquidity_component": round(liquidity_component, 4),
            "affordability_component": round(affordability_component, 4),
            "catalyst_component": round(catalyst_component, 4),
            "delta_component": round(delta_component, 4),
        },
        "catalyst_tags": sorted(set(catalyst_tags)),
    }


def build_risk_flags(
    guardrails: Dict[str, Any],
    reward_risk: Dict[str, Any],
    config: RuntimeConfig,
    post_flow_confirmation: Optional[Dict[str, Any]] = None,
) -> List[str]:
    flags: List[str] = ["unusual_options_flow", "long_premium_only", "downside_limited_to_premium"]
    if guardrails.get("warnings"):
        flags.extend(guardrails["warnings"])
    if not reward_risk.get("reward_risk_meets_target"):
        flags.append("below_5_to_1_target")
    if reward_risk.get("estimated_reward_to_risk") is None:
        flags.append("reward_risk_unavailable")
    elif reward_risk.get("estimated_reward_to_risk", 0) < config.min_reward_to_risk:
        flags.append("low_estimated_reward_to_risk")

    if post_flow_confirmation:
        status = safe_str(post_flow_confirmation.get("status"))
        if status == "CONFIRMED":
            flags.append("post_flow_underlying_confirmed")
        elif status in {"CONTRADICTED", "AMBIGUOUS", "UNAVAILABLE"}:
            flags.append(f"post_flow_{status.lower()}")
        synthetic = post_flow_confirmation.get("synthetic_flow_review", {}) or {}
        if synthetic.get("synthetic_or_complex_flow_possible"):
            flags.append("possible_synthetic_or_complex_flow")
    return sorted(set(flags))


def candidate_id_from_row(row: Dict[str, Any], idx: int) -> str:
    symbol = safe_str(row.get("underlying_symbol"), "UNKNOWN").upper()
    option_id = safe_str(row.get("option_chain_id")) or safe_str(row.get("string"))
    option_id = re.sub(r"[^A-Za-z0-9_]+", "_", option_id).strip("_")[:40]
    return f"UOA_{symbol}_{option_id or idx}_{idx}"


def build_candidate(
    row: Dict[str, Any],
    idx: int,
    guardrails: Dict[str, Any],
    score: Dict[str, Any],
    reward_risk: Dict[str, Any],
    config: RuntimeConfig,
    post_flow_confirmation: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    symbol = safe_str(row.get("underlying_symbol")).upper()
    company = safe_str(row.get("full_name")) or symbol
    direction = guardrails["flow"]["direction_info"]["direction"]
    structure = guardrails["structure"]
    contract = guardrails["contract"]
    quote = guardrails["quote"]
    flow = guardrails["flow"]
    dte = safe_int(contract.get("dte"), 0)
    horizon = "SHORT_TERM" if dte <= 60 else "LONG_TERM"
    rr = reward_risk.get("estimated_reward_to_risk")

    confidence = clamp(
        0.50 * score["fit_score"]
        + 0.20 * (1.0 if guardrails["passed"] else 0.0)
        + 0.15 * (1.0 if reward_risk.get("reward_risk_meets_minimum") else 0.0)
        + 0.15 * (1.0 if flow["direction_info"].get("buyer_initiated") else 0.0),
        0.0,
        1.0,
    )

    thesis_tags = [
        "uoa",
        "buyer_initiated" if flow["direction_info"].get("buyer_initiated") else "ambiguous_side",
        "ask_side" if flow.get("side_ratios", {}).get("ask_side_ratio", 0) else "flow_side_unknown",
        "premium_flow",
    ]
    thesis_tags.extend(score.get("catalyst_tags") or [])
    if flow.get("volume_open_interest_ratio") is not None and flow["volume_open_interest_ratio"] >= 1.0:
        thesis_tags.append("volume_gt_open_interest")
    if rr is not None and rr >= config.target_reward_to_risk:
        thesis_tags.append("five_to_one_candidate")
    if post_flow_confirmation and post_flow_confirmation.get("status") == "CONFIRMED":
        thesis_tags.append("post_flow_confirmed")
    if (post_flow_confirmation or {}).get("synthetic_flow_review", {}).get("synthetic_or_complex_flow_possible"):
        thesis_tags.append("synthetic_risk_reviewed")

    summary = (
        f"{direction} unusual options flow in {symbol}: {flow['premium']:,.0f} premium on "
        f"{contract['contract_string'] or (str(contract['strike']) + ' ' + str(contract['option_type']))}; "
        f"structure={structure}, DTE={contract['dte']}, option_price={quote['price']}."
    )

    return {
        "candidate_id": candidate_id_from_row(row, idx),
        "strategy_type": STRATEGY_TYPE,
        "direction": direction,
        "horizon": horizon,
        "structure_family": structure,
        "summary": summary,
        "confidence": round(confidence, 4),
        "fit_score": score["fit_score"],
        "thesis_tags": sorted(set(thesis_tags)),
        "risk_flags": build_risk_flags(guardrails, reward_risk, config, post_flow_confirmation),
        "implementation": {
            "signal_basis": "UNUSUAL_OPTIONS_FLOW",
            "symbol": symbol,
            "company_name": company,
            "sector": safe_str(row.get("sector")) or None,
            "industry_type": safe_str(row.get("industry_type")) or None,
            "observed_flow": {
                **flow,
                "date": safe_str(row.get("date")) or None,
                "time": safe_str(row.get("time")) or None,
                "exchange": safe_str(row.get("exchange")) or None,
                "report_flags": safe_str(row.get("report_flags")) or None,
                "tags": safe_str(row.get("tags")) or None,
            },
            "contract": contract,
            "quote": quote,
            "underlying": {
                "underlying_price": round(guardrails["underlying_price"], 4) if guardrails.get("underlying_price") is not None else None,
                "market_cap": round(guardrails["market_cap"], 2) if guardrails.get("market_cap") is not None else None,
            },
            "greeks": {
                "implied_volatility": safe_float(row.get("implied_volatility")),
                "delta": safe_float(row.get("delta")),
                "theta": safe_float(row.get("theta")),
                "gamma": safe_float(row.get("gamma")),
                "vega": safe_float(row.get("vega")),
                "rho": safe_float(row.get("rho")),
                "theo": safe_float(row.get("theo")),
            },
            "catalyst": {
                "next_earnings_date": safe_str(row.get("next_earnings_date")) or None,
                "er_time": safe_str(row.get("er_time")) or None,
                "catalyst_tags": score.get("catalyst_tags") or [],
            },
            "selected_trade": {
                "preferred_structure": structure,
                "expiration": contract.get("expiry"),
                "strike": contract.get("strike"),
                "option_type": contract.get("option_type"),
                "estimated_entry_price": quote.get("price"),
                "maximum_loss_per_contract": reward_risk.get("max_loss_per_contract"),
                "downside_limited_to_premium": True,
                "direct_order_routing_allowed": False,
            },
            "reward_risk": reward_risk,
            "post_flow_confirmation": post_flow_confirmation or {"enabled": False, "status": "NOT_EVALUATED"},
            "synthetic_flow_review": (post_flow_confirmation or {}).get("synthetic_flow_review") or infer_synthetic_risk(row, guardrails),
            "score_components": score["score_components"],
            "hard_limits": {
                "undefined_risk_allowed": False,
                "short_naked_options_allowed": False,
                "stock_short_allowed": False,
                "direct_order_routing_allowed": False,
            },
        },
    }


# -------------------- main analysis --------------------
def analyze_flow_dataframe(df: pd.DataFrame, config: RuntimeConfig) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    watchlist: List[Dict[str, Any]] = []
    reason_counts: Dict[str, int] = {}
    intraday_cache: Dict[str, Any] = {}

    for idx, row_series in df.iterrows():
        row = row_dict(row_series)
        guardrails = evaluate_row_guardrails(row, config)
        score = score_row(row, guardrails, config)
        quote = guardrails["quote"]
        contract = guardrails["contract"]
        flow = guardrails["flow"]
        reward_risk = estimate_reward_risk(
            direction=flow["direction_info"]["direction"],
            option_type=contract.get("option_type"),
            underlying=guardrails.get("underlying_price"),
            strike=safe_float(contract.get("strike")),
            option_price_value=safe_float(quote.get("price")),
            implied_volatility=safe_float(row.get("implied_volatility")),
            dte=safe_int(contract.get("dte"), 30),
        )

        post_flow_confirmation = evaluate_post_flow_confirmation(row, guardrails, config, intraday_cache)
        if config.require_post_flow_confirmation and not post_flow_confirmation.get("passed"):
            status = safe_str(post_flow_confirmation.get("status")).lower() or "failed"
            guardrails["reasons"].append(f"post_flow_confirmation_{status}")
            guardrails["reasons"].append(safe_str(post_flow_confirmation.get("reason"), "post_flow_confirmation_failed"))
            guardrails["passed"] = False

        # Reward/risk is a soft-but-important filter. If a candidate cannot even
        # clear 1:1 under the simple thesis-move model, keep it off the main queue.
        rr = safe_float(reward_risk.get("estimated_reward_to_risk"))
        if rr is not None and rr < config.min_reward_to_risk:
            guardrails["reasons"].append("reward_risk_below_minimum")
            guardrails["passed"] = False

        if guardrails["passed"] and score["fit_score"] >= config.min_fit_score:
            candidates.append(build_candidate(row, int(idx), guardrails, score, reward_risk, config, post_flow_confirmation))
        elif score["fit_score"] >= config.watchlist_min_fit_score:
            watchlist.append({
                "watchlist_id": candidate_id_from_row(row, int(idx)),
                "symbol": safe_str(row.get("underlying_symbol")).upper(),
                "status": "WATCHLIST",
                "fit_score": score["fit_score"],
                "reasons": guardrails["reasons"],
                "summary": safe_str(row.get("string")),
                "guardrails": guardrails,
                "reward_risk": reward_risk,
                "post_flow_confirmation": post_flow_confirmation,
            })
        else:
            rejected.append({
                "symbol": safe_str(row.get("underlying_symbol")).upper(),
                "row_number": int(idx) + 2,
                "fit_score": score["fit_score"],
                "reasons": guardrails["reasons"] or ["fit_score_below_threshold"],
                "contract": guardrails["contract"],
                "quote": guardrails["quote"],
            })

        for reason in guardrails["reasons"]:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    # Rank by fit first, then confidence, then larger premium.
    candidates.sort(
        key=lambda c: (
            safe_float(c.get("fit_score"), 0.0) or 0.0,
            safe_float(c.get("confidence"), 0.0) or 0.0,
            safe_float(c.get("implementation", {}).get("observed_flow", {}).get("premium"), 0.0) or 0.0,
        ),
        reverse=True,
    )
    watchlist.sort(key=lambda w: safe_float(w.get("fit_score"), 0.0) or 0.0, reverse=True)

    diagnostics = {
        "input_rows": int(len(df)),
        "candidate_count_before_cap": int(len(candidates)),
        "watchlist_count_before_cap": int(len(watchlist)),
        "rejected_count": int(len(rejected)),
        "reason_counts": dict(sorted(reason_counts.items(), key=lambda kv: kv[1], reverse=True)),
    }

    return candidates[: config.max_candidates], watchlist[: config.max_watchlist], {**diagnostics, "rejected": rejected[:100]}


def build_payload(config: RuntimeConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    input_path = resolve_runtime_path(config.input_file, DEFAULT_INPUT_FILE)
    log(f"Loading unusual options CSV: {input_path}", config.verbose)
    df = load_flow_csv(input_path)

    candidates, watchlist, diagnostics = analyze_flow_dataframe(df, config)
    event_id = now_iso()

    payload = {
        "agent": AGENT_NAME,
        "schema_version": SCHEMA_VERSION,
        "event_id": event_id,
        "strategy_type": STRATEGY_TYPE,
        "candidates": candidates,
        "summary": (
            f"Unusual Options Analyst identified {len(candidates)} tradeable long-premium candidates "
            f"and {len(watchlist)} watchlist setups from {len(df)} flow rows."
        ),
    }
    validate_output_schema(payload)

    full_report = {
        "output": payload,
        "diagnostics": diagnostics,
        "watchlist": watchlist,
        "config": asdict(config),
        "input_file": str(input_path),
        "notes": [
            "This script is analysis-only and does not place trades.",
            "Candidates are filtered to long-premium structures with maximum loss limited to premium paid.",
            "Reward/risk is estimated using a simple one-IV-move-to-expiration model, not a full options pricing model.",
            "UOA direction is now gated by 10-minute post-flow underlying confirmation to reduce synthetic/complex-flow false positives.",
        ],
    }
    return payload, full_report


def write_report(report: Dict[str, Any]) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = REPORTS_DIR / f"uoa_payload_{stamp}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=jsonable)
    return path


def parse_args() -> RuntimeConfig:
    parser = argparse.ArgumentParser(description="Unusual Options Activity Analyst")
    parser.add_argument("--input-file", default=DEFAULT_INPUT_FILE)
    parser.add_argument("--max-candidates", type=int, default=10)
    parser.add_argument("--max-watchlist", type=int, default=25)
    parser.add_argument("--min-total-premium", type=float, default=250_000.0)
    parser.add_argument("--min-option-volume", type=int, default=100)
    parser.add_argument("--min-open-interest", type=int, default=20)
    parser.add_argument("--max-bid-ask-spread-pct", type=float, default=0.35)
    parser.add_argument("--min-ask-side-ratio", type=float, default=0.60)
    parser.add_argument("--min-fit-score", type=float, default=0.55)
    parser.add_argument("--watchlist-min-fit-score", type=float, default=0.40)
    parser.add_argument("--min-reward-to-risk", type=float, default=1.00)
    parser.add_argument("--target-reward-to-risk", type=float, default=5.00)
    parser.add_argument("--min-underlying-price", type=float, default=5.00)
    parser.add_argument("--max-underlying-price", type=float, default=100.00)
    parser.add_argument("--min-option-price", type=float, default=0.50)
    parser.add_argument("--max-option-price", type=float, default=5.00)
    parser.add_argument("--max-dte", type=int, default=60)
    parser.add_argument("--max-market-cap", type=float, default=20_000_000_000.0)
    parser.add_argument("--allow-expensive-exceptional", action="store_true")
    parser.add_argument("--no-require-buyer-initiated", action="store_true")
    parser.add_argument("--no-require-post-flow-confirmation", action="store_true", help="Disable 10-minute underlying confirmation gate.")
    parser.add_argument("--allow-missing-post-flow-confirmation", action="store_true", help="Warn instead of reject if 1-minute bars are unavailable. Contradictions still fail.")
    parser.add_argument("--post-flow-window-minutes", type=int, default=10)
    parser.add_argument("--min-post-flow-confirm-move-pct", type=float, default=0.0010, help="Minimum 10-minute underlying move in thesis direction, e.g. 0.001 = 0.10%.")
    parser.add_argument("--trade-timestamp-timezone", default="America/New_York")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    return RuntimeConfig(
        input_file=args.input_file,
        max_candidates=args.max_candidates,
        max_watchlist=args.max_watchlist,
        min_total_premium=args.min_total_premium,
        min_option_volume=args.min_option_volume,
        min_open_interest=args.min_open_interest,
        max_bid_ask_spread_pct=args.max_bid_ask_spread_pct,
        min_ask_side_ratio=args.min_ask_side_ratio,
        min_fit_score=args.min_fit_score,
        watchlist_min_fit_score=args.watchlist_min_fit_score,
        min_reward_to_risk=args.min_reward_to_risk,
        target_reward_to_risk=args.target_reward_to_risk,
        min_underlying_price=args.min_underlying_price,
        max_underlying_price=args.max_underlying_price,
        min_option_price=args.min_option_price,
        max_option_price=args.max_option_price,
        max_dte=args.max_dte,
        max_market_cap=args.max_market_cap,
        allow_expensive_exceptional=bool(args.allow_expensive_exceptional),
        require_buyer_initiated=not bool(args.no_require_buyer_initiated),
        require_post_flow_confirmation=not bool(args.no_require_post_flow_confirmation),
        allow_missing_post_flow_confirmation=bool(args.allow_missing_post_flow_confirmation),
        post_flow_window_minutes=int(args.post_flow_window_minutes),
        min_post_flow_confirm_move_pct=float(args.min_post_flow_confirm_move_pct),
        trade_timestamp_timezone=str(args.trade_timestamp_timezone),
        verbose=bool(args.verbose),
    )


def main() -> None:
    config = parse_args()
    try:
        payload, report = build_payload(config)
        report_path = write_report(report)
        print(json.dumps(payload, indent=2, default=jsonable))
        print(f"[INFO] Wrote UOA diagnostics report to {report_path}", file=sys.stderr)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
