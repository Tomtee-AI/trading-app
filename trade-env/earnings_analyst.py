#!/usr/bin/env python3
"""
earnings_analyst.py

Deterministic Earnings Trade Analyst.

Mission
-------
Identify names where implied earnings volatility appears cheap versus historical
post-earnings moves and build defined-risk / long-premium trade templates.

Designed for the current multi-agent trading system:
- coordinator-compatible specialist payload on stdout
- richer diagnostics report written to reports/
- no direct order routing
- strict JSON schema validation

Input
-----
Default workbook: earnings.xlsx
Expected workbook shape is based on the user's example:
    Symbol | EVR | Recent Close | Avg Volume | Earning Date | Max One Day Move | ... | Position | Option Cost | Implied Move
Rows with a blank Symbol are treated as supporting prior earnings history rows.

Strategy families allowed
-------------------------
- LONG_CALL
- LONG_PUT
- CALL_DEBIT_SPREAD
- PUT_DEBIT_SPREAD
- STRADDLE
- STRANGLE

Earnings date validation
------------------------
Best-effort validation against multiple sources:
1. Input workbook date
2. yfinance calendar
3. yfinance earnings_dates
4. optional Finnhub, if FINNHUB_API_KEY is set
5. optional Financial Modeling Prep, if FMP_API_KEY is set

The script can still run without API keys. Date validation is reported in diagnostics.

Vol / expected-move analysis
----------------------------
If tos_options_agent_functions.py is available, the script calls
build_options_analysis_packet(...) using OHLC data and a deterministic IV series.
If the helper is unavailable, the script still runs and reports a fallback state.

Install
-------
pip install pandas numpy yfinance jsonschema requests openpyxl python-dotenv

Run
---
python earnings_analyst.py
python earnings_analyst.py --input-file earnings.xlsx
python earnings_analyst.py --min-cheap-ratio 1.05 --max-candidates 10
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone, time as dt_time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover - fallback for very old Python runtimes
    ZoneInfo = None  # type: ignore

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from jsonschema import Draft202012Validator, FormatChecker

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SCRIPT_DIR = Path(__file__).resolve().parent
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_INPUT_FILE = "earnings.xlsx"
DEFAULT_OPTIONS_MODULE_PATH = "tos_options_agent_functions.py"

ALLOWED_STRUCTURES = [
    "LONG_CALL",
    "LONG_PUT",
    "CALL_DEBIT_SPREAD",
    "PUT_DEBIT_SPREAD",
    "STRADDLE",
    "STRANGLE",
]

DATE_SOURCE_NAMES = [
    "input_workbook",
    "yfinance_calendar",
    "yfinance_earnings_dates",
    "finnhub",
    "fmp",
]

COORDINATOR_CANDIDATE_SCHEMA = {
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
        "strategy_type": {"type": "string", "const": "earnings"},
        "direction": {"type": "string", "enum": ["BULLISH", "BEARISH", "NEUTRAL"]},
        "horizon": {"type": "string", "const": "SHORT_TERM"},
        "structure_family": {"type": "string", "enum": ALLOWED_STRUCTURES},
        "summary": {"type": "string", "minLength": 1},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "fit_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "thesis_tags": {"type": "array", "items": {"type": "string"}},
        "risk_flags": {"type": "array", "items": {"type": "string"}},
        "implementation": {"type": "object", "additionalProperties": True},
    },
}

OUTPUT_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": ["agent", "schema_version", "event_id", "strategy_type", "candidates", "summary"],
    "properties": {
        "agent": {"type": "string", "const": "earnings_analyst"},
        "schema_version": {"type": "string", "const": "1.0.0"},
        "event_id": {"type": "string", "format": "date-time"},
        "strategy_type": {"type": "string", "const": "earnings"},
        "candidates": {"type": "array", "items": COORDINATOR_CANDIDATE_SCHEMA},
        "summary": {"type": "string"},
    },
}


@dataclass
class RuntimeConfig:
    input_file: str = DEFAULT_INPUT_FILE
    options_module_path: str = DEFAULT_OPTIONS_MODULE_PATH
    history_period: str = "2y"
    history_interval: str = "1d"
    min_avg_volume: int = 250_000
    min_cheap_ratio: float = 1.00
    cheap_ratio_watchlist: float = 0.85
    expensive_ratio: float = 1.25
    min_historical_samples: int = 2
    max_days_to_earnings: int = 45
    min_days_to_earnings: int = 0
    min_fit_score: float = 0.55
    watchlist_min_fit_score: float = 0.40
    max_candidates: int = 15
    max_watchlist: int = 25
    strangle_discount_threshold: float = 0.70
    debit_spread_iv_threshold: float = 0.90
    strict_date_validation: bool = False
    date_validation_tolerance_days: int = 2
    request_timeout: int = 12
    market_timezone: str = "America/New_York"
    market_open_time: str = "09:30"
    market_close_time: str = "16:00"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_clock_time(value: str, default: dt_time) -> dt_time:
    """Parse a HH:MM clock-time string safely.

    The earnings analyst uses this for coarse event-staleness checks.  The
    function intentionally falls back to a conservative default instead of
    raising, because bad CLI/config values should not crash the full agent.
    """
    try:
        hour_text, minute_text = str(value).strip().split(":", 1)
        hour = int(hour_text)
        minute = int(minute_text)
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            return dt_time(hour=hour, minute=minute)
    except Exception:
        pass
    return default


def get_market_timezone(config: RuntimeConfig):
    """Return the configured market timezone, falling back safely to UTC."""
    if ZoneInfo is None:
        return timezone.utc
    try:
        return ZoneInfo(config.market_timezone)
    except Exception:
        return timezone.utc


def assess_earnings_event_timing(
    earnings_date: date,
    earnings_time: Any,
    config: RuntimeConfig,
    as_of_utc: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Determine whether a listed earnings event is still tradable.

    Rule enforced here:
        Same-day BEFORE_OPEN earnings events must not be promoted after the
        regular cash session has opened.  Once the stock has already reported
        before the open, a new long-premium earnings-volatility entry is stale;
        the event catalyst has passed and implied volatility may already have
        collapsed.

    This function is intentionally conservative and deterministic.  It does
    not try to infer exact press-release times.  It uses the configured market
    timezone and regular session open as a practical cutoff.
    """
    tz = get_market_timezone(config)
    now_market = (as_of_utc or datetime.now(timezone.utc)).astimezone(tz)
    earnings_time_text = str(earnings_time or "UNKNOWN").upper()
    market_open = parse_clock_time(config.market_open_time, dt_time(9, 30))
    market_close = parse_clock_time(config.market_close_time, dt_time(16, 0))
    market_open_dt = datetime.combine(earnings_date, market_open, tzinfo=tz)
    market_close_dt = datetime.combine(earnings_date, market_close, tzinfo=tz)

    result = {
        "status": "PENDING",
        "eligible_for_new_entry": True,
        "reason": "event_not_yet_past",
        "market_timezone": getattr(tz, "key", str(tz)),
        "as_of_market_time": now_market.isoformat(),
        "market_open_cutoff": market_open_dt.isoformat(),
        "market_close_reference": market_close_dt.isoformat(),
    }

    # Previous-day events are stale regardless of the reported time bucket.
    if earnings_date < now_market.date():
        result.update({
            "status": "PASSED",
            "eligible_for_new_entry": False,
            "reason": "earnings_date_before_market_today",
        })
        return result

    # Future-dated events are still tradable from a timing perspective.
    if earnings_date > now_market.date():
        return result

    # Same-day before-open events are stale once the regular session opens.
    if earnings_time_text == "BEFORE_OPEN" and now_market >= market_open_dt:
        result.update({
            "status": "PASSED",
            "eligible_for_new_entry": False,
            "reason": "same_day_before_open_event_after_market_open",
        })
        return result

    # Same-day after-close and unknown events remain eligible here.  Other
    # validation layers still handle liquidity, cheap-vol, and date confidence.
    if earnings_time_text == "AFTER_CLOSE":
        result.update({"reason": "same_day_after_close_event_not_blocked_by_before_open_rule"})
    elif earnings_time_text == "UNKNOWN":
        result.update({"reason": "same_day_unknown_earnings_time_not_blocked"})

    return result


def as_jsonable(value: Any) -> Any:
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None or pd.isna(value):
            return default
        if isinstance(value, str):
            cleaned = value.strip().replace("%", "")
            if cleaned.lower() in {"none", "-none", "nan", "#value!", ""}:
                return default
            return float(cleaned)
        return float(value)
    except Exception:
        return default


def safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        f = safe_float(value)
        if f is None:
            return default
        return int(f)
    except Exception:
        return default


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def resolve_runtime_path(user_value: Optional[str], default_filename: str) -> str:
    """
    Resolve file paths in a Windows-friendly way.

    Order:
      1. absolute path
      2. current working directory
      3. script directory
      4. fallback literal default
    """
    raw = user_value or default_filename
    p = Path(raw)
    if p.is_absolute() and p.exists():
        return str(p)

    cwd_candidate = Path.cwd() / raw
    if cwd_candidate.exists():
        return str(cwd_candidate)

    script_candidate = SCRIPT_DIR / raw
    if script_candidate.exists():
        return str(script_candidate)

    return str(p)


def validate_output_schema(payload: Dict[str, Any]) -> None:
    validator = Draft202012Validator(OUTPUT_SCHEMA, format_checker=FormatChecker())
    errors = sorted(validator.iter_errors(payload), key=lambda e: list(e.path))
    if errors:
        lines = []
        for err in errors[:10]:
            path = ".".join(str(p) for p in err.path) if err.path else "<root>"
            lines.append(f"{path}: {err.message}")
        raise ValueError("Output JSON schema validation failed:\n" + "\n".join(lines))


def import_options_module(module_path: str):
    resolved = resolve_runtime_path(module_path, DEFAULT_OPTIONS_MODULE_PATH)
    module_file = Path(resolved)
    if not module_file.exists():
        return None

    try:
        spec = importlib.util.spec_from_file_location("tos_options_agent_functions", module_file)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not hasattr(module, "build_options_analysis_packet"):
            return None
        return module
    except Exception:
        return None


# -------------------- input parsing --------------------
def parse_excel_date(value: Any) -> Optional[date]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None

    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value

    # Excel serial dates. The example workbook has serials such as 45784.
    if isinstance(value, (int, float, np.integer, np.floating)) and not pd.isna(value):
        try:
            if 20_000 < float(value) < 80_000:
                return (pd.Timestamp("1899-12-30") + pd.to_timedelta(float(value), unit="D")).date()
        except Exception:
            pass

    text = str(value).strip()
    if not text or text.lower() in {"none", "nan", "#value!"}:
        return None

    # Remove common earnings time suffixes.
    text = re.sub(r"\b(AC|AMC|PM|AFTER CLOSE|AFTER MARKET|BO|BMO|AM|BEFORE OPEN|BEFORE MARKET)\b", "", text, flags=re.I).strip()
    text = text.replace(".", "")

    candidates = [
        text,
        text.replace(",", ""),
    ]
    formats = [
        "%b %d, %Y", "%B %d, %Y", "%b %d %Y", "%B %d %Y",
        "%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y",
    ]
    for c in candidates:
        for fmt in formats:
            try:
                return datetime.strptime(c, fmt).date()
            except Exception:
                continue

    try:
        parsed = pd.to_datetime(text, errors="coerce")
        if pd.notna(parsed):
            return parsed.date()
    except Exception:
        pass

    return None


def parse_earnings_time(value: Any) -> str:
    text = str(value or "").upper()
    if any(tok in text for tok in ["AC", "AMC", "PM", "AFTER"]):
        return "AFTER_CLOSE"
    if any(tok in text for tok in ["BO", "BMO", "AM", "BEFORE"]):
        return "BEFORE_OPEN"
    return "UNKNOWN"


def normalize_move_value(value: Any) -> Optional[float]:
    """
    Return move as decimal fraction, e.g. 0.0924 = 9.24%.
    """
    x = safe_float(value)
    if x is None:
        return None
    # If user accidentally supplies 9.24 instead of 0.0924, normalize.
    if abs(x) > 2.0:
        return x / 100.0
    return x


def load_earnings_workbook(path: str) -> List[Dict[str, Any]]:
    df = pd.read_excel(path, sheet_name=0, engine="openpyxl")
    df = df.replace({np.nan: None})

    records: List[Dict[str, Any]] = []
    for idx, row in df.iterrows():
        symbol_raw = row.get("Symbol")
        if symbol_raw is None:
            continue
        symbol = str(symbol_raw).strip().upper()
        if not symbol or symbol in {"#VALUE!", "NAN", "NONE"}:
            continue

        earnings_raw = row.get("Earning Date")
        earnings_date = parse_excel_date(earnings_raw)
        if earnings_date is None:
            continue

        # Historical move columns: in the example workbook, max one-day move and the next 3 columns hold moves.
        move_values: List[float] = []
        for col in ["Max One Day Move"]:
            mv = normalize_move_value(row.get(col))
            if mv is not None:
                move_values.append(mv)

        # Pull unlabeled historical move columns by index positions F:I if present.
        # pandas may name blank headers as Unnamed: 6, Unnamed: 7, etc.
        for col in df.columns:
            if str(col).startswith("Unnamed"):
                mv = normalize_move_value(row.get(col))
                if mv is not None and abs(mv) <= 2.0:
                    move_values.append(mv)

        prior_dates: List[str] = []
        if idx + 1 < len(df):
            next_row = df.iloc[idx + 1]
            if next_row.get("Symbol") is None:
                for value in next_row.values:
                    d = parse_excel_date(value)
                    if d is not None:
                        prior_dates.append(d.isoformat())

        implied_move = normalize_move_value(row.get("Implied Move"))
        option_cost = safe_float(row.get("Option Cost"))
        recent_close = safe_float(row.get("Recent Close"))

        abs_moves = [abs(x) for x in move_values if x is not None]
        avg_abs_move = float(np.mean(abs_moves)) if abs_moves else None
        max_abs_move = float(np.max(abs_moves)) if abs_moves else None

        records.append({
            "symbol": symbol,
            "evr": row.get("EVR"),
            "recent_close": recent_close,
            "avg_volume": safe_int(row.get("Avg Volume")),
            "earnings_date": earnings_date,
            "earnings_time": parse_earnings_time(earnings_raw),
            "earnings_raw": str(earnings_raw),
            "historical_moves": [round(float(x), 6) for x in move_values],
            "historical_abs_moves": [round(float(x), 6) for x in abs_moves],
            "avg_abs_earnings_move": round(avg_abs_move, 6) if avg_abs_move is not None else None,
            "max_abs_earnings_move": round(max_abs_move, 6) if max_abs_move is not None else None,
            "prior_earnings_dates": prior_dates,
            "position_hint": row.get("Position"),
            "sheet_option_cost": option_cost,
            "sheet_implied_move": implied_move,
            "row_number": int(idx + 2),
        })

    return records


# -------------------- earnings date validation --------------------
def date_diff_days(a: Optional[date], b: Optional[date]) -> Optional[int]:
    if a is None or b is None:
        return None
    return abs((a - b).days)


def source_date_entry(source: str, value: Optional[date], status: str, detail: str = "") -> Dict[str, Any]:
    return {
        "source": source,
        "date": value.isoformat() if value else None,
        "status": status,
        "detail": detail,
    }


def fetch_yfinance_calendar_date(symbol: str) -> Dict[str, Any]:
    try:
        cal = yf.Ticker(symbol).calendar
        if cal is None:
            return source_date_entry("yfinance_calendar", None, "UNAVAILABLE", "calendar is None")

        # yfinance has returned both DataFrame and dict forms across versions.
        if isinstance(cal, pd.DataFrame) and not cal.empty:
            for key in ["Earnings Date", "Earnings High", "Earnings Low"]:
                if key in cal.index:
                    vals = cal.loc[key].values
                    for v in vals:
                        d = parse_excel_date(v)
                        if d:
                            return source_date_entry("yfinance_calendar", d, "OK")
            for value in cal.values.flatten():
                d = parse_excel_date(value)
                if d:
                    return source_date_entry("yfinance_calendar", d, "OK")

        if isinstance(cal, dict):
            for key in ["Earnings Date", "Earnings High", "Earnings Low", "earningsDate"]:
                value = cal.get(key)
                if isinstance(value, (list, tuple)):
                    for item in value:
                        d = parse_excel_date(item)
                        if d:
                            return source_date_entry("yfinance_calendar", d, "OK")
                else:
                    d = parse_excel_date(value)
                    if d:
                        return source_date_entry("yfinance_calendar", d, "OK")

        return source_date_entry("yfinance_calendar", None, "NO_DATE", "No parseable date in calendar")
    except Exception as exc:
        return source_date_entry("yfinance_calendar", None, "ERROR", str(exc))


def fetch_yfinance_earnings_dates(symbol: str, target_date: date) -> Dict[str, Any]:
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.get_earnings_dates(limit=16)
        if df is None or df.empty:
            return source_date_entry("yfinance_earnings_dates", None, "UNAVAILABLE", "No earnings_dates rows")

        dates: List[date] = []
        # Often the earnings date is the index.
        if isinstance(df.index, pd.DatetimeIndex):
            dates.extend([ts.date() for ts in df.index if pd.notna(ts)])

        for col in df.columns:
            for val in df[col].values:
                d = parse_excel_date(val)
                if d:
                    dates.append(d)

        if not dates:
            return source_date_entry("yfinance_earnings_dates", None, "NO_DATE")

        nearest = min(dates, key=lambda d: abs((d - target_date).days))
        return source_date_entry("yfinance_earnings_dates", nearest, "OK")
    except Exception as exc:
        return source_date_entry("yfinance_earnings_dates", None, "ERROR", str(exc))


def fetch_finnhub_earnings_date(symbol: str, target_date: date, timeout: int = 12) -> Dict[str, Any]:
    token = os.getenv("FINNHUB_API_KEY")
    if not token:
        return source_date_entry("finnhub", None, "SKIPPED", "FINNHUB_API_KEY not set")

    start = (target_date - timedelta(days=30)).isoformat()
    end = (target_date + timedelta(days=30)).isoformat()
    url = "https://finnhub.io/api/v1/calendar/earnings"
    try:
        resp = requests.get(url, params={"symbol": symbol, "from": start, "to": end, "token": token}, timeout=timeout)
        resp.raise_for_status()
        rows = resp.json().get("earningsCalendar", [])
        dates = [parse_excel_date(row.get("date")) for row in rows]
        dates = [d for d in dates if d]
        if not dates:
            return source_date_entry("finnhub", None, "NO_DATE")
        nearest = min(dates, key=lambda d: abs((d - target_date).days))
        return source_date_entry("finnhub", nearest, "OK")
    except Exception as exc:
        return source_date_entry("finnhub", None, "ERROR", str(exc))


def fetch_fmp_earnings_date(symbol: str, target_date: date, timeout: int = 12) -> Dict[str, Any]:
    token = os.getenv("FMP_API_KEY")
    if not token:
        return source_date_entry("fmp", None, "SKIPPED", "FMP_API_KEY not set")

    # FMP has multiple earnings endpoints across plans. This endpoint works for many accounts.
    url = f"https://financialmodelingprep.com/api/v3/historical/earning_calendar/{symbol}"
    try:
        resp = requests.get(url, params={"apikey": token, "limit": 20}, timeout=timeout)
        resp.raise_for_status()
        rows = resp.json()
        if not isinstance(rows, list):
            return source_date_entry("fmp", None, "NO_DATE", "Unexpected response shape")
        dates = [parse_excel_date(row.get("date")) for row in rows if isinstance(row, dict)]
        dates = [d for d in dates if d]
        if not dates:
            return source_date_entry("fmp", None, "NO_DATE")
        nearest = min(dates, key=lambda d: abs((d - target_date).days))
        return source_date_entry("fmp", nearest, "OK")
    except Exception as exc:
        return source_date_entry("fmp", None, "ERROR", str(exc))


def validate_earnings_date(symbol: str, input_date: date, config: RuntimeConfig) -> Dict[str, Any]:
    sources = [
        source_date_entry("input_workbook", input_date, "OK"),
        fetch_yfinance_calendar_date(symbol),
        fetch_yfinance_earnings_dates(symbol, input_date),
        fetch_finnhub_earnings_date(symbol, input_date, config.request_timeout),
        fetch_fmp_earnings_date(symbol, input_date, config.request_timeout),
    ]

    ok_dates = []
    for item in sources:
        d = parse_excel_date(item.get("date"))
        if item.get("status") == "OK" and d:
            ok_dates.append((item["source"], d))

    agreements = []
    for source, d in ok_dates:
        diff = date_diff_days(input_date, d)
        if diff is not None and diff <= config.date_validation_tolerance_days:
            agreements.append(source)

    external_ok = [source for source, d in ok_dates if source != "input_workbook"]
    external_agree = [source for source in agreements if source != "input_workbook"]

    if len(external_agree) >= 2:
        status = "CONFIRMED_3_SOURCE"
    elif len(external_agree) >= 1:
        status = "PARTIAL_CONFIRMATION"
    elif len(external_ok) == 0:
        status = "UNCONFIRMED_NO_EXTERNAL_DATA"
    else:
        status = "CONFLICT"

    return {
        "input_date": input_date.isoformat(),
        "status": status,
        "agreement_sources": agreements,
        "external_sources_available": external_ok,
        "sources": sources,
    }


# -------------------- market/options data --------------------
def fetch_history(symbol: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=False)
    df = df.dropna(how="all")
    if df.empty:
        raise ValueError(f"No history returned for {symbol}")
    df.index = pd.to_datetime(df.index)
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    return df


def latest_close_from_history(df: pd.DataFrame) -> Optional[float]:
    if df is None or df.empty or "Close" not in df.columns:
        return None
    close = df["Close"].dropna()
    if close.empty:
        return None
    return safe_float(close.iloc[-1])


def get_option_expirations(symbol: str) -> List[str]:
    try:
        return list(yf.Ticker(symbol).options or [])
    except Exception:
        return []


def third_friday(year: int, month: int) -> date:
    first = date(year, month, 1)
    days_until_friday = (4 - first.weekday()) % 7
    return first + timedelta(days=days_until_friday + 14)


def monthly_opex_cycle_for_earnings(earnings_date: date) -> Dict[str, Any]:
    exp = third_friday(earnings_date.year, earnings_date.month)
    if earnings_date > exp:
        next_month = pd.Timestamp(earnings_date) + pd.offsets.MonthBegin(1)
        exp = third_friday(int(next_month.year), int(next_month.month))

    cycle_start = third_friday((pd.Timestamp(exp) - pd.offsets.MonthBegin(1)).year, (pd.Timestamp(exp) - pd.offsets.MonthBegin(1)).month)
    return {
        "earnings_date": earnings_date.isoformat(),
        "monthly_opex": exp.isoformat(),
        "days_earnings_to_monthly_opex": (exp - earnings_date).days,
        "cycle_start_estimate": cycle_start.isoformat(),
    }


def choose_expiration(symbol: str, earnings_date: date, monthly_opex: date) -> Dict[str, Any]:
    expirations = get_option_expirations(symbol)
    if not expirations:
        return {"has_options": False, "selected_expiration": None, "monthly_opex_listed": False, "all_expirations": []}

    parsed = []
    for exp in expirations:
        d = parse_excel_date(exp)
        if d:
            parsed.append((exp, d))

    if not parsed:
        return {"has_options": False, "selected_expiration": None, "monthly_opex_listed": False, "all_expirations": expirations}

    monthly_opex_listed = any(d == monthly_opex for _, d in parsed)
    # Prefer the monthly opex cycle if listed; otherwise nearest expiration on/after earnings.
    if monthly_opex_listed:
        selected = [s for s, d in parsed if d == monthly_opex][0]
    else:
        future = [(s, d) for s, d in parsed if d >= earnings_date]
        selected = min(future, key=lambda x: abs((x[1] - earnings_date).days))[0] if future else parsed[0][0]

    selected_date = parse_excel_date(selected)
    return {
        "has_options": True,
        "selected_expiration": selected,
        "selected_expiration_date": selected_date.isoformat() if selected_date else None,
        "monthly_opex_listed": monthly_opex_listed,
        "all_expirations": expirations[:20],
    }


def nearest_option_row(df: pd.DataFrame, underlying_price: float) -> Optional[Dict[str, Any]]:
    if df is None or df.empty or "strike" not in df.columns:
        return None
    work = df.copy()
    work["distance"] = (work["strike"].astype(float) - underlying_price).abs()
    row = work.sort_values("distance").iloc[0].to_dict()
    return {k: as_jsonable(v) for k, v in row.items()}


def option_mid(row: Optional[Dict[str, Any]]) -> Optional[float]:
    if not row:
        return None
    bid = safe_float(row.get("bid"))
    ask = safe_float(row.get("ask"))
    last = safe_float(row.get("lastPrice"))
    if bid is not None and ask is not None and ask > 0 and bid >= 0:
        return (bid + ask) / 2.0
    return last


def get_option_snapshot(symbol: str, expiration: Optional[str], underlying_price: Optional[float]) -> Dict[str, Any]:
    if not expiration or underlying_price is None:
        return {"status": "UNAVAILABLE", "reason": "missing expiration or underlying price"}

    try:
        chain = yf.Ticker(symbol).option_chain(expiration)
        call_row = nearest_option_row(chain.calls, underlying_price)
        put_row = nearest_option_row(chain.puts, underlying_price)
        call_mid = option_mid(call_row)
        put_mid = option_mid(put_row)
        call_iv = safe_float(call_row.get("impliedVolatility")) if call_row else None
        put_iv = safe_float(put_row.get("impliedVolatility")) if put_row else None

        atm_straddle_cost = None
        atm_straddle_implied_move = None
        if call_mid is not None and put_mid is not None:
            atm_straddle_cost = call_mid + put_mid
            if underlying_price > 0:
                atm_straddle_implied_move = atm_straddle_cost / underlying_price

        avg_iv = None
        iv_values = [x for x in [call_iv, put_iv] if x is not None and x > 0]
        if iv_values:
            avg_iv = float(np.mean(iv_values))

        return {
            "status": "OK",
            "expiration": expiration,
            "underlying_price": round(float(underlying_price), 4),
            "atm_call": call_row,
            "atm_put": put_row,
            "atm_call_mid": round(call_mid, 4) if call_mid is not None else None,
            "atm_put_mid": round(put_mid, 4) if put_mid is not None else None,
            "atm_straddle_cost": round(atm_straddle_cost, 4) if atm_straddle_cost is not None else None,
            "atm_straddle_implied_move": round(atm_straddle_implied_move, 6) if atm_straddle_implied_move is not None else None,
            "average_atm_iv": round(avg_iv, 6) if avg_iv is not None else None,
        }
    except Exception as exc:
        return {"status": "ERROR", "reason": str(exc)}


def build_iv_series_for_tools(df: pd.DataFrame, current_iv: Optional[float]) -> pd.Series:
    close = df["Close"].astype(float)
    log_ret = np.log(close / close.shift(1))
    realized_20 = log_ret.rolling(20).std(ddof=1) * np.sqrt(252.0)

    if current_iv is not None and current_iv > 0:
        # Blend current options IV with realized vol shape. This gives HVT/volviz a usable series while anchoring latest IV.
        rv_latest = safe_float(realized_20.dropna().iloc[-1]) if not realized_20.dropna().empty else None
        multiplier = current_iv / rv_latest if rv_latest and rv_latest > 0 else 1.0
        iv = (realized_20 * multiplier).clip(lower=0.05, upper=5.0)
    else:
        iv = (realized_20 * 1.10).clip(lower=0.05, upper=5.0)

    return iv.bfill().ffill()


def analyze_with_vol_tools(symbol: str, df: pd.DataFrame, option_snapshot: Dict[str, Any], options_module) -> Dict[str, Any]:
    current_iv = safe_float(option_snapshot.get("average_atm_iv"))
    iv_series = build_iv_series_for_tools(df, current_iv)
    px = df[["Open", "High", "Low", "Close"]].rename(columns=str.lower)

    if options_module is None:
        return {
            "status": "FALLBACK_NO_TOS_MODULE",
            "sigma_state": "NO_MODULE",
            "expected_move_state": "NO_MODULE",
            "vol_up": None,
            "current_iv_used": round(current_iv, 6) if current_iv else None,
        }

    try:
        packet = options_module.build_options_analysis_packet(px, implied_volatility=iv_series)
        return {
            "status": "OK",
            "sigma_state": packet.get("sigma_reentry", {}).get("state"),
            "expected_move_state": packet.get("expected_move", {}).get("state"),
            "vol_up": packet.get("vol_filter", {}).get("vol_up"),
            "current_iv_used": round(current_iv, 6) if current_iv else None,
            "packet": packet,
        }
    except Exception as exc:
        return {
            "status": "ERROR",
            "error": str(exc),
            "sigma_state": "ERROR",
            "expected_move_state": "ERROR",
            "vol_up": None,
            "current_iv_used": round(current_iv, 6) if current_iv else None,
        }


# -------------------- scoring / strategy selection --------------------
def calculate_vol_value(record: Dict[str, Any], option_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    hist_avg = safe_float(record.get("avg_abs_earnings_move"))
    sheet_iv_move = safe_float(record.get("sheet_implied_move"))
    chain_iv_move = safe_float(option_snapshot.get("atm_straddle_implied_move"))

    # Prefer live option-chain implied move, fallback to sheet implied move.
    selected_implied_move = chain_iv_move if chain_iv_move is not None else sheet_iv_move
    source = "option_chain_atm_straddle" if chain_iv_move is not None else "input_sheet"

    cheap_ratio = None
    if hist_avg is not None and selected_implied_move not in (None, 0):
        cheap_ratio = hist_avg / selected_implied_move

    if cheap_ratio is None:
        state = "UNKNOWN"
    elif cheap_ratio >= 1.25:
        state = "VERY_CHEAP"
    elif cheap_ratio >= 1.00:
        state = "CHEAP"
    elif cheap_ratio >= 0.85:
        state = "FAIR"
    elif cheap_ratio <= 0.70:
        state = "EXPENSIVE"
    else:
        state = "RICH"

    return {
        "historical_avg_abs_move": round(hist_avg, 6) if hist_avg is not None else None,
        "historical_max_abs_move": record.get("max_abs_earnings_move"),
        "selected_implied_move": round(selected_implied_move, 6) if selected_implied_move is not None else None,
        "selected_implied_move_source": source,
        "sheet_implied_move": round(sheet_iv_move, 6) if sheet_iv_move is not None else None,
        "chain_atm_straddle_implied_move": round(chain_iv_move, 6) if chain_iv_move is not None else None,
        "historical_to_implied_ratio": round(cheap_ratio, 4) if cheap_ratio is not None else None,
        "vol_value_state": state,
    }


def infer_direction_from_tools(vol_tools: Dict[str, Any]) -> str:
    sigma_state = str(vol_tools.get("sigma_state") or "")
    em_state = str(vol_tools.get("expected_move_state") or "")

    if sigma_state == "BUY_REENTRY" or "BELOW" in em_state:
        return "BULLISH"
    if sigma_state == "SELL_REENTRY" or "ABOVE" in em_state:
        return "BEARISH"
    return "NEUTRAL"


def choose_strategy(direction: str, vol_value: Dict[str, Any], option_snapshot: Dict[str, Any], config: RuntimeConfig) -> str:
    ratio = safe_float(vol_value.get("historical_to_implied_ratio"))
    implied_move = safe_float(vol_value.get("selected_implied_move"))

    if direction == "BULLISH":
        # If vol is cheap enough, use convex long call; otherwise define risk with spread.
        return "LONG_CALL" if ratio is not None and ratio >= 1.25 else "CALL_DEBIT_SPREAD"
    if direction == "BEARISH":
        return "LONG_PUT" if ratio is not None and ratio >= 1.25 else "PUT_DEBIT_SPREAD"

    # Non-directional cheap-vol earnings play.
    if implied_move is not None and ratio is not None and ratio >= 1.25:
        return "STRADDLE"
    return "STRANGLE"


def calculate_fit_score(record: Dict[str, Any], vol_value: Dict[str, Any], date_validation: Dict[str, Any], option_snapshot: Dict[str, Any], vol_tools: Dict[str, Any], config: RuntimeConfig) -> float:
    ratio = safe_float(vol_value.get("historical_to_implied_ratio"))
    if ratio is None:
        ratio_component = 0.0
    else:
        ratio_component = clamp((ratio - 0.70) / 0.80, 0.0, 1.0)

    samples = len(record.get("historical_abs_moves") or [])
    sample_component = clamp(samples / 4.0, 0.0, 1.0)

    option_component = 1.0 if option_snapshot.get("status") == "OK" else 0.0

    validation_status = date_validation.get("status")
    if validation_status == "CONFIRMED_3_SOURCE":
        date_component = 1.0
    elif validation_status == "PARTIAL_CONFIRMATION":
        date_component = 0.70
    elif validation_status == "UNCONFIRMED_NO_EXTERNAL_DATA":
        date_component = 0.40
    else:
        date_component = 0.20

    vol_tool_component = 0.0
    if vol_tools.get("status") == "OK":
        vol_tool_component += 0.20
        if vol_tools.get("expected_move_state") not in {None, "NO_MODULE", "ERROR"}:
            vol_tool_component += 0.20
        if vol_tools.get("sigma_state") in {"BUY_REENTRY", "SELL_REENTRY"}:
            vol_tool_component += 0.30
        if vol_tools.get("vol_up"):
            vol_tool_component += 0.30
        vol_tool_component = clamp(vol_tool_component, 0.0, 1.0)

    score = (
        0.45 * ratio_component
        + 0.15 * sample_component
        + 0.15 * option_component
        + 0.15 * date_component
        + 0.10 * vol_tool_component
    )
    return round(clamp(score, 0.0, 1.0), 4)


def build_risk_flags(record: Dict[str, Any], date_validation: Dict[str, Any], option_snapshot: Dict[str, Any], vol_value: Dict[str, Any], config: RuntimeConfig) -> List[str]:
    flags: List[str] = ["earnings_event", "long_premium_only", "defined_risk_preferred"]

    if option_snapshot.get("status") != "OK":
        flags.append("options_chain_unavailable")

    if date_validation.get("status") not in {"CONFIRMED_3_SOURCE", "PARTIAL_CONFIRMATION"}:
        flags.append("earnings_date_not_fully_confirmed")

    samples = len(record.get("historical_abs_moves") or [])
    if samples < config.min_historical_samples:
        flags.append("limited_earnings_history")

    avg_vol = safe_int(record.get("avg_volume"))
    if avg_vol is not None and avg_vol < config.min_avg_volume:
        flags.append("low_underlying_volume")

    if vol_value.get("vol_value_state") in {"RICH", "EXPENSIVE"}:
        flags.append("vol_not_cheap")

    return sorted(set(flags))


def build_trade_template(symbol: str, strategy: str, expiration_info: Dict[str, Any], option_snapshot: Dict[str, Any], vol_value: Dict[str, Any]) -> Dict[str, Any]:
    underlying = safe_float(option_snapshot.get("underlying_price"))
    implied_move = safe_float(vol_value.get("selected_implied_move"))
    selected_exp = expiration_info.get("selected_expiration")

    move_points = underlying * implied_move if underlying is not None and implied_move is not None else None
    lower = underlying - move_points if underlying is not None and move_points is not None else None
    upper = underlying + move_points if underlying is not None and move_points is not None else None

    if strategy == "LONG_CALL":
        legs = [{"action": "BUY", "type": "CALL", "strike_hint": "ATM to slightly OTM"}]
    elif strategy == "LONG_PUT":
        legs = [{"action": "BUY", "type": "PUT", "strike_hint": "ATM to slightly OTM"}]
    elif strategy == "CALL_DEBIT_SPREAD":
        legs = [
            {"action": "BUY", "type": "CALL", "strike_hint": "ATM or nearest liquid strike"},
            {"action": "SELL", "type": "CALL", "strike_hint": "near upper implied-move target"},
        ]
    elif strategy == "PUT_DEBIT_SPREAD":
        legs = [
            {"action": "BUY", "type": "PUT", "strike_hint": "ATM or nearest liquid strike"},
            {"action": "SELL", "type": "PUT", "strike_hint": "near lower implied-move target"},
        ]
    elif strategy == "STRADDLE":
        legs = [
            {"action": "BUY", "type": "CALL", "strike_hint": "ATM"},
            {"action": "BUY", "type": "PUT", "strike_hint": "ATM"},
        ]
    else:
        legs = [
            {"action": "BUY", "type": "CALL", "strike_hint": "upper expected-move wing"},
            {"action": "BUY", "type": "PUT", "strike_hint": "lower expected-move wing"},
        ]

    return {
        "symbol": symbol,
        "preferred_structure": strategy,
        "expiration": selected_exp,
        "monthly_opex": expiration_info.get("monthly_opex"),
        "underlying_price": round(underlying, 4) if underlying is not None else None,
        "implied_move_pct": round(implied_move, 6) if implied_move is not None else None,
        "implied_move_points": round(move_points, 4) if move_points is not None else None,
        "expected_range_lower": round(lower, 4) if lower is not None else None,
        "expected_range_upper": round(upper, 4) if upper is not None else None,
        "legs": legs,
        "hard_limits": {
            "undefined_risk_allowed": False,
            "short_naked_options_allowed": False,
            "direct_order_routing_allowed": False,
        },
    }


def analyze_symbol(record: Dict[str, Any], config: RuntimeConfig, options_module) -> Dict[str, Any]:
    symbol = record["symbol"]
    earnings_date = record["earnings_date"]
    today = datetime.now(timezone.utc).date()
    days_to_earnings = (earnings_date - today).days

    event_timing = assess_earnings_event_timing(earnings_date, record.get("earnings_time"), config)

    base = {
        "symbol": symbol,
        "earnings_date": earnings_date.isoformat(),
        "earnings_time": record.get("earnings_time"),
        "days_to_earnings": days_to_earnings,
        "event_timing": event_timing,
    }

    if not event_timing.get("eligible_for_new_entry", True):
        return {
            **base,
            "status": "DISQUALIFIED",
            "reason_code": "EARNINGS_EVENT_ALREADY_PASSED",
            "summary": (
                f"{symbol} earnings event is stale for new entry: "
                f"{event_timing.get('reason')}"
            ),
        }

    if days_to_earnings < config.min_days_to_earnings or days_to_earnings > config.max_days_to_earnings:
        return {
            **base,
            "status": "DISQUALIFIED",
            "reason_code": "EARNINGS_DATE_OUT_OF_WINDOW",
            "summary": f"{symbol} earnings date outside configured window.",
        }

    date_validation = validate_earnings_date(symbol, earnings_date, config)
    if config.strict_date_validation and date_validation["status"] not in {"CONFIRMED_3_SOURCE", "PARTIAL_CONFIRMATION"}:
        return {
            **base,
            "status": "DISQUALIFIED",
            "reason_code": "EARNINGS_DATE_NOT_VALIDATED",
            "date_validation": date_validation,
            "summary": f"{symbol} earnings date not validated.",
        }

    try:
        hist = fetch_history(symbol, config.history_period, config.history_interval)
        current_price = latest_close_from_history(hist) or record.get("recent_close")
    except Exception as exc:
        return {
            **base,
            "status": "ERROR",
            "reason_code": "HISTORY_UNAVAILABLE",
            "error": str(exc),
            "date_validation": date_validation,
        }

    opex = monthly_opex_cycle_for_earnings(earnings_date)
    expiration_info = choose_expiration(symbol, earnings_date, parse_excel_date(opex["monthly_opex"]))
    expiration_info["monthly_opex"] = opex["monthly_opex"]
    expiration_info["days_earnings_to_monthly_opex"] = opex["days_earnings_to_monthly_opex"]

    option_snapshot = get_option_snapshot(symbol, expiration_info.get("selected_expiration"), current_price)
    vol_tools = analyze_with_vol_tools(symbol, hist, option_snapshot, options_module)
    vol_value = calculate_vol_value(record, option_snapshot)
    direction = infer_direction_from_tools(vol_tools)
    strategy = choose_strategy(direction, vol_value, option_snapshot, config)
    fit_score = calculate_fit_score(record, vol_value, date_validation, option_snapshot, vol_tools, config)
    confidence = round(clamp(0.50 + 0.45 * fit_score, 0.05, 0.95), 4)
    risk_flags = build_risk_flags(record, date_validation, option_snapshot, vol_value, config)
    trade_template = build_trade_template(symbol, strategy, expiration_info, option_snapshot, vol_value)

    ratio = safe_float(vol_value.get("historical_to_implied_ratio"))
    vol_state = vol_value.get("vol_value_state")

    if fit_score >= config.min_fit_score and ratio is not None and ratio >= config.min_cheap_ratio:
        status = "CANDIDATE"
        reason_code = "PASSED_CHEAP_VOL"
    elif fit_score >= config.watchlist_min_fit_score and ratio is not None and ratio >= config.cheap_ratio_watchlist:
        status = "WATCHLIST"
        reason_code = "BORDERLINE_CHEAP_VOL"
    else:
        status = "DISQUALIFIED"
        reason_code = "VOL_NOT_CHEAP_ENOUGH"

    summary = (
        f"{symbol} earnings {earnings_date.isoformat()} {record.get('earnings_time')}; "
        f"historical avg move={vol_value.get('historical_avg_abs_move')}, "
        f"implied move={vol_value.get('selected_implied_move')}, "
        f"hist/implied={vol_value.get('historical_to_implied_ratio')}, "
        f"state={vol_state}, preferred={strategy}."
    )

    return {
        **base,
        "status": status,
        "reason_code": reason_code,
        "current_price": round(float(current_price), 4) if current_price is not None else None,
        "avg_volume": record.get("avg_volume"),
        "evr": record.get("evr"),
        "historical_moves": record.get("historical_moves"),
        "prior_earnings_dates": record.get("prior_earnings_dates"),
        "date_validation": date_validation,
        "monthly_opex_cycle": opex,
        "expiration_selection": expiration_info,
        "option_snapshot": option_snapshot,
        "vol_value": vol_value,
        "vol_tools": vol_tools,
        "direction": direction,
        "strategy": strategy,
        "fit_score": fit_score,
        "confidence": confidence,
        "risk_flags": risk_flags,
        "trade_template": trade_template,
        "summary": summary,
    }


def build_candidate(result: Dict[str, Any]) -> Dict[str, Any]:
    symbol = result["symbol"]
    strategy = result["strategy"]
    vol_value = result["vol_value"]
    ratio = vol_value.get("historical_to_implied_ratio")

    return {
        "candidate_id": f"earn_{symbol.lower()}_{result['earnings_date'].replace('-', '')}",
        "strategy_type": "earnings",
        "direction": result["direction"],
        "horizon": "SHORT_TERM",
        "structure_family": strategy,
        "summary": result["summary"],
        "confidence": result["confidence"],
        "fit_score": result["fit_score"],
        "thesis_tags": [
            "earnings",
            "cheap_volatility",
            str(result.get("direction", "neutral")).lower(),
            strategy.lower(),
        ],
        "risk_flags": result.get("risk_flags", []),
        "implementation": {
            "symbol": symbol,
            "earnings_date": result["earnings_date"],
            "earnings_time": result.get("earnings_time"),
            "days_to_earnings": result.get("days_to_earnings"),
            "event_timing": result.get("event_timing"),
            "vol_value": vol_value,
            "date_validation_status": result.get("date_validation", {}).get("status"),
            "monthly_opex_cycle": result.get("monthly_opex_cycle"),
            "expiration_selection": result.get("expiration_selection"),
            "vol_tools_summary": {
                "sigma_state": result.get("vol_tools", {}).get("sigma_state"),
                "expected_move_state": result.get("vol_tools", {}).get("expected_move_state"),
                "vol_up": result.get("vol_tools", {}).get("vol_up"),
            },
            "preferred_trade": result.get("trade_template"),
            "historical_moves": result.get("historical_moves"),
            "prior_earnings_dates": result.get("prior_earnings_dates"),
        },
    }


def build_output(config: RuntimeConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    input_path = resolve_runtime_path(config.input_file, DEFAULT_INPUT_FILE)
    options_module = import_options_module(config.options_module_path)

    records = load_earnings_workbook(input_path)
    analyses: List[Dict[str, Any]] = []

    for rec in records:
        analyses.append(analyze_symbol(rec, config, options_module))

    candidates = [build_candidate(r) for r in analyses if r.get("status") == "CANDIDATE"]
    watchlist = [r for r in analyses if r.get("status") == "WATCHLIST"]
    candidates.sort(key=lambda x: (x["fit_score"], x["confidence"]), reverse=True)
    candidates = candidates[: config.max_candidates]
    watchlist.sort(key=lambda x: (x.get("fit_score", 0), x.get("confidence", 0)), reverse=True)
    watchlist = watchlist[: config.max_watchlist]

    if candidates:
        top = candidates[0]
        summary = f"Top earnings cheap-vol candidate: {top['summary']}"
    else:
        summary = f"Earnings Analyst identified 0 approved cheap-vol candidates; watchlist={len(watchlist)}."

    output = {
        "agent": "earnings_analyst",
        "schema_version": "1.0.0",
        "event_id": now_iso(),
        "strategy_type": "earnings",
        "candidates": candidates,
        "summary": summary,
    }
    validate_output_schema(output)

    diagnostics = {
        "agent": "earnings_analyst",
        "schema_version": "1.0.0",
        "event_id": output["event_id"],
        "input_file": str(input_path),
        "options_module_loaded": options_module is not None,
        "config": {
            "min_avg_volume": config.min_avg_volume,
            "min_cheap_ratio": config.min_cheap_ratio,
            "cheap_ratio_watchlist": config.cheap_ratio_watchlist,
            "expensive_ratio": config.expensive_ratio,
            "min_historical_samples": config.min_historical_samples,
            "max_days_to_earnings": config.max_days_to_earnings,
            "min_fit_score": config.min_fit_score,
            "watchlist_min_fit_score": config.watchlist_min_fit_score,
            "strict_date_validation": config.strict_date_validation,
            "date_validation_tolerance_days": config.date_validation_tolerance_days,
            "market_timezone": config.market_timezone,
            "market_open_time": config.market_open_time,
            "market_close_time": config.market_close_time,
        },
        "counts": {
            "input_symbols": len(records),
            "approved_candidates": len(candidates),
            "watchlist": len(watchlist),
            "disqualified_or_error": len([r for r in analyses if r.get("status") not in {"CANDIDATE", "WATCHLIST"}]),
        },
        "watchlist": watchlist,
        "all_symbol_analyses": analyses,
        "output": output,
    }
    return output, diagnostics


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Earnings cheap-volatility analyst")
    parser.add_argument("--input-file", default=DEFAULT_INPUT_FILE, help="Input workbook path. Default: earnings.xlsx")
    parser.add_argument("--options-module-path", default=DEFAULT_OPTIONS_MODULE_PATH, help="Path to tos_options_agent_functions.py")
    parser.add_argument("--min-avg-volume", type=int, default=250_000)
    parser.add_argument("--min-cheap-ratio", type=float, default=1.00, help="Historical avg move / implied move required for approved candidate")
    parser.add_argument("--cheap-ratio-watchlist", type=float, default=0.85)
    parser.add_argument("--expensive-ratio", type=float, default=1.25)
    parser.add_argument("--min-historical-samples", type=int, default=2)
    parser.add_argument("--max-days-to-earnings", type=int, default=45)
    parser.add_argument("--min-days-to-earnings", type=int, default=0)
    parser.add_argument("--min-fit-score", type=float, default=0.55)
    parser.add_argument("--watchlist-min-fit-score", type=float, default=0.40)
    parser.add_argument("--max-candidates", type=int, default=15)
    parser.add_argument("--max-watchlist", type=int, default=25)
    parser.add_argument("--strict-date-validation", action="store_true")
    parser.add_argument("--date-validation-tolerance-days", type=int, default=2)
    parser.add_argument("--market-timezone", default="America/New_York", help="Timezone used for same-day earnings staleness checks")
    parser.add_argument("--market-open-time", default="09:30", help="Regular session open cutoff for BEFORE_OPEN earnings, HH:MM")
    parser.add_argument("--market-close-time", default="16:00", help="Regular session close reference for AFTER_CLOSE earnings, HH:MM")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    config = RuntimeConfig(
        input_file=args.input_file,
        options_module_path=args.options_module_path,
        min_avg_volume=args.min_avg_volume,
        min_cheap_ratio=args.min_cheap_ratio,
        cheap_ratio_watchlist=args.cheap_ratio_watchlist,
        expensive_ratio=args.expensive_ratio,
        min_historical_samples=args.min_historical_samples,
        max_days_to_earnings=args.max_days_to_earnings,
        min_days_to_earnings=args.min_days_to_earnings,
        min_fit_score=args.min_fit_score,
        watchlist_min_fit_score=args.watchlist_min_fit_score,
        max_candidates=args.max_candidates,
        max_watchlist=args.max_watchlist,
        strict_date_validation=args.strict_date_validation,
        date_validation_tolerance_days=args.date_validation_tolerance_days,
        market_timezone=args.market_timezone,
        market_open_time=args.market_open_time,
        market_close_time=args.market_close_time,
    )

    output, diagnostics = build_output(config)
    print(json.dumps(output, indent=2, default=as_jsonable))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    payload_path = REPORTS_DIR / f"earnings_payload_{ts}.json"
    report_path = REPORTS_DIR / f"earnings_report_{ts}.json"

    with open(payload_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=as_jsonable)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2, default=as_jsonable)

    print(f"\n[INFO] Coordinator payload written to {payload_path}")
    print(f"[INFO] Diagnostics report written to {report_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("[ERROR] Interrupted by user", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
