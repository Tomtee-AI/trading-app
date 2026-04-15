
from __future__ import annotations

from typing import Any, Dict, Literal, Optional

import numpy as np
import pandas as pd


PriceBasis = Literal["annual", "monthly", "weekly", "daily"]


def _normalize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy with lowercase OHLCV column names.

    Required columns:
      - high
      - low
      - close

    The DataFrame index must be a DatetimeIndex.
    """
    out = df.copy()
    rename = {}
    for col in out.columns:
        lc = str(col).lower()
        if lc in {"open", "high", "low", "close", "volume"} and col != lc:
            rename[col] = lc
    if rename:
        out = out.rename(columns=rename)

    required = {"high", "low", "close"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if not isinstance(out.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex.")

    return out.sort_index()


def _infer_bars_per_year(index: pd.DatetimeIndex, basis: PriceBasis = "annual") -> float:
    """
    Practical Python replacement for Thinkorswim aggregation-period logic.
    """
    if len(index) < 3:
        bars_per_year = 252.0
    else:
        deltas = index.to_series().diff().dropna()
        median_delta = deltas.median()

        if median_delta >= pd.Timedelta(days=27):
            bars_per_year = 12.0
        elif median_delta >= pd.Timedelta(days=6):
            bars_per_year = 52.0
        elif median_delta >= pd.Timedelta(days=1):
            bars_per_year = 252.0
        else:
            trading_minutes_per_day = 390.0
            bar_minutes = max(median_delta / pd.Timedelta(minutes=1), 1.0)
            bars_per_day = trading_minutes_per_day / bar_minutes
            bars_per_year = 252.0 * bars_per_day

    basis_divisor = {
        "annual": 1.0,
        "monthly": 12.0,
        "weekly": 52.0,
        "daily": 252.0,
    }[basis]

    return bars_per_year / basis_divisor


def compute_historical_volatility(
    close: pd.Series,
    length: int = 20,
    basis: PriceBasis = "annual",
) -> pd.Series:
    """
    Python translation of the HV calculation inside HVIV from hvt.txt.

    Thinkorswim source:
      HV = stdev(log(close / close[1]), length)
           * sqrt(barsPerYear / basisCoeff * length / (length - 1))
    """
    close = close.astype(float)
    log_ret = np.log(close / close.shift(1))
    annualization = _infer_bars_per_year(close.index, basis=basis)
    hv = log_ret.rolling(length).std(ddof=1) * np.sqrt(annualization * length / max(length - 1, 1))
    return hv


def compute_hviv_filter(
    close: pd.Series,
    implied_volatility: pd.Series,
    hv_length: int = 20,
    hv_basis: PriceBasis = "annual",
    lookback: int = 6,
    confirm_bars: int = 3,
) -> pd.DataFrame:
    """
    Recreates the HVIV sub-study from hvt.txt.

    Returns:
      hv
      hv_higher
      iv_higher
      vol_up

    Thinkorswim logic:
      HV_HIGHER = max(HV, hv[1..5]) > ((hv + hv[3]) / 2)
      IV_HIGHER = max(IV, iv[1..5]) > ((iv + iv[3]) / 2)
      vol_up = highest(IV_HIGHER, 3) and highest(HV_HIGHER, 3)
    """
    close = close.astype(float)
    iv = implied_volatility.astype(float)
    hv = compute_historical_volatility(close, length=hv_length, basis=hv_basis)

    hv_roll_max = hv.rolling(lookback).max()
    iv_roll_max = iv.rolling(lookback).max()

    hv_mid = (hv + hv.shift(3)) / 2.0
    iv_mid = (iv + iv.shift(3)) / 2.0

    hv_higher = hv_roll_max > hv_mid
    iv_higher = iv_roll_max > iv_mid

    vol_up = (
        hv_higher.rolling(confirm_bars).max().fillna(0).astype(bool)
        & iv_higher.rolling(confirm_bars).max().fillna(0).astype(bool)
    )

    return pd.DataFrame(
        {
            "hv": hv,
            "hv_higher": hv_higher.fillna(False),
            "iv_higher": iv_higher.fillna(False),
            "vol_up": vol_up.fillna(False),
        },
        index=close.index,
    )


def compute_sigma_reentry_signals(
    df: pd.DataFrame,
    implied_volatility: Optional[pd.Series] = None,
    price_col: str = "close",
    length: int = 21,
    num_dev_dn: float = -3.0,
    num_dev_up: float = 3.0,
    num_dev_dn_2: float = -4.0,
    num_dev_up_2: float = 4.0,
    average_type: str = "simple",
    require_vol_up: bool = False,
) -> pd.DataFrame:
    """
    Agent-friendly Python translation of the signal logic in hvt.txt.

    Core idea:
    - Track closes beyond the 4-sigma bands.
    - Wait for price to re-enter and cross back through the 3-sigma bands.
    - Require expanding band width.
    - Optionally require the HV/IV 'vol_up' filter.

    Returns columns including:
      midline, lower_band, upper_band, lower_band_2, upper_band_2
      expanding, counter_down_setup, counter_up_setup
      signal_down, signal_up
      stop_short, stop_long
      hv, hv_higher, iv_higher, vol_up (if IV is supplied)
    """
    px = _normalize_ohlc(df)
    close = px[price_col].astype(float)

    log_change = np.log(close / close.shift(1))
    sdev = log_change.rolling(length).std(ddof=1)

    ma_type = average_type.lower()
    if ma_type == "simple":
        midline = close.rolling(length).mean()
    elif ma_type == "ema":
        midline = close.ewm(span=length, adjust=False).mean()
    else:
        raise ValueError("average_type must be 'simple' or 'ema'.")

    lower_band = midline + (midline * num_dev_dn * sdev)
    upper_band = midline + (midline * num_dev_up * sdev)
    lower_band_2 = midline + (midline * num_dev_dn_2 * sdev)
    upper_band_2 = midline + (midline * num_dev_up_2 * sdev)

    band_width = upper_band - lower_band
    avg_band_width = band_width.rolling(12).mean()
    expanding = band_width > avg_band_width

    counter_down = np.zeros(len(px), dtype=float)
    counter_up = np.zeros(len(px), dtype=float)

    for i in range(1, len(px)):
        c = close.iloc[i]
        prev_down = counter_down[i - 1]
        prev_up = counter_up[i - 1]

        if pd.notna(upper_band_2.iloc[i]) and c > upper_band_2.iloc[i]:
            counter_down[i] = prev_down + 1
        elif prev_down > 0 and pd.notna(upper_band.iloc[i]) and c > upper_band.iloc[i] and c < upper_band_2.iloc[i]:
            counter_down[i] = max(1.0, prev_down - 1.0)
        elif prev_down > 0 and pd.notna(upper_band.iloc[i]) and c <= upper_band.iloc[i]:
            counter_down[i] = max(1.0, prev_down - 1.0)
        else:
            counter_down[i] = 0.0

        if pd.notna(lower_band_2.iloc[i]) and c < lower_band_2.iloc[i]:
            counter_up[i] = prev_up + 1
        elif prev_up > 0 and pd.notna(lower_band.iloc[i]) and c < lower_band.iloc[i] and c > lower_band_2.iloc[i]:
            counter_up[i] = max(1.0, prev_up - 1.0)
        elif prev_up > 0 and pd.notna(lower_band.iloc[i]) and c >= lower_band.iloc[i]:
            counter_up[i] = max(1.0, prev_up - 1.0)
        else:
            counter_up[i] = 0.0

    prev_close = close.shift(1)
    prev_upper = upper_band.shift(1)
    prev_lower = lower_band.shift(1)

    cross_below_upper_prevbar = (prev_close.shift(1) > prev_upper.shift(1)) & (prev_close <= prev_upper)
    cross_above_lower_prevbar = (prev_close.shift(1) < prev_lower.shift(1)) & (prev_close >= prev_lower)

    counter_down_recent = pd.Series(counter_down, index=px.index).rolling(3).max() > 0
    counter_up_recent = pd.Series(counter_up, index=px.index).rolling(3).max() > 0

    signal_down = cross_below_upper_prevbar & expanding & counter_down_recent
    signal_up = cross_above_lower_prevbar & expanding & counter_up_recent

    out = pd.DataFrame(
        {
            "midline": midline,
            "lower_band": lower_band,
            "upper_band": upper_band,
            "lower_band_2": lower_band_2,
            "upper_band_2": upper_band_2,
            "band_width": band_width,
            "avg_band_width_12": avg_band_width,
            "expanding": expanding.fillna(False),
            "counter_down_setup": counter_down,
            "counter_up_setup": counter_up,
            "signal_down": signal_down.fillna(False),
            "signal_up": signal_up.fillna(False),
            "stop_short": ((close > upper_band) & (close < upper_band_2)).fillna(False),
            "stop_long": ((close < lower_band) & (close > lower_band_2)).fillna(False),
        },
        index=px.index,
    )

    if implied_volatility is not None:
        hviv = compute_hviv_filter(close=close, implied_volatility=implied_volatility)
        out = out.join(hviv)
        if require_vol_up:
            out["signal_down"] = out["signal_down"] & out["vol_up"]
            out["signal_up"] = out["signal_up"] & out["vol_up"]

    return out


def summarize_sigma_signal_state(
    signal_df: pd.DataFrame,
    close: pd.Series,
) -> Dict[str, Any]:
    """
    Compact JSON-ready state for agents.
    """
    latest = signal_df.iloc[-1]
    return {
        "as_of": str(signal_df.index[-1]),
        "close": float(close.iloc[-1]),
        "signal_up": bool(latest.get("signal_up", False)),
        "signal_down": bool(latest.get("signal_down", False)),
        "expanding": bool(latest.get("expanding", False)),
        "counter_up_setup": float(latest.get("counter_up_setup", 0.0)),
        "counter_down_setup": float(latest.get("counter_down_setup", 0.0)),
        "vol_up": bool(latest.get("vol_up", False)) if "vol_up" in signal_df.columns else None,
        "state": (
            "BUY_REENTRY"
            if bool(latest.get("signal_up", False))
            else "SELL_REENTRY"
            if bool(latest.get("signal_down", False))
            else "NO_SIGNAL"
        ),
    }


def nth_weekday_of_month(year: int, month: int, weekday: int = 4, nth: int = 3) -> pd.Timestamp:
    """
    Return the nth weekday of a month.
    weekday: Monday=0 ... Friday=4
    """
    first = pd.Timestamp(year=year, month=month, day=1)
    offset = (weekday - first.weekday()) % 7
    day = 1 + offset + (nth - 1) * 7
    return pd.Timestamp(year=year, month=month, day=day)


def next_monthly_expiration_dates(
    index: pd.DatetimeIndex,
    nth_day_of_month: int = 3,
    opt_exp_day_of_week: str = "Friday",
) -> pd.Series:
    """
    Python replacement for the NextNFriday logic in volviz.txt.
    """
    weekday_map = {
        "Monday": 0,
        "Tuesday": 1,
        "Wednesday": 2,
        "Thursday": 3,
        "Friday": 4,
    }
    if opt_exp_day_of_week not in weekday_map:
        raise ValueError("opt_exp_day_of_week must be Monday, Tuesday, Wednesday, Thursday, or Friday.")

    wd = weekday_map[opt_exp_day_of_week]
    expiries = []

    for ts in index:
        exp = nth_weekday_of_month(ts.year, ts.month, weekday=wd, nth=nth_day_of_month)
        if ts.normalize() > exp.normalize():
            next_month = ts + pd.offsets.MonthBegin(1)
            exp = nth_weekday_of_month(next_month.year, next_month.month, weekday=wd, nth=nth_day_of_month)
        expiries.append(exp.normalize())

    return pd.Series(expiries, index=index)


def compute_expected_move_surface(
    df: pd.DataFrame,
    implied_volatility: pd.Series,
    opt_exp_day_of_week: str = "Friday",
    nth_day_of_month: int = 3,
    iv_multiplier_week: float = 0.8877,
    iv_multiplier_month: float = 1.0,
) -> pd.DataFrame:
    """
    Agent-friendly translation of the core weekly/monthly expected-move engine in volviz.txt.

    Preserved from the source:
    - Weekly cycle resets at the start of each week.
    - Monthly cycle resets at the configured monthly expiration boundary.
    - Weekly EM = prior close * adjusted weekly IV * sqrt(5/252)
    - Monthly EM = prior close * adjusted monthly IV * sqrt(21/252)

    Simplifications:
    - Expects an external implied_volatility series.
    - Focuses on EM levels and breach analytics instead of chart labels and wedges.
    """
    px = _normalize_ohlc(df)
    close = px["close"].astype(float)
    high = px["high"].astype(float)
    low = px["low"].astype(float)
    iv = implied_volatility.astype(float)

    daily_variance = np.sqrt((iv * iv / 365.0 * 30.0) / 21.0)
    daily_variance_points = daily_variance * close.shift(1)
    expected_daily_range = 0.8 * daily_variance_points
    previous_day_volatility = daily_variance * np.sqrt(252.0)

    weekday = px.index.to_series().dt.weekday
    start_week = (weekday < weekday.shift(1)).fillna(False)

    monthly_exp = next_monthly_expiration_dates(
        px.index,
        nth_day_of_month=nth_day_of_month,
        opt_exp_day_of_week=opt_exp_day_of_week,
    )
    start_month = (monthly_exp != monthly_exp.shift(1)).fillna(False)

    iv_week = pd.Series(np.nan, index=px.index, dtype=float)
    iv_month = pd.Series(np.nan, index=px.index, dtype=float)
    week_close = pd.Series(np.nan, index=px.index, dtype=float)
    month_close = pd.Series(np.nan, index=px.index, dtype=float)

    prev_close_series = close.shift(1)

    for i in range(1, len(px)):
        prev_iv = previous_day_volatility.iloc[i]
        prev_close = prev_close_series.iloc[i]

        if bool(start_week.iloc[i]) and pd.notna(prev_iv) and pd.notna(prev_close):
            iv_week.iloc[i] = prev_iv * iv_multiplier_week
            week_close.iloc[i] = prev_close
        else:
            iv_week.iloc[i] = iv_week.iloc[i - 1]
            week_close.iloc[i] = week_close.iloc[i - 1]

        if bool(start_month.iloc[i]) and pd.notna(prev_iv) and pd.notna(prev_close):
            iv_month.iloc[i] = prev_iv * iv_multiplier_month
            month_close.iloc[i] = prev_close
        else:
            iv_month.iloc[i] = iv_month.iloc[i - 1]
            month_close.iloc[i] = month_close.iloc[i - 1]

    week_expected_move = week_close * iv_week * np.sqrt(5.0 / 252.0)
    month_expected_move = month_close * iv_month * np.sqrt(21.0 / 252.0)

    week_upper = week_close + week_expected_move
    week_lower = week_close - week_expected_move
    month_upper = month_close + month_expected_move
    month_lower = month_close - month_expected_move

    weekly_breach = ((close > week_upper) | (close < week_lower)).fillna(False)
    monthly_breach = ((close > month_upper) | (close < month_lower)).fillna(False)

    week_group = px.index.to_series().dt.to_period("W-FRI")
    month_group = px.index.to_series().dt.to_period("M")

    week_high_running = high.groupby(week_group).cummax()
    week_low_running = low.groupby(week_group).cummin()
    month_high_running = high.groupby(month_group).cummax()
    month_low_running = low.groupby(month_group).cummin()

    return pd.DataFrame(
        {
            "implied_volatility": iv,
            "daily_variance": daily_variance,
            "daily_variance_points": daily_variance_points,
            "expected_daily_range": expected_daily_range,
            "previous_day_volatility": previous_day_volatility,
            "start_week": start_week,
            "start_month": start_month,
            "week_close": week_close,
            "month_close": month_close,
            "week_iv": iv_week,
            "month_iv": iv_month,
            "week_expected_move": week_expected_move,
            "month_expected_move": month_expected_move,
            "week_upper": week_upper,
            "week_lower": week_lower,
            "month_upper": month_upper,
            "month_lower": month_lower,
            "weekly_breach": weekly_breach,
            "monthly_breach": monthly_breach,
            "week_high_running": week_high_running,
            "week_low_running": week_low_running,
            "month_high_running": month_high_running,
            "month_low_running": month_low_running,
        },
        index=px.index,
    )


def summarize_expected_move_state(
    em_df: pd.DataFrame,
    close: pd.Series,
) -> Dict[str, Any]:
    """
    Compact JSON-ready state for agents.
    """
    latest = em_df.iloc[-1]
    c = float(close.iloc[-1])

    def _distance_pct(level: Any) -> Optional[float]:
        if pd.isna(level) or float(level) == 0.0:
            return None
        return round((c / float(level) - 1.0) * 100.0, 4)

    return {
        "as_of": str(em_df.index[-1]),
        "close": c,
        "weekly_expected_move": float(latest["week_expected_move"]) if pd.notna(latest["week_expected_move"]) else None,
        "monthly_expected_move": float(latest["month_expected_move"]) if pd.notna(latest["month_expected_move"]) else None,
        "weekly_breach": bool(latest.get("weekly_breach", False)),
        "monthly_breach": bool(latest.get("monthly_breach", False)),
        "distance_to_week_upper_pct": _distance_pct(latest.get("week_upper")),
        "distance_to_week_lower_pct": _distance_pct(latest.get("week_lower")),
        "distance_to_month_upper_pct": _distance_pct(latest.get("month_upper")),
        "distance_to_month_lower_pct": _distance_pct(latest.get("month_lower")),
        "state": (
            "OUTSIDE_MONTH_EM"
            if bool(latest.get("monthly_breach", False))
            else "OUTSIDE_WEEK_EM"
            if bool(latest.get("weekly_breach", False))
            else "INSIDE_EXPECTED_MOVES"
        ),
    }


def build_options_analysis_packet(
    df: pd.DataFrame,
    implied_volatility: pd.Series,
    sigma_length: int = 21,
    hv_length: int = 20,
    require_vol_up_for_sigma: bool = False,
) -> Dict[str, Any]:
    """
    Convenience wrapper for agent pipelines.

    Combines:
      - sigma re-entry state from hvt.txt
      - HV/IV filter state from hvt.txt
      - weekly/monthly expected-move state from volviz.txt
    """
    px = _normalize_ohlc(df)

    sigma_df = compute_sigma_reentry_signals(
        px,
        implied_volatility=implied_volatility,
        length=sigma_length,
        require_vol_up=require_vol_up_for_sigma,
    )
    em_df = compute_expected_move_surface(px, implied_volatility=implied_volatility)
    hviv_df = compute_hviv_filter(px["close"], implied_volatility=implied_volatility, hv_length=hv_length)

    return {
        "as_of": str(px.index[-1]),
        "sigma_reentry": summarize_sigma_signal_state(sigma_df, px["close"]),
        "expected_move": summarize_expected_move_state(em_df, px["close"]),
        "vol_filter": {
            "hv": float(hviv_df["hv"].iloc[-1]) if pd.notna(hviv_df["hv"].iloc[-1]) else None,
            "hv_higher": bool(hviv_df["hv_higher"].iloc[-1]),
            "iv_higher": bool(hviv_df["iv_higher"].iloc[-1]),
            "vol_up": bool(hviv_df["vol_up"].iloc[-1]),
        },
    }
