#!/usr/bin/env python3
"""
metals_trade_analyst.py

Standalone Trade Analyst focused on relative-value opportunities
in the metals complex only.

What it does
------------
- Pulls daily futures prices for:
    GC=F  Gold
    SI=F  Silver
    HG=F  Copper
    PL=F  Platinum
    PA=F  Palladium
- Tries to use a current price from intraday history
- Falls back to the previous daily close if current price is unavailable
- Calculates pairwise rolling return correlations
- Estimates hedge-ratio-adjusted log-spread z-scores
- Ranks relative-value opportunities
- Flags which leg appears expensive vs cheap

Install
-------
pip install yfinance pandas numpy

Run
---
python metals_trade_analyst.py
python metals_trade_analyst.py --json-out metals_report.json
python metals_trade_analyst.py --min-corr 0.50 --min-abs-z 1.0
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf


METALS_UNIVERSE = {
    "GC=F": "Gold",
    "SI=F": "Silver",
    "HG=F": "Copper",
    "PL=F": "Platinum",
    "PA=F": "Palladium",
}


@dataclass
class AnalystConfig:
    daily_period: str = "2y"
    daily_interval: str = "1d"
    current_intervals: Tuple[str, ...] = ("1m", "5m", "15m")
    current_periods: Tuple[str, ...] = ("1d", "5d", "5d")
    corr_window: int = 126
    spread_window: int = 126
    zscore_window: int = 63
    min_corr: float = 0.45
    min_abs_z: float = 1.25
    max_candidates: int = 10
    threads: bool = False
    use_current_price: bool = True


@dataclass
class PriceSnapshot:
    symbol: str
    name: str
    effective_price: float
    price_source: str
    source_timestamp_utc: Optional[str]
    previous_close: float


@dataclass
class PairSignal:
    pair: str
    symbol_a: str
    symbol_b: str
    name_a: str
    name_b: str
    correlation: float
    hedge_ratio_beta: float
    last_close_zscore: float
    current_zscore: float
    expensive_symbol: str
    cheap_symbol: str
    recommendation: str
    confidence: float
    score: float
    rationale: str


class MetalsTradeAnalyst:
    def __init__(self, config: AnalystConfig):
        self.config = config
        self.universe = METALS_UNIVERSE.copy()

    def fetch_daily_closes(self) -> pd.DataFrame:
        symbols = list(self.universe.keys())

        raw = yf.download(
            tickers=symbols,
            period=self.config.daily_period,
            interval=self.config.daily_interval,
            auto_adjust=False,
            progress=False,
            threads=self.config.threads,
            group_by="column",
        )

        if raw is None or raw.empty:
            raise RuntimeError("No daily market data returned from yfinance.")

        closes = self._extract_close_frame(raw, symbols)
        closes = closes.dropna(how="all").sort_index()

        if closes.empty:
            raise RuntimeError("Daily close frame is empty after cleaning.")

        available = [c for c in closes.columns if c in symbols]
        closes = closes[available]

        if closes.shape[1] < 2:
            raise RuntimeError("Need at least two symbols with data to run relative-value analysis.")

        return closes

    @staticmethod
    def _extract_close_frame(raw: pd.DataFrame, symbols: List[str]) -> pd.DataFrame:
        if isinstance(raw.columns, pd.MultiIndex):
            level0 = list(raw.columns.get_level_values(0))
            level1 = list(raw.columns.get_level_values(1))

            if "Close" in level0:
                closes = raw["Close"].copy()
            elif "Close" in level1:
                closes = raw.xs("Close", axis=1, level=1).copy()
            else:
                raise RuntimeError("Could not find Close columns in downloaded price frame.")
        else:
            if "Close" not in raw.columns:
                raise RuntimeError("Downloaded single-ticker frame missing Close column.")
            closes = raw[["Close"]].copy()
            closes.columns = [symbols[0]]

        closes = closes.apply(pd.to_numeric, errors="coerce")
        return closes

    def fetch_price_snapshots(self, daily_closes: pd.DataFrame) -> Dict[str, PriceSnapshot]:
        snapshots: Dict[str, PriceSnapshot] = {}

        for symbol, name in self.universe.items():
            if symbol not in daily_closes.columns:
                continue

            previous_close_series = daily_closes[symbol].dropna()
            if previous_close_series.empty:
                continue

            previous_close = float(previous_close_series.iloc[-1])

            effective_price = previous_close
            price_source = "previous_close"
            source_timestamp = None

            if self.config.use_current_price:
                current_price, current_ts = self._try_fetch_current_price(symbol)
                if current_price is not None and np.isfinite(current_price) and current_price > 0:
                    effective_price = float(current_price)
                    price_source = "current"
                    source_timestamp = current_ts

            snapshots[symbol] = PriceSnapshot(
                symbol=symbol,
                name=name,
                effective_price=float(effective_price),
                price_source=price_source,
                source_timestamp_utc=source_timestamp,
                previous_close=float(previous_close),
            )

        return snapshots

    def _try_fetch_current_price(self, symbol: str) -> Tuple[Optional[float], Optional[str]]:
        ticker = yf.Ticker(symbol)

        for period, interval in zip(self.config.current_periods, self.config.current_intervals):
            try:
                df = ticker.history(
                    period=period,
                    interval=interval,
                    auto_adjust=False,
                    prepost=True,
                )

                if df is None or df.empty or "Close" not in df.columns:
                    continue

                close_series = df["Close"].dropna()
                if close_series.empty:
                    continue

                px = float(close_series.iloc[-1])
                ts = close_series.index[-1]

                ts_utc = None
                try:
                    if getattr(ts, "tzinfo", None) is not None:
                        ts_utc = ts.tz_convert("UTC").isoformat()
                    else:
                        ts_utc = pd.Timestamp(ts).tz_localize("UTC").isoformat()
                except Exception:
                    ts_utc = None

                return px, ts_utc

            except Exception:
                continue

        return None, None

    def analyze(self, daily_closes: pd.DataFrame, snapshots: Dict[str, PriceSnapshot]) -> Dict[str, object]:
        usable_symbols = [s for s in daily_closes.columns if s in snapshots]
        px = daily_closes[usable_symbols].dropna(how="all").copy()

        if px.shape[1] < 2:
            raise RuntimeError("Not enough symbols with both history and snapshots.")

        px = px.where(px > 0)

        log_px = np.log(px)
        log_ret = log_px.diff()

        pair_signals: List[PairSignal] = []

        for symbol_a, symbol_b in itertools.combinations(usable_symbols, 2):
            signal = self._analyze_pair(
                symbol_a=symbol_a,
                symbol_b=symbol_b,
                log_px=log_px,
                log_ret=log_ret,
                snapshots=snapshots,
            )
            if signal is not None:
                pair_signals.append(signal)

        pair_signals.sort(key=lambda x: x.score, reverse=True)
        filtered_signals = [s for s in pair_signals if abs(s.current_zscore) >= self.config.min_abs_z]
        filtered_signals = filtered_signals[: self.config.max_candidates]

        diagnostics = [
            {
                "symbol": snap.symbol,
                "name": snap.name,
                "effective_price": snap.effective_price,
                "previous_close": snap.previous_close,
                "price_source": snap.price_source,
                "source_timestamp_utc": snap.source_timestamp_utc,
            }
            for snap in snapshots.values()
        ]

        report = {
            "agent_id": "trade_analyst_metals_rv",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "complex": "metals",
            "methodology": {
                "relative_value_measure": "hedge_ratio_adjusted_log_spread_zscore",
                "correlation_window_days": self.config.corr_window,
                "spread_window_days": self.config.spread_window,
                "zscore_window_days": self.config.zscore_window,
                "min_correlation": self.config.min_corr,
                "min_abs_zscore": self.config.min_abs_z,
                "current_price_preferred": self.config.use_current_price,
                "current_price_fallback": "previous_daily_close",
            },
            "snapshots": diagnostics,
            "trade_candidates": [asdict(s) for s in filtered_signals],
            "all_pair_count": len(pair_signals),
            "candidate_count": len(filtered_signals),
        }

        return report

    def _analyze_pair(
        self,
        symbol_a: str,
        symbol_b: str,
        log_px: pd.DataFrame,
        log_ret: pd.DataFrame,
        snapshots: Dict[str, PriceSnapshot],
    ) -> Optional[PairSignal]:
        pair_log_px = log_px[[symbol_a, symbol_b]].dropna()
        pair_ret = log_ret[[symbol_a, symbol_b]].dropna()

        min_needed = max(self.config.corr_window, self.config.spread_window, self.config.zscore_window) + 5
        if len(pair_log_px) < min_needed or len(pair_ret) < min_needed:
            return None

        corr_sample = pair_ret.tail(self.config.corr_window)
        corr = corr_sample[symbol_a].corr(corr_sample[symbol_b])

        if pd.isna(corr) or corr < self.config.min_corr:
            return None

        spread_sample = pair_log_px.tail(self.config.spread_window)

        x = spread_sample[symbol_b].values
        y = spread_sample[symbol_a].values

        var_x = np.var(x, ddof=1)
        if not np.isfinite(var_x) or var_x <= 0:
            return None

        cov_xy = np.cov(x, y, ddof=1)[0, 1]
        beta = float(cov_xy / var_x)

        if not np.isfinite(beta):
            return None

        hist_spread = pair_log_px[symbol_a] - beta * pair_log_px[symbol_b]
        hist_spread = hist_spread.dropna()
        if len(hist_spread) < self.config.zscore_window:
            return None

        z_window = hist_spread.tail(self.config.zscore_window)
        mu = float(z_window.mean())
        sigma = float(z_window.std(ddof=1))

        if not np.isfinite(sigma) or sigma <= 1e-10:
            return None

        last_close_spread = float(hist_spread.iloc[-1])
        last_close_z = float((last_close_spread - mu) / sigma)

        a_eff = snapshots[symbol_a].effective_price
        b_eff = snapshots[symbol_b].effective_price

        if a_eff <= 0 or b_eff <= 0:
            return None

        current_spread = math.log(a_eff) - beta * math.log(b_eff)
        current_z = float((current_spread - mu) / sigma)

        if current_z > 0:
            expensive_symbol = symbol_a
            cheap_symbol = symbol_b
        else:
            expensive_symbol = symbol_b
            cheap_symbol = symbol_a

        recommendation = f"SELL {expensive_symbol} / BUY {cheap_symbol}"

        z_strength = min(abs(current_z) / 3.0, 1.0)
        corr_strength = min(max((corr - self.config.min_corr) / (1.0 - self.config.min_corr + 1e-9), 0.0), 1.0)
        confidence = round(0.55 * z_strength + 0.45 * corr_strength, 4)
        score = round(abs(current_z) * max(corr, 0.0), 4)

        name_a = self.universe[symbol_a]
        name_b = self.universe[symbol_b]

        if expensive_symbol == symbol_a:
            rationale = (
                f"{name_a} screens expensive versus {name_b}. "
                f"The {name_a}/{name_b} hedge-adjusted spread is +{current_z:.2f} standard deviations "
                f"from its recent mean with {corr:.2f} return correlation."
            )
        else:
            rationale = (
                f"{name_b} screens expensive versus {name_a}. "
                f"The {name_a}/{name_b} hedge-adjusted spread is {current_z:.2f} standard deviations "
                f"from its recent mean with {corr:.2f} return correlation."
            )

        return PairSignal(
            pair=f"{symbol_a}__{symbol_b}",
            symbol_a=symbol_a,
            symbol_b=symbol_b,
            name_a=name_a,
            name_b=name_b,
            correlation=round(float(corr), 4),
            hedge_ratio_beta=round(beta, 4),
            last_close_zscore=round(last_close_z, 4),
            current_zscore=round(current_z, 4),
            expensive_symbol=expensive_symbol,
            cheap_symbol=cheap_symbol,
            recommendation=recommendation,
            confidence=confidence,
            score=score,
            rationale=rationale,
        )


def print_human_summary(report: Dict[str, object]) -> None:
    print("\n=== METALS TRADE ANALYST ===")
    print(f"Generated: {report['generated_at_utc']}")
    print(f"Complex:   {report['complex']}")
    print(f"Pairs analyzed: {report['all_pair_count']}")
    print(f"Candidates:     {report['candidate_count']}")

    print("\n--- Price Snapshots ---")
    for snap in report["snapshots"]:
        ts = snap["source_timestamp_utc"] or "n/a"
        print(
            f"{snap['symbol']:>5}  "
            f"{snap['name']:<10}  "
            f"effective={snap['effective_price']:.4f}  "
            f"prev_close={snap['previous_close']:.4f}  "
            f"source={snap['price_source']:<14}  "
            f"ts={ts}"
        )

    print("\n--- Ranked Trade Candidates ---")
    candidates = report["trade_candidates"]
    if not candidates:
        print("No candidates met the minimum z-score threshold.")
        return

    for idx, c in enumerate(candidates, start=1):
        print(f"\n[{idx}] {c['recommendation']}")
        print(f"    Pair:        {c['name_a']} vs {c['name_b']} ({c['symbol_a']} / {c['symbol_b']})")
        print(f"    Correlation: {c['correlation']:.2f}")
        print(f"    Beta:        {c['hedge_ratio_beta']:.4f}")
        print(f"    Close Z:     {c['last_close_zscore']:.2f}")
        print(f"    Current Z:   {c['current_zscore']:.2f}")
        print(f"    Confidence:  {c['confidence']:.2f}")
        print(f"    Score:       {c['score']:.2f}")
        print(f"    Expensive:   {c['expensive_symbol']}")
        print(f"    Cheap:       {c['cheap_symbol']}")
        print(f"    Rationale:   {c['rationale']}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone metals relative-value Trade Analyst")
    parser.add_argument("--json-out", type=str, default=None, help="Optional path to save JSON report")
    parser.add_argument("--min-corr", type=float, default=0.45, help="Minimum pair correlation threshold")
    parser.add_argument("--min-abs-z", type=float, default=1.25, help="Minimum absolute current z-score")
    parser.add_argument("--corr-window", type=int, default=126, help="Rolling window for return correlation")
    parser.add_argument("--spread-window", type=int, default=126, help="Window for hedge ratio estimation")
    parser.add_argument("--zscore-window", type=int, default=63, help="Window for spread z-score")
    parser.add_argument(
        "--disable-current-price",
        action="store_true",
        help="Use previous daily closes only",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    config = AnalystConfig(
        min_corr=args.min_corr,
        min_abs_z=args.min_abs_z,
        corr_window=args.corr_window,
        spread_window=args.spread_window,
        zscore_window=args.zscore_window,
        use_current_price=not args.disable_current_price,
    )

    analyst = MetalsTradeAnalyst(config)

    try:
        daily_closes = analyst.fetch_daily_closes()
        snapshots = analyst.fetch_price_snapshots(daily_closes)
        report = analyst.analyze(daily_closes, snapshots)

        print_human_summary(report)

        if args.json_out:
            with open(args.json_out, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            print(f"\nSaved JSON report to: {args.json_out}")

        return 0

    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())