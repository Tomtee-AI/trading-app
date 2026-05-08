"""
agents/market_analyst.py

Prototype implementation of the Market Analyst agent.

Design principles:
- Deterministic Python logic is the source of truth for regime classification.
- LLM (via CrewAI + Ollama) is used ONLY for natural language summary and factor commentary.
- All output is validated against Pydantic schema before return.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import yfinance as yf
import pandas as pd

from config.schemas import (
    MarketAnalystOutput,
    RegimeType,
    BiasType,
    FactorSignal,
    generate_event_id,  # from validation or duplicate here for standalone use
)


def _compute_regime_signals(ticker_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Pure deterministic Python logic for market regime classification.
    This should remain the authoritative source — LLM only explains it.
    """
    qqq = ticker_data.get("QQQ")
    vix = ticker_data.get("^VIX")

    if qqq is None or qqq.empty:
        return {"regime": RegimeType.UNKNOWN, "price_bias": BiasType.NEUTRAL, "error": "No QQQ data"}

    # Simple price structure
    recent_high = qqq["High"].tail(20).max()
    recent_low = qqq["Low"].tail(20).min()
    last_close = qqq["Close"].iloc[-1]
    prev_close = qqq["Close"].iloc[-2] if len(qqq) > 1 else last_close

    price_momentum = (last_close - prev_close) / prev_close if prev_close != 0 else 0

    # Basic regime logic (expand this significantly in production)
    if last_close > recent_high * 0.98:
        price_bias = BiasType.BULLISH
        regime = RegimeType.BULLISH
    elif last_close < recent_low * 1.02:
        price_bias = BiasType.BEARISH
        regime = RegimeType.BEARISH
    else:
        price_bias = BiasType.NEUTRAL
        regime = RegimeType.MIXED

    # Volatility bias (simplified)
    vol_bias = BiasType.NEUTRAL
    if vix is not None and not vix.empty:
        current_vix = vix["Close"].iloc[-1]
        if current_vix > 25:
            vol_bias = BiasType.BEARISH  # High vol = caution
        elif current_vix < 15:
            vol_bias = BiasType.BULLISH

    # Build factor signals
    factors: Dict[str, FactorSignal] = {
        "qqq_structure": FactorSignal(
            name="qqq_structure",
            value=round(last_close, 2),
            signal="strong" if price_bias == BiasType.BULLISH else "weak",
            weight=0.35,
            comment=f"Recent high: {recent_high:.2f}, low: {recent_low:.2f}",
        ),
        "price_momentum": FactorSignal(
            name="price_momentum",
            value=round(price_momentum * 100, 2),
            signal="positive" if price_momentum > 0 else "negative",
            weight=0.25,
        ),
    }

    if vix is not None and not vix.empty:
        factors["vix_level"] = FactorSignal(
            name="vix_level",
            value=round(current_vix, 2),
            signal="elevated" if current_vix > 20 else "normal",
            weight=0.20,
        )

    return {
        "regime": regime,
        "price_bias": price_bias,
        "vol_bias": vol_bias,
        "factors": factors,
        "key_levels": {
            "recent_high": round(recent_high, 2),
            "recent_low": round(recent_low, 2),
            "last_close": round(last_close, 2),
        },
    }


def fetch_market_data(period: str = "1mo", interval: str = "1d") -> Dict[str, pd.DataFrame]:
    """Fetch required market data using yfinance (public, no API key needed)."""
    tickers = ["QQQ", "^VIX"]
    data: Dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        try:
            df = yf.download(ticker, period=period, interval=interval, progress=False)
            if not df.empty:
                data[ticker] = df
        except Exception as e:
            print(f"Warning: Failed to fetch {ticker}: {e}")

    return data


def analyze_market_regime(
    model_name: str = "llama3.1",
    use_llm_summary: bool = False,  # Set True when Ollama + CrewAI is ready
) -> MarketAnalystOutput:
    """
    Main entry point for the Market Analyst.

    Returns a fully validated MarketAnalystOutput.
    """
    event_id = generate_event_id("market_regime")

    # 1. Fetch data (deterministic)
    market_data = fetch_market_data()

    # 2. Run deterministic regime engine
    signals = _compute_regime_signals(market_data)

    # 3. Build base output
    base_summary = (
        f"Market regime classified as {signals['regime'].value} with "
        f"{signals['price_bias'].value} price bias and {signals['vol_bias'].value} volatility bias."
    )

    # 4. Optional LLM enhancement (placeholder for now)
    if use_llm_summary:
        # TODO: Wire up CrewAI agent here with Ollama
        # For now we keep deterministic summary
        llm_comment = " [LLM summary would go here when enabled]"
        base_summary += llm_comment

    # 5. Construct and validate final output
    output = MarketAnalystOutput(
        event_id=event_id,
        regime=signals["regime"],
        price_bias=signals["price_bias"],
        vol_bias=signals["vol_bias"],
        confidence=0.78,  # Could be derived from signal strength in future
        summary=base_summary,
        factors=signals.get("factors", {}),
        key_levels=signals.get("key_levels", {}),
        macro_context="Prototype using public yfinance data only. FRED integration pending.",
        metadata={
            "data_source": "yfinance",
            "tickers_analyzed": list(market_data.keys()),
            "model_used_for_summary": "deterministic_python_v1" if not use_llm_summary else model_name,
            "analysis_version": "0.1.0-prototype",
        },
    )

    return output


# =============================================================================
# Example usage / quick test
# =============================================================================
if __name__ == "__main__":
    print("Running Market Analyst prototype...\n")
    result = analyze_market_regime(use_llm_summary=False)

    print("=== VALIDATED OUTPUT ===")
    print(result.model_dump_json(indent=2))

    print("\n=== Key Insights ===")
    print(f"Regime: {result.regime}")
    print(f"Price Bias: {result.price_bias}")
    print(f"Vol Bias: {result.vol_bias}")
    print(f"Confidence: {result.confidence}")
    print(f"Summary: {result.summary}")