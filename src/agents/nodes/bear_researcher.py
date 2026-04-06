"""
Builds the strongest bear case for selected movers.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.agents.providers import LLMRole


SYSTEM_PROMPT = """You are a bearish financial researcher.
Your job is to construct an aggressive, specific bear case for the given asset.
You have already seen the bull case and must directly counter it with opposing
evidence and risk analysis.

You MUST include:
- A specific downside price target with rationale
- A specific risk trigger with expected timeframe
- A concrete data point that contradicts the bull case

Do NOT use vague phrases like "vulnerable to downturns" without backing numbers.

If calibrated event-study data is available, treat it as objective evidence:
use adverse CAR data directly and explain when bullish event evidence looks
fully priced or overstretched.

You must argue the bear case even if you think the bull case is stronger.
This is adversarial analysis. Do not provide financial advice.
Confidence should reflect how compelling the bearish evidence is:
0.5 = slight lean, 0.7 = moderate conviction, 0.9 = strong conviction.

Respond in JSON:
{
  "thesis": "1-2 sentence core bear argument with a specific downside level when supported",
  "evidence": ["point 1", "point 2", "point 3"],
  "bull_weaknesses": ["weakness 1", "weakness 2"],
  "confidence": 0.7,
  "risk_trigger": "specific event that could prove bears right, with timeframe"
}"""


async def bear_researcher_node(state: Mapping[str, object], llm: Any) -> dict[str, object]:
    """Generate bearish arguments that counter the bull case."""
    movers = _dict_list(state.get("selected_movers"))
    bull_args = _dict_list(state.get("bull_arguments"))
    market_report = _as_mapping(state.get("market_report"))
    event_data = _dict_list(state.get("event_data"))
    ta_signals = _mapping_dict(state.get("ta_signals"))
    correlation_pairs = _mapping_dict(state.get("correlation_pairs"))
    composite_scores = _mapping_dict(state.get("composite_scores"))
    regime_state = _as_mapping(state.get("regime_state"))
    lead_lag_pairs = _mapping_dict(state.get("lead_lag_pairs"))

    arguments: list[dict[str, object]] = []
    call_count = _int_or_zero(state.get("llm_call_count"))

    all_prices = _dict_list(state.get("stock_prices")) + _dict_list(state.get("crypto_prices"))
    for index, mover in enumerate(movers):
        symbol = str(mover.get("symbol", "?"))
        asset_type = str(mover.get("asset_type", "unknown"))
        price_data = next(
            (item for item in all_prices if str(item.get("symbol", "")).upper() == symbol.upper()),
            {},
        )
        bull_case = bull_args[index] if index < len(bull_args) else {}
        relevant_events = [
            event
            for event in event_data
            if symbol.upper() in {related.upper() for related in _string_list(event.get("related_symbols"))}
        ]
        ta_data = _as_mapping(ta_signals.get(symbol.upper()))
        corr_data = _as_mapping(correlation_pairs.get(symbol.upper()))
        composite_data = _as_mapping(composite_scores.get(symbol.upper()))
        lead_lag = _as_mapping(_lead_lag_for_symbol(symbol, lead_lag_pairs))

        user_prompt = f"""Asset: {symbol} ({asset_type})
Current price: {_fmt_price(price_data.get('current_price'))}
24h change: {_float_or_zero(price_data.get('change_24h_pct')):+.1f}%

The bull case argues: {bull_case.get('thesis', 'N/A')}
Bull evidence: {bull_case.get('evidence', [])}
Bull confidence: {bull_case.get('confidence', 'N/A')}

Market context: {market_report.get('summary', 'N/A')}

Calibrated event data:
{_format_symbol_events(relevant_events)}

Technical analysis:
{_format_ta_data(ta_data)}

{_format_context_block(
    regime_state=regime_state,
    corr_data=corr_data,
    composite_data=composite_data,
    lead_lag=lead_lag,
)}

Construct an aggressive, specific bear case for {symbol}, directly countering the bull thesis with concrete downside risks and price levels where the data supports them."""

        result = await llm.generate_json(
            system_prompt=_system_prompt(state),
            user_prompt=user_prompt,
            temperature=0.0,
            role=LLMRole.BEAR,
        )
        result["symbol"] = symbol
        arguments.append(result)
        call_count += 1

    return {"bear_arguments": arguments, "llm_call_count": call_count}


def _dict_list(value: object) -> list[dict[str, object]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def _as_mapping(value: object) -> Mapping[str, object]:
    if isinstance(value, dict):
        return value
    return {}


def _mapping_dict(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return {str(key).upper(): item for key, item in value.items()}
    return {}


def _system_prompt(state: Mapping[str, object]) -> str:
    lessons = str(state.get("reflection_lessons", "")).strip()
    if not lessons:
        return SYSTEM_PROMPT
    return f"{SYSTEM_PROMPT}{lessons}"


def _fmt_price(value: object) -> str:
    if isinstance(value, int | float):
        numeric = float(value)
        if abs(numeric) >= 1_000:
            return f"${numeric:,.0f}"
        if abs(numeric) >= 1:
            return f"${numeric:,.2f}"
        return f"${numeric:.4g}"
    return "n/a"


def _float_or_zero(value: object) -> float:
    if isinstance(value, int | float):
        return float(value)
    return 0.0


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str)]


def _format_symbol_events(events: list[dict[str, object]]) -> str:
    if not events:
        return "  None"

    lines: list[str] = []
    for event in events:
        remaining = event.get("remaining_drift_pct")
        remaining_text = "n/a"
        if isinstance(remaining, int | float):
            remaining_text = f"{float(remaining):+.1f}%"
        lines.append(
            "  "
            f"{event.get('event_type', 'unknown')} "
            f"({(_float_or_zero(event.get('confidence'))):.0%} confidence) | "
            f"hist {_float_or_zero(event.get('expected_car_pct')):+.1f}% over "
            f"{event.get('expected_car_window', 'N/A')} | "
            f"current {_float_or_zero(event.get('actual_move_pct')):+.1f}% | "
            f"remaining {remaining_text}"
        )
    return "\n".join(lines)


def _format_ta_data(ta_data: Mapping[str, object]) -> str:
    if not ta_data:
        return "  None"
    trend_joiner = ">" if _float_or_zero(ta_data.get("ema_20")) > _float_or_zero(ta_data.get("ema_50")) else "<"
    squeeze_text = " (SQUEEZE)" if bool(ta_data.get("bb_squeeze")) else ""
    return "\n".join(
        [
            f"  RSI(14): {_float_or_zero(ta_data.get('rsi_value')):.1f} ({ta_data.get('rsi_signal', 'unknown')})",
            f"  MACD: {ta_data.get('macd_signal', 'unknown')} (histogram: {_float_or_zero(ta_data.get('macd_histogram')):+.3f})",
            f"  Bollinger: price {ta_data.get('bb_position', 'unknown')}{squeeze_text}",
            f"  Trend: {ta_data.get('trend', 'unknown')} (EMA20{trend_joiner}EMA50)",
            f"  ATR: {_float_or_zero(ta_data.get('atr_pct')):.1f}% ({ta_data.get('volatility', 'unknown')} volatility)",
            f"  Composite signal: {ta_data.get('overall_signal', 'unknown')} ({_float_or_zero(ta_data.get('confidence')):.0%})",
        ]
    )


def _format_context_block(
    *,
    regime_state: Mapping[str, object],
    corr_data: Mapping[str, object],
    composite_data: Mapping[str, object],
    lead_lag: Mapping[str, object],
) -> str:
    return "\n".join(
        [
            "--- Market Regime ---",
            f"Regime: {regime_state.get('regime', 'N/A')}",
            f"Description: {regime_state.get('description', 'N/A')}",
            "",
            "--- Correlation ---",
            f"Regime signal: {corr_data.get('regime_signal', 'N/A')}",
            f"20d/60d/90d: {_float_or_zero(corr_data.get('corr_20d')):.2f} / {_float_or_zero(corr_data.get('corr_60d')):.2f} / {_float_or_zero(corr_data.get('corr_90d')):.2f}",
            f"Divergence: {_float_or_zero(corr_data.get('divergence')):+.2f} | z-score: {_float_or_zero(corr_data.get('z_score')):+.2f}",
            "",
            "--- Lead-Lag ---",
            f"{lead_lag.get('interpretation', 'N/A')}",
            "",
            "--- Composite Signal ---",
            f"Score: {_float_or_zero(composite_data.get('score')):+.2f} ({composite_data.get('direction', 'N/A')})",
            f"Agreement: {_float_or_zero(composite_data.get('agreement')):.0%}",
            f"Summary: {composite_data.get('summary', 'N/A')}",
        ]
    )


def _lead_lag_for_symbol(symbol: str, lead_lag_pairs: Mapping[str, object]) -> object | None:
    symbol_upper = symbol.upper()
    for key, value in lead_lag_pairs.items():
        key_upper = str(key).upper()
        if symbol_upper in key_upper.split(":"):
            return value
    return None


def _int_or_zero(value: object) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0
