import json
from pathlib import Path
from typing import Dict, List

import streamlit as st

INDICATORS: Dict[str, str] = {
    "close": "Close Price",
    "SMA20": "SMA (20)",
    "SMA50": "SMA (50)",
    "SMA_custom": "Custom SMA",
    "RSI14": "RSI (14)",
    "MACD": "MACD",
    "MACD_signal": "MACD Signal",
    "MACD_hist": "MACD Histogram",
    "ATR14": "ATR (14)",
    "volume": "Volume",
}

OPERATORS = [
    ">",
    ">=",
    "<",
    "<=",
    "crosses above",
    "crosses below",
    "equal to",
]

STRATEGY_FILE = Path(__file__).resolve().parent.parent / "data" / "strategies.json"

st.set_page_config(page_title="Strategy Builder", layout="wide")

st.title("Strategy Builder")
st.caption("Define rule-based entry and exit criteria using stored indicators.")


def _load_strategies() -> Dict[str, Dict]:
    if STRATEGY_FILE.exists():
        try:
            return json.loads(STRATEGY_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            st.warning("Could not parse strategies.json. Starting with an empty collection.")
    return {}


def _save_strategies(data: Dict[str, Dict]) -> None:
    STRATEGY_FILE.parent.mkdir(parents=True, exist_ok=True)
    STRATEGY_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


strategies = _load_strategies()
strategy_names = ["<New Strategy>"] + sorted(strategies.keys())

if "strategy_name" not in st.session_state:
    st.session_state.strategy_name = ""
if "strategy_buy_rules" not in st.session_state:
    st.session_state.strategy_buy_rules = []
if "strategy_sell_rules" not in st.session_state:
    st.session_state.strategy_sell_rules = []
if "loaded_strategy" not in st.session_state:
    st.session_state.loaded_strategy = "<New Strategy>"


selected_strategy = st.selectbox(
    "Load existing strategy",
    strategy_names,
)

if selected_strategy != st.session_state.loaded_strategy:
    st.session_state.loaded_strategy = selected_strategy
    if selected_strategy != "<New Strategy>":
        payload = strategies[selected_strategy]
        st.session_state.strategy_name = selected_strategy
        st.session_state.strategy_buy_rules = payload.get("buy_rules", []).copy()
        st.session_state.strategy_sell_rules = payload.get("sell_rules", []).copy()
    else:
        st.session_state.strategy_name = ""
        st.session_state.strategy_buy_rules = []
        st.session_state.strategy_sell_rules = []


st.text_input("Strategy Name", key="strategy_name", placeholder="e.g., SMA crossover with RSI filter")

st.divider()
st.subheader("Add Rule")

with st.form("add_rule_form", clear_on_submit=True):
    rule_side = st.radio("Rule applies to", ["Buy", "Sell"], horizontal=True)
    indicator = st.selectbox(
        "Indicator",
        list(INDICATORS.keys()),
        format_func=lambda key: INDICATORS.get(key, key),
    )
    operator = st.selectbox("Operator", OPERATORS)
    compare_type = st.selectbox(
        "Compare against",
        ["Number", "Indicator"],
        key="compare_type_select",
    )
    if compare_type == "Number":
        compare_value = st.number_input(
            "Target value",
            value=0.0,
            format="%.4f",
            key="compare_number_value",
        )
        compare_payload = {"type": "number", "value": compare_value}
    else:
        compare_indicator = st.selectbox(
            "Target indicator",
            list(INDICATORS.keys()),
            format_func=lambda key: INDICATORS.get(key, key),
            key="compare_indicator_select",
        )
        compare_payload = {"type": "indicator", "value": compare_indicator}
    note = st.text_input("Optional note", placeholder="e.g., Trigger when RSI exits oversold zone")

    submitted = st.form_submit_button("Add Rule")
    if submitted:
        rule = {
            "indicator": indicator,
            "operator": operator,
            "compare": compare_payload,
            "note": note,
        }
        if rule_side == "Buy":
            st.session_state.strategy_buy_rules.append(rule)
        else:
            st.session_state.strategy_sell_rules.append(rule)
        st.success(f"{rule_side} rule added.")
        for reset_key in ("compare_type_select", "compare_number_value", "compare_indicator_select"):
            st.session_state.pop(reset_key, None)


def _describe_rule(rule: Dict) -> str:
    indicator_label = INDICATORS.get(rule["indicator"], rule["indicator"])
    compare = rule.get("compare", {})
    if compare.get("type") == "indicator":
        compare_label = INDICATORS.get(compare.get("value"), compare.get("value"))
    else:
        compare_label = compare.get("value")
    description = f"{indicator_label} {rule['operator']} {compare_label}"
    if rule.get("note"):
        description += f" — {rule['note']}"
    return description


def _render_rules(label: str, rules: List[Dict], key_prefix: str) -> None:
    st.markdown(f"**{label}**")
    if not rules:
        st.info(f"No {label.lower()} defined.")
        return
    for idx, rule in enumerate(rules):
        cols = st.columns([0.85, 0.15])
        cols[0].write(_describe_rule(rule))
        if cols[1].button("Remove", key=f"{key_prefix}_{idx}"):
            rules.pop(idx)
            st.rerun()


col_buy, col_sell = st.columns(2)
with col_buy:
    _render_rules("Buy rules", st.session_state.strategy_buy_rules, "buy_rule_remove")
with col_sell:
    _render_rules("Sell rules", st.session_state.strategy_sell_rules, "sell_rule_remove")

st.divider()

save_col, reset_col = st.columns([0.7, 0.3])
with save_col:
    if st.button("Save Strategy", type="primary"):
        name = st.session_state.strategy_name.strip()
        if not name:
            st.error("Provide a strategy name before saving.")
        else:
            strategies[name] = {
                "buy_rules": st.session_state.strategy_buy_rules,
                "sell_rules": st.session_state.strategy_sell_rules,
            }
            _save_strategies(strategies)
            st.success(f"Strategy '{name}' saved.")

with reset_col:
    if st.button("Reset Builder"):
        st.session_state.strategy_name = ""
        st.session_state.strategy_buy_rules = []
        st.session_state.strategy_sell_rules = []
        st.session_state.loaded_strategy = "<New Strategy>"
        st.rerun()
