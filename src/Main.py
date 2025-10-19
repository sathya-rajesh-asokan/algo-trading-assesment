from datetime import date, timedelta

import streamlit as st

from scripts.save_date_ranges import save_date_ranges
import scripts.fetch_trading_days as fetch_trading_days

st.set_page_config(page_title="Termbo Algo Trading Corp", layout="wide")

st.title("Termbo Algo Trading Corp")
st.caption("Grow your capital with us")
st.caption("We are in emulation mode. No live orders will be placed.")

st.page_link("pages/1_About_Us.py", label="About Us", icon="🏢")
st.page_link("pages/2_Time_Ranges_For_Simulation.py", label="Time Ranges For Simulation", icon="🔍")
st.page_link("pages/3_Market_Data_Explorer.py", label="Market Data Explorer", icon="📊")
st.page_link("pages/4_Strategy_Builder.py", label="Strategy Builder", icon="🛠️")
st.page_link("pages/5_Backtesting.py", label="Backtester", icon="📈")
st.page_link("pages/6_Trade_Simulation.py", label="Trade Simulator", icon="🤖" )
st.divider()
st.info(
    "Use the navigation links above or the built-in sidebar menu to move between pages."
)

