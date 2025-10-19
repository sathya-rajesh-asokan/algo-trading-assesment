import streamlit as st
from pathlib import Path


st.set_page_config(page_title="Termbo Algo Trading Corp", layout="wide")

st.title("Termbo Algo Trading Corp")
st.caption("Grow your capital with us")
st.caption("We are in emulation mode. No live orders will be placed.")

st.header("What we learned building this system")
st.markdown("""
- **Data Handling**: Learned to fetch, clean, and store large datasets efficiently using APIs
- **Backtesting**: Gained insights into simulating trading strategies against historical data
- **Strategy Development**: Developed skills in creating and refining algorithmic trading strategies
- **Risk Management**: Understood the importance of risk controls and position sizing in trading
- **Performance Analysis**: Analyzed strategy performance using key metrics and visualizations
- **Automation**: Built automated workflows for data updates, strategy execution, and reporting
- **Streamlit**: Gained experience in building interactive web apps for data exploration and strategy
""")

st.header("What we could do more")
st.markdown("""
- **More Indicators**: Implement additional technical indicators and machine learning models
- **Sector-Specific Strategies**: Implement ability to apply different strategies to different sectors and buckets to maximize returns
- **Non-Tangible Metrics**: Include non-tangible metrics like sentiment analysis, behavioral finance indicators
- **Consider Macro Factors**: Integrate macroeconomic data, index trends to be considered for strategy design/engine
""")