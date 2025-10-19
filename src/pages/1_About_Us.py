import streamlit as st
from pathlib import Path


st.set_page_config(page_title="Termbo Algo Trading Corp", layout="wide")

st.title("Termbo Algo Trading Corp")
st.caption("Grow your capital with us")
st.caption("We are in emulation mode. No live orders will be placed.")

st.header("About Us")
st.write("We are a small bunch of passionate IT engineers who are exploring how to build light weight algo trading systems.")

st.header("Our Portfolio")
st.write("For any investment you make, we ensure to diversify across multiple asset classes and strategies. They are clubbed into 3 main buckets")

# write a sub-header with a markdown table the headind is High-Return/High-Growth Bucket
st.subheader("📈 High-Return / Growth Bucket")
st.markdown("""
| Rank |   Ticker  | Company                  | Rationale                                                                                                    |
| :--: | :-------: | :----------------------- | :----------------------------------------------------------------------------------------------------------- |
|   1  |  **AAPL** | Apple Inc.               | Market leader in tech ecosystem; strong EPS growth, buybacks, and innovation cycle (iPhone → services → AI). |
|   2  |  **MSFT** | Microsoft Corp.          | Cloud and enterprise software powerhouse; consistent double-digit revenue growth since 2010.                 |
|   3  |  **NVDA** | NVIDIA Corp.             | GPU and AI infrastructure leader; explosive EPS compounding since 2016 AI/ML boom.                           |
|   4  |  **AMZN** | Amazon.com Inc.          | Dominant in e-commerce and AWS; reinvests heavily with steady long-term compounding.                         |
|   5  | **GOOGL** | Alphabet Inc.            | Diversified revenue via ads, YouTube, and AI; strong cash flow and R&D scale.                                |
|   6  |  **LLY**  | Eli Lilly & Co.          | Biopharma innovator (diabetes, obesity drugs); robust revenue growth with pricing power.                     |
|   7  |  **COST** | Costco Wholesale Corp.   | Steady membership-based retail growth; high renewal rates and strong free cash flow.                         |
|   8  |  **TMO**  | Thermo Fisher Scientific | Scientific instruments & diagnostics; stable high-margin growth and acquisitions.                            |
|   9  |  **ADBE** | Adobe Inc.               | Software subscription model (Creative Cloud, Acrobat); durable revenue compounding.                          |
|  10  |  **INTC** | Intel Corp.              | Long history, cyclical but vital semiconductor exposure; potential turnaround on foundry strategy.           |
""")

st.subheader("💵 High-Dividend / Income Bucket")
st.markdown("""
| Rank |  Ticker | Company                         | Dividend Yield | Comment                                                                 |
| :--: | :-----: | :------------------------------ | :---------------------: | :---------------------------------------------------------------------- |
|   1  | **JNJ** | Johnson & Johnson               |          ~3.0%          | Dividend Aristocrat > 60 yrs; stable healthcare earnings.               |
|   2  |  **PG** | Procter & Gamble                |          ~2.5%          | Consumer-staples giant; consistent cash flows and payout growth.        |
|   3  |  **KO** | Coca-Cola Co.                   |          ~3.1%          | Steady dividends backed by global brand strength and pricing power.     |
|   4  | **PEP** | PepsiCo Inc.                    |          ~2.8%          | Diversified beverages/snacks; reliable dividend compounder.             |
|   5  | **CVX** | Chevron Corp.                   |          ~4.0%          | High yield, strong balance sheet, long dividend track record.           |
|   6  | **XOM** | Exxon Mobil Corp.               |          ~3.5%          | Energy major with robust free cash flow and consistent payouts.         |
|   7  |  **VZ** | Verizon Communications          |          ~6.5%          | One of the highest large-cap yields; stable telecom cash flows.         |
|   8  | **PFE** | Pfizer Inc.                     |          ~5.0%          | Attractive yield; post-COVID earnings normalization expected.           |
|   9  | **MCD** | McDonald’s Corp.                |          ~2.3%          | Global brand; long dividend-growth history with strong franchise model. |
|  10  | **IBM** | International Business Machines |          ~3.7%          | High-yield tech; stable cash generation from enterprise software.       |
""")

st.subheader("🪙 Gold Hedge ETF")
st.markdown("""- SPDR Gold Shares (GLD) — Most liquid U.S. gold ETF.""")

st.header("Investment Philosophy")
st.write("We believe in a balanced approach to investing, combining high-growth equities with stable, income-generating assets and a hedge against market volatility through gold. Our portfolio is designed to capture long-term capital appreciation while managing risk effectively. Any investment you make will be diversified across these three buckets to ensure a robust and resilient portfolio. Please note this is only the distribution of your investment not the distribution of your portfolio")
st.markdown("""
| Asset Bucket                    | Target Allocation |
|---------------------------------|------------------:|
| 📈 High-Return / Growth Bucket   |               60% |
| 💵 High-Dividend / Income Bucket |               25% |
| 🪙 Gold Hedge ETF                |               15% |
""")

st.header("Our Charges")

st.markdown("""
Our charging structure is simple and transparent. We charge a flat 1% management fee on your total investment amount annually. This fee covers all our services, including portfolio management, trade execution, and regular performance reporting. There are no hidden fees or commissions. Our goal is to align our interests with yours by focusing on growing your capital over the long term.
""")

st.header("Our Responsibilities")

st.markdown("""
We are committed to managing your investments with the utmost integrity and professionalism. Our responsibilities include:
- Conducting thorough research and analysis to select the best assets for your portfolio.
- Regularly monitoring and rebalancing your portfolio to maintain the target asset allocation.
- Providing transparent and timely reports on portfolio performance.
- Ensuring compliance with all regulatory requirements and industry best practices.
""")

st.header("Our Design")

# Code showing the image in the images folder named portfolio_design.png
OUTPUT_FILE = Path(__file__).resolve().parent.parent / "images" / "Trading system design.drawio.png"
st.image(OUTPUT_FILE, caption="Portfolio Design Overview")

st.write("Our stock picks are bound to change over time as market conditions evolve. Please do your own due diligence before making any investment decisions.")