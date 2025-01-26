import streamlit as st
import yfinance as yf
import numpy as np
from math import exp, sqrt, log
from scipy.stats import norm
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt

# Set Page Configuration
st.set_page_config(page_title="FinEdge", layout="wide")

# Initialize Session State for Strike Price
if "strike_price" not in st.session_state:
    st.session_state["strike_price"] = None

# Add Custom CSS for Styling
st.markdown(
    """
    <style>
        .stApp { background-color: #f4f4f4; font-family: 'Arial', sans-serif; }
        h1 { color: #000000; font-family: 'Courier New', Courier, monospace; text-align: center; margin-bottom: 20px; }
        .price-box { 
            background-color: #d1f8d1; 
            border-radius: 10px; 
            padding: 20px; 
            margin: 10px; 
            text-align: center; 
        }
        .price-box-red {
            background-color: #e57373;
        }
        .price-title {
            font-size: 22px; 
            font-weight: bold; 
            color: #000000; 
            margin-bottom: 5px; 
        }
        .price-value {
            font-size: 28px; 
            color: #212121; 
            font-weight: bold;
        }
        .hedge-box {
            background-color: #f0f4c3; 
            padding: 20px; 
            border-radius: 10px; 
            border: 2px solid #cddc39; 
            margin-top: 20px;
        }
        .hedge-title {
            font-size: 20px; 
            font-weight: bold; 
            color: #33691e; 
        }
        .hedge-text {
            font-size: 18px; 
            margin-top: 10px; 
            color: #004d40;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# App Title
st.markdown("<h1>FinEdge</h1>", unsafe_allow_html=True)

# Helper Functions
def black_scholes(S, K, T, r, sigma, option_type):
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
    else:
        return K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def binomial_tree(S, K, T, r, sigma, N, option_type):
    dt = T / N
    u = exp(sigma * sqrt(dt))
    d = 1 / u
    q = (exp(r * dt) - d) / (u - d)
    prices = np.zeros((N + 1, N + 1))

    for i in range(N + 1):
        prices[N, i] = max(0, (S * (u**i) * (d**(N - i)) - K)) if option_type == "call" else max(0, (K - S * (u**i) * (d**(N - i))))
    for j in range(N - 1, -1, -1):
        for i in range(j + 1):
            prices[j, i] = exp(-r * dt) * (q * prices[j + 1, i + 1] + (1 - q) * prices[j + 1, i])
    return prices[0, 0]

def monte_carlo(S, K, T, r, sigma, simulations, option_type):
    np.random.seed(0)
    ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.random.normal(0, 1, simulations) * sqrt(T))
    if option_type == "call":
        return exp(-r * T) * np.mean(np.maximum(ST - K, 0))
    else:
        return exp(-r * T) * np.mean(np.maximum(K - ST, 0))

def calculate_volatility(ticker):
    try:
        data = ticker.history(period="1y")
        data["returns"] = data["Close"].pct_change()
        return data["returns"].std() * sqrt(252)
    except:
        return 0.2  # Default fallback

def delta(S, K, T, r, sigma, option_type):
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    if option_type == "call":
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1

def plot_heatmap(spot_range, vol_range, strike, T, r):
    call_prices = np.zeros((len(vol_range), len(spot_range)))
    put_prices = np.zeros((len(vol_range), len(spot_range)))

    for i, vol in enumerate(vol_range):
        for j, spot in enumerate(spot_range):
            call_prices[i, j] = black_scholes(spot, strike, T, r, vol, "call")
            put_prices[i, j] = black_scholes(spot, strike, T, r, vol, "put")

    # Plotting Call and Put Price Heatmaps side by side
    fig, (ax_call, ax_put) = plt.subplots(1, 2, figsize=(20, 8))

    sns.heatmap(call_prices, xticklabels=np.round(spot_range, 2), yticklabels=np.round(vol_range, 2), annot=True, fmt=".2f", cmap="RdYlGn", ax=ax_call)
    ax_call.set_title('CALL')
    ax_call.set_xlabel('Spot Price')
    ax_call.set_ylabel('Volatility')

    sns.heatmap(put_prices, xticklabels=np.round(spot_range, 2), yticklabels=np.round(vol_range, 2), annot=True, fmt=".2f", cmap="RdYlGn", ax=ax_put)
    ax_put.set_title('PUT')
    ax_put.set_xlabel('Spot Price')
    ax_put.set_ylabel('Volatility')

    return fig

# Sidebar Inputs
st.sidebar.header("Options Pricing Inputs")

# Stock Symbol Input
symbol = st.sidebar.text_input("Stock Symbol (e.g., AAPL, TSLA)", "AAPL").upper()
ticker = yf.Ticker(symbol)
try:
    stock_price = ticker.history(period="1d")["Close"][-1]
except:
    st.sidebar.error(f"Invalid symbol: {symbol}")
    st.stop()

# Display Current Stock Price
st.sidebar.markdown(f"### Current Price: **${stock_price:.2f}**")

# Strike Price Input with Session State
if st.session_state["strike_price"] is None:
    st.session_state["strike_price"] = round(stock_price, 2)

strike_price = st.sidebar.number_input(
    "Strike Price",
    value=st.session_state["strike_price"],
    step=1.0,
    format="%.2f",
    key="strike_price_input",
)
st.session_state["strike_price"] = strike_price  # Update session state

# Volatility
volatility = calculate_volatility(ticker)
st.sidebar.markdown(f"### Calculated Volatility: **{volatility:.2f}**")

# Sliders for Min and Max Volatility
min_volatility = st.sidebar.slider("Min Volatility", min_value=0.01, max_value=1.0, value=0.1)
max_volatility = st.sidebar.slider("Max Volatility", min_value=0.01, max_value=1.0, value=0.5)

# Sliders for Min and Max Spot Price
min_spot_price = st.sidebar.slider("Min Spot Price", min_value=1.0, max_value=stock_price * 2, value=stock_price * 0.8)
max_spot_price = st.sidebar.slider("Max Spot Price", min_value=1.0, max_value=stock_price * 2, value=stock_price * 1.2)

# Other Inputs
expiry_date = st.sidebar.date_input("Expiry Date", value=datetime.now() + timedelta(days=30))
risk_free_rate = st.sidebar.number_input("Risk-Free Rate (Annualized)", value=0.05)
N = st.sidebar.slider("Binomial Tree Steps", min_value=10, max_value=500, value=100)
simulations = st.sidebar.slider("Monte Carlo Simulations", min_value=1000, max_value=50000, value=10000)

# Time to Maturity
# Time to Maturity
expiry_datetime = datetime.combine(expiry_date, datetime.min.time())
T = (expiry_datetime - datetime.now()).days / 365.0
if T <= 0:
    st.sidebar.error("Expiry date must be in the future.")
    st.stop()

# Pricing Models
bs_price_call = black_scholes(stock_price, strike_price, T, risk_free_rate, volatility, "call")
bt_price_call = binomial_tree(stock_price, strike_price, T, risk_free_rate, volatility, N, "call")
mc_price_call = monte_carlo(stock_price, strike_price, T, risk_free_rate, volatility, simulations, "call")

bs_price_put = black_scholes(stock_price, strike_price, T, risk_free_rate, volatility, "put")
bt_price_put = binomial_tree(stock_price, strike_price, T, risk_free_rate, volatility, N, "put")
mc_price_put = monte_carlo(stock_price, strike_price, T, risk_free_rate, volatility, simulations, "put")

# Delta and Hedging
option_delta_call = delta(stock_price, strike_price, T, risk_free_rate, volatility, "call")
option_delta_put = delta(stock_price, strike_price, T, risk_free_rate, volatility, "put")

# Display Results
st.header("Options Pricing Results")

# Call/Put Prices in Green and Red Boxes
st.markdown(
    f"""
    <div style="display: flex; flex-wrap: wrap; justify-content: space-around;">
        <div class="price-box">
            <div class="price-title">Black-Scholes Call Price</div>
            <div class="price-value">${bs_price_call:.2f}</div>
        </div>
        <div class="price-box price-box-red">
            <div class="price-title">Black-Scholes Put Price</div>
            <div class="price-value">${bs_price_put:.2f}</div>
        </div>
        <div class="price-box">
            <div class="price-title">Binomial Tree Call Price</div>
            <div class="price-value">${bt_price_call:.2f}</div>
        </div>
        <div class="price-box price-box-red">
            <div class="price-title">Binomial Tree Put Price</div>
            <div class="price-value">${bt_price_put:.2f}</div>
        </div>
        <div class="price-box">
            <div class="price-title">Monte Carlo Call Price</div>
            <div class="price-value">${mc_price_call:.2f}</div>
        </div>
        <div class="price-box price-box-red">
            <div class="price-title">Monte Carlo Put Price</div>
            <div class="price-value">${mc_price_put:.2f}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Hedging Strategy Section
st.markdown(
    f"""
    <div class="hedge-box">
    <div class="hedge-title">Hedging Strategy</div>
    <div class="hedge-text">
        <b>Delta (Call):</b> {option_delta_call:.4f}<br>
        To hedge a call option, hold <b>{option_delta_call * 100:.2f}%</b> of the underlying stock. A positive delta indicates buying.
    </div>
    <div class="hedge-text">
        <b>Delta (Put):</b> {option_delta_put:.4f}<br>
        To hedge a put option, hold <b>{option_delta_put * 100:.2f}%</b> of the underlying stock. A negative delta suggests shorting.
    </div>
</div>

    """,
    unsafe_allow_html=True,
)

# Heatmaps for Volatility vs. Spot Price
volatilities = np.linspace(min_volatility, max_volatility, 10)
spot_prices = np.linspace(min_spot_price, max_spot_price, 10)

fig = plot_heatmap(spot_prices, volatilities, strike_price, T, risk_free_rate)

st.pyplot(fig) 
    
