"""
Complete Python Trading Simulator with Streamlit UI
Features:
- Angel One SmartAPI integration
- Yahoo Finance fallback (FREE)
- Moving Average Crossover Strategy
- RSI Strategy
- Backtesting engine
- Beautiful interactive charts

Installation:
pip install streamlit pandas numpy yfinance requests pyotp plotly

Run:
streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import pyotp
from datetime import datetime, timedelta
from collections import deque
import heapq
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================================
# ANGEL ONE API CLIENT
# ============================================================================

class AngelOneClient:
    """Angel One SmartAPI Client for Indian Stock Market"""

    def __init__(self, api_key=None, client_code=None, password=None, totp_secret=None):
        self.base_url = "https://apiconnect.angelbroking.com"
        self.api_key = api_key
        self.client_code = client_code
        self.password = password
        self.totp_secret = totp_secret
        self.jwt_token = None
        self.refresh_token = None
        self.feed_token = None

    def generate_totp(self):
        """Generate TOTP for 2FA authentication"""
        if not self.totp_secret:
            return "000000"
        totp = pyotp.TOTP(self.totp_secret)
        return totp.now()

    def login(self):
        """Authenticate with Angel One and get JWT token"""
        url = f"{self.base_url}/rest/auth/angelbroking/user/v1/loginByPassword"

        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'X-UserType': 'USER',
            'X-SourceID': 'WEB',
            'X-ClientLocalIP': '127.0.0.1',
            'X-ClientPublicIP': '127.0.0.1',
            'X-MACAddress': '00:00:00:00:00:00',
            'X-PrivateKey': self.api_key
        }

        payload = {
            'clientcode': self.client_code,
            'password': self.password,
            'totp': self.generate_totp()
        }

        try:
            response = requests.post(url, headers=headers, json=payload)
            data = response.json()

            if data.get('status'):
                self.jwt_token = data['data']['jwtToken']
                self.refresh_token = data['data']['refreshToken']
                self.feed_token = data['data'].get('feedToken', '')
                return True, "Login successful"
            else:
                return False, data.get('message', 'Login failed')
        except Exception as e:
            return False, f"Error: {str(e)}"

    def get_historical_data(self, symbol_token, exchange='NSE', interval='ONE_DAY',
                          from_date=None, to_date=None):
        """Get historical candlestick data"""
        url = f"{self.base_url}/rest/secure/angelbroking/historical/v1/getCandleData"

        if not from_date:
            from_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d 09:15')
        if not to_date:
            to_date = datetime.now().strftime('%Y-%m-%d 15:30')

        headers = {
            'Authorization': f'Bearer {self.jwt_token}',
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'X-UserType': 'USER',
            'X-SourceID': 'WEB',
            'X-ClientLocalIP': '127.0.0.1',
            'X-ClientPublicIP': '127.0.0.1',
            'X-MACAddress': '00:00:00:00:00:00',
            'X-PrivateKey': self.api_key
        }

        payload = {
            'exchange': exchange,
            'symboltoken': symbol_token,
            'interval': interval,
            'fromdate': from_date,
            'todate': to_date
        }

        try:
            response = requests.post(url, headers=headers, json=payload)
            data = response.json()

            if data.get('status') and data.get('data'):
                df = pd.DataFrame(data['data'],
                                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.rename(columns={'timestamp': 'date'}, inplace=True)
                return df
            return None
        except Exception as e:
            st.error(f"Error fetching historical data: {e}")
            return None

# ============================================================================
# YAHOO FINANCE CLIENT (FREE ALTERNATIVE)
# ============================================================================

class YahooFinanceClient:
    """Yahoo Finance client for free stock data"""

    STOCK_SYMBOLS = {
        'RELIANCE': 'RELIANCE.NS',
        'TCS': 'TCS.NS',
        'HDFCBANK': 'HDFCBANK.NS',
        'INFY': 'INFY.NS',
        'ICICIBANK': 'ICICIBANK.NS',
        'HINDUNILVR': 'HINDUNILVR.NS',
        'ITC': 'ITC.NS',
        'SBIN': 'SBIN.NS',
        'BHARTIARTL': 'BHARTIARTL.NS',
        'KOTAKBANK': 'KOTAKBANK.NS',
        'LT': 'LT.NS',
        'WIPRO': 'WIPRO.NS',
        'AXISBANK': 'AXISBANK.NS',
        'MARUTI': 'MARUTI.NS',
        'TITAN': 'TITAN.NS',
        'SUNPHARMA': 'SUNPHARMA.NS',
        'ULTRACEMCO': 'ULTRACEMCO.NS',
        'ASIANPAINT': 'ASIANPAINT.NS',
        'NESTLEIND': 'NESTLEIND.NS',
        'TATAMOTORS': 'TATAMOTORS.NS'
    }

    @staticmethod
    def get_historical_data(symbol, period='6mo'):
        """Get historical data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period)
            df.reset_index(inplace=True)
            df.columns = [col.lower() for col in df.columns]
            return df
        except Exception as e:
            st.error(f"Error fetching Yahoo Finance data: {e}")
            return None

# ============================================================================
# DSA: SLIDING WINDOW FOR MOVING AVERAGE
# ============================================================================

class SlidingWindow:
    """
    Sliding Window for efficient Moving Average calculation
    Time Complexity: O(1) per element
    """
    def __init__(self, window_size):
        self.window = deque(maxlen=window_size)
        self.sum = 0.0
        self.window_size = window_size

    def add(self, value):
        """Add new value and calculate MA in O(1)"""
        if len(self.window) == self.window_size:
            self.sum -= self.window[0]

        self.window.append(value)
        self.sum += value

        return self.sum / len(self.window)

# ============================================================================
# TRADING STRATEGIES
# ============================================================================

class MovingAverageCrossover:
    """Moving Average Crossover Strategy"""

    def __init__(self, short_window=5, long_window=20):
        self.short_window = short_window
        self.long_window = long_window

    def calculate_ma(self, prices, window):
        """Calculate moving average using Sliding Window"""
        ma = []
        sw = SlidingWindow(window)

        for price in prices:
            ma_value = sw.add(price)
            ma.append(ma_value if len(sw.window) == window else None)

        return ma

    def generate_signals(self, df):
        """Generate buy/sell signals"""
        df['ma_short'] = self.calculate_ma(df['close'].values, self.short_window)
        df['ma_long'] = self.calculate_ma(df['close'].values, self.long_window)

        df['signal'] = 'HOLD'
        df['reason'] = ''

        for i in range(1, len(df)):
            if df['ma_short'].iloc[i-1] is not None and df['ma_long'].iloc[i-1] is not None:
                # Golden Cross - BUY
                if (df['ma_short'].iloc[i-1] <= df['ma_long'].iloc[i-1] and
                    df['ma_short'].iloc[i] > df['ma_long'].iloc[i]):
                    df.at[i, 'signal'] = 'BUY'
                    df.at[i, 'reason'] = f'Golden Cross: {self.short_window}-MA crossed above {self.long_window}-MA'

                # Death Cross - SELL
                elif (df['ma_short'].iloc[i-1] >= df['ma_long'].iloc[i-1] and
                      df['ma_short'].iloc[i] < df['ma_long'].iloc[i]):
                    df.at[i, 'signal'] = 'SELL'
                    df.at[i, 'reason'] = f'Death Cross: {self.short_window}-MA crossed below {self.long_window}-MA'

        return df


class RSIStrategy:
    """RSI Strategy"""

    def __init__(self, period=14, oversold=30, overbought=70):
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def calculate_rsi(self, prices):
        """Calculate RSI indicator"""
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gains = []
        avg_losses = []

        avg_gain = np.mean(gains[:self.period])
        avg_loss = np.mean(losses[:self.period])

        avg_gains.append(avg_gain)
        avg_losses.append(avg_loss)

        for i in range(self.period, len(gains)):
            avg_gain = (avg_gain * (self.period - 1) + gains[i]) / self.period
            avg_loss = (avg_loss * (self.period - 1) + losses[i]) / self.period

            avg_gains.append(avg_gain)
            avg_losses.append(avg_loss)

        rsi = []
        for avg_gain, avg_loss in zip(avg_gains, avg_losses):
            if avg_loss == 0:
                rsi.append(100)
            else:
                rs = avg_gain / avg_loss
                rsi.append(100 - (100 / (1 + rs)))

        return [None] * self.period + rsi

    def generate_signals(self, df):
        """Generate buy/sell signals based on RSI"""
        df['rsi'] = self.calculate_rsi(df['close'].values)
        df['signal'] = 'HOLD'
        df['reason'] = ''

        for i in range(1, len(df)):
            if df['rsi'].iloc[i] is not None:
                # Oversold - BUY
                if df['rsi'].iloc[i] < self.oversold and df['rsi'].iloc[i-1] >= self.oversold:
                    df.at[i, 'signal'] = 'BUY'
                    df.at[i, 'reason'] = f'RSI oversold: {df["rsi"].iloc[i]:.1f} < {self.oversold}'

                # Overbought - SELL
                elif df['rsi'].iloc[i] > self.overbought and df['rsi'].iloc[i-1] <= self.overbought:
                    df.at[i, 'signal'] = 'SELL'
                    df.at[i, 'reason'] = f'RSI overbought: {df["rsi"].iloc[i]:.1f} > {self.overbought}'

        return df

# ============================================================================
# BACKTESTING ENGINE
# ============================================================================

class BacktestEngine:
    """Backtesting engine to test strategies on historical data"""

    def __init__(self, initial_capital=100000, brokerage_pct=0.0003):
        self.initial_capital = initial_capital
        self.brokerage_pct = brokerage_pct

    def run_backtest(self, df):
        """Run backtest on dataframe with signals"""
        cash = self.initial_capital
        shares = 0
        trades = []
        portfolio_values = []

        for idx, row in df.iterrows():
            current_price = row['close']
            signal = row['signal']

            # Execute BUY signal
            if signal == 'BUY' and cash >= current_price:
                shares_to_buy = int(cash / current_price)
                cost = shares_to_buy * current_price
                brokerage = cost * self.brokerage_pct
                total_cost = cost + brokerage

                if total_cost <= cash:
                    cash -= total_cost
                    shares += shares_to_buy

                    trades.append({
                        'date': row['date'] if 'date' in row else idx,
                        'type': 'BUY',
                        'price': current_price,
                        'shares': shares_to_buy,
                        'cost': total_cost,
                        'brokerage': brokerage,
                        'reason': row.get('reason', 'Buy signal')
                    })

            # Execute SELL signal
            elif signal == 'SELL' and shares > 0:
                revenue = shares * current_price
                brokerage = revenue * self.brokerage_pct
                net_revenue = revenue - brokerage

                cash += net_revenue

                trades.append({
                    'date': row['date'] if 'date' in row else idx,
                    'type': 'SELL',
                    'price': current_price,
                    'shares': shares,
                    'revenue': net_revenue,
                    'brokerage': brokerage,
                    'reason': row.get('reason', 'Sell signal')
                })

                shares = 0

            # Calculate portfolio value
            portfolio_value = cash + (shares * current_price)
            portfolio_values.append(portfolio_value)

        # Final portfolio value
        final_value = cash + (shares * df['close'].iloc[-1])
        profit = final_value - self.initial_capital
        profit_pct = (profit / self.initial_capital) * 100

        # Calculate metrics
        max_drawdown = self.calculate_max_drawdown(portfolio_values)
        win_rate = self.calculate_win_rate(trades)

        # Buy and hold comparison
        buy_hold_shares = int(self.initial_capital / df['close'].iloc[0])
        buy_hold_value = buy_hold_shares * df['close'].iloc[-1]
        buy_hold_profit = buy_hold_value - self.initial_capital
        buy_hold_pct = (buy_hold_profit / self.initial_capital) * 100

        return {
            'trades': trades,
            'portfolio_values': portfolio_values,
            'final_value': final_value,
            'profit': profit,
            'profit_pct': profit_pct,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'num_trades': len(trades),
            'buy_hold_value': buy_hold_value,
            'buy_hold_profit': buy_hold_profit,
            'buy_hold_pct': buy_hold_pct
        }

    def calculate_max_drawdown(self, portfolio_values):
        """Calculate maximum drawdown"""
        peak = portfolio_values[0]
        max_dd = 0

        for value in portfolio_values:
            if value > peak:
                peak = value
            dd = ((peak - value) / peak) * 100
            if dd > max_dd:
                max_dd = dd

        return max_dd

    def calculate_win_rate(self, trades):
        """Calculate win rate"""
        if len(trades) < 2:
            return 0

        wins = 0
        total_pairs = 0

        for i in range(0, len(trades) - 1, 2):
            if trades[i]['type'] == 'BUY' and i + 1 < len(trades) and trades[i + 1]['type'] == 'SELL':
                total_pairs += 1
                if trades[i + 1]['price'] > trades[i]['price']:
                    wins += 1

        return (wins / total_pairs * 100) if total_pairs > 0 else 0

# ============================================================================
# STREAMLIT UI
# ============================================================================
from sklearn.linear_model import LinearRegression

def predict_portfolio_growth(balance_history, days_to_predict=30):
    """
    Predicts future portfolio value using Linear Regression on historical performance.
    """
    if not balance_history or len(balance_history) < 5:
        return None, None
    
    # Prepare data for regression
    days = np.array(range(len(balance_history))).reshape(-1, 1)
    values = np.array(balance_history).reshape(-1, 1)
    
    # Train model
    model = LinearRegression()
    model.fit(days, values)
    
    # Predict future
    last_day = len(balance_history)
    future_days = np.array(range(last_day, last_day + days_to_predict)).reshape(-1, 1)
    future_values = model.predict(future_days)
    
    return future_days.flatten(), future_values.flatten()
def main():
    st.set_page_config(page_title="Python Trading Simulator", page_icon="📈", layout="wide")

    # Header
    st.title("🐍 Python Trading Simulator")
    st.markdown("**Complete backtesting & live trading system for Indian stock market**")

    # Sidebar
    st.sidebar.header("⚙️ Configuration")

    # Data source selection
    data_source = st.sidebar.selectbox(
        "Data Source",
        ["Yahoo Finance (FREE)", "Angel One API (Requires credentials)"]
    )

    # Stock selection
    if data_source == "Yahoo Finance (FREE)":
        stock_options = list(YahooFinanceClient.STOCK_SYMBOLS.keys())
        selected_stock = st.sidebar.selectbox("Select Stock", stock_options)
        symbol = YahooFinanceClient.STOCK_SYMBOLS[selected_stock]
    else:
        selected_stock = st.sidebar.text_input("Stock Symbol", "RELIANCE-EQ")
        symbol_token = st.sidebar.text_input("Symbol Token", "2885")

    # Strategy selection
    strategy_type = st.sidebar.selectbox(
        "Trading Strategy",
        ["Moving Average Crossover", "RSI Strategy"]
    )

    # Strategy parameters
    if strategy_type == "Moving Average Crossover":
        short_ma = st.sidebar.slider("Short MA Period", 3, 20, 5)
        long_ma = st.sidebar.slider("Long MA Period", 10, 50, 20)
    else:
        rsi_period = st.sidebar.slider("RSI Period", 7, 21, 14)
        oversold = st.sidebar.slider("Oversold Level", 20, 40, 30)
        overbought = st.sidebar.slider("Overbought Level", 60, 80, 70)

    # Capital
    initial_capital = st.sidebar.number_input(
        "Initial Capital (₹)",
        min_value=10000,
        max_value=10000000,
        value=100000,
        step=10000
    )

    # Time period
    period = st.sidebar.selectbox(
        "Time Period",
        ["1mo", "3mo", "6mo", "1y", "2y"]
    )

    # Angel One credentials (if selected)
    if data_source == "Angel One API (Requires credentials)":
        with st.sidebar.expander("🔐 Angel One Credentials"):
            api_key = st.text_input("API Key", type="password")
            client_code = st.text_input("Client Code")
            password = st.text_input("Password", type="password")
            totp_secret = st.text_input("TOTP Secret", type="password")

    # Run backtest button
    run_backtest = st.sidebar.button("🚀 Run Backtest", type="primary")

    # Main content
    if run_backtest:
        with st.spinner("Fetching data and running backtest..."):
            # Fetch data
            if data_source == "Yahoo Finance (FREE)":
                df = YahooFinanceClient.get_historical_data(symbol, period)
            else:
                # Angel One API
                client = AngelOneClient(api_key, client_code, password, totp_secret)
                success, message = client.login()

                if success:
                    st.success(f"✅ {message}")
                    df = client.get_historical_data(symbol_token)
                else:
                    st.error(f"❌ {message}")
                    st.info("💡 Using Yahoo Finance as fallback")
                    df = YahooFinanceClient.get_historical_data(symbol, period)

            if df is not None and not df.empty:
                # Apply strategy
                if strategy_type == "Moving Average Crossover":
                    strategy = MovingAverageCrossover(short_ma, long_ma)
                else:
                    strategy = RSIStrategy(rsi_period, oversold, overbought)

                df = strategy.generate_signals(df)

                # Run backtest
                engine = BacktestEngine(initial_capital)
                results = engine.run_backtest(df)

                # Display results
                st.success("✅ Backtest Complete!")

                # Metrics
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric(
                        "Strategy Return",
                        f"₹{results['profit']:,.0f}",
                        f"{results['profit_pct']:.2f}%"
                    )

                with col2:
                    st.metric(
                        "Buy & Hold Return",
                        f"₹{results['buy_hold_profit']:,.0f}",
                        f"{results['buy_hold_pct']:.2f}%"
                    )

                with col3:
                    st.metric(
                        "Win Rate",
                        f"{results['win_rate']:.1f}%",
                        f"{results['num_trades']} trades"
                    )

                with col4:
                    st.metric(
                        "Max Drawdown",
                        f"{results['max_drawdown']:.2f}%"
                    )

                # Price chart
                st.subheader("📊 Price & Indicators")

                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.03,
                    row_heights=[0.7, 0.3]
                )

                # Price and MAs
                fig.add_trace(
                    go.Scatter(x=df.index, y=df['close'], name='Price', line=dict(color='blue')),
                    row=1, col=1
                )

                if 'ma_short' in df.columns:
                    fig.add_trace(
                        go.Scatter(x=df.index, y=df['ma_short'], name=f'MA {short_ma}',
                                 line=dict(color='orange', dash='dash')),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=df.index, y=df['ma_long'], name=f'MA {long_ma}',
                                 line=dict(color='red', dash='dash')),
                        row=1, col=1
                    )

                # Buy/Sell signals
                buy_signals = df[df['signal'] == 'BUY']
                sell_signals = df[df['signal'] == 'SELL']

                fig.add_trace(
                    go.Scatter(x=buy_signals.index, y=buy_signals['close'],
                             mode='markers', name='Buy', marker=dict(color='green', size=10, symbol='triangle-up')),
                    row=1, col=1
                )

                fig.add_trace(
                    go.Scatter(x=sell_signals.index, y=sell_signals['close'],
                             mode='markers', name='Sell', marker=dict(color='red', size=10, symbol='triangle-down')),
                    row=1, col=1
                )

                # Portfolio value
                fig.add_trace(
                    go.Scatter(x=df.index, y=results['portfolio_values'], name='Portfolio Value',
                             line=dict(color='green', width=2)),
                    row=2, col=1
                )

                fig.update_layout(height=700, showlegend=True, title_text=f"{selected_stock} Trading Analysis")
                fig.update_xaxes(title_text="Date", row=2, col=1)
                fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
                fig.update_yaxes(title_text="Portfolio (₹)", row=2, col=1)

                st.plotly_chart(fig, use_container_width=True)

                # Trade history
                st.subheader("📋 Trade History")
                if results['trades']:
                    trades_df = pd.DataFrame(results['trades'])
                    st.dataframe(trades_df, use_container_width=True)
                else:
                    st.info("No trades executed during this period")
            else:
                st.error("❌ Failed to fetch data. Please check your inputs.")
    else:
        # Welcome screen
        st.info("👈 Configure settings in the sidebar and click 'Run Backtest' to start")

        st.markdown("### 🎯 Features")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **Data Sources:**
            - ✅ Yahoo Finance (Free, No signup)
            - ✅ Angel One SmartAPI (Requires account)

            **Strategies:**
            - ✅ Moving Average Crossover
            - ✅ RSI (Relative Strength Index)
            """)

        with col2:
            st.markdown("""
            **DSA Concepts:**
            - 📊 Arrays & Lists (price data)
            - 🧮 Sliding Window (moving averages)
            - 🔁 Loops & Iteration (backtesting)
            - 🗂 Dictionaries (symbol data)
            - ⏱ Time Complexity O(1) for MA
            """)

        st.markdown("---")
        st.markdown("### 📚 Quick Start")
        st.markdown("""
        1. Select **Yahoo Finance** as data source (free, no credentials needed)
        2. Choose a stock from the dropdown (e.g., RELIANCE, TCS)
        3. Select a trading strategy
        4. Adjust strategy parameters using sliders
        5. Click **Run Backtest** to see results
        """)


if __name__ == "__main__":
    main()
