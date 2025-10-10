import streamlit as st
import pandas as pd
import json
import os
import plotly.express as px
from streamlit_autorefresh import st_autorefresh
import subprocess
import sys

# ======================================================
# CONFIGURATION & CONSTANTS
# ======================================================
def load_config():
    """Loads the central configuration file."""
    if not os.path.exists("config.json"):
        st.error("❌ `config.json` not found! Please ensure it's in the same directory.")
        return None
    with open("config.json", "r") as f:
        return json.load(f)

config = load_config()
if not config:
    st.stop()

# Get file paths from config
PATHS = config.get("paths", {})
TRADES_FILE = PATHS.get("trades_file", "data/trades.csv")
PREDICTIONS_FILE = PATHS.get("predictions_file", "data/predictions.xlsx")
LOG_FILE = PATHS.get("log_file", "logs/bot.log")

# ======================================================
# PAGE SETUP
# ======================================================
st.set_page_config(
    page_title="AI Trading Bot Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("🤖 AI Trading Bot Dashboard")
st_autorefresh(interval=10 * 1000, key="auto_refresh_dashboard")

# ======================================================
# DATA LOADING (with caching for performance)
# ======================================================
@st.cache_data(ttl=10)  # Cache data for 10 seconds
def load_dataframe(file_path):
    """Loads a CSV or Excel file into a pandas DataFrame."""
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return pd.DataFrame()
    
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            return pd.DataFrame()
        
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
        return df
    except Exception as e:
        st.warning(f"Could not load {os.path.basename(file_path)}: {e}")
        return pd.DataFrame()

# ======================================================
# SIDEBAR - CONTROLS
# ======================================================
with st.sidebar:
    st.header("⚙️ Controls")
    st.info("These buttons start the bot processes. For best results, run these in separate terminals as described in the User Manual.")

    if 'trainer_running' not in st.session_state:
        st.session_state.trainer_running = False
    if 'trader_running' not in st.session_state:
        st.session_state.trader_running = False

    if st.button("Train Model (`run_pipline.py`)", disabled=st.session_state.trainer_running):
        try:
            subprocess.Popen([sys.executable, "run_pipline.py"])
            st.success("Started the training pipeline!")
            st.session_state.trainer_running = True
        except Exception as e:
            st.error(f"Failed to start trainer: {e}")

    if st.button("Run Live Bot (`live_trader.py`)", disabled=st.session_state.trader_running):
        try:
            subprocess.Popen([sys.executable, "live_trader.py"])
            st.success("Live trader has been launched!")
            st.session_state.trader_running = True
        except Exception as e:
            st.error(f"Failed to start trader: {e}")
    
    st.header("📖 Documentation")
    st.write("Refer to the `User_Manual.md` for full setup and operational instructions.")

# ======================================================
# MAIN DASHBOARD AREA
# ======================================================
trades_df = load_dataframe(TRADES_FILE)
preds_df = load_dataframe(PREDICTIONS_FILE)

# --- Performance Summary ---
st.subheader("📈 Trading Performance Summary")

if not trades_df.empty and "realized_pnl" in trades_df.columns:
    total_trades = len(trades_df)
    wins = trades_df[trades_df["realized_pnl"] > 0]
    losses = trades_df[trades_df["realized_pnl"] < 0]
    total_pnl = trades_df["realized_pnl"].sum()
    win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0
    avg_pnl = trades_df["realized_pnl"].mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Trades", f"{total_trades}")
    col2.metric("Win Rate", f"{win_rate:.2f}%")
    col3.metric("Total PnL ($)", f"{total_pnl:,.2f}")
    col4.metric("Avg PnL/Trade ($)", f"{avg_pnl:.2f}")
else:
    st.info("No trade data with 'realized_pnl' found yet. Execute some trades to see performance metrics.")

st.divider()

# --- Charts ---
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("💹 Cumulative PnL Over Time")
    if not trades_df.empty and "realized_pnl" in trades_df.columns and "time" in trades_df.columns:
        trades_df_sorted = trades_df.sort_values("time")
        trades_df_sorted["cumulative_pnl"] = trades_df_sorted["realized_pnl"].cumsum()
        fig_cum_pnl = px.line(trades_df_sorted, x="time", y="cumulative_pnl", title="Cumulative Profit & Loss", labels={"time": "Date", "cumulative_pnl": "Cumulative PnL ($)"})
        st.plotly_chart(fig_cum_pnl, use_container_width=True)
    else:
        st.info("Waiting for PnL data to plot cumulative performance.")

with col_b:
    st.subheader("📊 Win / Loss Distribution")
    if not trades_df.empty and "realized_pnl" in trades_df.columns:
        wins = trades_df[trades_df["realized_pnl"] > 0]
        losses = trades_df[trades_df["realized_pnl"] < 0]
        win_loss_counts = pd.DataFrame({
            "Result": ["Wins", "Losses"],
            "Count": [len(wins), len(losses)]
        })
        fig_pie = px.pie(win_loss_counts, values="Count", names="Result", title="Win vs. Loss Breakdown", color_discrete_map={"Wins": "green", "Losses": "red"})
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("Waiting for trade data to build the pie chart.")

st.divider()

# --- Data Tables and Logs ---
col_c, col_d = st.columns([3, 2]) # Give more space to tables

with col_c:
    st.subheader("📑 Recent Trades")
    if not trades_df.empty and 'time' in trades_df.columns:
        st.dataframe(trades_df.sort_values("time", ascending=False).head(10), use_container_width=True)
    else:
        st.info("No trades have been recorded in 'trades.csv' yet.")

    st.subheader("🤖 Recent AI Predictions")
    if not preds_df.empty and 'time' in preds_df.columns:
        st.dataframe(preds_df.sort_values("time", ascending=False).head(10), use_container_width=True)
    else:
        st.info("No predictions have been recorded in 'predictions.xlsx' yet.")


with col_d:
    st.subheader("📜 Live Log Stream")
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                log_lines = f.readlines()[-30:] # Get last 30 lines
            st.code("".join(log_lines), language="log")
        except Exception as e:
            st.error(f"Error reading log file: {e}")
    else:
        st.info("Log file not created yet. It will appear once the bot starts running.")


