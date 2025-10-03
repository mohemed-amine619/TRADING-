# dashboard.py
import streamlit as st
import pandas as pd
import subprocess
import os

st.set_page_config(layout="wide")
st.title("📈 AI Trading Bot Dashboard")

st.header("Bot Control")
col1, col2 = st.columns(2)
if col1.button("▶️ Start Bot"):
    try:
        subprocess.Popen(['python', 'main.py'])
        st.success("Bot process started in the background!")
    except Exception as e:
        st.error(f"Failed to start bot: {e}")

if col2.button("⏹️ Stop Bot"):
    try:
        # This is a simple stop method for Unix-like systems, may need adjustment for Windows
        if os.name == 'nt': # Windows
             os.system("taskkill /F /IM python.exe /T")
        else: # Unix
             os.system("pkill -f 'python main.py'")
        st.warning("Sent stop command to bot process.")
    except Exception as e:
        st.error(f"Failed to stop bot: {e}")

st.markdown("---")

st.header("Backtest Performance (Cumulative PnL)")
try:
    backtest_df = pd.read_excel(config.BACKTEST_FILE)
    st.line_chart(backtest_df['pnl'])
except Exception:
    st.info("Run the main bot once to generate backtest results.")

st.header("Bot Activity Log")
try:
    with open("logs/bot.log", "r", encoding='utf-8') as f:
        log_contents = f.readlines()
    st.text_area("Logs", "".join(log_contents[-50:]), height=300)
except Exception:
    st.info("Log file not found. Run the bot to create it.")