import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import FinanceDataReader as fdr
from datetime import datetime

# =========================================================
# 실전 매매 타이밍 & 밸류에이션(PER) 통합 대시보드
# =========================================================

st.set_page_config(page_title="실전 매매 타이밍 & PER 분석기", layout="wide")

st.title("📈 실전 트레이딩 & 밸류에이션 대시보드 (Forward PER 적용)")

# =========================================================
# 데이터 로드 및 계산 로직
# =========================================================

@st.cache_data(ttl=3600)
def get_ticker_info(ticker):
    """yfinance로부터 실시간 PER 정보 획득"""
    try:
        t = yf.Ticker(ticker)
        info = t.info
        # Forward PER 우선 사용, 없으면 Trailing PER 사용
        f_pe = info.get('forwardPE')
        t_pe = info.get('trailingPE')
        return f_pe if f_pe and f_pe > 0 else t_pe
    except:
        return None

@st.cache_data(ttl=3600)
def load_and_prepare_data(ticker, start_date):
    # 가격 데이터 로드
    df = yf.Ticker(ticker).history(start=start_date, auto_adjust=True)
    df.index = df.index.tz_localize(None)
    
    # PER 정보 로드 (Forward PER 우선)
    pe_val = get_ticker_info(ticker)
    
    # 지표 계산
    df["Peak"] = df["Close"].cummax()
    df["Current_Drawdown"] = df["Close"] / df["Peak"] - 1
    df["MA20"] = df["Close"].rolling(20).mean()
    
    # PER을 일별 데이터에 할당 (현재 시점의 PER을 과거에도 적용하여 밴드 확인)
    df["PER"] = pe_val if pe_val else np.nan
    
    # RSI 계산
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + (gain / loss)))
    
    # 매매 시그널
    buy_cond = (df["Current_Drawdown"] <= -0.15) & (df["RSI"] <= 30)
    sell_cond = (df["Current_Drawdown"] >= -0.02) & (df["RSI"] >= 70)
    
    df['Buy_Signal'] = df['Close'][buy_cond & (~buy_cond.shift(1).fillna(False))]
    df['Sell_Signal'] = df['Close'][sell_cond & (~sell_cond.shift(1).fillna(False))]
    
    return df

# =========================================================
# 차트 생성 (Y축 고정 및 PER 포함)
# =========================================================
def make_final_chart(df, ticker):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
    
    # 상단: 주가 + 매매 시그널
    ax1.plot(df.index, df["Close"], label="Price", color="#1F77B4", linewidth=2)
    ax1.scatter(df.index, df['Buy_Signal'], color='green', marker='^', s=150, label='BUY')
    ax1.scatter(df.index, df['Sell_Signal'], color='red', marker='v', s=150, label='SELL')
    ax1.set_title(f"{ticker} Price & Signals", fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # 하단: PER (왜곡 방지를 위해 0~50으로 Y축 고정)
    ax2.plot(df.index, df['PER'], color="#D69E2E", linewidth=2, label="Forward PER")
    ax2.set_ylim(0, 50)  # 핵심: 50배 이상의 이상치로 인한 왜곡 방지
    ax2.axhline(df['PER'].mean(), color="gray", linestyle="--", label="Avg PER")
    ax2.set_title("Forward PER (Range Locked 0-50x)", fontsize=12)
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    return fig

# =========================================================
# 메인 화면
# =========================================================
ticker_input = st.text_input("티커 입력 (예: NVDA, AAPL)", value="NVDA")
if st.button("분석 실행"):
    df = load_and_prepare_data(ticker_input, "2024-01-01")
    
    # 현재 상태 카드
    latest = df.iloc[-1]
    col1, col2, col3 = st.columns(3)
    col1.metric("현재가", f"{latest['Close']:,.2f}")
    col2.metric("현재 Forward PER", f"{latest['PER']:.2f}x" if pd.notna(latest['PER']) else "N/A")
    col3.metric("최대 낙폭(MDD)", f"{latest['Current_Drawdown']*100:.2f}%")
    
    # 차트 출력
    st.pyplot(make_final_chart(df, ticker_input))
