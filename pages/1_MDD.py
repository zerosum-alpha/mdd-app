import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import FinanceDataReader as fdr
from datetime import datetime

try:
    from pykrx import stock as krx_stock
except Exception:
    krx_stock = None

from auth import require_login, logout_button

# =========================================================
# MDD 저점매수 & 과열매도 실전 트레이딩 대시보드
# =========================================================

st.set_page_config(page_title="실전 매매 타이밍 분석기", layout="wide")

require_login()
logout_button()

st.title("📈 실전 매매 타이밍 대시보드 (Buy & Sell Overlay)")

# =========================================================
# 한국 종목 검색 & 데이터 로드
# =========================================================
@st.cache_data(ttl=86400)
def get_stock_list():
    try:
        df = fdr.StockListing("KRX")
        if df is None or df.empty: return pd.DataFrame()
        df["Code"] = df["Code"].astype(str).str.zfill(6)
        df["Name"] = df["Name"].astype(str).str.strip()
        return df
    except:
        return pd.DataFrame()

KR_FALLBACK_MAP = {
    "삼성전자": "005930", "SK하이닉스": "000660", "현대차": "005380", 
    "NAVER": "035420", "카카오": "035720", "LG에너지솔루션": "373220",
    "삼성SDI": "006400", "셀트리온": "068270", "POSCO홀딩스": "005490"
}

stock_list = get_stock_list()

def find_ticker(query):
    query = str(query).strip()
    if not query: return None, None, None
    if query.isdigit() and len(query) == 6: return "KR", query, query
    if query in KR_FALLBACK_MAP: return "KR", KR_FALLBACK_MAP[query], query
    
    if not stock_list.empty:
        exact = stock_list[stock_list["Name"] == query]
        if not exact.empty: return "KR", exact.iloc[0]["Code"], exact.iloc[0]["Name"]
        partial = stock_list[stock_list["Name"].str.contains(query, case=False, na=False)]
        if not partial.empty: return "KR", partial.iloc[0]["Code"], partial.iloc[0]["Name"]
        
    if any("가" <= ch <= "힣" for ch in query): return None, None, None
    return "US", query.upper(), query.upper()

@st.cache_data(ttl=3600)
def load_price_data(market, ticker, start_date):
    start = start_date.strftime("%Y-%m-%d")
    try:
        if market == "KR":
            df = fdr.DataReader(ticker, start)
            if not df.empty: df.index = pd.to_datetime(df.index)
            return df
        if market == "US":
            df = yf.Ticker(ticker).history(start=start, auto_adjust=True)
            if not df.empty: df.index = df.index.tz_localize(None)
            return df
    except: pass
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_vix_data(start_date):
    """시장 공포 지수(VIX) 로드 - 미국장 및 글로벌 투심 기준"""
    start = start_date.strftime("%Y-%m-%d")
    try:
        df = yf.Ticker("^VIX").history(start=start, auto_adjust=True)
        if not df.empty:
            df.index = df.index.tz_localize(None)
            return df[['Close']].rename(columns={'Close': 'VIX'})
    except: pass
    return pd.DataFrame()

# =========================================================
# 지표 및 시그널 계산 로직
# =========================================================
def calculate_rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_indicators_and_signals(df, vix_df=None):
    df = df.copy()
    if "Volume" not in df.columns: df["Volume"] = 0
    
    # 기본 지표
    df["Peak"] = df["Close"].cummax()
    df["Current_Drawdown"] = df["Close"] / df["Peak"] - 1
    df["Max_Drawdown"] = df["Current_Drawdown"].cummin()
    
    df["MA20"] = df["Close"].rolling(20).mean()
    df["RSI"] = calculate_rsi(df["Close"], 14)
    
    # VIX 결합
    if vix_df is not None and not vix_df.empty:
        df = df.join(vix_df, how='left')
        df['VIX'] = df['VIX'].ffill() # 공휴일 결측치 채우기
    else:
        df['VIX'] = np.nan

    # ---------------------------------------------------------
    # 실전 매매 타이밍 (Signal) 포착 로직
    # ---------------------------------------------------------
    # [매수 조건]: MDD가 -15% 이하로 깊고, RSI가 30 이하(과매도)이거나 VIX가 25 이상(공포)일 때
    buy_cond = (df["Current_Drawdown"] <= -0.15) & ((df["RSI"] <= 30) | (df["VIX"] >= 25))
    
    # [매도 조건]: MDD가 -2% 이상으로 전고점 회복 부근이고, RSI가 70 이상(과열)일 때
    sell_cond = (df["Current_Drawdown"] >= -0.02) & (df["RSI"] >= 70)

    # 연속된 시그널 중 첫 번째만 필터링하여 마커 표시
    df['Buy_Signal'] = df['Close'][buy_cond & (~buy_cond.shift(1).fillna(False))]
    df['Sell_Signal'] = df['Close'][sell_cond & (~sell_cond.shift(1).fillna(False))]

    return df

# =========================================================
# 실전 매매 오버레이 차트 생성
# =========================================================
def make_trading_overlay_chart(df, ticker):
    if df is None or df.empty: return None
    
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2.5, 1]}, sharex=True)
    
    # --- [상단 패널]: 주가 + 매수/매도 시그널 + VIX ---
    ax1.plot(df.index, df["Close"], label="Price", color="#1F77B4", linewidth=2)
    ax1.plot(df.index, df["MA20"], label="MA20", color="orange", alpha=0.6, linestyle="--")
    
    # 매수/매도 마커 오버레이
    ax1.scatter(df.index, df['Buy_Signal'] * 0.95, color='green', marker='^', s=150, zorder=5, label='BUY (Undervalued / Fear)')
    ax1.scatter(df.index, df['Sell_Signal'] * 1.05, color='red', marker='v', s=150, zorder=5, label='SELL (Overheated / Greed)')
    
    ax1.set_title(f"[{ticker}] Trading Execution Dashboard (Signal Overlay)", fontsize=16, fontweight='bold')
    ax1.set_ylabel("Stock Price", color="#1F77B4", fontweight='bold')
    ax1.tick_params(axis='y', labelcolor="#1F77B4")
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # 이중 축 (VIX가 있는 경우)
    if 'VIX' in df.columns and df['VIX'].notna().any():
        ax2 = ax1.twinx()
        ax2.plot(df.index, df['VIX'], color='purple', alpha=0.3, linewidth=1.5, label='VIX (Market Fear)')
        ax2.set_ylabel("VIX Index", color="purple")
        ax2.tick_params(axis='y', labelcolor="purple")
        
        # 범례 합치기
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    else:
        ax1.legend(loc='upper left')

    # --- [하단 패널]: MDD (최대 낙폭) 영역 차트 ---
    ax3.fill_between(df.index, df["Current_Drawdown"] * 100, 0, color="red", alpha=0.2, label="Current Drawdown")
    ax3.plot(df.index, df["Max_Drawdown"] * 100, color="darkred", linestyle="--", alpha=0.7, label="Max Drawdown")
    
    # 주요 지지선
    ax3.axhline(y=-10, color="gray", linestyle=":", alpha=0.6, label="-10% Line")
    ax3.axhline(y=-20, color="orange", linestyle=":", alpha=0.6, label="-20% Line")
    
    ax3.set_ylabel("Drawdown (%)", fontweight='bold')
    ax3.set_xlabel("Date", fontweight='bold')
    ax3.legend(loc="lower left")
    ax3.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    return fig

# =========================================================
# 메인 UI 흐름
# =========================================================
col_a, col_b, col_c = st.columns(3)

with col_a:
    user_input = st.text_input("종목명 / 종목코드 / 미국 티커", value="NVDA")
with col_b:
    start_date = st.date_input("기준 시작일", pd.to_datetime("2023-01-01"))
with col_c:
    run = st.button("실전 트레이딩 분석 실행", type="primary", use_container_width=True)

if run:
    market, ticker, display_name = find_ticker(user_input)

    if ticker is None:
        st.error("종목을 찾을 수 없습니다.")
        st.stop()

    with st.spinner(f"{display_name} 데이터 수집 및 시그널 계산 중..."):
        # 1. 데이터 로드
        price_df = load_price_data(market, ticker, start_date)
        vix_df = load_vix_data(start_date) if market == "US" else None

        if price_df.empty:
            st.error("가격을 가져오지 못했습니다. 티커를 확인하세요.")
            st.stop()

        # 2. 지표 및 시그널 계산
        df = calculate_indicators_and_signals(price_df, vix_df)
        
        latest = df.iloc[-1]
        current_price = latest["Close"]
        current_dd = latest["Current_Drawdown"]
        rsi = latest["RSI"]
        
        # 3. 상태판 출력
        st.markdown("---")
        st.markdown(f"## 📊 현재 시장 상태: **{display_name} ({ticker})**")
        
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("현재가", f"{current_price:,.2f}")
        s2.metric("최고점 대비 낙폭 (MDD)", f"{current_dd * 100:.2f}%")
        s3.metric("현재 RSI (과열/과매도)", f"{rsi:.1f}")
        if 'VIX' in latest and pd.notna(latest['VIX']):
            s4.metric("현재 VIX (공포지수)", f"{latest['VIX']:.1f}")
            
        # 매매 직관적 판단 코멘트
        if current_dd <= -0.15 and rsi <= 35:
            st.success("💡 **현재 상태**: 낙폭이 깊고 투심이 얼어붙은 **적극 매수 검토 구간**입니다.")
        elif current_dd >= -0.03 and rsi >= 70:
            st.error("🚨 **현재 상태**: 주가가 전고점에 근접하며 과열 양상입니다. **분할 매도 및 현금 확보 검토 구간**입니다.")
        else:
            st.info("⚖️ **현재 상태**: 명확한 매수/매도 극단값이 아닙니다. **관망 또는 보유 유지 구간**입니다.")

        # 4. 메인 차트 출력 (Overlay)
        st.markdown("---")
        st.markdown("### 📈 실전 매매 타이밍 오버레이 차트")
        st.caption("초록색 세모(▲)는 강력 매수 조건 충족일, 빨간색 세모(▼)는 과열 매도 조건 충족일을 의미합니다.")
        
        chart_fig = make_trading_overlay_chart(df, ticker)
        if chart_fig:
            st.pyplot(chart_fig)
            
        # 5. 데이터 테이블 확인
        with st.expander("최근 30일 시그널 발생 및 상세 데이터 보기"):
            view_df = df[["Close", "Current_Drawdown", "RSI", "Buy_Signal", "Sell_Signal"]].tail(30).copy()
            view_df["Current_Drawdown"] = (view_df["Current_Drawdown"] * 100).round(2)
            view_df["RSI"] = view_df["RSI"].round(2)
            view_df["Buy_Signal"] = view_df["Buy_Signal"].apply(lambda x: "BUY" if pd.notna(x) else "")
            view_df["Sell_Signal"] = view_df["Sell_Signal"].apply(lambda x: "SELL" if pd.notna(x) else "")
            st.dataframe(view_df.sort_index(ascending=False), use_container_width=True)
