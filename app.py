import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import FinanceDataReader as fdr

st.set_page_config(page_title="완전체 주식 분석기", layout="centered")

# --- [비밀번호] ---
MY_PASSWORD = "0914"
st.title("🔒 나만의 주식 분석기")
entered_password = st.text_input("비밀번호를 입력하세요", type="password")
if entered_password != MY_PASSWORD:
    if entered_password: st.error("❌ 비밀번호 틀림")
    st.stop()

# --- [설정] ---
@st.cache_data
def get_stock_list(): return fdr.StockListing('KRX')
stock_list = get_stock_list()

def find_ticker(query):
    if query.isdigit() and len(query) == 6:
        ticker_ks = f"{query}.KS"
        ticker_kq = f"{query}.KQ"
        if not yf.Ticker(ticker_ks).history(period="1d").empty: return ticker_ks
        if not yf.Ticker(ticker_kq).history(period="1d").empty: return ticker_kq
    match = stock_list[stock_list['Name'] == query]
    if not match.empty:
        code = match.iloc[0]['Code']
        market = match.iloc[0]['Market']
        return f"{code}.KQ" if market == 'KOSDAQ' else f"{code}.KS"
    return query.upper()

st.title("📈 완전체 주식 분석기")

# 검색창 비우기 체크박스
clear_click = st.checkbox("🗑️ 검색창 비우기 (새 종목 입력 시 체크)")
default_val = "" if clear_click else "삼성전자"

user_input = st.text_input("종목명 또는 번호 입력:", value=default_val)
start_date = st.date_input("기준 시작일", pd.to_datetime("2024-01-01"))
buy_target_pct = st.number_input("목표 하락률(%)", value=15.0)

if st.button("분석 실행"):
    ticker = find_ticker(user_input)
    if not ticker: st.error("종목을 찾을 수 없습니다.")
    else:
        with st.spinner('데이터 분석 중...'):
            df = yf.Ticker(ticker).history(start=start_date.strftime('%Y-%m-%d'))
            if df.empty: st.error("데이터 없음.")
            else:
                df.index = df.index.tz_localize(None)
                # 계산
                df['Peak'] = df['Close'].cummax()
                df['Drawdown'] = (df['Close'] - df['Peak']) / df['Peak']
                df['MA200'] = df['Close'].rolling(200).mean()
                df['std'] = df['Close'].rolling(20).std()
                df['Lower'] = df['Close'].rolling(20).mean() - (df['std'] * 2)
                df['Buy_Signal'] = (df['Close'] <= df['Lower']) & (df['Close'] >= df['MA200'] * 0.95)
                
                # 시각화
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
                
                # 1. 가격 + 고점선 + 스마트 매수 별표 + 오늘 가격 빨간 점
                ax1.plot(df.index, df['Close'], label='Price', color='black')
                ax1.plot(df.index, df['Peak'], label='Peak', color='blue', linestyle='--')
                buy_pts = df[df['Buy_Signal']]
                ax1.scatter(buy_pts.index, buy_pts['Close'], color='lime', marker='*', s=100, label='Smart Buy')
                
                # 오늘 가격 표시 (빨간 점)
                ax1.scatter(df.index[-1], df['Close'].iloc[-1], color='red', marker='o', s=100, label='Today Price')
                ax1.set_title(f"[{ticker}] Trend")
                ax1.legend()
                
                # 2. 낙폭 차트
                ax2.plot(df.index, df['Drawdown'] * 100, color='red', label='Drawdown')
                ax2.axhline(y=-buy_target_pct, color='green', linestyle='--')
                ax2.set_title('Drawdown (%)')
                ax2.legend()
                
                st.pyplot(fig)
