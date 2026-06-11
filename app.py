import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import FinanceDataReader as fdr

st.set_page_config(page_title="MDD 분석기", layout="centered")

# --- [비밀번호] ---
MY_PASSWORD = "0914"
st.title("🔒 나만의 주식 분석기")
entered_password = st.text_input("비밀번호를 입력하세요", type="password")
if entered_password != MY_PASSWORD:
    if entered_password: st.error("❌ 비밀번호 틀림")
    st.stop()

# 💡 핵심: 한국 전체 종목 목록을 내장함 (프로그램 실행 시 한 번만 로드)
@st.cache_data
def get_stock_list():
    return fdr.StockListing('KRX')

stock_list = get_stock_list()

def find_ticker(query):
    # 1. 입력값이 6자리 숫자(코드)일 경우
    if query.isdigit() and len(query) == 6:
        # 코스피/코스닥 판별 후 티커 반환
        ticker_ks = f"{query}.KS"
        ticker_kq = f"{query}.KQ"
        if not yf.Ticker(ticker_ks).history(period="1d").empty: return ticker_ks
        if not yf.Ticker(ticker_kq).history(period="1d").empty: return ticker_kq
        return None
    
    # 2. 입력값이 한글 종목명일 경우
    match = stock_list[stock_list['Name'] == query]
    if not match.empty:
        code = match.iloc[0]['Code']
        market = match.iloc[0]['Market']
        return f"{code}.KQ" if market == 'KOSDAQ' else f"{code}.KS"
    
    # 3. 미국 주식(알파벳)일 경우
    return query.upper()

st.title("📈 한글 종목명 검색 분석기")
user_input = st.text_input("한글 종목명 또는 번호 입력 (예: 삼성전자, 005930):")
start_date = st.date_input("기준 시작일", pd.to_datetime("2024-01-01"))
buy_target_pct = st.number_input("목표 하락률(%)", value=15.0)

if st.button("분석 실행"):
    if not user_input:
        st.warning("종목을 입력하세요.")
    else:
        ticker = find_ticker(user_input)
        if not ticker:
            st.error("종목을 찾을 수 없습니다. (데이터가 없는 종목일 수 있습니다.)")
        else:
            with st.spinner(f"'{ticker}' 데이터 분석 중..."):
                try:
                    df = yf.Ticker(ticker).history(start=start_date.strftime('%Y-%m-%d'))
                    if df.empty:
                        st.error("데이터 없음.")
                    else:
                        df.index = df.index.tz_localize(None)
                        df['Peak'] = df['Close'].cummax()
                        df['Drawdown'] = (df['Close'] - df['Peak']) / df['Peak']
                        df['MDD'] = df['Drawdown'].cummin()
                        
                        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
                        ax1.plot(df.index, df['Close'], label='Price')
                        ax1.set_title(f"[{ticker}] Trend")
                        ax2.plot(df.index, df['Drawdown'] * 100, color='red')
                        ax2.axhline(y=-buy_target_pct, color='green', linestyle='--')
                        ax2.set_title('Drawdown (%)')
                        st.pyplot(fig)
                except Exception as e:
                    st.error(f"오류: {e}")
