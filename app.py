import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import FinanceDataReader as fdr

st.set_page_config(page_title="MDD 분석기", layout="centered")

# --- [비밀번호] ---
MY_PASSWORD = "1234"
st.title("🔒 나만의 주식 분석기")
entered_password = st.text_input("비밀번호를 입력하세요", type="password")
if entered_password != MY_PASSWORD:
    if entered_password: st.error("❌ 비밀번호 틀림")
    st.stop()

@st.cache_data
def get_stock_list():
    return fdr.StockListing('KRX')

stock_list = get_stock_list()

def find_ticker(query):
    if query.isdigit() and len(query) == 6:
        ticker_ks = f"{query}.KS"
        ticker_kq = f"{query}.KQ"
        if not yf.Ticker(ticker_ks).history(period="1d").empty: return ticker_ks
        if not yf.Ticker(ticker_kq).history(period="1d").empty: return ticker_kq
        return None
    match = stock_list[stock_list['Name'] == query]
    if not match.empty:
        code = match.iloc[0]['Code']
        market = match.iloc[0]['Market']
        return f"{code}.KQ" if market == 'KOSDAQ' else f"{code}.KS"
    return query.upper()

st.title("📈 MDD & 매수 타이밍 분석기")
user_input = st.text_input("한글 종목명 또는 번호 입력:")
start_date = st.date_input("기준 시작일", pd.to_datetime("2024-01-01"))
buy_target_pct = st.number_input("목표 하락률(%)", value=15.0)

if st.button("분석 실행"):
    if not user_input:
        st.warning("종목을 입력하세요.")
    else:
        ticker = find_ticker(user_input)
        if not ticker:
            st.error("종목을 찾을 수 없습니다.")
        else:
            with st.spinner(f"'{ticker}' 데이터 분석 중..."):
                df = yf.Ticker(ticker).history(start=start_date.strftime('%Y-%m-%d'))
                if df.empty:
                    st.error("데이터 없음.")
                else:
                    df.index = df.index.tz_localize(None)
                    # 💡 고점 및 낙폭 계산
                    df['Peak'] = df['Close'].cummax()
                    df['Drawdown'] = (df['Close'] - df['Peak']) / df['Peak']
                    
                    buy_threshold = -(buy_target_pct / 100)
                    buy_signals = df[df['Drawdown'] <= buy_threshold]
                    
                    # 💡 시각화 복구
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
                    
                    # 1. 주가 및 고점 추적선
                    ax1.plot(df.index, df['Close'], label='Price', color='black')
                    ax1.plot(df.index, df['Peak'], label='Peak Line', color='blue', linestyle='--')
                    ax1.set_title(f"[{ticker}] Price & Peak Line")
                    ax1.legend()
                    
                    # 2. 매수 신호(초록 화살표) 표시
                    if not buy_signals.empty:
                        ax1.scatter(buy_signals.index, buy_signals['Close'], color='green', marker='^', s=50, label='Buy Point')
                    
                    # 3. 낙폭 및 목표 하락선
                    ax2.plot(df.index, df['Drawdown'] * 100, color='red', label='Drawdown')
                    ax2.axhline(y=-buy_target_pct, color='green', linestyle='--', label='Target Line')
                    ax2.set_title('Drawdown (%)')
                    ax2.legend()
                    
                    st.pyplot(fig)

# (기존 데이터 분석 로직 바로 아래에 붙여넣으세요)
        with st.spinner('스마트 매수 지표 계산 중...'):
            # 1. 지표 계산
            df['MA200'] = df['Close'].rolling(window=200).mean()
            df['std'] = df['Close'].rolling(window=20).std()
            df['Upper'] = df['Close'].rolling(window=20).mean() + (df['std'] * 2)
            df['Lower'] = df['Close'].rolling(window=20).mean() - (df['std'] * 2)
            
            # 2. 스마트 매수 신호 (볼린저 밴드 하단 터치 + 장기 추세 근접)
            df['Buy_Signal'] = (df['Close'] <= df['Lower']) & (df['Close'] >= df['MA200'] * 0.95)

            # 3. 차트 그리기
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(df.index, df['Close'], label='Price', color='black', alpha=0.7)
            ax.plot(df.index, df['MA200'], label='MA200', color='blue', linestyle='--')
            ax.fill_between(df.index, df['Upper'], df['Lower'], color='gray', alpha=0.2)
            
            # 4. 신호 표시
            buy_points = df[df['Buy_Signal']]
            ax.scatter(buy_points.index, buy_points['Close'], color='lime', marker='*', s=100, label='Smart Buy')
            
            ax.set_title("Smart Buy Signals (Bollinger Band + MA200)")
            ax.legend()
            st.pyplot(fig)
