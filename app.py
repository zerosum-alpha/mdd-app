import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

st.set_page_config(page_title="MDD 분석기", layout="centered")

# --- [비밀번호] ---
MY_PASSWORD = "1234"
st.title("🔒 나만의 주식 분석기")
entered_password = st.text_input("비밀번호를 입력하세요", type="password")
if entered_password != MY_PASSWORD:
    if entered_password: st.error("❌ 비밀번호 틀림")
    st.stop()

# 💡 접미사 자동 판별 함수
def get_full_ticker(code):
    if code.isalpha():  # 미국 주식(알파벳)이면 그대로 반환
        return code.upper()
    
    # 한국 주식 코드(6자리)일 경우, 야후 파이낸스에서 확인 후 자동 변환 시도
    ticker_ks = f"{code}.KS"
    ticker_kq = f"{code}.KQ"
    
    # 먼저 코스피로 확인
    if not yf.Ticker(ticker_ks).history(period="1d").empty:
        return ticker_ks
    # 코스피에 없으면 코스닥으로 확인
    elif not yf.Ticker(ticker_kq).history(period="1d").empty:
        return ticker_kq
    return None

st.title("📈 자동 코드 변환 MDD 분석기")
st.info("💡 이제 접미사 없이 6자리 종목번호(예: 005930)만 입력하세요!")

user_input = st.text_input("종목 번호 또는 미국 티커 입력:")
start_date = st.date_input("기준 시작일", pd.to_datetime("2024-01-01"))
buy_target_pct = st.number_input("목표 하락률(%)", value=15.0)

if st.button("분석 실행"):
    if not user_input:
        st.warning("코드를 입력하세요.")
    else:
        # 자동 변환 적용
        ticker = get_full_ticker(user_input)
        
        if not ticker:
            st.error("종목을 찾을 수 없습니다. 번호를 확인하세요.")
        else:
            with st.spinner(f"'{ticker}' 데이터를 분석 중..."):
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
                        ax1.set_title(f"[{ticker}] Price")
                        ax2.plot(df.index, df['Drawdown'] * 100, color='red')
                        ax2.axhline(y=-buy_target_pct, color='green', linestyle='--')
                        ax2.set_title('Drawdown (%)')
                        st.pyplot(fig)
                except Exception as e:
                    st.error(f"오류: {e}")
