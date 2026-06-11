import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import urllib.parse
import json

st.set_page_config(page_title="MDD & 줍줍 분석기", layout="centered")

# --- [비밀번호 설정] ---
MY_PASSWORD = "1234"

st.title("🔒 나만의 주식 분석기")
entered_password = st.text_input("비밀번호를 입력하세요", type="password")

if entered_password != MY_PASSWORD:
    if entered_password: 
        st.error("❌ 비밀번호가 틀렸습니다.")
    st.stop()

# --- [메인 프로그램] ---
st.success("✅ 인증 완료! 분석기를 시작합니다.")

# 💡 에러 해결: 서버 차단 위험이 있는 KRX 전체 로딩 대신, 네이버 금융 직통 검색 엔진 탑재 (속도 대폭 향상)
def search_ticker_naver(user_input):
    user_input = user_input.strip()
    
    # 1. 미국 주식 티커인 경우 (순수 영문)
    if user_input.isalpha():
        return user_input.upper()
        
    # 2. 이미 .KS나 .KQ가 붙어있는 경우
    if user_input.upper().endswith('.KS') or user_input.upper().endswith('.KQ'):
        return user_input.upper()
        
    # 3. 6자리 종목 코드 직접 입력 시 (숫자/알파벳 혼용 포함)
    if len(user_input) == 6 and user_input.isalnum():
        # 임의로 코스피(.KS)로 시도 후 에러 시 대안을 찾도록 처리하기 위해 우선 반환
        return f"{user_input.upper()}.KS"
        
    # 4. 한글 종목명 입력 시 -> 네이버 금융 자동완성 API 활용 (서버 차단 없음)
    try:
        encoded_keyword = urllib.parse.quote(user_input)
        url = f"https://ac.finance.naver.com/ac?q={encoded_keyword}&q_enc=utf-8&st=1&r_format=json"
        
        import requests
        response = requests.get(url, timeout=5)
        data = json.loads(response.text)
        
        # 네이버 검색 결과에서 가장 첫 번째 종목 정보 추출
        if data and 'items' in data and data['items'] and data['items'][0]:
            stock_info = data['items'][0][0] # 종목코드 리스트
            market_info = data['items'][0][1] # 시장 정보 리스트
            
            code = stock_info[0]
            market_name = market_info[0] if market_info else ""
            
            if "코스닥" in market_name:
                return f"{code}.KQ"
            else:
                return f"{code}.KS"
    except Exception:
        pass
        
    return None

st.title("📈 실시간 MDD & 매수 타이밍 분석기")

col1, col2, col3 = st.columns(3)
with col1:
    user_input = st.text_input("종목명 또는 티커", value="KoAct 미국나스닥성장기업액티브")
with col2:
    start_date = st.date_input("기준 시작일", pd.to_datetime("2024-01-01"))
with col3:
    buy_target_pct = st.number_input("줍줍 목표 하락률(%)", min_value=1.0, value=15.0)

if st.button("분석 차트 그리기", use_container_width=True):
    ticker = search_ticker_naver(user_input)
    if not ticker:
        st.error(f"'{user_input}' 종목을 찾을 수 없습니다. 오타를 확인해주세요.")
    else:
        with st.spinner('데이터 분석 중...'):
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date.strftime('%Y-%m-%d'))
            
            # 코스피(.KS)로 검색 안 될 경우 코스닥(.KQ)으로 한 번 더 교차 검증 자동 전환
            if df.empty and ticker.endswith('.KS') and len(ticker) == 9:
                ticker = ticker.replace('.KS', '.KQ')
                stock = yf.Ticker(ticker)
                df = stock.history(start=start_date.strftime('%Y-%m-%d'))
                
            if df.empty:
                st.error(f"❌ '{user_input}'의 데이터를 야후 파이낸스에서 불러오지 못했습니다. 코드를 다시 확인해주세요.")
            else:
                df.index = df.index.tz_localize(None)
                actual_start_date = df.index[0]
                
                df['Peak'] = df['Close'].cummax()
                df['Drawdown'] = (df['Close'] - df['Peak']) / df['Peak']
                df['MDD'] = df['Drawdown'].cummin()
                
                current_mdd = float(df['Drawdown'].iloc[-1]) * 100
                max_mdd = float(df['MDD'].min()) * 100
                buy_threshold_decimal = -(buy_target_pct / 100)
                buy_signals = df[df['Drawdown'] <= buy_threshold_decimal]
                
                st.success(f"분석 완료! (데이터 시작일: {actual_start_date.strftime('%Y-%m-%d')})")
                metric1, metric2, metric3 = st.columns(3)
                metric1.metric("현재 낙폭", f"{current_mdd:.2f}%")
                metric2.metric("최대 낙폭(MDD)", f"{max_mdd:.2f}%")
                metric3.metric(f"-{buy_target_pct}% 진입 기회", f"{len(buy_signals)}일")

                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
                ax1.plot(df.index, df['Close'], label='Price', color='black', alpha=0.7)
                ax1.plot(df.index, df['Peak'], label='Peak', color='blue', linestyle='--', alpha=0.5)
                if not buy_signals.empty:
                    ax1.scatter(buy_signals.index, buy_signals['Close'], color='green', marker='^', s=80, zorder=5)
                ax1.set_title(f"[{ticker}] Price Trend")
                ax1.grid(True, alpha=0.3)
                
                ax2.plot(df.index, df['Drawdown'] * 100, color='red')
                ax2.fill_between(df.index, df['Drawdown'] * 100, 0, color='red', alpha=0.2)
                ax2.axhline(y=-buy_target_pct, color='green', linestyle='--')
                if not buy_signals.empty:
                    ax2.scatter(buy_signals.index, buy_signals['Drawdown'] * 100, color='green', marker='^', s=80, zorder=5)
                ax2.set_title('Drawdown (%) & Buy Zones')
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
