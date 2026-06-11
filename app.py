import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import urllib.parse
import json
import requests
import FinanceDataReader as fdr

st.set_page_config(page_title="MDD & 줍줍 분석기", layout="centered")

# --- [비밀번호 설정] ---
MY_PASSWORD = "0914"

st.title("🔒 나만의 주식 분석기")
entered_password = st.text_input("비밀번호를 입력하세요", type="password")

if entered_password != MY_PASSWORD:
    if entered_password: 
        st.error("❌ 비밀번호가 틀렸습니다.")
    st.stop()

# --- [메인 프로그램] ---
st.success("✅ 인증 완료! 분석기를 시작합니다.")

def search_ticker_naver(user_input):
    user_input = user_input.strip().upper()
    
    # 💡 치명적 오류 수정: 순수 영어 알파벳(A-Z)일 때만 미국 주식으로 인식!
    if user_input.isascii() and user_input.isalpha():
        return user_input
        
    if user_input.endswith('.KS') or user_input.endswith('.KQ'):
        return user_input[:-3]
        
    if len(user_input) == 6 and user_input.isalnum() and user_input.isascii():
        return user_input
        
    # 한글 이름일 경우 네이버를 통해 6자리 번호로 통역
    try:
        encoded_keyword = urllib.parse.quote(user_input)
        url = f"https://ac.finance.naver.com/ac?q={encoded_keyword}&q_enc=utf-8&st=1&r_format=json"
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36"
        }
        
        response = requests.get(url, headers=headers, timeout=5)
        data = json.loads(response.text)
        
        if data and 'items' in data and data['items'] and data['items'][0]:
            code = data['items'][0][0][0]
            return code
    except Exception:
        pass
        
    return None

st.title("📈 실시간 MDD & 매수 타이밍 분석기")

clear_click = st.checkbox("🗑️ 검색창 비우기 (새 종목 입력 시 체크하세요)")
default_value = "" if clear_click else "삼성전자"

col1, col2, col3 = st.columns(3)
with col1:
    user_input = st.text_input("종목명 또는 티커", value=default_value)
with col2:
    start_date = st.date_input("기준 시작일", pd.to_datetime("2024-01-01"))
with col3:
    buy_target_pct = st.number_input("줍줍 목표 하락률(%)", min_value=1.0, value=15.0)

if st.button("분석 차트 그리기", use_container_width=True):
    if not user_input:
        st.warning("⚠️ 검색할 종목명이나 코드를 먼저 입력해 주세요.")
    else:
        ticker = search_ticker_naver(user_input)
        if not ticker:
            st.error(f"❌ '{user_input}' 종목을 찾을 수 없습니다. 오타를 확인해 주세요.")
        else:
            with st.spinner('안전한 서버에서 데이터를 분석 중입니다...'):
                try:
                    df = fdr.DataReader(ticker, start_date.strftime('%Y-%m-%d'))
                    
                    if df.empty:
                        st.error("❌ 데이터를 찾을 수 없습니다. (상장일 이전 날짜인지 확인해 주세요)")
                    else:
                        if df.index.tzinfo is not None:
                            df.index = df.index.tz_localize(None)
                            
                        actual_start_date = df.index[0]
                        
                        df['Peak'] = df['Close'].cummax()
                        df['Drawdown'] = (df['Close'] - df['Peak']) / df['Peak']
                        df['MDD'] = df['Drawdown'].cummin()
                        
                        current_mdd = float(df['Drawdown'].iloc[-1]) * 100
                        max_mdd = float(df['MDD'].min()) * 100
                        buy_threshold_decimal = -(buy_target_pct / 100)
                        buy_signals = df[df['Drawdown'] <= buy_threshold_decimal]
                        
                        st.success(f"분석 완료! (실제 데이터 시작일: {actual_start_date.strftime('%Y-%m-%d')})")
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
                except Exception as e:
                    st.error(f"❌ 데이터 수집 중 오류가 발생했습니다: {e}")
