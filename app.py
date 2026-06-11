import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import urllib.parse
import json
import requests
import FinanceDataReader as fdr
import yfinance as yf

st.set_page_config(page_title="MDD & 줍줍 분석기", layout="centered")

# --- [비밀번호 설정] ---
MY_PASSWORD = "0914"

st.title("🔒 나만의 주식 분석기")
entered_password = st.text_input("비밀번호를 입력하세요", type="password")

if entered_password != MY_PASSWORD:
    if entered_password: 
        st.error("❌ 비밀번호가 틀렸습니다.")
    st.stop()

st.success("✅ 인증 완료! 분석기를 시작합니다.")

# 💡 무적의 검색 엔진: 네이버가 차단하면 즉시 다음(Daum) 금융으로 자동 우회
def translate_name_to_ticker(user_input):
    user_input = user_input.strip().upper()
    
    if user_input.isascii() and user_input.isalpha(): return user_input
    if user_input.endswith('.KS') or user_input.endswith('.KQ'): return user_input[:-3]
    if len(user_input) == 6 and user_input.isalnum() and user_input.isascii(): return user_input
    
    # 1. 네이버 금융 우회 시도
    headers_n = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://finance.naver.com/"
    }
    try:
        url_n = f"https://ac.finance.naver.com/ac?q={urllib.parse.quote(user_input)}&q_enc=utf-8&st=1&r_format=json"
        res_n = requests.get(url_n, headers=headers_n, timeout=3)
        data_n = res_n.json()
        if data_n.get('items') and data_n['items'][0]:
            return data_n['items'][0][0][0]
    except:
        pass
        
    # 2. 네이버 차단 시 다음(Daum) 금융 API로 2차 시도 (스트림릿 클라우드 차단 확률 매우 낮음)
    headers_d = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://finance.daum.net/"
    }
    try:
        url_d = f"https://finance.daum.net/api/search/search?q={urllib.parse.quote(user_input)}"
        res_d = requests.get(url_d, headers=headers_d, timeout=3)
        data_d = res_d.json()
        if data_d.get('assets'):
            for item in data_d['assets']:
                if item.get('country') == 'KOREA':
                    code = item.get('symbolCode')
                    return code[1:] if code.startswith('A') else code
    except:
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
        ticker = translate_name_to_ticker(user_input)
        if not ticker:
            st.error(f"❌ '{user_input}' 종목을 찾을 수 없습니다. (만약 서버 차단이 계속된다면 번거로우시더라도 6자리 코드 번호를 직접 입력해 주세요.)")
        else:
            with st.spinner('안전한 이중 서버망에서 데이터를 분석 중입니다...'):
                try:
                    # 💡 이중 데이터 호출 엔진: FDR이 고장나면 야후 파이낸스로 즉시 대체 가동
                    df = pd.DataFrame()
                    
                    try:
                        df = fdr.DataReader(ticker, start_date.strftime('%Y-%m-%d'))
                    except:
                        pass
                        
                    if df.empty:
                        if ticker.isascii() and ticker.isalpha(): # 미국 주식일 경우
                            df = yf.Ticker(ticker).history(start=start_date.strftime('%Y-%m-%d'))
                        else: # 한국 주식일 경우
                            df = yf.Ticker(f"{ticker}.KS").history(start=start_date.strftime('%Y-%m-%d'))
                            if df.empty:
                                df = yf.Ticker(f"{ticker}.KQ").history(start=start_date.strftime('%Y-%m-%d'))
                    
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
                    st.error(f"❌ 데이터 수집
