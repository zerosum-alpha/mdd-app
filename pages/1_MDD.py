import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import FinanceDataReader as fdr
from datetime import datetime

# 한국 주식 PER 데이터를 가져오기 위한 pykrx 
try:
    from pykrx import stock as krx_stock
except Exception:
    krx_stock = None

from auth import require_login, logout_button

# =========================================================
# 실전 매매 타이밍 & 밸류에이션(PER) 통합 대시보드
# =========================================================

st.set_page_config(page_title="실전 매매 타이밍 & PER 분석기", layout="wide")

require_login()
logout_button()

st.title("📈 실전 트레이딩 & 밸류에이션 대시보드 (PER + Signals)")

# =========================================================
# 1. 종목 검색 및 가격/VIX 데이터 로드
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
    start = start_date.strftime("%Y-%m-%d")
    try:
        df = yf.Ticker("^VIX").history(start=start, auto_adjust=True)
        if not df.empty:
            df.index = df.index.tz_localize(None)
            return df[['Close']].rename(columns={'Close': 'VIX'})
    except: pass
    return pd.DataFrame()

# =========================================================
# 2. PER(밸류에이션) 데이터 로드 및 계산 (복구됨)
# =========================================================
@st.cache_data(ttl=86400)
def load_krx_per_data(ticker, start_date):
    """한국 주식용 과거 PER 시계열 데이터 로드"""
    if krx_stock is None: return pd.DataFrame()
    
    start_str = start_date.strftime("%Y%m%d")
    end_str = datetime.today().strftime("%Y%m%d")
    
    try:
        df = krx_stock.get_market_fundamental(start_str, end_str, ticker)
        if not df.empty:
            df.index = pd.to_datetime(df.index)
            df = df.replace(0, np.nan)
            return df[['PER', 'PBR']]
    except: pass
    return pd.DataFrame()

@st.cache_data(ttl=86400)
def load_us_historical_per(ticker, price_df):
    """미국 주식 EPS 추적 -> 일별 PER 시계열 데이터 생성 (복구 완료)"""
    if price_df is None or price_df.empty: return price_df
    
    try:
        t = yf.Ticker(ticker)
        eps_data = pd.DataFrame()
        
        # 1. earnings_dates 시도 (정확한 발표일 기준)
        try:
            ed = t.get_earnings_dates(limit=40)
            if ed is not None and not ed.empty:
                ed = ed.reset_index()
                date_col = ed.columns[0]
                eps_col = [c for c in ed.columns if 'Reported' in str(c) and 'EPS' in str(c)]
                if eps_col:
                    ed = ed.dropna(subset=[date_col, eps_col[0]])
                    ed['date'] = pd.to_datetime(ed[date_col], utc=True).dt.tz_localize(None)
                    ed['eps'] = pd.to_numeric(ed[eps_col[0]], errors='coerce')
                    ed = ed.sort_values('date').dropna(subset=['eps'])
                    ed['eps_ttm'] = ed['eps'].rolling(4).sum()
                    ed = ed.dropna(subset=['eps_ttm'])
                    if not ed.empty:
                        ed['report_date'] = ed['date'] + pd.Timedelta(days=1)
                        eps_data = ed[['report_date', 'eps_ttm']]
        except: pass

        # 2. 실패 시 재무제표 시도 (분기 종료일 기준 + 45일 지연 반영)
        if eps_data.empty:
            try:
                inc = t.quarterly_income_stmt
                if inc is not None and not inc.empty:
                    inc = inc.T
                    eps_cols = [c for c in inc.columns if 'Diluted EPS' in str(c) or 'Basic EPS' in str(c)]
                    if eps_cols:
                        inc['eps'] = pd.to_numeric(inc[eps_cols[0]], errors='coerce')
                        inc = inc.dropna(subset=['eps']).sort_index()
                        inc['eps_ttm'] = inc['eps'].rolling(4).sum()
                        inc = inc.dropna(subset=['eps_ttm'])
                        if not inc.empty:
                            inc['date'] = pd.to_datetime(inc.index).tz_localize(None)
                            inc['report_date'] = inc['date'] + pd.Timedelta(days=45)
                            eps_data = inc[['report_date', 'eps_ttm']]
            except: pass

        # 3. 주가 데이터와 병합 (일별 PER 계산)
        if not eps_data.empty:
            p_df = price_df.copy()
            p_df['join_date'] = pd.to_datetime(p_df.index).tz_localize(None)
            
            eps_data = eps_data.sort_values('report_date')
            p_df = p_df.sort_values('join_date')
            
            merged = pd.merge_asof(
                p_df, 
                eps_data, 
                left_on='join_date', 
                right_on='report_date', 
                direction='backward'
            )
            
            merged['PER'] = merged['Close'] / merged['eps_ttm']
            merged.loc[(merged['PER'] <= 0) | (merged['PER'] > 300), 'PER'] = np.nan
            
            price_df['PER'] = merged['PER'].values
            return price_df
    except: pass
    
    price_df['PER'] = np.nan
    return price_df

@st.cache_data(ttl=3600)
def load_us_valuation(ticker):
    """미국 주식용 현재 밸류에이션 요약"""
    try:
        info = yf.Ticker(ticker).info
        return {
            "trailing_pe": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "price_to_sales": info.get("priceToSalesTrailing12Months"),
            "peg_ratio": info.get("pegRatio")
        }
    except: return {}

# =========================================================
# 3. 타이밍 지표 및 매매 시그널 결합
# =========================================================
def calculate_indicators_and_signals(df, vix_df=None, kr_per_df=None):
    df = df.copy()
    
    # 1. 기술적 지표 (MDD, RSI)
    df["Peak"] = df["Close"].cummax()
    df["Current_Drawdown"] = df["Close"] / df["Peak"] - 1
    df["Max_Drawdown"] = df["Current_Drawdown"].cummin()
    
    df["MA20"] = df["Close"].rolling(20).mean()
    
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + (gain / loss)))
    
    # 2. VIX 결합
    if vix_df is not None and not vix_df.empty:
        df = df.join(vix_df, how='left')
        df['VIX'] = df['VIX'].ffill()
    else:
        df['VIX'] = np.nan

    # 3. 한국 주식 PER 결합 (미국 주식은 이미 위에서 병합됨)
    if kr_per_df is not None and not kr_per_df.empty:
        df = df.join(kr_per_df, how='left')
        df['PER'] = df['PER'].ffill()
    elif 'PER' not in df.columns:
        df['PER'] = np.nan

    # 4. 실전 매매 타이밍 (Signal) 조건
    buy_cond = (df["Current_Drawdown"] <= -0.15) & ((df["RSI"] <= 30) | (df["VIX"] >= 25))
    sell_cond = (df["Current_Drawdown"] >= -0.02) & (df["RSI"] >= 70)

    df['Buy_Signal'] = df['Close'][buy_cond & (~buy_cond.shift(1).fillna(False))]
    df['Sell_Signal'] = df['Close'][sell_cond & (~sell_cond.shift(1).fillna(False))]

    return df

# =========================================================
# 4. 3단 통합 차트 생성 (Price / PER / MDD+VIX)
# =========================================================
def make_comprehensive_chart(df, ticker, market):
    if df is None or df.empty: return None
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [2.5, 1.2, 1.2]}, sharex=True)
    
    # --- [Panel 1]: 주가 + 매수/매도 시그널 오버레이 ---
    ax1.plot(df.index, df["Close"], label="Price", color="#1F77B4", linewidth=2)
    ax1.plot(df.index, df["MA20"], label="MA20", color="orange", alpha=0.6, linestyle="--")
    
    ax1.scatter(df.index, df['Buy_Signal'] * 0.95, color='green', marker='^', s=150, zorder=5, label='BUY Signal (Deep DD + Fear)')
    ax1.scatter(df.index, df['Sell_Signal'] * 1.05, color='red', marker='v', s=150, zorder=5, label='SELL Signal (Recovery + Overheated)')
    
    ax1.set_title(f"[{ticker}] Trading & Valuation Dashboard", fontsize=16, fontweight='bold')
    ax1.set_ylabel("Stock Price", color="#1F77B4", fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle=':', alpha=0.6)

    # --- [Panel 2]: 밸류에이션 (PER) 추이 복구 ---
    if 'PER' in df.columns and df['PER'].notna().any():
        color_per = "#D69E2E" if market == "KR" else "#2CA02C" 
        label_per = "Historical PER (KRX)" if market == "KR" else "Estimated TTM P/E (US)"
        
        ax2.plot(df.index, df['PER'], color=color_per, linewidth=2, label=label_per)
        per_mean = df['PER'].mean()
        ax2.axhline(per_mean, color="gray", linestyle="--", alpha=0.6, label=f"Average PER ({per_mean:.1f}x)")
        
        ax2.set_ylabel("PER (x)", color=color_per, fontweight='bold')
        ax2.legend(loc='upper left')
        ax2.grid(True, linestyle=':', alpha=0.6)
    else:
        ax2.text(0.5, 0.5, 'PER Data not available for this ticker.', horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes, color='gray')
        ax2.set_ylabel("Valuation", color="gray")
        ax2.set_yticks([])

    # --- [Panel 3]: 리스크 (MDD + VIX) ---
    ax3.fill_between(df.index, df["Current_Drawdown"] * 100, 0, color="red", alpha=0.2, label="Current Drawdown")
    ax3.plot(df.index, df["Max_Drawdown"] * 100, color="darkred", linestyle="--", alpha=0.7, label="Max Drawdown")
    ax3.set_ylabel("Drawdown (%)", color="darkred", fontweight='bold')
    ax3.grid(True, linestyle=':', alpha=0.6)
    
    if 'VIX' in df.columns and df['VIX'].notna().any():
        ax4 = ax3.twinx()
        ax4.plot(df.index, df['VIX'], color='purple', alpha=0.4, linewidth=1.5, label='VIX Index')
        ax4.set_ylabel("VIX Index", color="purple")
        
        lines3, labels3 = ax3.get_legend_handles_labels()
        lines4, labels4 = ax4.get_legend_handles_labels()
        ax3.legend(lines3 + lines4, labels3 + labels4, loc="lower left")
    else:
        ax3.legend(loc="lower left")

    ax3.set_xlabel("Date", fontweight='bold')
    plt.tight_layout()
    return fig

# =========================================================
# 메인 UI
# =========================================================
col_a, col_b, col_c = st.columns(3)

with col_a:
    user_input = st.text_input("종목명 / 종목코드 / 미국 티커", value="NVDA")
with col_b:
    start_date = st.date_input("기준 시작일", pd.to_datetime("2024-01-01"))
with col_c:
    run = st.button("트레이딩 & 밸류에이션 분석 실행", type="primary", use_container_width=True)

if run:
    market, ticker, display_name = find_ticker(user_input)

    if ticker is None:
        st.error("종목을 찾을 수 없습니다.")
        st.stop()

    with st.spinner(f"{display_name} 데이터 분석 중..."):
        
        price_df = load_price_data(market, ticker, start_date)
        vix_df = load_vix_data(start_date) if market == "US" else None
        
        if price_df.empty:
            st.error("가격을 가져오지 못했습니다.")
            st.stop()

        # 한국/미국 분기하여 PER 데이터 획득
        kr_per_df = None
        us_val = {}
        if market == "KR":
            kr_per_df = load_krx_per_data(ticker, start_date)
        else:
            price_df = load_us_historical_per(ticker, price_df) # 미국주식 일별 PER 차트용 계산
            us_val = load_us_valuation(ticker) # 미국주식 상단 카드용 계산

        # 데이터 통합 및 시그널 계산
        df = calculate_indicators_and_signals(price_df, vix_df, kr_per_df)
        latest = df.iloc[-1]
        
        # ---------------------------------------------------------
        # 1. 밸류에이션(PER) 요약 카드 
        # ---------------------------------------------------------
        st.markdown("---")
        st.markdown(f"## 📊 1. 현재 밸류에이션 (PER) 상태: **{display_name}**")
        
        v1, v2, v3, v4 = st.columns(4)
        if market == "KR":
            current_per = latest.get("PER", np.nan)
            current_pbr = latest.get("PBR", np.nan)
            v1.metric("현재 PER (KRX)", f"{current_per:.2f}x" if pd.notna(current_per) else "N/A")
            v2.metric("현재 PBR (KRX)", f"{current_pbr:.2f}x" if pd.notna(current_pbr) else "N/A")
            v3.metric("과거 평균 PER", f"{df['PER'].mean():.2f}x" if 'PER' in df.columns and df['PER'].notna().any() else "N/A")
            v4.info("👉 차트의 PER 추이와 평균선을 비교하여 고평가 여부를 확인하세요.")
        else:
            # yfinance API가 고장나서 0.0이나 None을 뱉을 경우, 방금 직접 계산한 TTM PER로 강제 대체 (방어 로직)
            calc_per = df['PER'].dropna().iloc[-1] if 'PER' in df.columns and df['PER'].notna().any() else None
            
            t_pe = us_val.get('trailing_pe')
            if not t_pe or pd.isna(t_pe) or t_pe == 0: t_pe = calc_per
                
            f_pe = us_val.get('forward_pe')
            ps = us_val.get('price_to_sales')
            peg = us_val.get('peg_ratio')

            v1.metric("Trailing P/E (TTM)", f"{t_pe:.2f}x" if t_pe else "N/A")
            v2.metric("Forward P/E (예상)", f"{f_pe:.2f}x" if f_pe and f_pe != 0 else "N/A")
            v3.metric("P/S (매출배수)", f"{ps:.2f}x" if ps and ps != 0 else "N/A")
            v4.metric("PEG Ratio", f"{peg:.2f}" if peg and peg != 0 else "N/A")

        # ---------------------------------------------------------
        # 2. 실전 매매 타이밍 (Signal & MDD)
        # ---------------------------------------------------------
        st.markdown("## 🎯 2. 타이밍 및 시장 공포 상태")
        
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("현재가", f"{latest['Close']:,.2f}")
        s2.metric("최대 낙폭 (MDD)", f"{latest['Current_Drawdown'] * 100:.2f}%")
        s3.metric("현재 RSI (과열/과매도)", f"{latest['RSI']:.1f}" if pd.notna(latest['RSI']) else "N/A")
        
        if market == "US" and 'VIX' in latest and pd.notna(latest['VIX']):
            s4.metric("현재 VIX (공포지수)", f"{latest['VIX']:.1f}")
        else:
            s4.metric("시장 구분", "한국(KRX) - 개별 종목 수급 우선")

        # ---------------------------------------------------------
        # 3. 3단 통합 시각화 (Price + PER + MDD)
        # ---------------------------------------------------------
        st.markdown("## 📈 3. 통합 매매 시그널 차트")
        st.caption("▲(매수): 깊은 낙폭(MDD -15% 이하) + 과매도/공포 구간 | ▼(매도): 전고점 회복 + 단기 과열(RSI 70 이상)")
        
        chart_fig = make_comprehensive_chart(df, ticker, market)
        if chart_fig:
            st.pyplot(chart_fig)
            
        # ---------------------------------------------------------
        # 4. 최근 시그널 이력
        # ---------------------------------------------------------
        with st.expander("최근 30일 데이터 및 시그널 발생 내역 보기"):
            view_cols = ["Close", "Current_Drawdown", "RSI", "Buy_Signal", "Sell_Signal"]
            if "PER" in df.columns: view_cols.insert(1, "PER")
            if market == "US" and "VIX" in df.columns: view_cols.insert(3, "VIX")
                
            view_df = df[view_cols].tail(30).copy()
            view_df["Current_Drawdown"] = (view_df["Current_Drawdown"] * 100).round(2)
            view_df["RSI"] = view_df["RSI"].round(2)
            if "PER" in view_df.columns: view_df["PER"] = view_df["PER"].round(2)
            
            view_df["Buy_Signal"] = view_df["Buy_Signal"].apply(lambda x: "🟢 BUY" if pd.notna(x) else "")
            view_df["Sell_Signal"] = view_df["Sell_Signal"].apply(lambda x: "🔴 SELL" if pd.notna(x) else "")
            st.dataframe(view_df.sort_index(ascending=False), use_container_width=True)
