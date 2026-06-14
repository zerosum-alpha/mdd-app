import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
import FinanceDataReader as fdr
from datetime import datetime

try:
    from pykrx import stock as pkstock
    PYKRX_AVAILABLE = True
except Exception:
    pkstock = None
    PYKRX_AVAILABLE = False

try:
    from auth import require_login, logout_button
except Exception:
    def require_login():
        return True
    def logout_button():
        return None

# =========================================================
# MDD 저점매수 분석기 - Core Dashboard Final
# 핵심만 표시:
# 1) Price + PER + MA + Signal
# 2) MDD + VIX
# 3) Current valuation snapshot
# 불필요한 바차트/밴드/가짜 PER/예측선 제거
# =========================================================

st.set_page_config(page_title="MDD 분석기", layout="wide")
require_login()
logout_button()

st.title("📈 MDD 저점매수 분석기 - Core Dashboard")
st.caption("핵심만 봅니다: 주가 / PER / MDD / VIX / 이동평균 / 매수·현금확보 후보")

# =========================================================
# Helpers
# =========================================================

def safe_float(x):
    try:
        if x is None or pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def fmt_num(x, digits=2):
    v = safe_float(x)
    if v is None:
        return "N/A"
    return f"{v:,.{digits}f}"


def fmt_pct(x, digits=2):
    v = safe_float(x)
    if v is None:
        return "N/A"
    return f"{v * 100:.{digits}f}%"


def fmt_pct_points(x, digits=2):
    v = safe_float(x)
    if v is None:
        return "N/A"
    return f"{v:.{digits}f}%"


def to_datetime_ns_series(values):
    """pandas/Streamlit Cloud 버전 차이로 생기는 M8[s]/M8[us] merge 오류 방지."""
    dt = pd.to_datetime(values, errors="coerce", utc=True)
    if isinstance(dt, pd.Series):
        return dt.dt.tz_convert(None).astype("datetime64[ns]")
    return pd.Series(dt).dt.tz_convert(None).astype("datetime64[ns]")


def to_datetime_ns_index(index):
    dt = pd.to_datetime(index, errors="coerce", utc=True)
    return dt.tz_convert(None).astype("datetime64[ns]")


# =========================================================
# 한국 종목 검색
# =========================================================
@st.cache_data(ttl=86400)
def get_stock_list():
    try:
        df = fdr.StockListing("KRX")
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.copy()
        df["Code"] = df["Code"].astype(str).str.zfill(6)
        df["Name"] = df["Name"].astype(str).str.strip()
        return df
    except Exception:
        return pd.DataFrame()


KR_FALLBACK_MAP = {
    "삼성전자": "005930",
    "삼성전자우": "005935",
    "SK하이닉스": "000660",
    "sk하이닉스": "000660",
    "현대차": "005380",
    "기아": "000270",
    "NAVER": "035420",
    "네이버": "035420",
    "카카오": "035720",
    "LG에너지솔루션": "373220",
    "삼성SDI": "006400",
    "삼성바이오로직스": "207940",
    "셀트리온": "068270",
    "POSCO홀딩스": "005490",
    "포스코홀딩스": "005490",
    "한화에어로스페이스": "012450",
    "두산에너빌리티": "034020",
}

stock_list = get_stock_list()


def find_ticker(query):
    query = str(query).strip()
    if query == "":
        return None, None, None

    if query.isdigit() and len(query) == 6:
        return "KR", query, query

    if query in KR_FALLBACK_MAP:
        return "KR", KR_FALLBACK_MAP[query], query

    if not stock_list.empty and "Name" in stock_list.columns and "Code" in stock_list.columns:
        exact = stock_list[stock_list["Name"] == query]
        if not exact.empty:
            return "KR", exact.iloc[0]["Code"], exact.iloc[0]["Name"]

        partial = stock_list[stock_list["Name"].str.contains(query, case=False, na=False)]
        if not partial.empty:
            return "KR", partial.iloc[0]["Code"], partial.iloc[0]["Name"]

    if any("가" <= ch <= "힣" for ch in query):
        return None, None, None

    return "US", query.upper(), query.upper()


# =========================================================
# Price / Indicator
# =========================================================
@st.cache_data(ttl=1800)
def load_price_data(market, ticker, start_date):
    start = pd.to_datetime(start_date).strftime("%Y-%m-%d")
    try:
        if market == "KR":
            df = fdr.DataReader(ticker, start)
            if df is None or df.empty:
                return pd.DataFrame()
            df.index = to_datetime_ns_index(df.index)
            return df

        df = yf.Ticker(ticker).history(start=start, auto_adjust=True)
        if df is None or df.empty:
            return pd.DataFrame()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df
    except Exception:
        return pd.DataFrame()


def calc_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def add_indicators(df):
    df = df.copy()
    if "Volume" not in df.columns:
        df["Volume"] = np.nan

    df["Peak"] = df["Close"].cummax()
    df["Current_Drawdown"] = df["Close"] / df["Peak"] - 1
    df["Max_Drawdown"] = df["Current_Drawdown"].cummin()
    df["Recovery_To_Peak"] = 1 / (1 + df["Current_Drawdown"]) - 1

    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA60"] = df["Close"].rolling(60).mean()
    df["MA200"] = df["Close"].rolling(200).mean()
    df["RSI"] = calc_rsi(df["Close"])

    df["Volume_MA20"] = df["Volume"].rolling(20).mean()
    df["Volume_Ratio"] = df["Volume"] / df["Volume_MA20"].replace(0, np.nan)
    df["Return"] = df["Close"].pct_change()
    return df


# =========================================================
# PER data
# =========================================================
@st.cache_data(ttl=3600)
def load_kr_per_series(ticker, start_date, end_date):
    """한국 종목: pykrx에서 과거 PER/PBR/EPS를 최대한 가져온다."""
    if not PYKRX_AVAILABLE:
        return pd.DataFrame(), "pykrx 미설치"

    start = pd.to_datetime(start_date).strftime("%Y%m%d")
    end = pd.to_datetime(end_date).strftime("%Y%m%d")

    attempts = []

    # 1) 공식적으로 많이 쓰이는 by_date 함수
    try:
        df = pkstock.get_market_fundamental_by_date(start, end, ticker)
        attempts.append(("get_market_fundamental_by_date", df))
    except Exception as e:
        attempts.append((f"get_market_fundamental_by_date 실패: {e}", pd.DataFrame()))

    # 2) 다른 pykrx 버전 호환
    for freq in ["d", "m"]:
        try:
            df = pkstock.get_market_fundamental(start, end, ticker, freq=freq)
            attempts.append((f"get_market_fundamental freq={freq}", df))
        except Exception as e:
            attempts.append((f"get_market_fundamental freq={freq} 실패: {e}", pd.DataFrame()))

    for source, df in attempts:
        if df is None or df.empty:
            continue
        df = df.copy()
        df.index = to_datetime_ns_index(df.index)
        df = df.dropna(axis=0, how="all")
        if "PER" not in df.columns:
            continue
        keep_cols = [c for c in ["PER", "PBR", "EPS", "BPS"] if c in df.columns]
        out = df[keep_cols].copy()
        for col in keep_cols:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        out = out[(out["PER"] > 0) & (out["PER"] < 300)]
        out = out.dropna(subset=["PER"])
        if len(out) >= 5:
            return out, source

    reason = " / ".join([src for src, _ in attempts[:3]])
    return pd.DataFrame(), reason


@st.cache_data(ttl=3600)
def load_us_valuation_info(ticker):
    empty = {
        "trailing_pe": None,
        "forward_pe": None,
        "price_to_sales": None,
        "peg_ratio": None,
        "market_cap": None,
    }
    try:
        info = yf.Ticker(ticker).info
        return {
            "trailing_pe": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "price_to_sales": info.get("priceToSalesTrailing12Months"),
            "peg_ratio": info.get("pegRatio"),
            "market_cap": info.get("marketCap"),
        }
    except Exception:
        return empty


@st.cache_data(ttl=3600)
def load_us_ttm_pe_series(ticker, price_df):
    """미국 종목: 분기 EPS 또는 Reported EPS로 TTM EPS를 만들고 일별 주가와 결합한다."""
    try:
        tk = yf.Ticker(ticker)
        eps_series = None
        source = ""

        # 1) earnings_dates Reported EPS가 가장 단순하게 동작하는 경우가 많다.
        try:
            ed = tk.get_earnings_dates(limit=40)
            if ed is not None and not ed.empty and "Reported EPS" in ed.columns:
                eps_series = pd.to_numeric(ed["Reported EPS"], errors="coerce").dropna()
                eps_series.index = to_datetime_ns_index(eps_series.index)
                eps_series = eps_series.sort_index()
                source = "yfinance earnings_dates Reported EPS"
        except Exception:
            pass

        # 2) quarterly_income_stmt EPS row
        if eps_series is None or len(eps_series) < 4:
            for attr in ["quarterly_income_stmt", "quarterly_financials"]:
                try:
                    stmt = getattr(tk, attr)
                    if stmt is None or stmt.empty:
                        continue
                    stmt = stmt.copy()
                    stmt.columns = pd.to_datetime(stmt.columns, errors="coerce")
                    for row_name in ["Diluted EPS", "Basic EPS", "Normalized EPS"]:
                        if row_name in stmt.index:
                            s = pd.to_numeric(stmt.loc[row_name], errors="coerce").dropna()
                            s.index = to_datetime_ns_index(s.index)
                            s = s.sort_index()
                            if len(s) >= 4:
                                eps_series = s
                                source = f"yfinance {attr} {row_name}"
                                break
                    if eps_series is not None and len(eps_series) >= 4:
                        break
                except Exception:
                    continue

        if eps_series is None or len(eps_series) < 4:
            return pd.DataFrame(), "미국 EPS 원자료 부족"

        eps_series = eps_series.sort_index()
        eps_ttm = eps_series.rolling(4).sum().dropna()
        eps_ttm = eps_ttm[eps_ttm > 0]
        if eps_ttm.empty:
            return pd.DataFrame(), "EPS TTM 계산 불가 또는 EPS <= 0"

        # 실적이 시장에 반영되는 단순 지연값. 너무 복잡하게 하지 않음.
        eps_df = pd.DataFrame({
            "Date": pd.to_datetime(eps_ttm.index) + pd.Timedelta(days=1),
            "EPS_TTM": eps_ttm.values,
        }).dropna()
        eps_df["Date"] = to_datetime_ns_series(eps_df["Date"])
        eps_df = eps_df.dropna(subset=["Date"]).sort_values("Date")

        daily = price_df[["Close"]].copy().reset_index()
        daily.columns = ["Date", "Close"]
        daily["Date"] = to_datetime_ns_series(daily["Date"])
        daily = daily.dropna(subset=["Date"]).sort_values("Date")

        # merge_asof는 양쪽 Date dtype이 정확히 같아야 함
        merged = pd.merge_asof(daily, eps_df, on="Date", direction="backward")
        merged["PER"] = merged["Close"] / merged["EPS_TTM"]
        merged["PER"] = pd.to_numeric(merged["PER"], errors="coerce")
        merged = merged[(merged["PER"] > 0) & (merged["PER"] < 300)].dropna(subset=["PER"])

        if len(merged) < 30:
            return pd.DataFrame(), "PER 일별 차트 데이터 부족"

        out = merged.set_index("Date")[["PER", "EPS_TTM"]]
        return out, source
    except Exception as e:
        return pd.DataFrame(), f"미국 PER 계산 실패: {e}"


@st.cache_data(ttl=1800)
def load_vix(start_date):
    try:
        start = pd.to_datetime(start_date).strftime("%Y-%m-%d")
        df = yf.Ticker("^VIX").history(start=start, auto_adjust=True)
        if df is None or df.empty:
            return pd.DataFrame()
        df.index = to_datetime_ns_index(df.index)
        return df[["Close"]].rename(columns={"Close": "VIX"})
    except Exception:
        return pd.DataFrame()


# =========================================================
# Signals / Decision
# =========================================================
def _first_signal_with_gap(condition, min_gap=20):
    """연속 신호가 차트를 뒤덮지 않도록 최소 간격을 둔다."""
    result = pd.Series(False, index=condition.index)
    last_idx = -10_000
    values = condition.fillna(False).to_numpy()

    for i, flag in enumerate(values):
        if flag and i - last_idx >= min_gap:
            result.iloc[i] = True
            last_idx = i
    return result


def build_signals(df):
    sig = df.copy()

    # 매수 후보: 깊은 MDD + 과매도 + 단기 추세 아래쪽. 신호는 참고 마커일 뿐 Buy Score를 바꾸지 않는다.
    sig["Buy_Signal"] = (
        (sig["Current_Drawdown"] <= -0.12) &
        (sig["RSI"] <= 40) &
        (sig["Close"] <= sig["MA20"] * 1.02)
    )

    # 현금확보 후보: 전고점 근처 + 과열. 너무 자주 찍히지 않도록 간격 필터를 적용한다.
    sig["Cash_Signal"] = (
        ((sig["Current_Drawdown"] >= -0.03) & (sig["RSI"] >= 70)) |
        ((sig["Close"] >= sig["Peak"] * 0.99) & (sig["RSI"] >= 65))
    )

    buy_mark = _first_signal_with_gap(sig["Buy_Signal"], min_gap=20)
    cash_mark = _first_signal_with_gap(sig["Cash_Signal"], min_gap=20)

    sig["Buy_Display"] = sig["Close"].where(buy_mark)
    sig["Cash_Display"] = sig["Close"].where(cash_mark)
    return sig[["Buy_Display", "Cash_Display"]]


def make_core_action(latest, per_available):
    dd = latest["Current_Drawdown"]
    rsi = latest["RSI"]
    close = latest["Close"]
    ma20 = latest["MA20"]

    if dd <= -0.20 and close < ma20:
        return "매수 금지 / 추세 훼손 확인", "DD -20% 이하 + MA20 하회"
    if dd <= -0.12 and rsi <= 40:
        if per_available:
            return "1차 소액 가능", "DD 깊음 + RSI 과매도 + PER 확인 가능"
        return "1차 소액 가능하나 PER 확인 불가", "DD 깊음 + RSI 과매도, PER 데이터 없음"
    if dd >= -0.03 and rsi >= 65:
        return "현금확보 검토", "고점 근처 + RSI 과열"
    if close > ma20 and rsi >= 50:
        return "확인매수 후보", "MA20 위 + RSI 회복"
    return "대기", "가격 매력 또는 반등 확인 부족"


# =========================================================
# Chart
# =========================================================
def plot_core_dashboard(df, per_df, vix_df, signal_df, title, valuation=None):
    df = df.copy()
    df.index = to_datetime_ns_index(df.index)
    chart = df[["Close", "MA20", "MA60", "MA200", "Current_Drawdown"]].copy()
    chart = chart.rename(columns={"Close": "Price"})

    if per_df is not None and not per_df.empty and "PER" in per_df.columns:
        per_df = per_df.copy()
        per_df.index = to_datetime_ns_index(per_df.index)
        chart = chart.join(per_df[["PER"]], how="left")
        chart["PER"] = chart["PER"].ffill()
    else:
        chart["PER"] = np.nan

    if vix_df is not None and not vix_df.empty:
        vix_df = vix_df.copy()
        vix_df.index = to_datetime_ns_index(vix_df.index)
        chart = chart.join(vix_df[["VIX"]], how="left")
        chart["VIX"] = chart["VIX"].ffill()
    else:
        chart["VIX"] = np.nan

    if signal_df is not None and not signal_df.empty:
        signal_df = signal_df.copy()
        signal_df.index = to_datetime_ns_index(signal_df.index)
        chart = chart.join(signal_df[["Buy_Display", "Cash_Display"]], how="left")

    fig, (ax1, ax3) = plt.subplots(
        2, 1,
        figsize=(15, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.4]}
    )

    # Top: Price + PER
    ax1.plot(chart.index, chart["Price"], label="Price", linewidth=2.0, color="#1f77b4")
    ax1.plot(chart.index, chart["MA20"], label="MA20", linewidth=1.1, color="#ff7f0e", alpha=0.9)
    ax1.plot(chart.index, chart["MA60"], label="MA60", linewidth=1.1, color="#2ca02c", alpha=0.85)
    ax1.plot(chart.index, chart["MA200"], label="MA200", linewidth=1.1, color="#9467bd", alpha=0.8)
    ax1.set_ylabel("Price")
    ax1.grid(True, linestyle=":", alpha=0.5)

    if "Buy_Display" in chart.columns:
        ax1.scatter(chart.index, chart["Buy_Display"] * 0.97, color="green", marker="^", s=110, zorder=5, label="BUY candidate")
    if "Cash_Display" in chart.columns:
        ax1.scatter(chart.index, chart["Cash_Display"] * 1.03, color="red", marker="v", s=110, zorder=5, label="Cash candidate")

    ax2 = ax1.twinx()
    if chart["PER"].dropna().empty:
        ax2.text(0.02, 0.92, "PER data unavailable", transform=ax2.transAxes, fontsize=10, color="crimson")
    else:
        ax2.plot(chart.index, chart["PER"], label="P/E", linewidth=1.8, color="crimson", alpha=0.95)
        per_valid = chart["PER"].dropna()
        if len(per_valid) >= 20:
            per_avg = per_valid.mean()
            ax2.axhline(per_avg, color="crimson", linestyle="--", alpha=0.35, linewidth=1.0, label="TTM P/E avg")

        # 현재 Forward P/E는 과거 시계열이 아니라 현재 기준선이다.
        if valuation:
            fpe = valuation.get("forward_pe")
            tpe = valuation.get("trailing_pe")
            try:
                if fpe is not None and pd.notna(fpe) and float(fpe) > 0:
                    ax2.axhline(float(fpe), color="black", linestyle="-.", alpha=0.65, linewidth=1.0, label="Current forward P/E")
                if tpe is not None and pd.notna(tpe) and float(tpe) > 0:
                    ax2.axhline(float(tpe), color="crimson", linestyle=":", alpha=0.55, linewidth=1.0, label="Current trailing P/E")
            except Exception:
                pass

    ax2.set_ylabel("TTM P/E", color="crimson")
    ax2.tick_params(axis="y", labelcolor="crimson")

    ax1.set_title(title, fontsize=14, fontweight="bold")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

    # Bottom: MDD + VIX
    ax3.plot(chart.index, chart["Current_Drawdown"] * 100, color="darkred", linewidth=1.6, label="Current DD")
    for level, label in [(-8, "Watch -8%"), (-12, "Buy zone -12%"), (-15, "Deep -15%"), (-20, "Risk -20%")]:
        ax3.axhline(level, linestyle="--", linewidth=0.9, alpha=0.55, label=label)
    ax3.set_ylabel("MDD (%)", color="darkred")
    ax3.tick_params(axis="y", labelcolor="darkred")
    ax3.grid(True, linestyle=":", alpha=0.5)

    ax4 = ax3.twinx()
    if not chart["VIX"].dropna().empty:
        ax4.plot(chart.index, chart["VIX"], color="green", linestyle="--", linewidth=1.2, alpha=0.7, label="VIX")
    ax4.set_ylabel("VIX", color="green")
    ax4.tick_params(axis="y", labelcolor="green")

    lines3, labels3 = ax3.get_legend_handles_labels()
    lines4, labels4 = ax4.get_legend_handles_labels()
    ax3.legend(lines3 + lines4, labels3 + labels4, loc="lower left", fontsize=8, ncol=3)

    plt.tight_layout()
    st.pyplot(fig)


# =========================================================
# UI
# =========================================================
col1, col2, col3 = st.columns(3)
with col1:
    user_input = st.text_input("종목명 / 코드 / 티커", value="삼성전자")
with col2:
    start_date = st.date_input("기준 시작일", pd.to_datetime("2024-01-01"))
with col3:
    planned_buy_amount = st.number_input("추가매수 예정금", value=1_000_000, step=100_000)

run = st.button("분석 실행")

if run:
    market, ticker, display_name = find_ticker(user_input)

    if ticker is None:
        st.error("종목을 찾을 수 없습니다. 예: 삼성전자, 005930, NVDA")
        st.stop()

    price_df = load_price_data(market, ticker, start_date)
    if price_df.empty:
        st.error("가격 데이터를 가져오지 못했습니다. 종목명/코드/티커를 확인하세요.")
        st.stop()

    df = add_indicators(price_df)
    latest = df.iloc[-1]

    # PER data
    per_source = ""
    if market == "KR":
        per_df, per_source = load_kr_per_series(ticker, start_date, datetime.today())
        valuation = {
            "trailing_pe": per_df["PER"].iloc[-1] if not per_df.empty and "PER" in per_df.columns else None,
            "forward_pe": None,
            "price_to_sales": None,
            "peg_ratio": None,
        }
    else:
        per_df, per_source = load_us_ttm_pe_series(ticker, df)
        valuation = load_us_valuation_info(ticker)

    vix_df = load_vix(start_date)
    signal_df = build_signals(df)
    per_available = per_df is not None and not per_df.empty and "PER" in per_df.columns
    action, action_reason = make_core_action(latest, per_available)

    # =====================================================
    # Core cards
    # =====================================================
    st.subheader(f"분석 대상: {display_name} / {ticker} / {market}")

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("최종 행동", action)
    k2.metric("Current DD", fmt_pct(latest["Current_Drawdown"]))
    k3.metric("MDD", fmt_pct(df["Max_Drawdown"].min()))
    k4.metric("RSI", fmt_num(latest["RSI"]))
    k5.metric("MA20 상태", "위" if latest["Close"] > latest["MA20"] else "아래")
    k6.metric("Vol Ratio", fmt_num(latest["Volume_Ratio"]))

    st.info(f"판단 이유: {action_reason}")

    # =====================================================
    # Main chart
    # =====================================================
    st.markdown("## 핵심 차트: Price + P/E + MDD + VIX")
    st.caption("상단: 주가·이평선·추정 TTM P/E·현재 Forward P/E 기준선 / 하단: MDD·VIX")

    chart_title = f"{ticker} Price + P/E + MDD + VIX"
    plot_core_dashboard(df, per_df, vix_df, signal_df, chart_title, valuation)

    if market == "US":
        st.caption(
            "미국 종목 차트의 빨간 P/E 선은 과거 4분기 EPS로 계산한 Estimated TTM P/E입니다. "
            "Forward P/E는 현재 컨센서스 기준값이라 과거 시계열이 아니며, 차트에서는 검은 점선 기준선으로만 표시합니다."
        )
    elif market == "KR":
        st.caption(
            "한국 종목 차트의 P/E는 pykrx/KRX 기반 과거 P/E입니다. 증권사 리포트의 12개월 예상 PER과는 다를 수 있습니다."
        )

    # =====================================================
    # Current valuation
    # =====================================================
    st.markdown("## 현재 Valuation")
    v1, v2, v3, v4 = st.columns(4)
    v1.metric("TTM/KRX P/E", fmt_num(valuation.get("trailing_pe")))
    v2.metric("Forward P/E", fmt_num(valuation.get("forward_pe")))
    v3.metric("P/S", fmt_num(valuation.get("price_to_sales")))
    v4.metric("PEG", fmt_num(valuation.get("peg_ratio")))

    if per_available:
        recent = per_df["PER"].dropna()
        current_per = recent.iloc[-1]
        per_avg = recent.mean()
        per_std = recent.std() if len(recent) > 2 else np.nan
        p1, p2, p3 = st.columns(3)
        p1.metric("현재 P/E", fmt_num(current_per))
        p2.metric("P/E 평균", fmt_num(per_avg))
        if safe_float(per_std) is not None:
            p3.metric("평균 대비", fmt_num((current_per - per_avg) / per_std), "표준편차")
        else:
            p3.metric("평균 대비", "N/A")
    else:
        st.warning(f"PER 시계열을 가져오지 못했습니다. 사유: {per_source}")

    # =====================================================
    # What to look at
    # =====================================================
    st.markdown("## 보는 기준")
    guide = pd.DataFrame({
        "봐야 할 것": [
            "주가 상승 + PER 하락",
            "주가 상승 + PER 상승",
            "주가 하락 + PER 하락",
            "주가 하락 + PER 상승",
            "DD -12% 이하 + RSI 40 이하",
            "VIX 급등 + DD 확대",
            "전고점 근처 + RSI 65 이상"
        ],
        "해석": [
            "이익 개선이 주가 상승을 정당화. 강한 구간 가능",
            "기대감 선반영. 추격 주의",
            "밸류 부담 완화. 업황 훼손 여부 확인",
            "이익 악화 가능성. 저점매수 주의",
            "1차 소액 후보",
            "공포성 눌림 가능. 분할 접근만",
            "현금확보 또는 추가매수 중단 후보"
        ]
    })
    st.table(guide)

    with st.expander("최근 20거래일 데이터"):
        show_cols = ["Close", "Current_Drawdown", "Max_Drawdown", "RSI", "MA20", "MA60", "MA200", "Volume_Ratio"]
        show = df[show_cols].tail(20).copy()
        show["Current_Drawdown"] = show["Current_Drawdown"] * 100
        show["Max_Drawdown"] = show["Max_Drawdown"] * 100
        st.dataframe(show, use_container_width=True)

    with st.expander("PER 원자료 확인"):
        st.write(f"PER source: {per_source}")
        if per_available:
            st.dataframe(per_df.tail(60), use_container_width=True)
        else:
            st.info("PER 원자료가 없습니다.")

    st.warning(
        "주의: 이 화면은 매수·매도 자동 신호가 아니라 판단 보조 도구입니다. "
        "미국 종목의 PER은 yfinance EPS 기반 Estimated TTM P/E이며, 증권사 리포트식 12개월 Forward P/E 시계열이 아닙니다."
    )
