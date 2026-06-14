
import streamlit as st
import pandas as pd
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
# MDD 저점매수 분석기 FINAL
# + 한국 종목 검색 보강
# + Valuation Summary
# + P/E Band Chart
# + KR Price vs PER/PBR/EPS Trend
# + US Price vs estimated TTM P/E Trend separated chart
# + MDD + Valuation Matrix
# + Target Price
# + Cash Warning Light
#
# 원칙:
# - 기존 MDD / Buy Score 계산 로직은 유지
# - Valuation / Target / Cash Warning은 참고용 보조 필터
# - Buy Score에 강제 반영하지 않음
# =========================================================

st.set_page_config(page_title="MDD 분석기", layout="wide")

require_login()
logout_button()

st.title("📈 MDD 저점매수 분석기 FINAL")


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
        exact_match = stock_list[stock_list["Name"] == query]
        if not exact_match.empty:
            return "KR", exact_match.iloc[0]["Code"], exact_match.iloc[0]["Name"]

        partial_match = stock_list[
            stock_list["Name"].str.contains(query, case=False, na=False)
        ]
        if not partial_match.empty:
            return "KR", partial_match.iloc[0]["Code"], partial_match.iloc[0]["Name"]

    if any("가" <= ch <= "힣" for ch in query):
        return None, None, None

    return "US", query.upper(), query.upper()


# =========================================================
# 가격 데이터
# =========================================================
@st.cache_data(ttl=3600)
def load_price_data(market, ticker, start_date):
    start = start_date.strftime("%Y-%m-%d")

    try:
        if market == "KR":
            df = fdr.DataReader(ticker, start)
            if df.empty:
                return pd.DataFrame()
            df.index = pd.to_datetime(df.index)
            return df

        if market == "US":
            df = yf.Ticker(ticker).history(start=start, auto_adjust=True)
            if df.empty:
                return pd.DataFrame()
            df.index = df.index.tz_localize(None)
            return df

    except Exception:
        return pd.DataFrame()

    return pd.DataFrame()


@st.cache_data(ttl=1800)
def load_us_benchmark(ticker, start_date):
    start = start_date.strftime("%Y-%m-%d")

    try:
        df = yf.Ticker(ticker).history(start=start, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        df.index = df.index.tz_localize(None)
        return df
    except Exception:
        return pd.DataFrame()


# =========================================================
# 공통 포맷
# =========================================================
def is_valid_number(value):
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
        float(value)
        return True
    except Exception:
        return False


def format_price(value):
    if not is_valid_number(value):
        return "N/A"
    return f"{float(value):,.2f}"


def format_pct_value(value):
    if not is_valid_number(value):
        return "N/A"
    return f"{float(value):.2f}%"


def format_valuation_value(value):
    if not is_valid_number(value):
        return "N/A"
    return f"{float(value):,.2f}"


def format_market_cap(value):
    if not is_valid_number(value):
        return "N/A"

    value = float(value)

    if value >= 1_000_000_000_000:
        return f"{value / 1_000_000_000_000:.2f}T"
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"

    return f"{value:,.0f}"


def is_etf_like(asset_type, display_name, ticker):
    text = f"{asset_type} {display_name} {ticker}".upper()
    etf_keywords = [
        "ETF", "TIGER", "KODEX", "ACE", "TIME", "RISE",
        "SOL", "KOACT", "WON", "PLUS", "QQQ", "SPY",
        "SOXX", "IWM", "EWY", "SMH"
    ]
    return any(keyword in text for keyword in etf_keywords)


# =========================================================
# Valuation
# =========================================================
@st.cache_data(ttl=3600)
def load_valuation_data(market, ticker):
    empty_data = {
        "trailing_pe": None,
        "forward_pe": None,
        "price_to_sales": None,
        "peg_ratio": None,
        "market_cap": None,
        "enterprise_to_ebitda": None,
        "data_status": "N/A"
    }

    try:
        if market != "US":
            return empty_data

        info = yf.Ticker(ticker).info

        return {
            "trailing_pe": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "price_to_sales": info.get("priceToSalesTrailing12Months"),
            "peg_ratio": info.get("pegRatio"),
            "market_cap": info.get("marketCap"),
            "enterprise_to_ebitda": info.get("enterpriseToEbitda"),
            "data_status": "OK"
        }

    except Exception:
        return empty_data


def interpret_forward_pe(value):
    if not is_valid_number(value):
        return "예상PER 데이터 없음"

    value = float(value)

    if value <= 0:
        return "해석 제외"
    if value <= 15:
        return "밸류 부담 낮음"
    if value <= 30:
        return "보통"
    if value <= 50:
        return "성장 기대 반영, 부담 있음"
    return "고평가·추격 주의"


def interpret_price_to_sales(value):
    if not is_valid_number(value):
        return "매출배수 데이터 없음"

    value = float(value)

    if value <= 3:
        return "매출 대비 부담 낮음"
    if value <= 10:
        return "보통~성장주 구간"
    if value <= 30:
        return "고성장 기대 반영"
    return "과열 가능성, 추격 주의"


def interpret_peg(value):
    if not is_valid_number(value):
        return "PEG 데이터 없음"

    value = float(value)

    if value <= 0:
        return "해석 제외"
    if value <= 1:
        return "성장 대비 밸류 양호"
    if value <= 2:
        return "보통"
    return "성장 대비 밸류 부담"


def interpret_trailing_pe(value):
    if not is_valid_number(value):
        return "과거PER 데이터 없음"

    value = float(value)

    if value <= 0:
        return "해석 제외"
    if value <= 15:
        return "현재 이익 기준 부담 낮음"
    if value <= 30:
        return "현재 이익 기준 보통"
    if value <= 50:
        return "현재 이익 기준 부담 있음"
    return "현재 이익 기준 고평가 주의"


def interpret_ev_ebitda(value):
    if not is_valid_number(value):
        return "EV/EBITDA 데이터 없음"

    value = float(value)

    if value <= 0:
        return "해석 제외"
    if value <= 15:
        return "현금창출 대비 부담 낮음"
    if value <= 30:
        return "현금창출 대비 보통"
    return "현금창출 대비 부담 있음"


def interpret_market_cap(value):
    if not is_valid_number(value):
        return "시가총액 데이터 없음"

    value = float(value)

    if value >= 1_000_000_000_000:
        return "초대형주"
    if value >= 100_000_000_000:
        return "대형주"
    if value >= 10_000_000_000:
        return "중형주"
    return "소형주"


def make_valuation_table(valuation):
    return pd.DataFrame([
        {
            "항목": "Trailing P/E(과거PER)",
            "값": format_valuation_value(valuation["trailing_pe"]),
            "해석": interpret_trailing_pe(valuation["trailing_pe"])
        },
        {
            "항목": "Forward P/E(예상PER)",
            "값": format_valuation_value(valuation["forward_pe"]),
            "해석": interpret_forward_pe(valuation["forward_pe"])
        },
        {
            "항목": "P/S(매출배수)",
            "값": format_valuation_value(valuation["price_to_sales"]),
            "해석": interpret_price_to_sales(valuation["price_to_sales"])
        },
        {
            "항목": "PEG(성장대비PER)",
            "값": format_valuation_value(valuation["peg_ratio"]),
            "해석": interpret_peg(valuation["peg_ratio"])
        },
        {
            "항목": "EV/EBITDA",
            "값": format_valuation_value(valuation["enterprise_to_ebitda"]),
            "해석": interpret_ev_ebitda(valuation["enterprise_to_ebitda"])
        },
        {
            "항목": "Market Cap(시가총액)",
            "값": format_market_cap(valuation["market_cap"]),
            "해석": interpret_market_cap(valuation["market_cap"])
        }
    ])


def make_mdd_valuation_comment(current_dd, valuation):
    forward_pe = valuation["forward_pe"]
    ps = valuation["price_to_sales"]

    forward_pe_valid = is_valid_number(forward_pe) and float(forward_pe) > 0
    ps_valid = is_valid_number(ps)

    if not forward_pe_valid and not ps_valid:
        return "밸류 판단 불가. MDD·차트·수급 중심으로 판단해야 합니다."

    comments = []

    if forward_pe_valid:
        pe = float(forward_pe)

        if current_dd <= -0.12 and pe <= 30:
            comments.append("MDD가 깊고 Forward P/E도 30 이하라 밸류 부담이 완화된 구간입니다.")
        elif current_dd <= -0.12 and pe > 50:
            comments.append("MDD는 깊지만 Forward P/E가 50 초과라 밸류 부담이 여전히 큽니다. 소액 접근만 적합합니다.")
        elif current_dd > -0.08 and pe > 50:
            comments.append("낙폭은 얕고 Forward P/E가 50 초과라 추격 매수 금지 구간입니다.")

    if ps_valid:
        ps_value = float(ps)
        if current_dd <= -0.15 and ps_value > 30:
            comments.append("Current DD는 깊지만 P/S가 30 초과라 고성장 기대가 여전히 과도하게 반영된 구간입니다.")

    if not comments:
        comments.append("MDD와 밸류에이션이 명확한 극단 구간은 아닙니다. 기존 MDD 신호와 시장 필터를 함께 확인하세요.")

    return " ".join(comments)


def make_valuation_summary(valuation):
    forward_pe = valuation["forward_pe"]
    ps = valuation["price_to_sales"]
    peg = valuation["peg_ratio"]

    pe_text = format_valuation_value(forward_pe)
    ps_text = format_valuation_value(ps)
    peg_text = format_valuation_value(peg)

    parts = [
        f"Forward P/E {pe_text}",
        f"P/S {ps_text}",
        f"PEG {peg_text}"
    ]

    if is_valid_number(forward_pe):
        pe = float(forward_pe)
        if pe <= 30:
            summary = "밸류 부담 보통 이하"
        elif pe <= 50:
            summary = "성장 기대 반영"
        else:
            summary = "추격 주의"
    elif is_valid_number(ps):
        ps_value = float(ps)
        if ps_value <= 10:
            summary = "매출배수 부담 보통 이하"
        elif ps_value <= 30:
            summary = "고성장 기대 반영"
        else:
            summary = "과열 가능성"
    else:
        summary = "밸류 판단 불가"

    return " / ".join(parts) + f" → {summary}"


# =========================================================
# Financial trend
# =========================================================
@st.cache_data(ttl=86400)
def load_financial_trend_data(ticker):
    """
    yfinance 분기 손익계산서에서 TTM EPS / TTM Revenue를 만든다.

    핵심:
    - 주가처럼 PER 시계열이 바로 제공되는 것이 아니므로,
      실제 분기 EPS 또는 Net Income / Diluted Shares로 EPS를 만든다.
    - 최근 4개 분기 합산 EPS = EPS TTM
    - 최근 4개 분기 합산 매출 = Revenue TTM
    - 이후 일별 주가와 결합해서 일별 TTM P/E를 계산한다.
    """
    try:
        t = yf.Ticker(ticker)

        q_inc = t.quarterly_income_stmt
        q_fin = t.quarterly_financials

        if q_inc is not None and not q_inc.empty:
            financials = q_inc.copy()
        elif q_fin is not None and not q_fin.empty:
            financials = q_fin.copy()
        else:
            return pd.DataFrame()

        financials.columns = pd.to_datetime(financials.columns, errors="coerce")
        financials = financials.loc[:, financials.columns.notna()]
        financials = financials.sort_index(axis=1)

        if financials.empty or len(financials.columns) < 4:
            return pd.DataFrame()

        def pick_row(candidates):
            for c in candidates:
                if c in financials.index:
                    return pd.to_numeric(financials.loc[c], errors="coerce")
            return None

        revenue = pick_row([
            "Total Revenue", "TotalRevenue", "Operating Revenue", "Revenue"
        ])

        eps = pick_row([
            "Diluted EPS", "DilutedEPS", "Basic EPS", "BasicEPS"
        ])

        # yfinance에서 EPS 행이 없는 경우 Net Income / Diluted Average Shares로 직접 계산
        if eps is None:
            net_income = pick_row([
                "Net Income Common Stockholders",
                "Net Income",
                "NetIncome",
                "Net Income From Continuing Operation Net Minority Interest"
            ])
            shares = pick_row([
                "Diluted Average Shares",
                "DilutedAverageShares",
                "Basic Average Shares",
                "BasicAverageShares"
            ])

            if net_income is not None and shares is not None:
                shares = shares.replace(0, pd.NA)
                eps = net_income / shares

        if eps is None:
            return pd.DataFrame()

        eps = pd.to_numeric(eps, errors="coerce")
        eps_ttm = eps.rolling(4).sum()

        if revenue is not None:
            revenue = pd.to_numeric(revenue, errors="coerce")
            revenue_ttm = revenue.rolling(4).sum()
        else:
            revenue_ttm = None

        rows = []

        for dt in financials.columns:
            eps_value = eps_ttm.get(dt)
            revenue_value = revenue_ttm.get(dt) if revenue_ttm is not None else None

            rows.append({
                "fiscal_date": pd.Timestamp(dt),
                "revenue_ttm": None if revenue_value is None or pd.isna(revenue_value) else float(revenue_value),
                "eps_ttm": None if eps_value is None or pd.isna(eps_value) else float(eps_value)
            })

        trend_df = pd.DataFrame(rows)
        trend_df = trend_df.dropna(subset=["eps_ttm"]).copy()
        trend_df = trend_df[trend_df["eps_ttm"] > 0].copy()
        trend_df = trend_df.sort_values("fiscal_date")

        return trend_df.tail(12)

    except Exception:
        return pd.DataFrame()


def add_price_to_financial_trend(financial_df, price_df, report_lag_days=45):
    """
    일별 주가 + EPS TTM을 결합해 일별 TTM P/E 추정값을 만든다.

    계산 방식:
    - 재무제표의 fiscal_date + 45일을 실적 반영일로 근사한다.
    - 그 이후 다음 실적 반영일까지 EPS TTM을 유지한다.
    - 일별 TTM P/E = 일별 주가 / EPS TTM

    주의:
    - 이 값은 FactSet식 12개월 Forward P/E가 아니다.
    - 무료 데이터로 가능한 Trailing P/E 추정 시계열이다.
    """
    if financial_df is None or financial_df.empty or price_df is None or price_df.empty:
        return pd.DataFrame()

    try:
        fin = financial_df.copy()
        if "fiscal_date" not in fin.columns or "eps_ttm" not in fin.columns:
            return pd.DataFrame()

        fin["fiscal_date"] = pd.to_datetime(fin["fiscal_date"], errors="coerce")
        fin["eps_ttm"] = pd.to_numeric(fin["eps_ttm"], errors="coerce")
        if "revenue_ttm" in fin.columns:
            fin["revenue_ttm"] = pd.to_numeric(fin["revenue_ttm"], errors="coerce")
        else:
            fin["revenue_ttm"] = pd.NA

        fin = fin.dropna(subset=["fiscal_date", "eps_ttm"]).copy()
        fin = fin[fin["eps_ttm"] > 0].copy()
        fin = fin.sort_values("fiscal_date")

        if fin.empty:
            return pd.DataFrame()

        fin["report_date"] = fin["fiscal_date"] + pd.Timedelta(days=report_lag_days)
        fin = fin.sort_values("report_date")

        price = price_df.copy().sort_index()
        price = price.reset_index()

        date_col = price.columns[0]
        price = price.rename(columns={date_col: "date", "Close": "price"})
        price["date"] = pd.to_datetime(price["date"], errors="coerce")
        price["price"] = pd.to_numeric(price["price"], errors="coerce")
        price = price.dropna(subset=["date", "price"]).copy()
        price = price[price["price"] > 0].copy()
        price = price.sort_values("date")

        if price.empty:
            return pd.DataFrame()

        merged = pd.merge_asof(
            price[["date", "price"]],
            fin[["report_date", "fiscal_date", "revenue_ttm", "eps_ttm"]],
            left_on="date",
            right_on="report_date",
            direction="backward"
        )

        merged = merged.dropna(subset=["eps_ttm"]).copy()
        merged["pe_ttm"] = merged["price"] / merged["eps_ttm"]

        # 차트를 망가뜨리는 극단값 제거. 음수 EPS는 이미 제외.
        merged = merged[(merged["pe_ttm"] > 0) & (merged["pe_ttm"] < 300)].copy()

        return merged[["date", "price", "pe_ttm", "eps_ttm", "revenue_ttm", "fiscal_date", "report_date"]]

    except Exception:
        return pd.DataFrame()


def make_monthly_view(df, date_col="date"):
    if df is None or df.empty or date_col not in df.columns:
        return pd.DataFrame()

    temp = df.copy()
    temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
    temp = temp.dropna(subset=[date_col]).sort_values(date_col)

    if temp.empty:
        return pd.DataFrame()

    temp = temp.set_index(date_col)

    # 월말 기준. pandas 버전 호환을 위해 M 사용.
    monthly = temp.resample("M").last().dropna(how="all").reset_index()

    if len(monthly) < 6:
        weekly = temp.resample("W-FRI").last().dropna(how="all").reset_index()
        return weekly

    return monthly


def normalize_series(value_series):
    s = pd.Series(value_series).astype(float)
    s = s.replace([float("inf"), float("-inf")], pd.NA)
    first_valid = s.dropna()

    if first_valid.empty:
        return s

    base = first_valid.iloc[0]

    if base == 0 or pd.isna(base):
        return s

    return s / base * 100


def make_financial_trend_chart(fin_trend_df, ticker):
    if fin_trend_df is None or fin_trend_df.empty:
        return None

    existing_cols = [col for col in ["price", "revenue_ttm", "eps_ttm"] if col in fin_trend_df.columns]

    if "date" not in fin_trend_df.columns or not existing_cols:
        return None

    chart_df = fin_trend_df.copy()
    chart_df = chart_df.dropna(how="all", subset=existing_cols)

    if chart_df.empty:
        return None

    chart_df = make_monthly_view(chart_df)

    if chart_df.empty or "date" not in chart_df.columns:
        return None

    fig, ax = plt.subplots(figsize=(12, 5))
    plotted = False

    if "price" in chart_df.columns and chart_df["price"].notna().sum() >= 2:
        chart_df["price_index"] = normalize_series(chart_df["price"])
        ax.plot(chart_df["date"], chart_df["price_index"], marker="o", label="Price Index")
        plotted = True

    if "revenue_ttm" in chart_df.columns and chart_df["revenue_ttm"].notna().sum() >= 2:
        chart_df["revenue_index"] = normalize_series(chart_df["revenue_ttm"])
        ax.plot(chart_df["date"], chart_df["revenue_index"], marker="o", label="Revenue TTM Index")
        plotted = True

    if "eps_ttm" in chart_df.columns and chart_df["eps_ttm"].notna().sum() >= 2:
        chart_df["eps_index"] = normalize_series(chart_df["eps_ttm"])
        ax.plot(chart_df["date"], chart_df["eps_index"], marker="o", label="EPS TTM Index")
        plotted = True

    if not plotted:
        plt.close(fig)
        return None

    ax.axhline(y=100, linestyle="--", alpha=0.5)
    ax.set_title(f"{ticker} Price vs Fundamentals Index")
    ax.set_ylabel("Index = 100 at first available point")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    return fig


def make_price_pe_trend_chart(fin_trend_df, valuation, ticker):
    """
    US 종목용 Price vs Estimated TTM P/E 차트.

    한국 종목의 Price vs PER 차트와 유사하게,
    - 위 패널: Price
    - 아래 패널: Estimated TTM P/E + 평균/±1SD/±2SD + 현재 Forward P/E 참고선
    으로 분리해서 표시한다.

    주의:
    - 이 값은 FactSet식 12개월 Forward P/E 시계열이 아니다.
    - yfinance 재무제표 기반 Estimated TTM P/E다.
    - 하지만 "주가가 오르는데 P/E가 낮아지는지"는 판단 가능하다.
    """
    if fin_trend_df is None or fin_trend_df.empty:
        return None, pd.DataFrame()

    needed = {"date", "price", "pe_ttm"}
    if not needed.issubset(set(fin_trend_df.columns)):
        return None, pd.DataFrame()

    chart_df = fin_trend_df.copy()
    chart_df["date"] = pd.to_datetime(chart_df["date"], errors="coerce")
    chart_df["price"] = pd.to_numeric(chart_df["price"], errors="coerce")
    chart_df["pe_ttm"] = pd.to_numeric(chart_df["pe_ttm"], errors="coerce")

    chart_df = chart_df.dropna(subset=["date", "price", "pe_ttm"]).copy()
    chart_df = chart_df[
        (chart_df["price"] > 0)
        & (chart_df["pe_ttm"] > 0)
        & (chart_df["pe_ttm"] < 300)
    ].copy()
    chart_df = chart_df.sort_values("date")

    if chart_df.empty or len(chart_df) < 20:
        return None, chart_df

    # 보기 쉽게 월말 기준. 데이터가 부족하면 주간 기준으로 대체.
    plot_df = make_monthly_view(chart_df)
    if plot_df.empty or len(plot_df) < 6:
        plot_df = chart_df.copy()

    if plot_df.empty or len(plot_df) < 4:
        return None, plot_df

    fig = plt.figure(figsize=(13.5, 7.2))
    gs = fig.add_gridspec(4, 1, hspace=0.10)

    ax_price = fig.add_subplot(gs[0:3, 0])
    ax_pe = fig.add_subplot(gs[3, 0], sharex=ax_price)

    # 상단: 주가
    ax_price.plot(plot_df["date"], plot_df["price"], label="Price", linewidth=2)
    ax_price.set_title(f"{ticker} Price vs Estimated TTM P/E")
    ax_price.set_ylabel("Price")
    ax_price.grid(True, alpha=0.3)
    ax_price.legend(loc="upper left")

    # 하단: P/E
    ax_pe.plot(plot_df["date"], plot_df["pe_ttm"], label="Estimated TTM P/E", linewidth=2)
    ax_pe.set_ylabel("P/E")
    ax_pe.grid(True, alpha=0.3)

    pe_mean = plot_df["pe_ttm"].mean()
    pe_std = plot_df["pe_ttm"].std()

    if is_valid_number(pe_mean):
        ax_pe.axhline(pe_mean, linestyle="-", alpha=0.45, label="P/E avg")

    if is_valid_number(pe_std) and pe_std > 0:
        ax_pe.axhline(pe_mean + pe_std, linestyle=":", alpha=0.45, label="P/E +1SD")
        ax_pe.axhline(max(pe_mean - pe_std, 0), linestyle=":", alpha=0.45, label="P/E -1SD")
        ax_pe.axhline(pe_mean + 2 * pe_std, linestyle="-.", alpha=0.28, label="P/E +2SD")
        ax_pe.axhline(max(pe_mean - 2 * pe_std, 0), linestyle="-.", alpha=0.28, label="P/E -2SD")

    current_forward_pe = valuation.get("forward_pe") if isinstance(valuation, dict) else None
    if is_valid_number(current_forward_pe) and float(current_forward_pe) > 0:
        ax_pe.axhline(float(current_forward_pe), linestyle="--", alpha=0.35, label="Current forward P/E")

    ax_pe.legend(loc="upper left", fontsize=8)

    # 상단 x축 라벨 겹침 방지
    plt.setp(ax_price.get_xticklabels(), visible=False)

    plt.tight_layout()
    return fig, plot_df

def make_price_pe_comment(price_pe_df):
    if price_pe_df is None or price_pe_df.empty or not {"date", "price", "pe_ttm"}.issubset(set(price_pe_df.columns)):
        return (
            "Price + P/E 추세 차트를 그릴 데이터가 부족합니다. "
            "분기 EPS 또는 주가 데이터가 충분하지 않습니다."
        )

    df = price_pe_df.dropna(subset=["date", "price", "pe_ttm"]).copy()

    if df.empty or len(df) < 3:
        return "Price + P/E 추세 해석을 하기에는 데이터가 부족합니다."

    df = df.sort_values("date")
    last_date = pd.to_datetime(df["date"].iloc[-1])
    cutoff = last_date - pd.DateOffset(months=6)
    base_df = df[pd.to_datetime(df["date"]) >= cutoff]

    if len(base_df) < 3:
        base_df = df

    first_price = base_df["price"].iloc[0]
    last_price = base_df["price"].iloc[-1]
    first_pe = base_df["pe_ttm"].iloc[0]
    last_pe = base_df["pe_ttm"].iloc[-1]

    price_change = (last_price / first_price - 1) * 100 if first_price else None
    pe_change = (last_pe / first_pe - 1) * 100 if first_pe else None

    if price_change is None or pe_change is None:
        return "Price + P/E 추세 해석이 불가능합니다."

    prefix = f"최근 비교 구간: 주가 {price_change:+.1f}%, 추정 TTM P/E {pe_change:+.1f}%. "

    if price_change > 0 and pe_change < 0:
        return prefix + "주가는 상승했지만 P/E는 하락했습니다. 실적 개선이 주가 상승을 정당화하는 구간일 수 있습니다."

    if price_change > 0 and pe_change > 0:
        return prefix + "주가와 P/E가 함께 상승했습니다. 실적보다 기대감이 더 빠르게 반영되는 과열 구간인지 확인해야 합니다."

    if price_change < 0 and pe_change < 0:
        return prefix + "주가와 P/E가 함께 하락했습니다. 밸류 부담은 완화됐지만 업황 훼손 여부를 확인해야 합니다."

    if price_change < 0 and pe_change > 0:
        return prefix + "주가는 하락했지만 P/E는 상승했습니다. 이익이 더 빠르게 악화된 구간일 수 있어 저점매수 주의가 필요합니다."

    return prefix + "주가와 P/E 변화가 뚜렷하지 않습니다. MDD, RSI, 이벤트 리스크를 함께 확인하세요."


# =========================================================
# Valuation charts
# =========================================================
def make_pe_band_data(current_price, valuation):
    pe = valuation["forward_pe"]

    if not is_valid_number(pe) or float(pe) <= 0 or current_price <= 0:
        return pd.DataFrame()

    pe = float(pe)
    implied_eps = current_price / pe

    bands = [
        ("15x", implied_eps * 15),
        ("30x", implied_eps * 30),
        ("50x", implied_eps * 50),
        ("Current", current_price)
    ]

    return pd.DataFrame(bands, columns=["Band", "Price"])


def make_pe_band_chart(current_price, valuation, ticker):
    band_df = make_pe_band_data(current_price, valuation)

    if band_df.empty:
        return None, band_df

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.bar(band_df["Band"], band_df["Price"])
    ax.axhline(y=current_price, linestyle="--", alpha=0.7, label="Current Price")

    ax.set_title(f"{ticker} Forward P/E Band")
    ax.set_ylabel("Price")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    plt.tight_layout()

    return fig, band_df


def make_valuation_zone(forward_pe, ps):
    if not is_valid_number(forward_pe) and not is_valid_number(ps):
        return "Unknown"

    if is_valid_number(forward_pe):
        pe = float(forward_pe)

        if pe <= 30:
            return "Low/Normal"
        if pe <= 50:
            return "High"
        return "Very High"

    ps_value = float(ps)

    if ps_value <= 10:
        return "Low/Normal"
    if ps_value <= 30:
        return "High"
    return "Very High"


def make_mdd_zone(current_dd):
    if current_dd > -0.08:
        return "Shallow DD"
    if current_dd > -0.12:
        return "Watch DD"
    if current_dd > -0.15:
        return "Buy1 DD"
    return "Deep DD"


def make_mdd_valuation_matrix(current_dd, valuation):
    valuation_zone = make_valuation_zone(valuation["forward_pe"], valuation["price_to_sales"])
    mdd_zone = make_mdd_zone(current_dd)

    rules = {
        ("Shallow DD", "Low/Normal"): "관망",
        ("Shallow DD", "High"): "대기",
        ("Shallow DD", "Very High"): "추격 금지",
        ("Shallow DD", "Unknown"): "밸류 확인 불가",

        ("Watch DD", "Low/Normal"): "관심",
        ("Watch DD", "High"): "소액 후보",
        ("Watch DD", "Very High"): "대기",
        ("Watch DD", "Unknown"): "MDD 중심 판단",

        ("Buy1 DD", "Low/Normal"): "1차 가능",
        ("Buy1 DD", "High"): "1차 소액",
        ("Buy1 DD", "Very High"): "소액만",
        ("Buy1 DD", "Unknown"): "MDD 중심 1차 후보",

        ("Deep DD", "Low/Normal"): "강한 후보",
        ("Deep DD", "High"): "분할 가능",
        ("Deep DD", "Very High"): "리스크 확인",
        ("Deep DD", "Unknown"): "MDD 중심 분할 가능",
    }

    decision = rules.get((mdd_zone, valuation_zone), "판단 보류")

    matrix_rows = []
    for row_zone in ["Shallow DD", "Watch DD", "Buy1 DD", "Deep DD"]:
        row = {"MDD 구간": row_zone}
        for col_zone in ["Low/Normal", "High", "Very High", "Unknown"]:
            value = rules.get((row_zone, col_zone), "-")
            if row_zone == mdd_zone and col_zone == valuation_zone:
                value = f"▶ {value}"
            row[col_zone] = value
        matrix_rows.append(row)

    matrix_df = pd.DataFrame(matrix_rows)

    return matrix_df, mdd_zone, valuation_zone, decision


# =========================================================
# Cash Warning Light
# =========================================================
def make_default_event_schedule():
    return pd.DataFrame({
        "date": [
            "2026-06-18",
            "2026-06-19",
            "2026-06-19",
            "2026-06-24",
            "2026-06-26",
            "2026-07-02",
            "2026-07-10",
            "2026-07-15",
            "2026-07-16",
            "2026-07-29",
            "2026-08-01",
            "2026-08-15",
        ],
        "event": [
            "미국 PPI 발표",
            "미국 네마녀의 날",
            "한국 ETF 리밸런싱",
            "주요 기업 실적 발표",
            "대형 IPO/상장 이벤트",
            "미국 고용보고서",
            "지정학 이벤트 점검",
            "미국 CPI 발표",
            "미국 PPI 발표",
            "FOMC",
            "주요 기업 실적 시즌",
            "옵션만기/선물만기"
        ],
        "category": [
            "PPI",
            "QuadWitching",
            "ETF_Rebalance",
            "Earnings",
            "IPO",
            "기타",
            "Geopolitical",
            "CPI",
            "PPI",
            "FOMC",
            "Earnings",
            "기타"
        ],
        "market": [
            "US",
            "US",
            "KR",
            "US",
            "US",
            "US",
            "Global",
            "US",
            "US",
            "US",
            "US",
            "KR"
        ],
        "impact": [
            "High",
            "High",
            "Medium",
            "Medium",
            "Medium",
            "Medium",
            "High",
            "High",
            "High",
            "High",
            "Medium",
            "Medium"
        ],
        "memo": [
            "물가 재부담 여부 확인",
            "옵션·선물 만기 수급 변동성 확대 가능",
            "국내 ETF 구성종목 수급 왜곡 가능",
            "AI·반도체·빅테크 가이던스 확인",
            "대형 상장 이벤트 전후 유동성 이동 가능",
            "고용 강세 시 금리 부담 가능",
            "전쟁·제재·해상운송·유가 리스크 점검",
            "물가 핵심 이벤트",
            "CPI 이후 생산자물가 확인",
            "금리·점도표·파월 발언 확인",
            "실적과 가이던스에 따라 테마 변동 가능",
            "국내 수급 변동성 가능"
        ]
    })


def normalize_event_schedule(df):
    required_cols = ["date", "event", "category", "market", "impact", "memo"]

    for col in required_cols:
        if col not in df.columns:
            df[col] = ""

    df = df[required_cols].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["impact"] = df["impact"].astype(str).str.strip().str.capitalize()
    df["impact"] = df["impact"].replace({
        "HIGH": "High",
        "MEDIUM": "Medium",
        "LOW": "Low",
        "high": "High",
        "medium": "Medium",
        "low": "Low"
    })

    return df


def business_days_until(today, event_date):
    today = pd.Timestamp(today).normalize()
    event_date = pd.Timestamp(event_date).normalize()

    if event_date < today:
        return -1

    days = pd.bdate_range(today, event_date)
    return max(len(days) - 1, 0)


def calculate_cash_warning(event_df, manual_score):
    today = pd.Timestamp(datetime.today().date())

    df = event_df.copy()
    df["D-Day"] = df["date"].apply(lambda x: business_days_until(today, x))
    df = df[df["D-Day"] >= 0].copy()
    df = df.sort_values("date")

    auto_score = 0
    warning_messages = []

    for _, row in df.iterrows():
        d_day = int(row["D-Day"])
        impact = row["impact"]

        if impact == "High":
            if d_day <= 3:
                auto_score += 2
            elif d_day <= 5:
                auto_score += 1

            if d_day <= 1:
                warning_messages.append(
                    f"High 이벤트가 D-{d_day}입니다: {row['event']}. 신규매수보다 리스크 관리 우선."
                )

        elif impact == "Medium":
            if d_day <= 3:
                auto_score += 1

    total_score = auto_score + manual_score

    if total_score <= 1:
        status = "유지"
        final_action = "유지"
    elif total_score == 2:
        status = "주의"
        final_action = "추가매수 중단"
    elif total_score == 3:
        status = "현금확보 검토"
        final_action = "현금 10% 확보 검토"
    else:
        status = "위험"
        final_action = "현금 20~30% 확보 검토"

    near_events = df[df["D-Day"] <= 10].copy()
    near_events["날짜"] = near_events["date"].dt.strftime("%Y-%m-%d")
    near_events = near_events.rename(columns={
        "event": "이벤트명",
        "category": "구분",
        "market": "시장",
        "impact": "중요도",
        "memo": "메모"
    })

    near_events = near_events[
        ["D-Day", "날짜", "이벤트명", "구분", "시장", "중요도", "메모"]
    ]

    full_events = df.copy()
    full_events["date"] = full_events["date"].dt.strftime("%Y-%m-%d")

    return {
        "auto_score": auto_score,
        "manual_score": manual_score,
        "total_score": total_score,
        "status": status,
        "final_action": final_action,
        "near_events": near_events,
        "full_events": full_events,
        "warning_messages": warning_messages
    }


def make_cash_mdd_comment(current_dd, rsi, recovery_needed, total_score):
    rsi_valid = is_valid_number(rsi)
    rsi_value = float(rsi) if rsi_valid else None

    if recovery_needed <= 0.05 and rsi_valid and rsi_value >= 70 and total_score >= 2:
        return "MDD 회복 + RSI 과열 + 경고 높음: 현금확보 우선 구간입니다."

    if current_dd <= -0.12 and total_score <= 1:
        return "MDD 깊음 + 경고 낮음: 저점매수 가능 구간입니다. 단, 분할 접근이 우선입니다."

    if current_dd <= -0.12 and total_score >= 2:
        return "MDD 깊음 + 경고 높음: 가격 매력은 있으나 이벤트 리스크가 있어 소액만 가능합니다."

    if current_dd > -0.08 and total_score >= 2:
        return "MDD 얕음 + 경고 높음: 신규매수 금지에 가깝습니다. 이벤트 확인 후 판단하세요."

    if recovery_needed <= 0.05 and total_score >= 2:
        return "MDD 거의 회복 + 경고 높음: 일부 현금화 검토 구간입니다."

    if total_score >= 4:
        return "경고 점수가 높습니다. 신규매수보다 현금확보와 리스크 관리가 우선입니다."

    return "MDD와 이벤트 리스크가 극단 구간은 아닙니다. 기존 Buy Score와 시장 필터를 함께 확인하세요."



# =========================================================
# Korean valuation trend - KRX PER/PBR/EPS
# =========================================================
@st.cache_data(ttl=86400)
def load_kr_fundamental_by_date(ticker, start_date):
    """
    pykrx에서 한국 종목의 PER/PBR/EPS를 가져온다.

    중요:
    - pykrx 공식 사용 함수는 get_market_fundamental(start, end, ticker, freq="m")다.
    - get_market_fundamental_by_date()는 환경에 따라 없어서 Streamlit Cloud에서 실패할 수 있다.
    - 월말 기준 PER/PBR/EPS를 가져와 주가와 날짜 기준으로 병합한다.
    - Dataguide/FnGuide의 12개월 예상 PER은 아니고 KRX 실적 기반 PER이다.
    """
    if krx_stock is None:
        return pd.DataFrame()

    try:
        start = pd.Timestamp(start_date).strftime("%Y%m%d")
        end = datetime.today().strftime("%Y%m%d")

        # pykrx 공식 방식: 특정 종목 기간별 PER/PBR/EPS
        try:
            f = krx_stock.get_market_fundamental(start, end, ticker, freq="m")
        except TypeError:
            # 구버전 호환: freq 파라미터가 없으면 일별로 시도
            f = krx_stock.get_market_fundamental(start, end, ticker)

        if f is None or f.empty:
            return pd.DataFrame()

        f = f.copy()
        f.index = pd.to_datetime(f.index)
        f = f.reset_index()

        # 첫 컬럼명을 date로 통일
        first_col = f.columns[0]
        f = f.rename(columns={first_col: "date"})

        rename_map = {
            "PER": "per",
            "PBR": "pbr",
            "EPS": "eps",
            "BPS": "bps",
            "DIV": "div",
            "DPS": "dps",
        }
        f = f.rename(columns=rename_map)

        keep_cols = [c for c in ["date", "per", "pbr", "eps", "bps", "div", "dps"] if c in f.columns]
        f = f[keep_cols].copy()

        f["date"] = pd.to_datetime(f["date"], errors="coerce")
        f = f.dropna(subset=["date"])

        for col in ["per", "pbr", "eps", "bps", "div", "dps"]:
            if col in f.columns:
                f[col] = pd.to_numeric(f[col], errors="coerce")

        # 의미 없는 0 또는 극단값 제거
        if "per" in f.columns:
            f.loc[(f["per"] <= 0) | (f["per"] > 300), "per"] = pd.NA
        if "pbr" in f.columns:
            f.loc[(f["pbr"] <= 0) | (f["pbr"] > 30), "pbr"] = pd.NA

        return f.sort_values("date")

    except Exception:
        return pd.DataFrame()


def make_kr_price_valuation_df(ticker, start_date, price_df):
    """
    FinanceDataReader 주가와 pykrx PER/PBR/EPS를 날짜 기준으로 병합한다.
    """
    if price_df is None or price_df.empty:
        return pd.DataFrame()

    fundamental = load_kr_fundamental_by_date(ticker, start_date)
    if fundamental.empty:
        return pd.DataFrame()

    try:
        price = price_df.copy().reset_index()
        date_col = price.columns[0]
        price = price.rename(columns={date_col: "date", "Close": "price"})
        price["date"] = pd.to_datetime(price["date"], errors="coerce")
        price["price"] = pd.to_numeric(price["price"], errors="coerce")
        price = price.dropna(subset=["date", "price"]).sort_values("date")

        f = fundamental.copy()
        f["date"] = pd.to_datetime(f["date"], errors="coerce")
        f = f.dropna(subset=["date"]).sort_values("date")

        merged = pd.merge_asof(
            price[["date", "price"]],
            f,
            on="date",
            direction="backward"
        )

        for col in ["per", "pbr", "eps", "bps"]:
            if col in merged.columns:
                merged[col] = pd.to_numeric(merged[col], errors="coerce")

        if "per" in merged.columns:
            merged.loc[(merged["per"] <= 0) | (merged["per"] > 300), "per"] = pd.NA
        if "pbr" in merged.columns:
            merged.loc[(merged["pbr"] <= 0) | (merged["pbr"] > 30), "pbr"] = pd.NA

        return merged.sort_values("date")

    except Exception:
        return pd.DataFrame()


def make_kr_price_per_chart(kr_val_df, ticker):
    """
    한국 종목용 Price vs PER 차트.
    네가 보여준 이미지처럼 주가와 PER을 같은 날짜축에 보여준다.
    """
    if kr_val_df is None or kr_val_df.empty:
        return None, pd.DataFrame()

    needed = {"date", "price", "per"}
    if not needed.issubset(set(kr_val_df.columns)):
        return None, pd.DataFrame()

    chart_df = kr_val_df.copy()
    chart_df["date"] = pd.to_datetime(chart_df["date"], errors="coerce")
    chart_df["price"] = pd.to_numeric(chart_df["price"], errors="coerce")
    chart_df["per"] = pd.to_numeric(chart_df["per"], errors="coerce")
    chart_df = chart_df.dropna(subset=["date", "price", "per"]).copy()
    chart_df = chart_df[(chart_df["price"] > 0) & (chart_df["per"] > 0) & (chart_df["per"] < 300)].copy()
    chart_df = chart_df.sort_values("date")

    if chart_df.empty or len(chart_df) < 20:
        return None, chart_df

    # 너무 촘촘하면 보기 어려워 월말 기준으로 표시. 데이터가 적으면 주간으로 대체.
    plot_df = make_monthly_view(chart_df)
    if plot_df.empty or len(plot_df) < 6:
        plot_df = chart_df.copy()

    fig, ax_price = plt.subplots(figsize=(13, 5.8))

    ax_price.plot(plot_df["date"], plot_df["price"], label="Price", linewidth=2)
    ax_price.set_ylabel("Price")
    ax_price.grid(True, alpha=0.3)

    ax_per = ax_price.twinx()
    ax_per.plot(plot_df["date"], plot_df["per"], linestyle="--", label="PER", linewidth=2)
    ax_per.set_ylabel("PER")

    per_mean = plot_df["per"].mean()
    per_std = plot_df["per"].std()

    if is_valid_number(per_mean):
        ax_per.axhline(per_mean, linestyle="-", alpha=0.35, label="PER avg")

    if is_valid_number(per_std) and per_std > 0:
        ax_per.axhline(per_mean + per_std, linestyle=":", alpha=0.35, label="PER +1SD")
        ax_per.axhline(max(per_mean - per_std, 0), linestyle=":", alpha=0.35, label="PER -1SD")
        ax_per.axhline(per_mean + 2 * per_std, linestyle="-.", alpha=0.25, label="PER +2SD")
        ax_per.axhline(max(per_mean - 2 * per_std, 0), linestyle="-.", alpha=0.25, label="PER -2SD")

    ax_price.set_title(f"{ticker} Price vs PER")

    lines1, labels1 = ax_price.get_legend_handles_labels()
    lines2, labels2 = ax_per.get_legend_handles_labels()
    ax_price.legend(lines1 + lines2, labels1 + labels2, loc="best")

    plt.tight_layout()
    return fig, plot_df


def make_kr_price_pbr_chart(kr_val_df, ticker):
    if kr_val_df is None or kr_val_df.empty:
        return None
    if not {"date", "price", "pbr"}.issubset(set(kr_val_df.columns)):
        return None

    chart_df = kr_val_df.copy()
    chart_df["date"] = pd.to_datetime(chart_df["date"], errors="coerce")
    chart_df["price"] = pd.to_numeric(chart_df["price"], errors="coerce")
    chart_df["pbr"] = pd.to_numeric(chart_df["pbr"], errors="coerce")
    chart_df = chart_df.dropna(subset=["date", "price", "pbr"]).copy()
    chart_df = chart_df[(chart_df["price"] > 0) & (chart_df["pbr"] > 0) & (chart_df["pbr"] < 30)].copy()
    chart_df = chart_df.sort_values("date")

    if chart_df.empty or len(chart_df) < 20:
        return None

    plot_df = make_monthly_view(chart_df)
    if plot_df.empty or len(plot_df) < 6:
        plot_df = chart_df.copy()

    fig, ax_price = plt.subplots(figsize=(13, 5.2))
    ax_price.plot(plot_df["date"], plot_df["price"], label="Price", linewidth=2)
    ax_price.set_ylabel("Price")
    ax_price.grid(True, alpha=0.3)

    ax_pbr = ax_price.twinx()
    ax_pbr.plot(plot_df["date"], plot_df["pbr"], linestyle="--", label="PBR", linewidth=2)
    ax_pbr.set_ylabel("PBR")

    pbr_mean = plot_df["pbr"].mean()
    pbr_std = plot_df["pbr"].std()

    if is_valid_number(pbr_mean):
        ax_pbr.axhline(pbr_mean, linestyle="-", alpha=0.35, label="PBR avg")
    if is_valid_number(pbr_std) and pbr_std > 0:
        ax_pbr.axhline(pbr_mean + pbr_std, linestyle=":", alpha=0.35, label="PBR +1SD")
        ax_pbr.axhline(max(pbr_mean - pbr_std, 0), linestyle=":", alpha=0.35, label="PBR -1SD")

    ax_price.set_title(f"{ticker} Price vs PBR")

    lines1, labels1 = ax_price.get_legend_handles_labels()
    lines2, labels2 = ax_pbr.get_legend_handles_labels()
    ax_price.legend(lines1 + lines2, labels1 + labels2, loc="best")

    plt.tight_layout()
    return fig


def make_kr_eps_chart(kr_val_df, ticker):
    if kr_val_df is None or kr_val_df.empty:
        return None
    if not {"date", "eps"}.issubset(set(kr_val_df.columns)):
        return None

    chart_df = kr_val_df.copy()
    chart_df["date"] = pd.to_datetime(chart_df["date"], errors="coerce")
    chart_df["eps"] = pd.to_numeric(chart_df["eps"], errors="coerce")
    chart_df = chart_df.dropna(subset=["date", "eps"]).copy()
    chart_df = chart_df.sort_values("date")

    if chart_df.empty or len(chart_df) < 20:
        return None

    plot_df = make_monthly_view(chart_df)
    if plot_df.empty or len(plot_df) < 6:
        plot_df = chart_df.copy()

    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(plot_df["date"], plot_df["eps"], label="EPS", linewidth=2)
    ax.axhline(y=0, linestyle="--", alpha=0.4)
    ax.set_title(f"{ticker} EPS Trend")
    ax.set_ylabel("EPS")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    return fig


def make_kr_price_per_comment(kr_plot_df):
    if kr_plot_df is None or kr_plot_df.empty or not {"date", "price", "per"}.issubset(set(kr_plot_df.columns)):
        return "KRX PER 데이터를 가져오지 못했습니다. pykrx 설치 또는 종목코드를 확인하세요."

    df = kr_plot_df.dropna(subset=["date", "price", "per"]).copy()
    if df.empty or len(df) < 3:
        return "주가와 PER 방향성을 해석하기에는 데이터가 부족합니다."

    df = df.sort_values("date")
    last_date = pd.to_datetime(df["date"].iloc[-1])
    cutoff = last_date - pd.DateOffset(months=6)
    base = df[pd.to_datetime(df["date"]) >= cutoff]
    if len(base) < 3:
        base = df.tail(6)

    first_price = float(base["price"].iloc[0])
    last_price = float(base["price"].iloc[-1])
    first_per = float(base["per"].iloc[0])
    last_per = float(base["per"].iloc[-1])

    price_change = (last_price / first_price - 1) * 100 if first_price > 0 else None
    per_change = (last_per / first_per - 1) * 100 if first_per > 0 else None

    per_mean = df["per"].mean()
    per_std = df["per"].std()

    band_msg = ""
    if is_valid_number(per_mean) and is_valid_number(per_std) and per_std > 0:
        if last_per >= per_mean + 2 * per_std:
            band_msg = "현재 PER은 +2SD 이상으로 과열 구간에 가깝습니다."
        elif last_per >= per_mean + per_std:
            band_msg = "현재 PER은 +1SD 이상으로 밸류 부담이 있는 구간입니다."
        elif last_per <= per_mean - per_std:
            band_msg = "현재 PER은 -1SD 이하로 밸류 부담이 완화된 구간입니다."
        else:
            band_msg = "현재 PER은 평균권 구간입니다."

    if price_change is None or per_change is None:
        return band_msg if band_msg else "주가/PER 변화율 계산 불가"

    if price_change > 0 and per_change < 0:
        direction_msg = "최근 주가는 상승했지만 PER은 하락했습니다. 이익 개선이 주가 상승을 정당화하는 구간일 수 있습니다."
    elif price_change > 0 and per_change > 0:
        direction_msg = "최근 주가와 PER이 같이 상승했습니다. 기대감 선반영 또는 밸류 부담 확대 여부를 확인해야 합니다."
    elif price_change < 0 and per_change < 0:
        direction_msg = "최근 주가와 PER이 같이 하락했습니다. 밸류 부담은 완화됐지만 업황 훼손 여부를 확인해야 합니다."
    elif price_change < 0 and per_change > 0:
        direction_msg = "최근 주가는 하락했지만 PER은 상승했습니다. 이익 하향 또는 실적 둔화 가능성이 있어 저점매수 주의가 필요합니다."
    else:
        direction_msg = "최근 주가/PER 방향성이 뚜렷하지 않습니다."

    return f"최근 약 6개월 주가 변화 {price_change:.2f}%, PER 변화 {per_change:.2f}%. {direction_msg} {band_msg}"

# =========================================================
# Target Price
# =========================================================
def make_mdd_target_table(peak_price, current_price, profile):
    rows = []

    levels = [
        ("관심가", profile["watch"], "관심 구간"),
        ("1차 매수가", profile["buy1"], "1차 선진입 후보 기준"),
        ("2차 매수가", profile["buy2"], "2차 매수 후보 기준"),
        ("위험가", profile["risk"], "추세 훼손 주의 기준")
    ]

    for name, dd_level, memo in levels:
        target_price = peak_price * (1 + dd_level)
        gap_pct = (target_price / current_price - 1) * 100 if current_price > 0 else None
        status = "도달" if current_price <= target_price else "미도달"

        rows.append({
            "구분": name,
            "MDD 기준": f"{dd_level * 100:.2f}%",
            "가격": format_price(target_price),
            "현재가 대비": format_pct_value(gap_pct),
            "상태": status,
            "해석": memo
        })

    return pd.DataFrame(rows)


def make_valuation_target_table(current_price, valuation):
    rows = []

    forward_pe = valuation["forward_pe"]
    ps = valuation["price_to_sales"]

    if is_valid_number(forward_pe) and float(forward_pe) > 0:
        forward_pe = float(forward_pe)
        forward_eps = current_price / forward_pe

        for label, multiple, memo in [
            ("Forward P/E 15x", 15, "밸류 부담 낮은 기준"),
            ("Forward P/E 30x", 30, "성장주 보통 상단 기준"),
            ("Forward P/E 50x", 50, "고평가 경계 기준")
        ]:
            target_price = forward_eps * multiple
            gap_pct = (target_price / current_price - 1) * 100 if current_price > 0 else None

            rows.append({
                "기준": label,
                "목표 배수": f"{multiple}x",
                "참고 목표가": format_price(target_price),
                "현재가 대비": format_pct_value(gap_pct),
                "해석": memo
            })

    if is_valid_number(ps) and float(ps) > 0:
        ps = float(ps)

        for label, multiple, memo in [
            ("P/S 3x", 3, "매출 대비 부담 낮은 기준"),
            ("P/S 10x", 10, "성장주 보통~상단 기준"),
            ("P/S 30x", 30, "고성장 기대 과열 경계")
        ]:
            target_price = current_price * (multiple / ps)
            gap_pct = (target_price / current_price - 1) * 100 if current_price > 0 else None

            rows.append({
                "기준": label,
                "목표 배수": f"{multiple}x",
                "참고 목표가": format_price(target_price),
                "현재가 대비": format_pct_value(gap_pct),
                "해석": memo
            })

    if not rows:
        return pd.DataFrame({
            "기준": ["N/A"],
            "목표 배수": ["N/A"],
            "참고 목표가": ["N/A"],
            "현재가 대비": ["N/A"],
            "해석": ["Forward P/E 또는 P/S 데이터가 없어 밸류 기준 목표가 계산 불가"]
        })

    return pd.DataFrame(rows)


# =========================================================
# MDD 계산
# =========================================================
def calculate_rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_indicators(df):
    df = df.copy()

    if "Volume" not in df.columns:
        df["Volume"] = 0

    df["Peak"] = df["Close"].cummax()

    df["Current_Drawdown"] = df["Close"] / df["Peak"] - 1
    df["Max_Drawdown"] = df["Current_Drawdown"].cummin()
    df["Recovery_To_Peak"] = 1 / (1 + df["Current_Drawdown"]) - 1

    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA60"] = df["Close"].rolling(60).mean()
    df["MA200"] = df["Close"].rolling(200).mean()

    df["RSI"] = calculate_rsi(df["Close"], 14)

    df["Volume_MA20"] = df["Volume"].rolling(20).mean()
    df["Volume_Ratio"] = df["Volume"] / df["Volume_MA20"]

    df["Return"] = df["Close"].pct_change()

    df["Low20"] = df["Close"].rolling(20).min()
    df["Prev_Low20"] = df["Close"].shift(1).rolling(20).min()
    df["High20"] = df["Close"].rolling(20).max()

    df["BB_Mid"] = df["Close"].rolling(20).mean()
    df["BB_Std"] = df["Close"].rolling(20).std()
    df["BB_Lower"] = df["BB_Mid"] - 2 * df["BB_Std"]
    df["BB_Upper"] = df["BB_Mid"] + 2 * df["BB_Std"]

    return df


def get_type_profile(asset_type):
    profiles = {
        "일반 주식/ETF": {"watch": -0.08, "buy1": -0.12, "buy2": -0.15, "risk": -0.20},
        "나스닥형 ETF": {"watch": -0.06, "buy1": -0.08, "buy2": -0.12, "risk": -0.15},
        "반도체/메모리 ETF": {"watch": -0.10, "buy1": -0.12, "buy2": -0.15, "risk": -0.20},
        "전력/인프라 ETF": {"watch": -0.08, "buy1": -0.10, "buy2": -0.15, "risk": -0.18},
        "우주/소형 테마": {"watch": -0.15, "buy1": -0.20, "buy2": -0.25, "risk": -0.30}
    }

    return profiles.get(asset_type, profiles["일반 주식/ETF"])


def get_market_filter(start_date):
    benchmarks = {
        "QQQ": {"name": "Nasdaq", "risk_dd": -0.08},
        "SOXX": {"name": "Semiconductor", "risk_dd": -0.12},
        "NVDA": {"name": "NVIDIA", "risk_dd": -0.10},
        "MU": {"name": "Memory", "risk_dd": -0.12}
    }

    rows = []
    risk_points = 0

    for ticker, info in benchmarks.items():
        df_b = load_us_benchmark(ticker, start_date)

        if df_b.empty:
            rows.append({
                "Ticker": ticker,
                "Name": info["name"],
                "Close": None,
                "Current DD(%)": None,
                "MA5": None,
                "Status": "No Data"
            })
            continue

        df_b = calculate_indicators(df_b)
        latest = df_b.iloc[-1]

        close = latest["Close"]
        dd = latest["Current_Drawdown"]
        ma5 = latest["MA5"]

        status = "Good"

        if dd <= info["risk_dd"]:
            risk_points += 1
            status = "Risk"

        if pd.notna(ma5) and close < ma5:
            risk_points += 0.5
            if status != "Risk":
                status = "Caution"

        rows.append({
            "Ticker": ticker,
            "Name": info["name"],
            "Close": close,
            "Current DD(%)": dd * 100,
            "MA5": ma5,
            "Status": status
        })

    if risk_points >= 3:
        market_status = "Risk"
    elif risk_points >= 1.5:
        market_status = "Caution"
    else:
        market_status = "Good"

    return market_status, risk_points, pd.DataFrame(rows)


def get_dynamic_market_penalty(current_dd, market_status):
    if market_status == "Good":
        return 0

    if current_dd > -0.05:
        risk_penalty = -25
    elif current_dd > -0.08:
        risk_penalty = -20
    elif current_dd > -0.12:
        risk_penalty = -15
    elif current_dd > -0.15:
        risk_penalty = -10
    else:
        risk_penalty = -5

    if market_status == "Caution":
        return int(risk_penalty * 0.5)

    return risk_penalty


def calculate_buy_score(row, profile, market_status, prev_row=None):
    score = 0
    reasons = []
    danger_reasons = []
    confirm_conditions = []

    hard_stop = False
    entry_type = "대기"
    market_risk_override = "OFF"

    dd = row["Current_Drawdown"]
    rsi = row["RSI"]
    close = row["Close"]

    ma5 = row["MA5"]
    ma20 = row["MA20"]
    ma200 = row["MA200"]
    vol_ratio = row["Volume_Ratio"]
    ret = row["Return"]
    low20 = row["Low20"]
    prev_low20 = row["Prev_Low20"]
    bb_lower = row["BB_Lower"]

    watch_dd = profile["watch"]
    buy1_dd = profile["buy1"]
    buy2_dd = profile["buy2"]
    risk_dd = profile["risk"]

    if (
        pd.notna(prev_low20)
        and dd <= -0.20
        and close < prev_low20
        and (
            (pd.notna(vol_ratio) and vol_ratio >= 1.2 and pd.notna(ret) and ret < 0)
            or (pd.notna(ma20) and close < ma20)
        )
    ):
        hard_stop = True
        danger_reasons.append("Current DD -20% 이하 + 직전 20일 저점 이탈: 매수 금지 조건")
        danger_reasons.append("저점 이탈 구간에서는 단기 과매도가 아니라 추세 훼손 가능성 우선")

    if dd <= -0.20:
        score += 25
        reasons.append("Current DD -20% 이하: 매우 깊은 조정권")
        danger_reasons.append("Current DD -20% 이하: 추세 훼손 여부 확인 필요")
    elif dd <= -0.15:
        score += 40
        reasons.append("Current DD -15% 이하: 1차 선진입 후보 강화 구간")
    elif dd <= -0.12:
        score += 35
        reasons.append("Current DD -12~-15% 구간: 1차 소액 선진입 후보")
    elif dd <= -0.08:
        score += 25
        reasons.append("Current DD -8~-12% 구간: 관심 / 소액 후보")
    elif dd <= watch_dd:
        score += 15
        reasons.append("종목 유형 기준 Watch 구간 진입")

    if dd <= risk_dd:
        score += 5
        danger_reasons.append("종목 유형 기준 Risk 구간: 추가 하락 가능성 확인 필요")
    elif dd <= buy2_dd:
        score += 10
        reasons.append("종목 유형 기준 Buy 2 구간")
    elif dd <= buy1_dd:
        score += 8
        reasons.append("종목 유형 기준 Buy 1 구간")

    if pd.notna(rsi):
        if rsi <= 25:
            score += 25
            reasons.append("RSI 25 이하: 강한 과매도")
        elif rsi <= 30:
            score += 20
            reasons.append("RSI 30 이하: 과매도")
        elif rsi <= 40:
            score += 10
            reasons.append("RSI 40 이하: 약한 과매도")

    if prev_row is not None:
        prev_rsi = prev_row["RSI"]
        if pd.notna(prev_rsi) and pd.notna(rsi):
            if prev_rsi < 30 <= rsi:
                score += 15
                reasons.append("RSI 30 회복: 과매도 탈출 신호")
                confirm_conditions.append("RSI 30 recovery")

    if pd.notna(ma5) and close > ma5:
        score += 10
        reasons.append("종가 MA5 회복: 단기 반등 신호")
        confirm_conditions.append("Close above MA5")

    if pd.notna(ma20) and close > ma20:
        score += 15
        reasons.append("종가 MA20 회복: 반등 신뢰 상승")
        confirm_conditions.append("Close above MA20")

    if pd.notna(ma200):
        if close > ma200:
            score += 10
            reasons.append("MA200 위: 장기 추세 유지")
        elif close < ma200 * 0.90:
            score -= 20
            danger_reasons.append("MA200 대비 -10% 이상 이탈: 장기 추세 훼손 가능")

    if pd.notna(vol_ratio) and pd.notna(ret):
        if vol_ratio >= 1.5 and ret > 0:
            score += 15
            reasons.append("거래량 증가 양봉: 매수세 유입")
            confirm_conditions.append("High volume bullish candle")
        elif vol_ratio >= 1.5 and ret < 0:
            score -= 12
            danger_reasons.append("거래량 증가 음봉: 투매 또는 기관 매도 가능")

    if pd.notna(bb_lower):
        if close <= bb_lower:
            score += 10
            reasons.append("볼린저 하단 이하: 단기 과매도")

    if pd.notna(prev_low20):
        if close < prev_low20:
            score -= 15
            danger_reasons.append("직전 20일 저점 이탈: 추가 하락 주의")
        elif pd.notna(low20) and close <= low20 * 1.005:
            score -= 5
            danger_reasons.append("20일 저점 근처: 분할 진입만 가능")
        elif pd.notna(low20) and close >= low20 * 1.03:
            score += 10
            reasons.append("최근 저점 대비 3% 이상 회복")

    market_penalty = get_dynamic_market_penalty(dd, market_status)

    if market_penalty < 0:
        score += market_penalty
        danger_reasons.append(f"시장 필터 감점 {market_penalty}점: Current DD 구간별 완화 적용")

    if market_status == "Risk" and dd <= -0.12 and not hard_stop:
        market_risk_override = "ON"
        reasons.append("시장 Risk지만 Current DD가 깊어 소액 선진입 허용 가능")

    score = max(0, min(100, score))

    confirm_count = len(confirm_conditions)

    if hard_stop:
        decision = "매수 금지: 저점 이탈 또는 추세 훼손"
        entry_type = "금지"
    elif score >= 80 and confirm_count >= 2:
        decision = "2차 확인매수 후보"
        entry_type = "확인매수"
    elif dd <= -0.15 and score >= 60:
        decision = "1차 선진입 후보"
        entry_type = "선진입"
    elif dd <= -0.12 and score >= 55:
        decision = "1차 선진입 후보"
        entry_type = "선진입"
    elif score >= 65 and confirm_count >= 2:
        decision = "2차 확인매수 후보"
        entry_type = "확인매수"
    elif score >= 50 or dd <= -0.08:
        decision = "관심 / 소액 후보"
        entry_type = "대기"
    else:
        decision = "대기"
        entry_type = "대기"

    if not confirm_conditions:
        confirm_condition_text = "MA5 회복, RSI 30 회복, 거래량 증가 양봉 필요"
    else:
        confirm_condition_text = " / ".join(confirm_conditions)

    return (
        score,
        decision,
        reasons,
        danger_reasons,
        entry_type,
        confirm_condition_text,
        market_risk_override,
        market_penalty,
        hard_stop
    )


def apply_buy_score(df, profile, market_status):
    scores = []
    decisions = []
    reason_list = []
    danger_list = []
    entry_types = []
    confirm_condition_list = []
    market_override_list = []
    market_penalty_list = []
    hard_stop_list = []

    for i in range(len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i - 1] if i > 0 else None

        (
            score,
            decision,
            reasons,
            dangers,
            entry_type,
            confirm_condition,
            market_override,
            market_penalty,
            hard_stop
        ) = calculate_buy_score(row, profile, market_status, prev_row)

        scores.append(score)
        decisions.append(decision)
        reason_list.append(" / ".join(reasons))
        danger_list.append(" / ".join(dangers))
        entry_types.append(entry_type)
        confirm_condition_list.append(confirm_condition)
        market_override_list.append(market_override)
        market_penalty_list.append(market_penalty)
        hard_stop_list.append(hard_stop)

    df["Buy_Score"] = scores
    df["Decision"] = decisions
    df["Reasons"] = reason_list
    df["Danger_Reasons"] = danger_list
    df["Entry_Type"] = entry_types
    df["Confirm_Buy_Condition"] = confirm_condition_list
    df["Market_Risk_Override"] = market_override_list
    df["Market_Penalty"] = market_penalty_list
    df["Hard_Stop"] = hard_stop_list

    return df


def get_buy_ratio(decision, market_status, current_dd, buy_score, hard_stop=False):
    if hard_stop or "매수 금지" in decision:
        return 0.00

    if decision == "대기":
        return 0.00

    if "관심" in decision:
        if market_status == "Good":
            return 0.05
        if market_status == "Caution":
            if current_dd <= -0.08 and buy_score >= 50:
                return 0.05
            return 0.00
        if market_status == "Risk":
            return 0.00

    if "1차" in decision:
        if market_status == "Good":
            return 0.20
        if market_status == "Caution":
            return 0.10
        if market_status == "Risk":
            if current_dd <= -0.15 and buy_score >= 65:
                return 0.10
            if current_dd <= -0.12 and buy_score >= 60:
                return 0.05
            return 0.00

    if "2차" in decision:
        if market_status == "Good":
            return 0.30
        if market_status == "Caution":
            return 0.20
        if market_status == "Risk":
            if current_dd <= -0.15 and buy_score >= 75:
                return 0.15
            return 0.10

    return 0.00


def simulate_avg_price(current_price, current_qty, avg_price, buy_amount):
    if current_qty <= 0 or avg_price <= 0 or buy_amount <= 0:
        return None

    add_qty = buy_amount / current_price
    total_qty = current_qty + add_qty
    total_cost = current_qty * avg_price + buy_amount
    new_avg = total_cost / total_qty
    recovery_to_new_avg = new_avg / current_price - 1

    return add_qty, total_qty, new_avg, recovery_to_new_avg


def make_final_trade_view(decision, cash_result, matrix_decision, current_dd, rsi, ma5, current_price, vol_ratio):
    if cash_result["total_score"] >= 4:
        return "현금확보 우선", "Event Risk가 높아 신규매수보다 현금확보와 리스크 관리 우선"

    if "매수 금지" in decision:
        return "매수 금지", "MDD/추세 조건상 저점 이탈 또는 추세 훼손 가능성 우선"

    if current_dd <= -0.12 and cash_result["total_score"] <= 1 and matrix_decision in ["1차 가능", "강한 후보", "분할 가능"]:
        return "1차 소액 가능", "MDD가 깊고 이벤트 리스크가 낮으며 밸류 부담도 과도하지 않음"

    if current_dd <= -0.12 and cash_result["total_score"] >= 2:
        return "소액만 가능", "가격 매력은 있으나 이벤트 리스크가 있어 비중 확대 금지"

    if current_dd > -0.08 and cash_result["total_score"] >= 2:
        return "추격 금지", "낙폭이 얕고 이벤트 리스크가 높음"

    if pd.notna(ma5) and current_price > ma5 and pd.notna(rsi) and rsi >= 30:
        return "확인매수 후보", "MA5 회복과 RSI 회복이 확인됨"

    return "대기", "가격·수급·이벤트 조건 중 명확한 우위 부족"


# =========================================================
# Main screen
# =========================================================
col_a, col_b, col_c = st.columns(3)

with col_a:
    user_input = st.text_input("종목명 / 종목코드 / 미국 티커", value="삼성전자")

with col_b:
    start_date = st.date_input("기준 시작일", pd.to_datetime("2024-01-01"))

with col_c:
    asset_type = st.selectbox(
        "종목 유형",
        ["일반 주식/ETF", "나스닥형 ETF", "반도체/메모리 ETF", "전력/인프라 ETF", "우주/소형 테마"],
        index=0
    )

col_d, col_e, col_f = st.columns(3)

with col_d:
    planned_buy_amount = st.number_input("추가매수 예정금", value=1000000, step=100000)

with col_e:
    current_qty = st.number_input("현재 보유수량", value=0.0, step=1.0)

with col_f:
    avg_price = st.number_input("현재 평균단가", value=0.0, step=100.0)

run = st.button("분석 실행")

if run:
    market, ticker, display_name = find_ticker(user_input)

    if ticker is None:
        st.error("종목을 찾을 수 없습니다. 한국 종목은 종목명 또는 6자리 코드로 입력하세요. 예: 삼성전자 또는 005930")
        st.stop()

    with st.spinner("데이터 분석 중..."):
        profile = get_type_profile(asset_type)
        market_status, market_risk_points, market_df = get_market_filter(start_date)

        df = load_price_data(market, ticker, start_date)

        if df.empty:
            st.error("가격 데이터를 가져오지 못했습니다. 예: 삼성전자, 005930, SK하이닉스, 000660, NVDA")
            st.stop()

        df = calculate_indicators(df)
        df = apply_buy_score(df, profile, market_status)

        latest = df.iloc[-1]

        current_price = latest["Close"]
        peak_price = latest["Peak"]
        current_dd = latest["Current_Drawdown"]
        period_mdd = df["Max_Drawdown"].min()
        recovery_needed = latest["Recovery_To_Peak"]
        rsi = latest["RSI"]
        ma5 = latest["MA5"]
        vol_ratio = latest["Volume_Ratio"]
        buy_score = latest["Buy_Score"]
        decision = latest["Decision"]
        entry_type = latest["Entry_Type"]
        confirm_buy_condition = latest["Confirm_Buy_Condition"]
        market_risk_override = latest["Market_Risk_Override"]
        market_penalty = latest["Market_Penalty"]
        hard_stop = latest["Hard_Stop"]

        buy_ratio = get_buy_ratio(decision, market_status, current_dd, buy_score, hard_stop)
        recommended_buy_amount = planned_buy_amount * buy_ratio

        valuation = load_valuation_data(market, ticker)
        valuation_df = make_valuation_table(valuation)
        valuation_comment = make_mdd_valuation_comment(current_dd, valuation)
        valuation_summary = make_valuation_summary(valuation)
        etf_flag = is_etf_like(asset_type, display_name, ticker)

        mdd_target_df = make_mdd_target_table(peak_price, current_price, profile)
        valuation_target_df = make_valuation_target_table(current_price, valuation)

        matrix_df, mdd_zone, valuation_zone, matrix_decision = make_mdd_valuation_matrix(current_dd, valuation)

        pe_band_fig, pe_band_df = make_pe_band_chart(current_price, valuation, ticker)

        financial_trend = pd.DataFrame()
        financial_chart = None
        price_pe_df = pd.DataFrame()
        price_pe_chart = None
        price_pe_comment = ""

        kr_valuation_trend = pd.DataFrame()
        kr_price_per_chart = None
        kr_price_pbr_chart = None
        kr_eps_chart = None
        kr_price_per_comment = ""

        if market == "US":
            raw_financial_trend = load_financial_trend_data(ticker)
            financial_trend = add_price_to_financial_trend(raw_financial_trend, df)
            price_pe_chart, price_pe_df = make_price_pe_trend_chart(financial_trend, valuation, ticker)
            price_pe_comment = make_price_pe_comment(price_pe_df)
            financial_chart = make_financial_trend_chart(financial_trend, ticker)
        elif market == "KR":
            kr_valuation_trend = make_kr_price_valuation_df(ticker, start_date, df)
            kr_price_per_chart, kr_price_per_plot_df = make_kr_price_per_chart(kr_valuation_trend, ticker)
            kr_price_pbr_chart = make_kr_price_pbr_chart(kr_valuation_trend, ticker)
            kr_eps_chart = make_kr_eps_chart(kr_valuation_trend, ticker)
            kr_price_per_comment = make_kr_price_per_comment(kr_price_per_plot_df)
            price_pe_comment = kr_price_per_comment
        else:
            price_pe_comment = "P/E 추세 데이터를 표시할 수 없습니다."

        default_event_df = normalize_event_schedule(make_default_event_schedule())

        with st.expander("일정 CSV 업로드 / 전체 일정표 보기"):
            uploaded_csv = st.file_uploader(
                "일정 CSV 업로드",
                type=["csv"],
                help="컬럼은 date,event,category,market,impact,memo 형식이어야 합니다."
            )

            if uploaded_csv is not None:
                try:
                    event_df = pd.read_csv(uploaded_csv)
                    event_df = normalize_event_schedule(event_df)
                    st.success("업로드한 일정표를 사용합니다.")
                except Exception:
                    event_df = default_event_df
                    st.error("CSV 형식 오류로 기본 일정표를 사용합니다.")
            else:
                event_df = default_event_df
                st.info("CSV가 없으므로 기본 일정표를 사용합니다.")

            st.dataframe(event_df, use_container_width=True)

        st.markdown("### 돌발 리스크 수동 체크")
        m1, m2, m3, m4 = st.columns(4)

        manual_score = 0

        with m1:
            if st.checkbox("유가 급등"):
                manual_score += 1
        with m2:
            if st.checkbox("미국 10년물 금리 급등"):
                manual_score += 1
        with m3:
            if st.checkbox("좋은 뉴스에도 주가 반응 약함"):
                manual_score += 1
        with m4:
            if st.checkbox("주도주 둔화"):
                manual_score += 1

        cash_result = calculate_cash_warning(event_df, manual_score)
        cash_comment = make_cash_mdd_comment(current_dd, rsi, recovery_needed, cash_result["total_score"])

        final_action, final_reason = make_final_trade_view(
            decision,
            cash_result,
            matrix_decision,
            current_dd,
            rsi,
            ma5,
            current_price,
            vol_ratio
        )

        st.subheader(f"분석 대상: {display_name} / {ticker} / {market}")
        st.write(f"종목 유형: **{asset_type}**")
        st.write(
            f"시장 필터: **{market_status}** / "
            f"위험점수: **{market_risk_points:.1f}** / "
            f"현재 적용 감점: **{market_penalty}점**"
        )

        # =================================================
        # 1. 매매 상태판
        # =================================================
        st.markdown("## 1. 매매 상태판")

        if final_action in ["1차 소액 가능", "확인매수 후보"]:
            st.success(f"최종 행동: {final_action}")
        elif final_action in ["소액만 가능", "대기"]:
            st.info(f"최종 행동: {final_action}")
        elif final_action in ["추격 금지", "현금확보 우선"]:
            st.error(f"최종 행동: {final_action}")
        else:
            st.warning(f"최종 행동: {final_action}")

        st.write(f"핵심 이유: **{final_reason}**")

        s1, s2, s3, s4, s5, s6 = st.columns(6)
        s1.metric("Current DD", f"{current_dd * 100:.2f}%")
        s2.metric("RSI", "N/A" if pd.isna(rsi) else f"{rsi:.2f}")
        s3.metric("MA5 상태", "회복" if pd.notna(ma5) and current_price > ma5 else "미회복")
        s4.metric("거래량", "N/A" if pd.isna(vol_ratio) else f"{vol_ratio:.2f}x")
        s5.metric("Event Risk", f"{cash_result['total_score']}점")
        s6.metric("Buy Score", f"{buy_score:.0f}점")

        g1, g2, g3 = st.columns(3)
        with g1:
            st.caption("Buy Score")
            st.progress(min(max(int(buy_score), 0), 100))
        with g2:
            st.caption("RSI")
            st.progress(0 if pd.isna(rsi) else min(max(int(rsi), 0), 100))
        with g3:
            st.caption("Event Risk")
            st.progress(min(cash_result["total_score"], 5) / 5)

        # =================================================
        # 2. 권장 행동
        # =================================================
        st.markdown("## 2. 권장 행동")

        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Entry Type", entry_type)
        a2.metric("First Buy Ratio", f"{buy_ratio * 100:.0f}%")
        a3.metric("Market Override", market_risk_override)
        a4.metric("Cash Action", cash_result["final_action"])

        st.write(f"확인매수 조건: **{confirm_buy_condition}**")

        if recommended_buy_amount > 0:
            st.success(f"권장 추가매수: 예정금의 {buy_ratio * 100:.0f}% ≈ {recommended_buy_amount:,.0f}")
        else:
            st.warning("현재 권장 추가매수는 0원 또는 대기입니다.")

        # =================================================
        # 3. 현금확보 경고등
        # =================================================
        st.markdown("## 3. 현금확보 경고등")

        wc1, wc2, wc3, wc4 = st.columns(4)
        wc1.metric("현재 상태", cash_result["status"])
        wc2.metric("경고 점수", cash_result["total_score"])
        wc3.metric("자동 점수", cash_result["auto_score"])
        wc4.metric("수동 점수", cash_result["manual_score"])

        if cash_result["status"] == "유지":
            st.success(f"최종 판단: {cash_result['final_action']}")
        elif cash_result["status"] == "주의":
            st.warning(f"최종 판단: {cash_result['final_action']}")
        else:
            st.error(f"최종 판단: {cash_result['final_action']}")

        if cash_result["warning_messages"]:
            for msg in cash_result["warning_messages"]:
                st.error(msg)

        st.write(cash_comment)

        if not cash_result["near_events"].empty:
            nearest_event = cash_result["near_events"].iloc[0]
            st.info(
                f"가까운 이벤트: D-{nearest_event['D-Day']} / "
                f"{nearest_event['이벤트명']} / {nearest_event['중요도']}"
            )

        with st.expander("가까운 이벤트 표 보기"):
            if cash_result["near_events"].empty:
                st.success("앞으로 10영업일 이내 주요 일정이 없습니다.")
            else:
                st.dataframe(cash_result["near_events"], use_container_width=True)

        # =================================================
        # 4. MDD 기준 매수가 카드
        # =================================================
        st.markdown("## 4. MDD 기준 매수가")

        mt1, mt2, mt3, mt4 = st.columns(4)

        for idx, col in enumerate([mt1, mt2, mt3, mt4]):
            row = mdd_target_df.iloc[idx]
            with col:
                st.metric(row["구분"], row["가격"], row["상태"])

        with st.expander("MDD 기준가 상세 보기"):
            st.dataframe(mdd_target_df, use_container_width=True)

        # =================================================
        # 5. Valuation Summary + Matrix
        # =================================================
        st.markdown("## 5. Valuation Summary")

        st.info(valuation_summary)

        if etf_flag:
            st.warning("ETF는 자체 PER보다 구성종목 가중평균 밸류에이션이 중요합니다. 이 값은 참고용으로만 사용하세요.")

        if market != "US":
            st.warning("한국 종목은 yfinance 밸류에이션 데이터가 없거나 부정확할 수 있습니다. 값이 없으면 N/A로 표시합니다.")

        v1, v2, v3, v4 = st.columns(4)
        v1.metric("Forward P/E", format_valuation_value(valuation["forward_pe"]))
        v2.metric("P/S", format_valuation_value(valuation["price_to_sales"]))
        v3.metric("PEG", format_valuation_value(valuation["peg_ratio"]))
        v4.metric("Matrix", matrix_decision)

        st.markdown("### MDD + Valuation Matrix")
        st.dataframe(matrix_df, use_container_width=True)
        st.write(f"현재 구간: **{mdd_zone} / {valuation_zone} → {matrix_decision}**")
        st.write(valuation_comment)

        # =================================================
        # 6. Price + P/E Trend
        # =================================================
        st.markdown("## 6. Price + P/E Trend")

        st.info(
            "주가와 P/E를 같이 봅니다. "
            "주가가 올라도 P/E가 낮아지면 실적 개선이 주가 상승을 정당화하는 구간일 수 있습니다. "
            "반대로 주가가 빠졌는데 P/E가 높아지면 이익 악화 가능성을 확인해야 합니다."
        )

        if market == "KR":
            st.caption("한국 종목은 pykrx의 KRX 월말 PER/PBR/EPS 데이터를 사용합니다. Dataguide식 12개월 예상 PER은 아니지만, 주가와 PER 방향성을 함께 보는 용도입니다.")
            if krx_stock is None:
                st.error("pykrx가 설치되지 않았습니다. requirements.txt에 pykrx를 추가해야 합니다.")
            elif kr_price_per_chart is None:
                st.warning(kr_price_per_comment)
                if not kr_valuation_trend.empty:
                    with st.expander("KRX PER/PBR/EPS 원자료 보기"):
                        show_kr_df = kr_valuation_trend.copy()
                        show_kr_df["date"] = pd.to_datetime(show_kr_df["date"]).dt.strftime("%Y-%m-%d")
                        for col in ["price", "per", "pbr", "eps", "bps"]:
                            if col in show_kr_df.columns:
                                show_kr_df[col] = show_kr_df[col].apply(lambda x: None if pd.isna(x) else round(float(x), 2))
                        cols = [c for c in ["date", "price", "per", "pbr", "eps", "bps"] if c in show_kr_df.columns]
                        st.dataframe(show_kr_df[cols].tail(120), use_container_width=True)
            else:
                st.pyplot(kr_price_per_chart)
                st.write(kr_price_per_comment)

                with st.expander("Price vs PBR / EPS 추가 차트 보기"):
                    if kr_price_pbr_chart is not None:
                        st.pyplot(kr_price_pbr_chart)
                    else:
                        st.info("PBR 차트를 표시할 데이터가 부족합니다.")

                    if kr_eps_chart is not None:
                        st.pyplot(kr_eps_chart)
                    else:
                        st.info("EPS 차트를 표시할 데이터가 부족합니다.")

                with st.expander("KRX PER/PBR/EPS 데이터 보기"):
                    show_kr_df = kr_valuation_trend.copy()
                    show_kr_df["date"] = pd.to_datetime(show_kr_df["date"]).dt.strftime("%Y-%m-%d")
                    for col in ["price", "per", "pbr", "eps", "bps"]:
                        if col in show_kr_df.columns:
                            show_kr_df[col] = show_kr_df[col].apply(lambda x: None if pd.isna(x) else round(float(x), 2))
                    cols = [c for c in ["date", "price", "per", "pbr", "eps", "bps"] if c in show_kr_df.columns]
                    st.dataframe(show_kr_df[cols].tail(120), use_container_width=True)

        elif market == "US":
            st.caption("미국 종목은 yfinance 재무제표 기반 Estimated TTM P/E를 사용합니다. 한국 종목처럼 상단 Price / 하단 P/E로 분리 표시합니다. FactSet식 12개월 Forward P/E 시계열은 아닙니다.")
            if price_pe_chart is None:
                st.warning(price_pe_comment)
                if not price_pe_df.empty:
                    st.caption("아래는 yfinance에서 계산 가능한 원자료입니다. 점이 너무 적으면 추세 차트로 쓰지 않습니다.")
                    show_price_pe_df = price_pe_df.copy()
                    show_price_pe_df["date"] = pd.to_datetime(show_price_pe_df["date"]).dt.strftime("%Y-%m-%d")
                    for col in ["price", "eps_ttm", "pe_ttm"]:
                        if col in show_price_pe_df.columns:
                            show_price_pe_df[col] = show_price_pe_df[col].apply(lambda x: None if pd.isna(x) else round(float(x), 2))
                    with st.expander("Price + P/E 원자료 보기"):
                        st.dataframe(show_price_pe_df[["date", "price", "eps_ttm", "pe_ttm"]], use_container_width=True)
            else:
                st.pyplot(price_pe_chart)
                st.write(price_pe_comment)

                show_price_pe_df = price_pe_df.copy()
                show_price_pe_df["date"] = pd.to_datetime(show_price_pe_df["date"]).dt.strftime("%Y-%m-%d")
                for col in ["price", "eps_ttm", "pe_ttm"]:
                    if col in show_price_pe_df.columns:
                        show_price_pe_df[col] = show_price_pe_df[col].apply(lambda x: None if pd.isna(x) else round(float(x), 2))

                with st.expander("Price + P/E 데이터 보기"):
                    st.dataframe(show_price_pe_df[["date", "price", "eps_ttm", "pe_ttm"]], use_container_width=True)
        else:
            st.info("P/E 추세 차트를 표시할 수 없습니다.")

        with st.expander("Forward P/E Band Chart 보기"):
            if pe_band_fig is None:
                st.info("Forward P/E 데이터가 없어 P/E Band Chart를 표시할 수 없습니다.")
            else:
                st.pyplot(pe_band_fig)
                st.dataframe(pe_band_df, use_container_width=True)

        # =================================================
        # 7. 차트
        # =================================================
        st.markdown("## 7. Price / MDD Chart")

        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        axes[0].plot(df.index, df["Close"], label="Close", color="black")
        axes[0].plot(df.index, df["Peak"], label="Peak", color="blue", linestyle="--", alpha=0.7)
        axes[0].plot(df.index, df["MA20"], label="MA20", color="orange", alpha=0.8)
        axes[0].plot(df.index, df["MA60"], label="MA60", color="green", alpha=0.8)
        axes[0].plot(df.index, df["MA200"], label="MA200", color="purple", alpha=0.8)
        axes[0].scatter(df.index[-1], df["Close"].iloc[-1], color="red", s=120, label="Today")

        first_buy_points = df[df["Entry_Type"] == "선진입"]
        confirm_buy_points = df[df["Entry_Type"] == "확인매수"]

        axes[0].scatter(first_buy_points.index, first_buy_points["Close"], color="lime", marker="*", s=150, label="Early Entry")
        axes[0].scatter(confirm_buy_points.index, confirm_buy_points["Close"], color="gold", marker="^", s=120, label="Confirm Buy")

        axes[0].set_title(f"{ticker} Price / Moving Averages")
        axes[0].set_ylabel("Price")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(df.index, df["Current_Drawdown"] * 100, color="red", label="Current DD")
        axes[1].plot(df.index, df["Max_Drawdown"] * 100, color="darkred", linestyle="--", alpha=0.7, label="Max DD")
        axes[1].axhline(y=-8, color="gray", linestyle="--", alpha=0.6, label="-8% Watch")
        axes[1].axhline(y=-12, color="green", linestyle="--", alpha=0.8, label="-12% Early Entry")
        axes[1].axhline(y=-15, color="orange", linestyle="--", alpha=0.8, label="-15% Strong Entry")
        axes[1].axhline(y=-20, color="red", linestyle="--", alpha=0.8, label="-20% Hard Risk")
        axes[1].set_title("Current Drawdown / Max Drawdown")
        axes[1].set_ylabel("Drawdown (%)")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)

        # =================================================
        # 8. 상세 정보
        # =================================================
        with st.expander("Price vs EPS/Revenue Trend 보기"):
            if market != "US":
                st.info("한국 종목은 yfinance 재무제표 추세 데이터가 제한적이라 표시하지 않습니다.")
            elif financial_chart is None:
                st.info("재무제표 추세 데이터를 가져오지 못했습니다.")
            else:
                st.pyplot(financial_chart)
                st.dataframe(financial_trend, use_container_width=True)

        with st.expander("Valuation 상세표 보기"):
            st.dataframe(valuation_df, use_container_width=True)

        with st.expander("Valuation 기준 목표가 보기"):
            st.dataframe(valuation_target_df, use_container_width=True)

        with st.expander("Buy Score 차트 보기"):
            fig_score, ax_score = plt.subplots(figsize=(14, 4))
            ax_score.plot(df.index, df["Buy_Score"], color="darkgreen", label="Buy Score")
            ax_score.axhline(y=50, color="gray", linestyle="--", alpha=0.6, label="Watch")
            ax_score.axhline(y=65, color="green", linestyle="--", alpha=0.8, label="Early Entry")
            ax_score.axhline(y=80, color="orange", linestyle="--", alpha=0.8, label="Confirm Buy")
            ax_score.set_title("Buy Score")
            ax_score.set_ylabel("Score")
            ax_score.set_xlabel("Date")
            ax_score.legend()
            ax_score.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig_score)

        with st.expander("최근 20거래일 데이터 보기"):
            view_cols = [
                "Close", "Peak", "Current_Drawdown", "Max_Drawdown",
                "Recovery_To_Peak", "RSI", "Volume_Ratio",
                "Market_Penalty", "Buy_Score", "Entry_Type",
                "Decision", "Confirm_Buy_Condition", "Market_Risk_Override"
            ]

            show_df = df[view_cols].tail(20).copy()
            show_df["Current_Drawdown"] = show_df["Current_Drawdown"] * 100
            show_df["Max_Drawdown"] = show_df["Max_Drawdown"] * 100
            show_df["Recovery_To_Peak"] = show_df["Recovery_To_Peak"] * 100

            show_df = show_df.rename(columns={
                "Close": "Close(종가)",
                "Peak": "Peak(기간고점)",
                "Current_Drawdown": "Current DD(현재낙폭%)",
                "Max_Drawdown": "Max DD(최대낙폭%)",
                "Recovery_To_Peak": "Recovery(회복필요%)",
                "RSI": "RSI(과매수/과매도)",
                "Volume_Ratio": "Vol Ratio(거래량비율)",
                "Market_Penalty": "Market Penalty(시장감점)",
                "Buy_Score": "Buy Score(매수점수)",
                "Entry_Type": "Entry Type(진입유형)",
                "Decision": "Decision(판단)",
                "Confirm_Buy_Condition": "Confirm Condition(확인조건)",
                "Market_Risk_Override": "Market Override(시장위험보정)"
            })

            show_df.index.name = "Date(날짜)"
            st.dataframe(show_df, use_container_width=True)

        with st.expander("시장 필터 보기"):
            show_market_df = market_df.copy()

            if not show_market_df.empty:
                show_market_df["Close"] = show_market_df["Close"].apply(lambda x: None if pd.isna(x) else round(x, 2))
                show_market_df["Current DD(%)"] = show_market_df["Current DD(%)"].apply(lambda x: None if pd.isna(x) else round(x, 2))
                show_market_df["MA5"] = show_market_df["MA5"].apply(lambda x: None if pd.isna(x) else round(x, 2))
                st.dataframe(show_market_df, use_container_width=True)

        with st.expander("해석 기준 보기"):
            guide_df = pd.DataFrame({
                "항목": [
                    "Current DD",
                    "RSI",
                    "MA5 상태",
                    "Volume Ratio",
                    "Event Risk",
                    "Valuation Matrix",
                    "P/E Band",
                    "Price vs Fundamentals"
                ],
                "매매 활용": [
                    "낙폭이 충분한지 확인",
                    "과매도/과열 확인",
                    "선진입과 확인매수 구분",
                    "반등 신뢰도 확인",
                    "추가매수 중단·현금확보 판단",
                    "MDD와 밸류 부담 결합 판단",
                    "현재 주가가 밸류 밴드상 어디인지 확인",
                    "주가 상승이 실적 성장으로 정당화되는지 확인"
                ],
                "주의점": [
                    "낙폭만으로 매수 금지",
                    "RSI 과매도는 더 빠질 수 있음",
                    "장중 회복보다 종가 확인 우선",
                    "거래량 증가 음봉은 위험",
                    "Buy Score에 직접 반영하지 않음",
                    "밸류 데이터 없으면 판단 제한",
                    "Forward P/E 기반 단순 환산",
                    "yfinance 재무 데이터 누락 가능"
                ]
            })
            st.table(guide_df)

        st.warning(
            "주의: 이 도구는 매수 판단 보조용입니다. "
            "Valuation, P/E Band, 현금확보 경고등은 Buy Score를 바꾸지 않습니다. "
            "주가가 올랐더라도 실적 성장으로 밸류가 낮아질 수 있고, 반대로 낙폭이 커도 밸류 부담이 여전히 클 수 있습니다."
        )
