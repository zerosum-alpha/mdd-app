import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import FinanceDataReader as fdr
from datetime import datetime, timedelta
from auth import require_login, logout_button

# =========================================================
# MDD 저점매수 분석기 FINAL
# + Valuation
# + Target Price
# + Cash Warning Light
#
# 핵심:
# - 기존 MDD / Buy Score 계산 로직 유지
# - 현금확보 경고등은 Buy Score에 반영하지 않음
# - 일정표 기반 자동 경고 + 돌발 리스크 수동 체크 4개만 반영
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


def make_cash_mdd_comment(current_dd, rsi, recovery_needed, cash_status, total_score):
    rsi_valid = is_valid_number(rsi)
    rsi_value = float(rsi) if rsi_valid else None

    if current_dd <= -0.12 and total_score <= 1:
        return "MDD 깊음 + 경고 낮음: 저점매수 가능 구간입니다. 단, 분할 접근이 우선입니다."

    if current_dd <= -0.12 and total_score >= 2:
        return "MDD 깊음 + 경고 높음: 가격 매력은 있으나 이벤트 리스크가 있어 소액만 가능합니다."

    if current_dd > -0.08 and total_score >= 2:
        return "MDD 얕음 + 경고 높음: 신규매수 금지에 가깝습니다. 이벤트 확인 후 판단하세요."

    if recovery_needed <= 0.05 and total_score >= 2:
        return "MDD 거의 회복 + 경고 높음: 일부 현금화 검토 구간입니다."

    if recovery_needed <= 0.05 and rsi_valid and rsi_value >= 70 and total_score >= 2:
        return "MDD 회복 + RSI 과열 + 경고 높음: 현금확보 우선 구간입니다."

    if total_score >= 4:
        return "경고 점수가 높습니다. 신규매수보다 현금확보와 리스크 관리가 우선입니다."

    return "MDD와 이벤트 리스크가 극단 구간은 아닙니다. 기존 Buy Score와 시장 필터를 함께 확인하세요."


# =========================================================
# Target Price
# =========================================================
def make_mdd_target_table(peak_price, current_price, profile):
    rows = []

    levels = [
        ("Watch 기준가", profile["watch"], "관심 구간"),
        ("Buy 1 기준가", profile["buy1"], "1차 선진입 후보 기준"),
        ("Buy 2 기준가", profile["buy2"], "2차 매수 후보 기준"),
        ("Risk 기준가", profile["risk"], "추세 훼손 주의 기준")
    ]

    for name, dd_level, memo in levels:
        target_price = peak_price * (1 + dd_level)
        gap_pct = (target_price / current_price - 1) * 100 if current_price > 0 else None
        status = "이미 해당 MDD 구간 도달" if current_price <= target_price else "추가 하락 시 도달"

        rows.append({
            "구분": name,
            "MDD 기준": f"{dd_level * 100:.2f}%",
            "목표가": format_price(target_price),
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


def make_target_comment(mdd_target_df, valuation_target_df):
    comments = []

    buy1_row = mdd_target_df[mdd_target_df["구분"] == "Buy 1 기준가"]
    buy2_row = mdd_target_df[mdd_target_df["구분"] == "Buy 2 기준가"]

    if not buy1_row.empty:
        comments.append(f"MDD 기준 1차 관심가는 **{buy1_row.iloc[0]['목표가']}** 입니다.")

    if not buy2_row.empty:
        comments.append(f"MDD 기준 2차 관심가는 **{buy2_row.iloc[0]['목표가']}** 입니다.")

    if not valuation_target_df.empty and valuation_target_df.iloc[0]["참고 목표가"] != "N/A":
        comments.append("Valuation 기준 목표가는 Forward P/E 또는 P/S를 단순 환산한 참고값입니다.")
    else:
        comments.append("밸류 데이터가 없어 Valuation 기준 목표가는 계산하지 않았습니다.")

    comments.append("목표가는 자동 매수 가격이 아니라 MDD·밸류 부담을 비교하기 위한 참고선입니다.")

    return " ".join(comments)


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
        etf_flag = is_etf_like(asset_type, display_name, ticker)

        mdd_target_df = make_mdd_target_table(peak_price, current_price, profile)
        valuation_target_df = make_valuation_target_table(current_price, valuation)
        target_comment = make_target_comment(mdd_target_df, valuation_target_df)

        st.subheader(f"분석 대상: {display_name} / {ticker} / {market}")
        st.write(f"종목 유형: **{asset_type}**")
        st.write(
            f"시장 필터: **{market_status}** / "
            f"위험점수: **{market_risk_points:.1f}** / "
            f"현재 적용 감점: **{market_penalty}점**"
        )

        # 1. 핵심 지표 카드
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("현재가", f"{current_price:,.2f}")
        c2.metric("기간 고점", f"{peak_price:,.2f}")
        c3.metric("현재 낙폭", f"{current_dd * 100:.2f}%")
        c4.metric("최대 낙폭", f"{period_mdd * 100:.2f}%")
        c5.metric("회복 필요", f"{recovery_needed * 100:.2f}%")
        c6.metric("매수 점수", f"{buy_score:.0f}점")

        # 2. 최종 판단
        st.markdown("## 최종 판단")

        if "매수 금지" in decision:
            st.error(f"🚫 {decision}")
        elif "2차" in decision:
            st.warning(f"🟠 {decision}")
        elif "1차" in decision:
            st.success(f"🟢 {decision}")
        elif "관심" in decision:
            st.info(f"🔵 {decision}")
        else:
            st.info(f"⚪ {decision}")

        # 3. 권장 행동
        st.markdown("## 권장 행동")

        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Entry Type", entry_type)
        e2.metric("First Buy Ratio", f"{buy_ratio * 100:.0f}%")
        e3.metric("Market Override", market_risk_override)
        e4.metric("Market Penalty", f"{market_penalty}점")

        st.write(f"확인매수 조건: **{confirm_buy_condition}**")

        if recommended_buy_amount > 0:
            st.success(f"권장 추가매수: 예정금의 {buy_ratio * 100:.0f}% ≈ {recommended_buy_amount:,.0f}")
        else:
            st.warning("현재 권장 추가매수는 0원 또는 대기입니다.")

        # 4. Valuation
        st.markdown("## Valuation(밸류에이션)")

        st.info("밸류에이션은 매수 신호가 아니라 참고용 보조 필터입니다. Buy Score 계산에는 반영하지 않습니다.")

        if etf_flag:
            st.warning("ETF는 자체 PER보다 구성종목 가중평균 밸류에이션이 중요합니다. 이 값은 참고용으로만 사용하세요.")

        if market != "US":
            st.warning("한국 종목은 yfinance 밸류에이션 데이터가 없거나 부정확할 수 있습니다. 값이 없으면 N/A로 표시합니다.")

        st.dataframe(valuation_df, use_container_width=True)

        st.markdown("### MDD + Valuation 참고 해석")
        st.write(valuation_comment)

        # 5. Target Price
        st.markdown("## Target Price(목표가 참고선)")
        st.info("목표가는 자동 매수/매도 가격이 아닙니다. MDD 기준 가격 부담과 Valuation 기준 가격 부담을 비교하기 위한 참고선입니다.")

        st.markdown("### MDD 기준 매수 목표가")
        st.dataframe(mdd_target_df, use_container_width=True)

        st.markdown("### Valuation 기준 참고 목표가")
        st.dataframe(valuation_target_df, use_container_width=True)

        st.markdown("### 목표가 참고 해석")
        st.write(target_comment)

        # 6. 현금확보 경고등
        st.markdown("## 🚦 현금확보 경고등")

        default_event_df = make_default_event_schedule()

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
                    event_df = normalize_event_schedule(default_event_df)
                    st.error("CSV 형식 오류로 기본 일정표를 사용합니다.")
            else:
                event_df = normalize_event_schedule(default_event_df)
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
        cash_comment = make_cash_mdd_comment(
            current_dd,
            rsi,
            recovery_needed,
            cash_result["status"],
            cash_result["total_score"]
        )

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

        st.markdown("### MDD + 현금확보 경고등 참고 해석")
        st.write(cash_comment)

        st.markdown("### 가까운 이벤트")
        if cash_result["near_events"].empty:
            st.success("앞으로 10영업일 이내 주요 일정이 없습니다.")
        else:
            st.dataframe(cash_result["near_events"], use_container_width=True)

        # 7. 물타기 후 평단
        st.markdown("## 물타기 후 평단 시뮬레이션")

        sim = simulate_avg_price(current_price, current_qty, avg_price, recommended_buy_amount)

        if sim is None:
            st.info("보유수량과 평균단가를 입력하면 물타기 후 평단이 계산됩니다.")
        else:
            add_qty, total_qty, new_avg, recovery_to_new_avg = sim
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("추가매수 수량", f"{add_qty:,.2f}")
            s2.metric("총 보유수량", f"{total_qty:,.2f}")
            s3.metric("새 평균단가", f"{new_avg:,.2f}")
            s4.metric("새 평단 회복 필요", f"{recovery_to_new_avg * 100:.2f}%")

        # 8. 긍정/위험 신호
        if latest["Reasons"]:
            st.markdown("### 긍정 신호")
            for r in latest["Reasons"].split(" / "):
                st.write(f"- {r}")

        if latest["Danger_Reasons"]:
            st.markdown("### 위험 신호")
            for r in latest["Danger_Reasons"].split(" / "):
                st.write(f"- {r}")

        # 9. 시장 필터
        st.markdown("## 시장 필터: QQQ / SOXX / NVDA / MU")
        show_market_df = market_df.copy()

        if not show_market_df.empty:
            show_market_df["Close"] = show_market_df["Close"].apply(lambda x: None if pd.isna(x) else round(x, 2))
            show_market_df["Current DD(%)"] = show_market_df["Current DD(%)"].apply(lambda x: None if pd.isna(x) else round(x, 2))
            show_market_df["MA5"] = show_market_df["MA5"].apply(lambda x: None if pd.isna(x) else round(x, 2))
            st.dataframe(show_market_df, use_container_width=True)

        # 10. 차트
        fig, axes = plt.subplots(3, 1, figsize=(14, 13), sharex=True)

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

        axes[2].plot(df.index, df["Buy_Score"], color="darkgreen", label="Buy Score")
        axes[2].axhline(y=50, color="gray", linestyle="--", alpha=0.6, label="Watch")
        axes[2].axhline(y=65, color="green", linestyle="--", alpha=0.8, label="Early Entry")
        axes[2].axhline(y=80, color="orange", linestyle="--", alpha=0.8, label="Confirm Buy")
        axes[2].set_title("Buy Score")
        axes[2].set_ylabel("Score")
        axes[2].set_xlabel("Date")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)

        # 11. 최근 데이터
        st.markdown("## 최근 20거래일 데이터")

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

        # 12. 해석 기준
        st.markdown("## 해석 기준")

        guide_df = pd.DataFrame({
            "항목": [
                "Current DD",
                "Buy Score",
                "Valuation",
                "Target Price",
                "현금확보 경고등",
                "Cash Warning Score"
            ],
            "의미": [
                "기간 고점 대비 현재 낙폭",
                "MDD·RSI·이평·거래량·시장필터 기반 매수점수",
                "밸류 부담 참고",
                "MDD·밸류 기준 참고 가격",
                "다가오는 일정 기반 현금확보 필요성",
                "자동 일정 점수 + 수동 돌발 리스크 점수"
            ],
            "주의점": [
                "낙폭만으로 매수 판단 금지",
                "절대 매수 신호 아님",
                "Buy Score에 반영 안 됨",
                "자동 매수 가격 아님",
                "Buy Score에 반영 안 됨",
                "일정표 정확도에 따라 달라짐"
            ]
        })

        st.table(guide_df)

        st.warning(
            "주의: 이 도구는 매수 판단 보조용입니다. "
            "현금확보 경고등은 Buy Score를 바꾸지 않습니다. "
            "다가오는 이벤트·수급 일정·매크로 리스크를 확인해 추가매수 중단 또는 현금확보 필요성을 참고하는 용도입니다."
        )
