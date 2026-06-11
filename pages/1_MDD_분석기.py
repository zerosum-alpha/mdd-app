import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import FinanceDataReader as fdr
from auth import require_login, logout_button

# =========================================================
# MDD 저점매수 분석기 FINAL
# - 비밀번호는 auth.py + Streamlit Secrets에서 처리
# - 기존 MDD 계산 로직 유지
# - 시장 Risk 감점은 Current DD 구간별 완화
# - 1차 선진입 / 2차 확인매수 단계 분리
# =========================================================

st.set_page_config(page_title="MDD 분석기", layout="wide")

require_login()
logout_button()

st.title("📈 MDD 저점매수 분석기 FINAL")


# =========================
# Stock list
# =========================
@st.cache_data
def get_stock_list():
    try:
        return fdr.StockListing("KRX")
    except Exception:
        return pd.DataFrame()


stock_list = get_stock_list()


def find_ticker(query):
    query = str(query).strip()

    if query == "":
        return None, None, None

    if query.isdigit() and len(query) == 6:
        return "KR", query, query

    if not stock_list.empty and "Name" in stock_list.columns:
        match = stock_list[stock_list["Name"] == query]
        if not match.empty:
            code = match.iloc[0]["Code"]
            name = match.iloc[0]["Name"]
            return "KR", code, name

    return "US", query.upper(), query.upper()


# =========================
# Load data
# =========================
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


# =========================
# RSI
# =========================
def calculate_rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


# =========================
# Indicators
# =========================
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


# =========================
# Type profile
# =========================
def get_type_profile(asset_type):
    profiles = {
        "일반 주식/ETF": {
            "watch": -0.08,
            "buy1": -0.12,
            "buy2": -0.15,
            "risk": -0.20
        },
        "나스닥형 ETF": {
            "watch": -0.06,
            "buy1": -0.08,
            "buy2": -0.12,
            "risk": -0.15
        },
        "반도체/메모리 ETF": {
            "watch": -0.10,
            "buy1": -0.12,
            "buy2": -0.15,
            "risk": -0.20
        },
        "전력/인프라 ETF": {
            "watch": -0.08,
            "buy1": -0.10,
            "buy2": -0.15,
            "risk": -0.18
        },
        "우주/소형 테마": {
            "watch": -0.15,
            "buy1": -0.20,
            "buy2": -0.25,
            "risk": -0.30
        }
    }

    return profiles.get(asset_type, profiles["일반 주식/ETF"])


# =========================
# Market filter
# =========================
def get_market_filter(start_date):
    benchmarks = {
        "QQQ": {
            "name": "Nasdaq",
            "risk_dd": -0.08
        },
        "SOXX": {
            "name": "Semiconductor",
            "risk_dd": -0.12
        },
        "NVDA": {
            "name": "NVIDIA",
            "risk_dd": -0.10
        },
        "MU": {
            "name": "Memory",
            "risk_dd": -0.12
        }
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


# =========================
# Dynamic market penalty
# =========================
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


# =========================
# Buy score
# =========================
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
        danger_reasons.append(
            f"시장 필터 감점 {market_penalty}점: Current DD 구간별 완화 적용"
        )

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
        ) = calculate_buy_score(
            row,
            profile,
            market_status,
            prev_row
        )

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


# =========================
# Buy amount suggestion
# =========================
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


# =========================
# Main screen
# =========================
col_a, col_b, col_c = st.columns(3)

with col_a:
    user_input = st.text_input("종목명 / 종목코드 / 미국 티커", value="삼성전자")

with col_b:
    start_date = st.date_input("기준 시작일", pd.to_datetime("2024-01-01"))

with col_c:
    asset_type = st.selectbox(
        "종목 유형",
        [
            "일반 주식/ETF",
            "나스닥형 ETF",
            "반도체/메모리 ETF",
            "전력/인프라 ETF",
            "우주/소형 테마"
        ],
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
        st.error("종목을 찾을 수 없습니다.")
        st.stop()

    with st.spinner("데이터 분석 중..."):
        profile = get_type_profile(asset_type)

        market_status, market_risk_points, market_df = get_market_filter(start_date)

        df = load_price_data(market, ticker, start_date)

        if df.empty:
            st.error("가격 데이터를 가져오지 못했습니다. 종목명/코드/티커를 확인하세요.")
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

        buy_ratio = get_buy_ratio(
            decision,
            market_status,
            current_dd,
            buy_score,
            hard_stop
        )

        recommended_buy_amount = planned_buy_amount * buy_ratio

        st.subheader(f"분석 대상: {display_name} / {ticker} / {market}")
        st.write(f"종목 유형: **{asset_type}**")
        st.write(
            f"시장 필터: **{market_status}** / "
            f"위험점수: **{market_risk_points:.1f}** / "
            f"현재 적용 감점: **{market_penalty}점**"
        )

        c1, c2, c3, c4, c5, c6 = st.columns(6)

        c1.metric("현재가", f"{current_price:,.2f}")
        c2.metric("기간 고점", f"{peak_price:,.2f}")
        c3.metric("현재 낙폭", f"{current_dd * 100:.2f}%")
        c4.metric("최대 낙폭", f"{period_mdd * 100:.2f}%")
        c5.metric("회복 필요", f"{recovery_needed * 100:.2f}%")
        c6.metric("매수 점수", f"{buy_score:.0f}점")

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

        st.markdown("## 진입 유형")

        e1, e2, e3, e4 = st.columns(4)

        e1.metric("Entry Type(진입유형)", entry_type)
        e2.metric("First Buy Ratio(1차 매수비율)", f"{buy_ratio * 100:.0f}%")
        e3.metric("Market Risk Override", market_risk_override)
        e4.metric("Market Penalty", f"{market_penalty}점")

        st.write(f"확인매수 조건: **{confirm_buy_condition}**")

        st.markdown("## 권장 행동")

        if recommended_buy_amount > 0:
            st.success(
                f"권장 추가매수: 예정금의 {buy_ratio * 100:.0f}% "
                f"≈ {recommended_buy_amount:,.0f}"
            )
        else:
            if current_dd <= -0.12 and not hard_stop:
                st.info(
                    "현재 권장 추가매수는 0원이다. "
                    "다만 Current DD가 깊은 구간이므로 MA5 회복, RSI 30 회복, "
                    "거래량 증가 양봉 중 2개 이상 확인 시 2차 확인매수 후보로 전환될 수 있다."
                )
            else:
                st.warning("현재는 대기. 가격 매력 또는 반등 확인 부족.")

        if pd.notna(rsi):
            st.write(f"현재 RSI: **{rsi:.2f}**")
        else:
            st.write("현재 RSI: 계산 불가")

        st.markdown("## 물타기 후 평단 시뮬레이션")

        sim = simulate_avg_price(
            current_price,
            current_qty,
            avg_price,
            recommended_buy_amount
        )

        if sim is None:
            st.info("보유수량과 평균단가를 입력하면 물타기 후 평단이 계산된다.")
        else:
            add_qty, total_qty, new_avg, recovery_to_new_avg = sim

            s1, s2, s3, s4 = st.columns(4)
            s1.metric("추가매수 수량", f"{add_qty:,.2f}")
            s2.metric("총 보유수량", f"{total_qty:,.2f}")
            s3.metric("새 평균단가", f"{new_avg:,.2f}")
            s4.metric("새 평단 회복 필요", f"{recovery_to_new_avg * 100:.2f}%")

        if latest["Reasons"]:
            st.markdown("### 긍정 신호")
            for r in latest["Reasons"].split(" / "):
                st.write(f"- {r}")

        if latest["Danger_Reasons"]:
            st.markdown("### 위험 신호")
            for r in latest["Danger_Reasons"].split(" / "):
                st.write(f"- {r}")

        st.markdown("## 시장 필터: QQQ / SOXX / NVDA / MU")

        show_market_df = market_df.copy()

        if not show_market_df.empty:
            show_market_df["Close"] = show_market_df["Close"].apply(
                lambda x: None if pd.isna(x) else round(x, 2)
            )
            show_market_df["Current DD(%)"] = show_market_df["Current DD(%)"].apply(
                lambda x: None if pd.isna(x) else round(x, 2)
            )
            show_market_df["MA5"] = show_market_df["MA5"].apply(
                lambda x: None if pd.isna(x) else round(x, 2)
            )

            st.dataframe(show_market_df, use_container_width=True)

        fig, axes = plt.subplots(3, 1, figsize=(14, 13), sharex=True)

        axes[0].plot(df.index, df["Close"], label="Close", color="black")
        axes[0].plot(df.index, df["Peak"], label="Peak", color="blue", linestyle="--", alpha=0.7)
        axes[0].plot(df.index, df["MA20"], label="MA20", color="orange", alpha=0.8)
        axes[0].plot(df.index, df["MA60"], label="MA60", color="green", alpha=0.8)
        axes[0].plot(df.index, df["MA200"], label="MA200", color="purple", alpha=0.8)

        axes[0].scatter(df.index[-1], df["Close"].iloc[-1], color="red", s=120, label="Today")

        first_buy_points = df[df["Entry_Type"] == "선진입"]
        confirm_buy_points = df[df["Entry_Type"] == "확인매수"]

        axes[0].scatter(
            first_buy_points.index,
            first_buy_points["Close"],
            color="lime",
            marker="*",
            s=150,
            label="Early Entry"
        )

        axes[0].scatter(
            confirm_buy_points.index,
            confirm_buy_points["Close"],
            color="gold",
            marker="^",
            s=120,
            label="Confirm Buy"
        )

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

        st.markdown("## 최근 20거래일 데이터")

        view_cols = [
            "Close",
            "Peak",
            "Current_Drawdown",
            "Max_Drawdown",
            "Recovery_To_Peak",
            "RSI",
            "Volume_Ratio",
            "Market_Penalty",
            "Buy_Score",
            "Entry_Type",
            "Decision",
            "Confirm_Buy_Condition",
            "Market_Risk_Override"
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

        st.markdown("## 해석 기준")

        guide_df = pd.DataFrame({
            "항목": [
                "Current DD(현재낙폭%)",
                "Max DD(최대낙폭%)",
                "Recovery(회복필요%)",
                "RSI(과매수/과매도)",
                "Vol Ratio(거래량비율)",
                "Market Penalty(시장감점)",
                "Entry Type(진입유형)",
                "Confirm Condition(확인조건)",
                "Market Override(시장위험보정)",
                "Decision(판단)"
            ],
            "의미": [
                "현재가가 시작일 이후 기간고점 대비 몇 % 빠졌는지",
                "시작일 이후 가장 크게 빠졌던 최대 낙폭",
                "현재가에서 기간고점까지 회복하려면 필요한 상승률",
                "30 이하 과매도, 70 이상 과매수",
                "현재 거래량이 20일 평균 거래량 대비 몇 배인지",
                "QQQ/SOXX/NVDA/MU 상태에 따른 감점. Current DD가 깊을수록 감점 완화",
                "선진입 / 확인매수 / 대기 / 금지",
                "MA5 회복, RSI 30 회복, 거래량 증가 양봉 등 추가매수 확인 조건",
                "시장 Risk지만 Current DD가 깊어 소액 허용 여부",
                "대기 / 관심 / 1차 선진입 후보 / 2차 확인매수 후보 / 매수 금지"
            ],
            "활용": [
                "저점매수 판단 핵심값",
                "종목 위험도 참고",
                "회복 난이도 판단",
                "과매도 반등 가능성 확인",
                "투매인지 매수세 유입인지 확인",
                "시장 위험 속 저점매수 가능성 보정",
                "매수 단계를 구분",
                "2차 매수 전 확인해야 할 조건",
                "시장 Risk에서도 소액 선진입 가능한지 판단",
                "최종 행동 판단"
            ]
        })

        st.table(guide_df)

        score_df = pd.DataFrame({
            "구분": [
                "대기",
                "관심 / 소액 후보",
                "1차 선진입 후보",
                "2차 확인매수 후보",
                "매수 금지"
            ],
            "조건": [
                "가격 매력 또는 반등 확인 부족",
                "Current DD -8~-12% 또는 점수 50점 이상",
                "Current DD -12% 이하 + 점수 충족",
                "MA5 회복, RSI 30 회복, 거래량 양봉 등 확인",
                "Current DD -20% 이하 + 저점 이탈 또는 추세 훼손"
            ],
            "권장 행동": [
                "매수 없음",
                "관찰 또는 Good 시장에서 5%",
                "예정금의 5~20% 분할",
                "예정금의 10~30% 추가 분할",
                "매수 금지"
            ]
        })

        st.markdown("## 매수 단계 기준")
        st.table(score_df)

        penalty_df = pd.DataFrame({
            "Current DD 구간": [
                "0 ~ -5%",
                "-5 ~ -8%",
                "-8 ~ -12%",
                "-12 ~ -15%",
                "-15% 이하"
            ],
            "시장 Risk 감점": [
                "-25점",
                "-20점",
                "-15점",
                "-10점",
                "-5점"
            ],
            "의미": [
                "하락이 얕아 시장 위험을 강하게 반영",
                "아직 가격 매력 부족",
                "관심 구간, 감점 일부 완화",
                "1차 선진입 후보 구간",
                "깊은 과매도, 시장 감점 최소화"
            ]
        })

        st.markdown("## 시장 Risk 감점 기준")
        st.table(penalty_df)

        st.warning(
            "주의: 이 도구는 매수 판단 보조용이다. "
            "업황 훼손 여부는 코드가 직접 판별하지 못한다. "
            "Current DD가 깊어도 실적, 뉴스, 지수, 금리, 환율, 외국인 수급, 미국 선물 흐름과 함께 확인해야 한다."
        )
