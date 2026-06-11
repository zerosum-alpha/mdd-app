import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import FinanceDataReader as fdr

# =========================================================
# MDD 저점매수 분석기 FINAL
# 기능:
# 1. 한국 종목명 / 종목코드 / 미국 티커 입력
# 2. 시작일 이후 Current DD / Max DD / Recovery 계산
# 3. RSI / 이동평균 / 거래량 / 최근 저점 / 볼린저밴드 반영
# 4. QQQ / SOXX / NVDA / MU 시장 필터 반영
# 5. 종목 유형별 MDD 기준 차등 적용
# 6. 추가매수 예정금 기준 권장 매수금액 계산
# 7. 보유수량 / 평균단가 입력 시 물타기 후 평단 계산
# 8. 차트 내부는 영어, 화면 설명은 한글
# =========================================================

st.set_page_config(page_title="MDD 저점매수 분석기 FINAL", layout="wide")

# =========================
# Password
# =========================
MY_PASSWORD = "1234"

st.title("🔒 MDD 저점매수 분석기 FINAL")
entered_password = st.text_input("비밀번호를 입력하세요", type="password")

if entered_password != MY_PASSWORD:
    if entered_password:
        st.error("❌ 비밀번호 틀림")
    st.stop()


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

    # 한국 6자리 종목코드
    if query.isdigit() and len(query) == 6:
        return "KR", query, query

    # 한국 종목명
    if not stock_list.empty and "Name" in stock_list.columns:
        match = stock_list[stock_list["Name"] == query]
        if not match.empty:
            code = match.iloc[0]["Code"]
            name = match.iloc[0]["Name"]
            return "KR", code, name

    # 미국 티커
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
        penalty = -25
    elif risk_points >= 1.5:
        market_status = "Caution"
        penalty = -10
    else:
        market_status = "Good"
        penalty = 0

    return market_status, penalty, pd.DataFrame(rows)


# =========================
# Buy score
# =========================
def calculate_buy_score(row, profile, market_penalty, prev_row=None):
    score = 0
    reasons = []
    danger = False
    danger_reasons = []

    dd = row["Current_Drawdown"]
    rsi = row["RSI"]
    close = row["Close"]

    ma5 = row["MA5"]
    ma20 = row["MA20"]
    ma200 = row["MA200"]
    vol_ratio = row["Volume_Ratio"]
    ret = row["Return"]
    low20 = row["Low20"]
    bb_lower = row["BB_Lower"]

    watch_dd = profile["watch"]
    buy1_dd = profile["buy1"]
    buy2_dd = profile["buy2"]
    risk_dd = profile["risk"]

    # 1. Drawdown score
    if dd <= risk_dd:
        score += 20
        reasons.append("Current DD가 Risk 구간 이하: 가격은 깊은 조정권")
        danger = True
        danger_reasons.append("Current DD가 Risk 구간 이하: 추세 훼손 가능성")
    elif dd <= buy2_dd:
        score += 35
        reasons.append("Current DD가 Buy 2 구간: 강한 조정")
    elif dd <= buy1_dd:
        score += 30
        reasons.append("Current DD가 Buy 1 구간: 1차 매수 후보권")
    elif dd <= watch_dd:
        score += 20
        reasons.append("Current DD가 Watch 구간: 관심 구간")

    # 2. RSI score
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

    # RSI recovery
    if prev_row is not None:
        prev_rsi = prev_row["RSI"]
        if pd.notna(prev_rsi) and pd.notna(rsi):
            if prev_rsi < 30 <= rsi:
                score += 15
                reasons.append("RSI 30 회복: 과매도 탈출 신호")

    # 3. Moving average recovery
    if pd.notna(ma5) and close > ma5:
        score += 10
        reasons.append("종가 MA5 회복: 단기 반등 신호")

    if pd.notna(ma20) and close > ma20:
        score += 15
        reasons.append("종가 MA20 회복: 반등 신뢰 상승")

    # 4. Long-term trend filter
    if pd.notna(ma200):
        if close > ma200:
            score += 10
            reasons.append("MA200 위: 장기 추세 유지")
        elif close < ma200 * 0.90:
            score -= 25
            danger = True
            danger_reasons.append("MA200 대비 -10% 이상 이탈: 장기 추세 훼손 가능")

    # 5. Volume filter
    if pd.notna(vol_ratio):
        if vol_ratio >= 1.5 and ret > 0:
            score += 15
            reasons.append("거래량 증가 양봉: 매수세 유입")
        elif vol_ratio >= 1.5 and ret < 0:
            score -= 15
            danger_reasons.append("거래량 증가 음봉: 투매 또는 기관 매도 가능")

    # 6. Bollinger lower band
    if pd.notna(bb_lower):
        if close <= bb_lower:
            score += 10
            reasons.append("볼린저 하단 이하: 단기 과매도")

    # 7. Recent low filter
    if pd.notna(low20):
        if close <= low20 * 1.005:
            score -= 20
            danger = True
            danger_reasons.append("20일 저점 근처 또는 이탈: 추가 하락 주의")
        elif close >= low20 * 1.03:
            score += 10
            reasons.append("최근 저점 대비 3% 이상 회복")

    # 8. Market filter penalty
    if market_penalty < 0:
        score += market_penalty
        danger_reasons.append(f"시장 필터 감점 {market_penalty}점: QQQ/SOXX/NVDA/MU 상태 불안")

    score = max(0, min(100, score))

    if danger and score < 70:
        decision = "매수 금지 / 추세 확인"
    elif score >= 80:
        decision = "2차 매수 후보"
    elif score >= 65:
        decision = "1차 매수 후보"
    elif score >= 50:
        decision = "관심 / 대기"
    else:
        decision = "대기"

    return score, decision, reasons, danger_reasons


def apply_buy_score(df, profile, market_penalty):
    scores = []
    decisions = []
    reason_list = []
    danger_list = []

    for i in range(len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i - 1] if i > 0 else None

        score, decision, reasons, dangers = calculate_buy_score(
            row,
            profile,
            market_penalty,
            prev_row
        )

        scores.append(score)
        decisions.append(decision)
        reason_list.append(" / ".join(reasons))
        danger_list.append(" / ".join(dangers))

    df["Buy_Score"] = scores
    df["Decision"] = decisions
    df["Reasons"] = reason_list
    df["Danger_Reasons"] = danger_list

    return df


# =========================
# Buy amount suggestion
# =========================
def get_buy_ratio(decision, market_status):
    if market_status == "Risk":
        if "2차" in decision:
            return 0.10
        if "1차" in decision:
            return 0.05
        return 0.00

    if market_status == "Caution":
        if "2차" in decision:
            return 0.20
        if "1차" in decision:
            return 0.10
        if "관심" in decision:
            return 0.05
        return 0.00

    if market_status == "Good":
        if "2차" in decision:
            return 0.30
        if "1차" in decision:
            return 0.20
        if "관심" in decision:
            return 0.10
        return 0.00

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
st.title("📈 MDD 저점매수 분석기 FINAL")

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

        market_status, market_penalty, market_df = get_market_filter(start_date)

        df = load_price_data(market, ticker, start_date)

        if df.empty:
            st.error("가격 데이터를 가져오지 못했습니다. 종목명/코드/티커를 확인하세요.")
            st.stop()

        df = calculate_indicators(df)
        df = apply_buy_score(df, profile, market_penalty)

        latest = df.iloc[-1]

        current_price = latest["Close"]
        peak_price = latest["Peak"]
        current_dd = latest["Current_Drawdown"]
        period_mdd = df["Max_Drawdown"].min()
        recovery_needed = latest["Recovery_To_Peak"]
        rsi = latest["RSI"]
        buy_score = latest["Buy_Score"]
        decision = latest["Decision"]

        buy_ratio = get_buy_ratio(decision, market_status)
        recommended_buy_amount = planned_buy_amount * buy_ratio

        st.subheader(f"분석 대상: {display_name} / {ticker} / {market}")
        st.write(f"종목 유형: **{asset_type}**")
        st.write(f"시장 필터: **{market_status}** / 감점: **{market_penalty}점**")

        # =========================
        # Metrics
        # =========================
        c1, c2, c3, c4, c5, c6 = st.columns(6)

        c1.metric("현재가", f"{current_price:,.2f}")
        c2.metric("기간 고점", f"{peak_price:,.2f}")
        c3.metric("현재 낙폭", f"{current_dd * 100:.2f}%")
        c4.metric("최대 낙폭", f"{period_mdd * 100:.2f}%")
        c5.metric("회복 필요", f"{recovery_needed * 100:.2f}%")
        c6.metric("매수 점수", f"{buy_score:.0f}점")

        # =========================
        # Decision
        # =========================
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

        st.markdown("## 권장 행동")

        if recommended_buy_amount > 0:
            st.success(
                f"권장 추가매수: 예정금의 {buy_ratio * 100:.0f}% "
                f"≈ {recommended_buy_amount:,.0f}"
            )
        else:
            st.warning("권장 추가매수: 0원 / 대기")

        if pd.notna(rsi):
            st.write(f"현재 RSI: **{rsi:.2f}**")
        else:
            st.write("현재 RSI: 계산 불가")

        # =========================
        # Average price simulation
        # =========================
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

        # =========================
        # Signal details
        # =========================
        if latest["Reasons"]:
            st.markdown("### 긍정 신호")
            for r in latest["Reasons"].split(" / "):
                st.write(f"- {r}")

        if latest["Danger_Reasons"]:
            st.markdown("### 위험 신호")
            for r in latest["Danger_Reasons"].split(" / "):
                st.write(f"- {r}")

        # =========================
        # Market filter table
        # =========================
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

        # =========================
        # Charts - English only
        # =========================
        fig, axes = plt.subplots(3, 1, figsize=(14, 13), sharex=True)

        # 1. Price chart
        axes[0].plot(df.index, df["Close"], label="Close", color="black")
        axes[0].plot(df.index, df["Peak"], label="Peak", color="blue", linestyle="--", alpha=0.7)
        axes[0].plot(df.index, df["MA20"], label="MA20", color="orange", alpha=0.8)
        axes[0].plot(df.index, df["MA60"], label="MA60", color="green", alpha=0.8)
        axes[0].plot(df.index, df["MA200"], label="MA200", color="purple", alpha=0.8)

        axes[0].scatter(df.index[-1], df["Close"].iloc[-1], color="red", s=120, label="Today")

        buy_points = df[df["Buy_Score"] >= 65]
        axes[0].scatter(
            buy_points.index,
            buy_points["Close"],
            color="lime",
            marker="*",
            s=150,
            label="Buy Candidate"
        )

        axes[0].set_title(f"{ticker} Price / Moving Averages")
        axes[0].set_ylabel("Price")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 2. Drawdown chart
        axes[1].plot(df.index, df["Current_Drawdown"] * 100, color="red", label="Current DD")
        axes[1].plot(df.index, df["Max_Drawdown"] * 100, color="darkred", linestyle="--", alpha=0.7, label="Max DD")

        axes[1].axhline(y=profile["watch"] * 100, color="gray", linestyle="--", alpha=0.6, label="Watch")
        axes[1].axhline(y=profile["buy1"] * 100, color="green", linestyle="--", alpha=0.8, label="Buy 1")
        axes[1].axhline(y=profile["buy2"] * 100, color="orange", linestyle="--", alpha=0.8, label="Buy 2")
        axes[1].axhline(y=profile["risk"] * 100, color="red", linestyle="--", alpha=0.8, label="Risk")

        axes[1].set_title("Current Drawdown / Max Drawdown")
        axes[1].set_ylabel("Drawdown (%)")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # 3. Buy score chart
        axes[2].plot(df.index, df["Buy_Score"], color="darkgreen", label="Buy Score")
        axes[2].axhline(y=50, color="gray", linestyle="--", alpha=0.6, label="Watch")
        axes[2].axhline(y=65, color="green", linestyle="--", alpha=0.8, label="Buy 1")
        axes[2].axhline(y=80, color="orange", linestyle="--", alpha=0.8, label="Buy 2")

        axes[2].set_title("Buy Score")
        axes[2].set_ylabel("Score")
        axes[2].set_xlabel("Date")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)

        # =========================
        # Recent data
        # =========================
        st.markdown("## 최근 20거래일 데이터")

        view_cols = [
            "Close",
            "Peak",
            "Current_Drawdown",
            "Max_Drawdown",
            "Recovery_To_Peak",
            "RSI",
            "Volume_Ratio",
            "Buy_Score",
            "Decision"
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
            "Buy_Score": "Buy Score(매수점수)",
            "Decision": "Decision(판단)"
        })

        show_df.index.name = "Date(날짜)"

        st.dataframe(show_df, use_container_width=True)

        # =========================
        # Guide
        # =========================
        st.markdown("## 해석 기준")

        guide_df = pd.DataFrame({
            "항목": [
                "Current DD(현재낙폭%)",
                "Max DD(최대낙폭%)",
                "Recovery(회복필요%)",
                "RSI(과매수/과매도)",
                "Vol Ratio(거래량비율)",
                "Buy Score(매수점수)",
                "시장 필터",
                "Decision(판단)"
            ],
            "의미": [
                "현재가가 시작일 이후 기간고점 대비 몇 % 빠졌는지",
                "시작일 이후 가장 크게 빠졌던 최대 낙폭",
                "현재가에서 기간고점까지 회복하려면 필요한 상승률",
                "30 이하 과매도, 70 이상 과매수",
                "현재 거래량이 20일 평균 거래량 대비 몇 배인지",
                "MDD, RSI, 이동평균, 거래량, 저점 방어, 시장 필터를 종합한 점수",
                "QQQ, SOXX, NVDA, MU 상태로 시장 위험도 판단",
                "대기 / 관심 / 1차 매수 후보 / 2차 매수 후보 / 매수 금지"
            ],
            "활용": [
                "저점매수 판단 핵심값",
                "종목 위험도 참고",
                "회복 난이도 판단",
                "과매도 반등 가능성 확인",
                "투매인지 매수세 유입인지 확인",
                "65점 이상이면 소액 분할매수 후보",
                "시장 위험 시 매수점수 감점",
                "최종 행동 판단"
            ]
        })

        st.table(guide_df)

        score_df = pd.DataFrame({
            "Buy Score(매수점수)": ["0~49", "50~64", "65~79", "80 이상"],
            "Decision(판단)": ["대기", "관심 / 대기", "1차 매수 후보", "2차 매수 후보"],
            "권장 행동": [
                "매수 없음",
                "관찰",
                "예정금 일부 소액",
                "조건 충족 시 추가 분할"
            ]
        })

        st.markdown("## 매수 점수 기준")
        st.table(score_df)

        st.warning(
            "주의: 이 도구는 매수 판단 보조용이다. "
            "Current DD가 깊다고 무조건 매수하면 안 된다. "
            "지수, 금리, 환율, 외국인 수급, 미국 선물, 뉴스와 함께 판단해야 한다."
        )

# =========================================================
# 설명 주석
# =========================================================
#
# 1. Close(종가)
#    해당 날짜의 종가.
#
# 2. Peak(기간고점)
#    사용자가 입력한 시작일 이후 현재까지의 누적 최고가.
#
# 3. Current DD(현재낙폭%)
#    현재 종가가 기간고점 대비 몇 % 빠졌는지.
#    물타기·저점매수 판단에서 가장 중요한 값.
#
# 4. Max DD(최대낙폭%)
#    시작일 이후 가장 크게 빠졌던 낙폭.
#    현재 낙폭이 아니라 기간 중 최악의 낙폭.
#
# 5. Recovery(회복필요%)
#    현재가에서 다시 기간고점까지 회복하려면 필요한 상승률.
#    예: Current DD -20%이면 Recovery +25%.
#
# 6. RSI
#    30 이하 과매도, 70 이상 과매수.
#    RSI 30 이하에서 30 위로 회복하면 과매도 탈출 신호.
#
# 7. Vol Ratio(거래량비율)
#    현재 거래량이 20일 평균 거래량 대비 몇 배인지.
#    1.5 이상이면 평소보다 거래량이 큰 상태.
#
# 8. Buy Score(매수점수)
#    Current DD, RSI, 이동평균, 거래량, 볼린저밴드,
#    최근 저점 방어, 시장 필터를 종합한 점수.
#
# 9. 시장 필터
#    QQQ, SOXX, NVDA, MU를 기준으로 시장 위험도를 계산.
#    시장이 불안하면 개별 종목 점수가 좋아도 감점.
#
# 10. 권장 추가매수
#    입력한 추가매수 예정금에서 현재 판단에 따라 일부만 계산.
#    시장 Risk 상태에서는 매수 비중을 자동으로 줄임.
#
# 11. 물타기 후 평단 시뮬레이션
#    보유수량과 평균단가를 입력하면 권장 추가매수 후 새 평균단가 계산.
#
# 12. 핵심 사용법
#    Current DD만 보고 매수하지 않는다.
#    Buy Score, Decision, 시장 필터, 뉴스, 지수 흐름을 함께 본다.
#
# =========================================================
