import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import FinanceDataReader as fdr

# =========================================================
# MDD 저점매수 분석기 - Final Version
# =========================================================

st.set_page_config(page_title="MDD 저점매수 분석기", layout="wide")

# =========================
# Password
# =========================
MY_PASSWORD = "1234"

st.title("🔒 MDD 저점매수 분석기")
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
    """
    한국: 종목명 또는 6자리 코드 입력
    미국: NVDA, QQQ, SOXX 등 티커 입력
    """
    query = query.strip()

    if query == "":
        return None, None, None

    # 한국 6자리 종목코드
    if query.isdigit() and len(query) == 6:
        return "KR", query, query

    # 한국 종목명 검색
    if not stock_list.empty and "Name" in stock_list.columns:
        match = stock_list[stock_list["Name"] == query]
        if not match.empty:
            code = match.iloc[0]["Code"]
            name = match.iloc[0]["Name"]
            return "KR", code, name

    # 미국 티커
    return "US", query.upper(), query.upper()


# =========================
# Load price data
# =========================
@st.cache_data(ttl=3600)
def load_price_data(market, ticker, start_date):
    start = start_date.strftime("%Y-%m-%d")

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

    # 핵심 MDD 계산
    df["Peak"] = df["Close"].cummax()
    df["Current_Drawdown"] = df["Close"] / df["Peak"] - 1
    df["Max_Drawdown"] = df["Current_Drawdown"].cummin()
    df["Recovery_To_Peak"] = 1 / (1 + df["Current_Drawdown"]) - 1

    # 기존 코드 호환용 컬럼
    df["Drawdown"] = df["Current_Drawdown"]
    df["MDD"] = df["Max_Drawdown"]
    df["Recovery_Needed"] = df["Recovery_To_Peak"]

    # 이동평균
    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA60"] = df["Close"].rolling(60).mean()
    df["MA200"] = df["Close"].rolling(200).mean()

    # RSI
    df["RSI"] = calculate_rsi(df["Close"], 14)

    # 거래량
    df["Volume_MA20"] = df["Volume"].rolling(20).mean()
    df["Volume_Ratio"] = df["Volume"] / df["Volume_MA20"]

    # 단기 고저점
    df["Return"] = df["Close"].pct_change()
    df["Low20"] = df["Close"].rolling(20).min()
    df["High20"] = df["Close"].rolling(20).max()

    # 볼린저밴드
    df["BB_Mid"] = df["Close"].rolling(20).mean()
    df["BB_Std"] = df["Close"].rolling(20).std()
    df["BB_Lower"] = df["BB_Mid"] - 2 * df["BB_Std"]
    df["BB_Upper"] = df["BB_Mid"] + 2 * df["BB_Std"]

    return df


# =========================
# Buy score
# =========================
def calculate_buy_score(row, prev_row=None):
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

    # 1. Current drawdown score
    if dd <= -0.20:
        score += 20
        reasons.append("Current DD -20% 이하: 가격은 깊은 조정권")
        danger = True
        danger_reasons.append("Current DD -20% 이하: 추세 훼손 가능성")
    elif dd <= -0.15:
        score += 35
        reasons.append("Current DD -15% 이하: 강한 조정")
    elif dd <= -0.12:
        score += 30
        reasons.append("Current DD -12% 이하: 2차 매수 후보권")
    elif dd <= -0.08:
        score += 20
        reasons.append("Current DD -8% 이하: 1차 매수 후보권")
    elif dd <= -0.05:
        score += 10
        reasons.append("Current DD -5% 이하: 관심 구간")

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


def apply_buy_score(df):
    scores = []
    decisions = []
    reason_list = []
    danger_list = []

    for i in range(len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i - 1] if i > 0 else None

        score, decision, reasons, dangers = calculate_buy_score(row, prev_row)

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
# Main screen
# =========================
st.title("📈 MDD 저점매수 분석기")

col_a, col_b, col_c = st.columns(3)

with col_a:
    user_input = st.text_input("종목명 / 종목코드 / 미국 티커", value="삼성전자")

with col_b:
    start_date = st.date_input("기준 시작일", pd.to_datetime("2024-01-01"))

with col_c:
    target_dd = st.number_input("관심 MDD 기준(%)", value=12.0, step=1.0)

run = st.button("분석 실행")

if run:
    market, ticker, display_name = find_ticker(user_input)

    if ticker is None:
        st.error("종목을 찾을 수 없습니다.")
        st.stop()

    with st.spinner("데이터 분석 중..."):
        df = load_price_data(market, ticker, start_date)

        if df.empty:
            st.error("가격 데이터를 가져오지 못했습니다. 종목명/코드/티커를 확인하세요.")
            st.stop()

        df = calculate_indicators(df)
        df = apply_buy_score(df)

        latest = df.iloc[-1]

        current_price = latest["Close"]
        peak_price = latest["Peak"]
        current_dd = latest["Current_Drawdown"]
        max_mdd = latest["Max_Drawdown"]
        period_mdd = df["Max_Drawdown"].min()
        recovery_needed = latest["Recovery_To_Peak"]
        rsi = latest["RSI"]
        buy_score = latest["Buy_Score"]
        decision = latest["Decision"]

        st.subheader(f"분석 대상: {display_name} / {ticker} / {market}")

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

        if pd.notna(rsi):
            st.write(f"**현재 RSI:** {rsi:.2f}")
        else:
            st.write("**현재 RSI:** 계산 불가")

        if latest["Reasons"]:
            st.markdown("### 긍정 신호")
            for r in latest["Reasons"].split(" / "):
                st.write(f"- {r}")

        if latest["Danger_Reasons"]:
            st.markdown("### 위험 신호")
            for r in latest["Danger_Reasons"].split(" / "):
                st.write(f"- {r}")

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

        axes[1].axhline(y=-8, color="gray", linestyle="--", alpha=0.6, label="-8% Watch")
        axes[1].axhline(y=-12, color="green", linestyle="--", alpha=0.8, label="-12% Buy 1")
        axes[1].axhline(y=-15, color="orange", linestyle="--", alpha=0.8, label="-15% Buy 2")
        axes[1].axhline(y=-20, color="red", linestyle="--", alpha=0.8, label="-20% Risk")
        axes[1].axhline(y=-target_dd, color="blue", linestyle=":", alpha=0.8, label="User Target")

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
                "Decision(판단)"
            ],
            "의미": [
                "현재가가 시작일 이후 기간고점 대비 몇 % 빠졌는지",
                "시작일 이후 가장 크게 빠졌던 최대 낙폭",
                "현재가에서 기간고점까지 회복하려면 필요한 상승률",
                "30 이하 과매도, 70 이상 과매수",
                "현재 거래량이 20일 평균 거래량 대비 몇 배인지",
                "MDD, RSI, 이동평균, 거래량, 저점 방어를 종합한 점수",
                "대기 / 관심 / 1차 매수 후보 / 2차 매수 후보 / 매수 금지"
            ],
            "활용": [
                "물타기·저점매수 판단의 핵심값",
                "종목의 위험도 참고용",
                "회복 난이도 판단",
                "과매도 반등 가능성 확인",
                "투매인지 매수세 유입인지 확인",
                "65점 이상이면 소액 분할매수 후보",
                "최종 행동 판단"
            ]
        })

        st.table(guide_df)

        score_df = pd.DataFrame({
            "Buy Score(매수점수)": ["0~49", "50~64", "65~79", "80 이상"],
            "Decision(판단)": ["대기", "관심 / 대기", "1차 매수 후보", "2차 매수 후보"],
            "설명": [
                "가격 매력 또는 반등 확인 부족",
                "관찰 구간",
                "소액 분할매수 검토 가능",
                "강한 과매도 + 반등 신호"
            ]
        })

        st.markdown("## 매수 점수 기준")
        st.table(score_df)

        st.warning(
            "주의: 이 도구는 매수 판단 보조용이다. "
            "Current DD가 깊다고 무조건 매수하면 안 된다. "
            "지수, 금리, 환율, 수급, 뉴스, 미국 선물시장 반응을 함께 확인해야 한다."
        )

# =========================================================
# 설명 주석
# =========================================================
#
# 1. Close(종가)
#    - 해당 날짜의 종가다.
#
# 2. Peak(기간고점)
#    - 사용자가 입력한 시작일 이후 현재까지의 누적 최고가다.
#    - 예: 시작일 이후 가장 높았던 가격.
#
# 3. Current DD(현재낙폭%)
#    - 현재 종가가 Peak(기간고점) 대비 몇 % 빠졌는지 보여준다.
#    - 물타기·저점매수 판단에서 가장 먼저 봐야 하는 값이다.
#    - 예: -10%면 기간고점 대비 10% 하락한 상태.
#
# 4. Max DD(최대낙폭%)
#    - 시작일 이후 가장 크게 빠졌던 낙폭이다.
#    - 현재 낙폭이 아니라, 기간 중 최악의 하락폭이다.
#    - 그래서 한 번 -20%까지 빠진 적이 있으면 이후 회복해도 Max DD는 계속 -20%로 남는다.
#
# 5. Recovery(회복필요%)
#    - 현재 가격에서 다시 Peak(기간고점)까지 회복하려면 몇 % 상승이 필요한지 보여준다.
#    - 예: Current DD가 -20%면 Recovery는 +25%다.
#
# 6. RSI(과매수/과매도)
#    - 보통 RSI 30 이하이면 과매도, 70 이상이면 과매수로 본다.
#    - RSI가 30 이하에서 다시 30 위로 올라오면 과매도 탈출 신호로 본다.
#
# 7. Vol Ratio(거래량비율)
#    - 현재 거래량이 20일 평균 거래량 대비 몇 배인지 보여준다.
#    - 1.5 이상이면 평소보다 거래량이 크게 증가한 것이다.
#    - 거래량 증가 + 양봉이면 매수세 유입 가능성.
#    - 거래량 증가 + 음봉이면 투매 또는 기관 매도 가능성.
#
# 8. Buy Score(매수점수)
#    - Current DD, RSI, 이동평균, 거래량, 볼린저밴드, 저점 방어 여부를 합산한 점수다.
#    - 65점 이상이면 1차 매수 후보.
#    - 80점 이상이면 2차 매수 후보.
#    - 단, 추세 훼손 위험 신호가 있으면 점수가 높아도 매수 금지가 나올 수 있다.
#
# 9. Decision(판단)
#    - 대기: 아직 매수 근거 부족.
#    - 관심 / 대기: 관찰 구간.
#    - 1차 매수 후보: 소액 분할매수 검토 가능.
#    - 2차 매수 후보: 강한 과매도 + 반등 신호.
#    - 매수 금지 / 추세 확인: 낙폭은 크지만 추세 훼손 위험이 커서 물타기 금지.
#
# 10. 핵심 사용법
#    - Current DD가 -8~-12%면 관심.
#    - Current DD가 -12~-15%면 1차 매수 후보.
#    - Current DD가 -15~-20%면 조건부 매수 후보.
#    - Current DD가 -20% 이하이면 무조건 매수하지 말고 추세 훼손 여부를 먼저 확인해야 한다.
#
# 11. 주의
#    - MDD 계산기는 가격 위치를 보는 도구다.
#    - 실제 매수는 시장 지수, 금리, 환율, 외국인 수급, 미국 선물, 뉴스와 함께 판단해야 한다.
#    - 특히 한국 ETF는 미국 본주 움직임과 환율, ETF 리밸런싱 수급까지 함께 봐야 한다.
#
# =========================================================
