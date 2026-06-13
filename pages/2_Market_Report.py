import streamlit as st
import pandas as pd
import yfinance as yf
import FinanceDataReader as fdr
import feedparser
from datetime import datetime, timedelta
from auth import require_login, logout_button

# =========================================================
# Market Report Page - Simple Dashboard Version
# 목적:
# - 30초 안에 오늘 시장 판단 확인
# - 기본 화면은 간소화
# - 상세 뉴스 / 리스크 / SNS / 테마 / 다운로드는 expander 안에 배치
# - 매수/매도 추천 아님
# - MDD Buy Score와 직접 연동 없음
# =========================================================

st.set_page_config(page_title="시장 리포트", layout="wide")

require_login()
logout_button()

st.title("📰 시장 리포트")

st.caption(
    "목적: 오늘 시장을 움직이는 핵심 이슈를 빠르게 확인하는 화면입니다. "
    "MDD Buy Score에는 직접 반영하지 않습니다."
)

# =========================================================
# RSS Sources
# =========================================================
RSS_SOURCES = {
    "Investing KR 전체 뉴스": "https://kr.investing.com/rss/news.rss",
    "Investing KR 주식 뉴스": "https://kr.investing.com/rss/news_25.rss",
    "Investing KR 경제 뉴스": "https://kr.investing.com/rss/news_14.rss",
    "Investing KR 경제지표 뉴스": "https://kr.investing.com/rss/news_95.rss",
    "Investing KR 상품 뉴스": "https://kr.investing.com/rss/news_11.rss",
    "Investing KR 시장 개황": "https://kr.investing.com/rss/market_overview.rss",
    "Investing KR 주식 분석": "https://kr.investing.com/rss/stock.rss",
}

# =========================================================
# Helper functions
# =========================================================
@st.cache_data(ttl=900)
def load_yfinance_latest(ticker):
    try:
        end = datetime.today()
        start = end - timedelta(days=10)

        df = yf.Ticker(ticker).history(
            start=start.strftime("%Y-%m-%d"),
            auto_adjust=True
        )

        if df.empty or len(df) < 2:
            return None, None

        df.index = pd.to_datetime(df.index).tz_localize(None)

        latest_close = df["Close"].iloc[-1]
        prev_close = df["Close"].iloc[-2]
        change_pct = (latest_close / prev_close - 1) * 100

        return latest_close, change_pct

    except Exception:
        return None, None


@st.cache_data(ttl=900)
def load_fdr_index_latest(symbol):
    try:
        end = datetime.today()
        start = end - timedelta(days=10)

        df = fdr.DataReader(symbol, start.strftime("%Y-%m-%d"))

        if df.empty or len(df) < 2:
            return None, None

        latest_close = df["Close"].iloc[-1]
        prev_close = df["Close"].iloc[-2]
        change_pct = (latest_close / prev_close - 1) * 100

        return latest_close, change_pct

    except Exception:
        return None, None


def format_number(value):
    if value is None or pd.isna(value):
        return "-"
    return f"{value:,.2f}"


def format_pct(value):
    if value is None or pd.isna(value):
        return "-"
    return f"{value:.2f}%"


def market_comment(ticker, change_pct):
    if change_pct is None or pd.isna(change_pct):
        return "데이터 확인 필요"

    ticker = ticker.upper()

    # VIX: 하락이 긍정, 상승이 부정
    if ticker == "^VIX":
        if change_pct <= -3.0:
            return "공포 완화"
        if change_pct < -0.5:
            return "위험심리 개선"
        if change_pct <= 0.5:
            return "중립"
        if change_pct < 3.0:
            return "변동성 확대"
        return "리스크 확대"

    # 미국 10년물 금리: 상승은 성장주 부담
    if ticker == "^TNX":
        if change_pct >= 2.0:
            return "금리 부담 확대"
        if change_pct >= 0.5:
            return "성장주 부담"
        if change_pct > -0.5:
            return "중립"
        if change_pct > -2.0:
            return "금리 부담 완화"
        return "성장주 우호"

    # 유가: 급등은 물가 부담
    if ticker in ["USO", "CL=F", "BZ=F"]:
        if change_pct >= 2.0:
            return "유가 부담 확대"
        if change_pct >= 0.5:
            return "물가 부담 주의"
        if change_pct > -0.5:
            return "중립"
        if change_pct > -2.0:
            return "유가 안정"
        return "유가 급락 / 경기 우려 확인"

    # 달러: 상승은 한국장·위험자산 부담
    if ticker in ["UUP", "DX-Y.NYB"]:
        if change_pct >= 1.0:
            return "달러 강세 부담"
        if change_pct >= 0.3:
            return "환율 부담"
        if change_pct > -0.3:
            return "중립"
        if change_pct > -1.0:
            return "달러 부담 완화"
        return "위험자산 우호"

    # 일반 주식/ETF/지수
    if change_pct >= 1.0:
        return "강한 반등"
    if change_pct >= 0.3:
        return "상승 우위"
    if change_pct > -0.3:
        return "중립"
    if change_pct > -1.0:
        return "약세"
    return "리스크 확대"


@st.cache_data(ttl=900)
def load_rss_news(selected_sources, max_items_per_source):
    rows = []

    for source_name in selected_sources:
        url = RSS_SOURCES[source_name]

        try:
            feed = feedparser.parse(
                url,
                request_headers={"User-Agent": "Mozilla/5.0"}
            )

            for entry in feed.entries[:max_items_per_source]:
                title = getattr(entry, "title", "")
                link = getattr(entry, "link", "")
                published = getattr(entry, "published", "")

                if title:
                    rows.append({
                        "출처": source_name,
                        "뉴스": title,
                        "발행": published,
                        "링크": link
                    })

        except Exception:
            continue

    if not rows:
        return pd.DataFrame(columns=["출처", "뉴스", "발행", "링크"])

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["뉴스"])
    return df


def classify_news(title):
    text = str(title).lower()

    issue_rules = {
        "전쟁·지정학": [
            "war", "attack", "missile", "iran", "israel", "gaza", "russia", "ukraine",
            "conflict", "geopolitical", "sanction", "military", "red sea", "houthi",
            "전쟁", "공격", "미사일", "이란", "이스라엘", "가자", "러시아", "우크라이나",
            "분쟁", "지정학", "제재", "군사", "홍해", "후티", "호르무즈"
        ],
        "유가·원자재": [
            "oil", "crude", "brent", "wti", "opec", "gas", "energy", "commodity",
            "gold", "copper", "유가", "원유", "브렌트", "천연가스", "원자재",
            "금", "구리", "에너지", "opec"
        ],
        "금리·물가·연준": [
            "fed", "fomc", "rate", "yield", "treasury", "inflation", "cpi", "ppi",
            "powell", "jobless", "payroll", "employment", "연준", "금리", "국채",
            "수익률", "물가", "인플레이션", "파월", "고용", "실업수당", "비농업", "채권"
        ],
        "AI·반도체": [
            "nvidia", "ai", "artificial intelligence", "semiconductor", "chip", "chips",
            "micron", "broadcom", "amd", "intel", "tsmc", "memory", "hbm", "dram",
            "엔비디아", "인공지능", "반도체", "칩", "마이크론", "브로드컴",
            "인텔", "메모리", "sk하이닉스", "삼성전자"
        ],
        "전력·AI인프라": [
            "power", "electricity", "grid", "data center", "datacenter", "utility",
            "energy infrastructure", "vertiv", "전력", "전력망", "전기", "데이터센터",
            "데이터 센터", "전력인프라", "인프라", "버티브", "냉각"
        ],
        "빅테크·나스닥": [
            "nasdaq", "apple", "microsoft", "amazon", "google", "alphabet", "meta",
            "tesla", "tech", "growth stocks", "나스닥", "애플", "마이크로소프트",
            "아마존", "구글", "알파벳", "메타", "테슬라", "빅테크", "성장주", "기술주"
        ],
        "한국장": [
            "korea", "kospi", "kosdaq", "won", "samsung", "sk hynix", "south korea",
            "한국", "코스피", "코스닥", "원화", "환율", "삼성전자", "sk하이닉스", "외국인", "기관"
        ],
        "중국·아시아": [
            "china", "hong kong", "asia", "japan", "taiwan", "yuan", "nikkei",
            "hang seng", "중국", "홍콩", "아시아", "일본", "대만", "위안", "니케이", "항셍"
        ],
        "우주·방산": [
            "spacex", "space", "rocket", "satellite", "defense", "aerospace",
            "스페이스x", "우주", "로켓", "위성", "방산", "항공우주"
        ],
        "실적·가이던스": [
            "earnings", "revenue", "profit", "forecast", "guidance", "outlook",
            "beat", "misses", "results", "실적", "매출", "이익", "가이던스",
            "전망", "어닝", "컨센서스", "상회", "하회"
        ],
        "금융·신용": [
            "bank", "credit", "debt", "default", "liquidity", "bond", "loan",
            "financial", "은행", "신용", "부채", "디폴트", "유동성", "채권", "대출", "금융"
        ]
    }

    theme_map = {
        "전쟁·지정학": "전체, 유가, 방산",
        "유가·원자재": "전체, 나스닥, 한국장",
        "금리·물가·연준": "나스닥, 반도체, 성장주",
        "AI·반도체": "메모리/반도체, 엔비디아, AI",
        "전력·AI인프라": "전력/AI 인프라",
        "빅테크·나스닥": "나스닥, 구글, 빅테크",
        "한국장": "한국장, 반도체",
        "중국·아시아": "한국장, 아시아 ETF",
        "우주·방산": "우주/SpaceX, 방산",
        "실적·가이던스": "해당 종목, 섹터",
        "금융·신용": "전체, 금융 리스크"
    }

    risk_issues = ["전쟁·지정학", "유가·원자재", "금리·물가·연준", "금융·신용"]

    matched_issues = []

    for issue, keywords in issue_rules.items():
        if any(keyword in text for keyword in keywords):
            matched_issues.append(issue)

    if not matched_issues:
        matched_issues = ["기타"]

    issue_text = ", ".join(matched_issues)

    themes = []
    for issue in matched_issues:
        if issue in theme_map:
            themes.append(theme_map[issue])

    related_theme = " / ".join(themes) if themes else "확인 필요"

    negative_words = [
        "fall", "drop", "plunge", "risk", "war", "attack", "concern", "worry",
        "fear", "loss", "miss", "cut", "weak", "slowdown", "inflation",
        "higher rates", "하락", "급락", "위험", "전쟁", "공격", "우려",
        "공포", "손실", "부진", "둔화", "인플레이션", "금리 상승", "악화",
        "부담", "제재", "관세"
    ]

    positive_words = [
        "rise", "gain", "jump", "surge", "rally", "beat", "record",
        "optimism", "growth", "strong", "rebound", "upgrade", "상승",
        "급등", "반등", "랠리", "호조", "상회", "기록", "낙관",
        "성장", "강세", "개선", "상향"
    ]

    if any(word in text for word in negative_words):
        tone = "부정"
    elif any(word in text for word in positive_words):
        tone = "긍정"
    else:
        tone = "중립"

    risk_flag = "리스크 확인" if any(issue in risk_issues for issue in matched_issues) else "일반 이슈"

    return issue_text, related_theme, tone, risk_flag


def classify_market_condition(market_df):
    if market_df.empty:
        return "중립", "보통", "대기", "금지"

    memo_text = " ".join(market_df["판단 메모"].astype(str).tolist())

    risk_words = ["리스크 확대", "변동성 확대", "금리 부담", "유가 부담", "달러 강세", "약세"]
    good_words = ["강한 반등", "상승 우위", "위험심리 개선", "공포 완화", "유가 안정"]

    risk_count = sum(word in memo_text for word in risk_words)
    good_count = sum(word in memo_text for word in good_words)

    if risk_count >= 3:
        return "부정", "높음", "대기", "금지"
    if risk_count >= 2:
        return "중립", "보통", "소액 가능", "금지"
    if good_count >= 3:
        return "긍정", "낮음", "가능", "금지"

    return "중립", "보통", "소액 가능", "금지"


def make_core_news(news_df, top_n=3):
    if news_df.empty:
        return pd.DataFrame({
            "순위": [1, 2, 3],
            "뉴스": ["뉴스 수집 실패", "-", "-"],
            "영향": ["중립", "중립", "중립"],
            "관련 테마": ["확인 필요", "-", "-"],
            "판단 메모": ["RSS 소스 또는 네트워크 확인 필요", "-", "-"]
        })

    score_map = {
        "부정": 3,
        "혼재": 2,
        "긍정": 2,
        "중립": 1
    }

    df = news_df.copy()
    df["우선순위"] = df["영향"].map(score_map).fillna(1)

    # 리스크 확인 뉴스 우선
    df.loc[df["구분"] == "리스크 확인", "우선순위"] += 1

    df = df.sort_values("우선순위", ascending=False).head(top_n).copy()
    df = df.reset_index(drop=True)

    rows = []
    for i, row in df.iterrows():
        rows.append({
            "순위": i + 1,
            "뉴스": row["뉴스"],
            "영향": row["영향"],
            "관련 테마": row["관련 테마"],
            "판단 메모": f"{row['이슈']} / {row['구분']}"
        })

    return pd.DataFrame(rows)


def make_core_risks(news_df, top_n=3):
    risk_df = news_df[
        (news_df["구분"] == "리스크 확인") | (news_df["영향"] == "부정")
    ].copy()

    if risk_df.empty:
        return pd.DataFrame({
            "순위": [1, 2, 3],
            "리스크": ["뚜렷한 리스크 뉴스 부족", "-", "-"],
            "영향": ["중립", "중립", "중립"],
            "관련 테마": ["전체", "-", "-"],
            "확인 상태": ["RSS 기준", "-", "-"],
            "판단 메모": ["지표와 추가 뉴스 확인 필요", "-", "-"]
        })

    risk_df = risk_df.head(top_n).reset_index(drop=True)

    rows = []
    for i, row in risk_df.iterrows():
        rows.append({
            "순위": i + 1,
            "리스크": row["뉴스"],
            "영향": row["영향"],
            "관련 테마": row["관련 테마"],
            "확인 상태": row["구분"],
            "판단 메모": row["이슈"]
        })

    return pd.DataFrame(rows)


def make_theme_summary(news_df):
    theme_issue_map = {
        "나스닥": ["빅테크·나스닥", "금리·물가·연준"],
        "메모리/반도체": ["AI·반도체"],
        "전력/AI 인프라": ["전력·AI인프라"],
        "우주/SpaceX": ["우주·방산"],
        "구글": ["빅테크·나스닥"],
        "한국장": ["한국장", "중국·아시아", "금리·물가·연준"],
        "유가/원자재": ["유가·원자재"],
        "금리/연준": ["금리·물가·연준"]
    }

    rows = []

    for theme, issue_keys in theme_issue_map.items():
        matched = news_df[
            news_df["이슈"].apply(lambda x: any(issue_key in x for issue_key in issue_keys))
        ]

        related_count = len(matched)
        neg_count = int((matched["영향"] == "부정").sum()) if not matched.empty else 0
        pos_count = int((matched["영향"] == "긍정").sum()) if not matched.empty else 0

        if related_count == 0:
            judgment = "중립"
            action = "대기"
            reason = "관련 뉴스 부족"
        elif neg_count > pos_count:
            judgment = "부정"
            action = "추격 금지"
            reason = "부정 뉴스 우위"
        elif pos_count > neg_count:
            judgment = "긍정"
            action = "소액 가능"
            reason = "긍정 뉴스 우위"
        else:
            judgment = "혼재"
            action = "대기"
            reason = "긍정·부정 혼재"

        rows.append({
            "테마": theme,
            "오늘 판단": judgment,
            "행동": action,
            "핵심 이유": f"{reason} / 관련 뉴스 {related_count}건"
        })

    return pd.DataFrame(rows)


def make_mdd_memo(market_mood, risk_level, buy_judgment):
    if risk_level in ["높음", "매우 높음"]:
        return "오늘 MDD 판단: 리스크가 높아 MDD가 깊어도 추격 금지. 1차 소액도 지표 안정 확인 후 접근."
    if buy_judgment == "가능":
        return "오늘 MDD 판단: MDD 깊은 종목은 눌림 시 1차 소액 가능. 단, 시초가 급등 추격은 금지."
    if buy_judgment == "소액 가능":
        return "오늘 MDD 판단: MDD 깊은 종목만 1차 소액 가능. 회복 확인 전 비중 확대는 금지."
    return "오늘 MDD 판단: 가격 매력보다 리스크 확인이 우선. 신규 매수는 대기."


def df_to_markdown(df):
    if df.empty:
        return "데이터 없음"

    headers = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for _, row in df.iterrows():
        values = [str(row[col]).replace("\n", " ").replace("|", "/") for col in headers]
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


# =========================================================
# Data Load
# =========================================================
default_main_tickers = ["QQQ", "SPY", "SOXX", "EWY", "^VIX", "USO"]
extra_tickers = ["IWM", "^TNX", "UUP", "NVDA", "MU", "VRT", "GOOGL"]

market_rows = []

for ticker in default_main_tickers:
    close, change_pct = load_yfinance_latest(ticker)
    market_rows.append({
        "표시명": ticker,
        "티커": ticker,
        "현재가": format_number(close),
        "등락률": format_pct(change_pct),
        "판단 메모": market_comment(ticker, change_pct)
    })

main_market_df = pd.DataFrame(market_rows)

extra_rows = []

for ticker in extra_tickers:
    close, change_pct = load_yfinance_latest(ticker)
    extra_rows.append({
        "표시명": ticker,
        "티커": ticker,
        "현재가": format_number(close),
        "등락률": format_pct(change_pct),
        "판단 메모": market_comment(ticker, change_pct)
    })

kospi_close, kospi_change = load_fdr_index_latest("KS11")
kosdaq_close, kosdaq_change = load_fdr_index_latest("KQ11")

extra_rows.append({
    "표시명": "KOSPI",
    "티커": "KS11",
    "현재가": format_number(kospi_close),
    "등락률": format_pct(kospi_change),
    "판단 메모": market_comment("KS11", kospi_change)
})

extra_rows.append({
    "표시명": "KOSDAQ",
    "티커": "KQ11",
    "현재가": format_number(kosdaq_close),
    "등락률": format_pct(kosdaq_change),
    "판단 메모": market_comment("KQ11", kosdaq_change)
})

extra_market_df = pd.DataFrame(extra_rows)

with st.sidebar:
    st.markdown("### 뉴스 수집 설정")
    selected_sources = st.multiselect(
        "뉴스 소스",
        list(RSS_SOURCES.keys()),
        default=[
            "Investing KR 전체 뉴스",
            "Investing KR 주식 뉴스",
            "Investing KR 경제지표 뉴스",
            "Investing KR 상품 뉴스"
        ]
    )

    max_items = st.number_input(
        "소스별 뉴스 개수",
        min_value=5,
        max_value=30,
        value=10,
        step=5
    )

raw_news_df = load_rss_news(selected_sources, max_items)

classified_rows = []

if not raw_news_df.empty:
    for _, row in raw_news_df.iterrows():
        issue, related_theme, tone, risk_flag = classify_news(row["뉴스"])

        classified_rows.append({
            "뉴스": row["뉴스"],
            "출처": row["출처"],
            "이슈": issue,
            "관련 테마": related_theme,
            "영향": tone,
            "구분": risk_flag,
            "발행": row["발행"],
            "링크": row["링크"]
        })

news_df = pd.DataFrame(classified_rows)

if news_df.empty:
    news_df = pd.DataFrame(columns=["뉴스", "출처", "이슈", "관련 테마", "영향", "구분", "발행", "링크"])

core_news_df = make_core_news(news_df)
core_risk_df = make_core_risks(news_df)
theme_df = make_theme_summary(news_df)

main_theme_df = theme_df[
    theme_df["테마"].isin(["나스닥", "메모리/반도체", "전력/AI 인프라", "우주/SpaceX"])
].copy()

detail_theme_df = theme_df[
    ~theme_df["테마"].isin(["나스닥", "메모리/반도체", "전력/AI 인프라", "우주/SpaceX"])
].copy()

auto_market_mood, auto_risk_level, auto_buy_judgment, auto_chase = classify_market_condition(main_market_df)

# =========================================================
# 1. 오늘 판단 카드
# =========================================================
st.markdown("## 1. 오늘 판단 카드")

c1, c2, c3, c4 = st.columns(4)

with c1:
    market_mood = st.selectbox(
        "시장 분위기",
        ["긍정", "중립", "부정"],
        index=["긍정", "중립", "부정"].index(auto_market_mood)
    )

with c2:
    risk_level = st.selectbox(
        "위험도",
        ["낮음", "보통", "높음", "매우 높음"],
        index=["낮음", "보통", "높음", "매우 높음"].index(auto_risk_level)
    )

with c3:
    buy_judgment = st.selectbox(
        "매수 판단",
        ["가능", "소액 가능", "대기", "금지"],
        index=["가능", "소액 가능", "대기", "금지"].index(auto_buy_judgment)
    )

with c4:
    chase_buy = st.selectbox(
        "추격매수",
        ["가능", "금지"],
        index=["가능", "금지"].index(auto_chase)
    )

default_one_line = make_mdd_memo(market_mood, risk_level, buy_judgment)

core_line = st.text_input(
    "핵심 한 줄",
    value=default_one_line
)

st.info(core_line)

# =========================================================
# 2. 주요 시장 지표
# =========================================================
st.markdown("## 2. 주요 시장 지표")

st.dataframe(main_market_df, use_container_width=True)

with st.expander("추가 지표 보기"):
    st.dataframe(extra_market_df, use_container_width=True)

# =========================================================
# 3. 핵심 뉴스 TOP 3
# =========================================================
st.markdown("## 3. 핵심 뉴스 TOP 3")

core_news_df = st.data_editor(
    core_news_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "영향": st.column_config.SelectboxColumn(
            "영향",
            options=["긍정", "중립", "부정", "혼재"]
        )
    },
    key="core_news_editor"
)

with st.expander("상세 뉴스 보기"):
    st.dataframe(
        news_df,
        use_container_width=True,
        column_config={
            "링크": st.column_config.LinkColumn("링크")
        }
    )

# =========================================================
# 4. 주요 리스크 TOP 3
# =========================================================
st.markdown("## 4. 주요 리스크 TOP 3")

core_risk_df = st.data_editor(
    core_risk_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "영향": st.column_config.SelectboxColumn(
            "영향",
            options=["긍정", "중립", "부정", "혼재"]
        ),
        "확인 상태": st.column_config.SelectboxColumn(
            "확인 상태",
            options=["공식", "보도", "비공식", "찌라시", "리스크 확인", "RSS 기준"]
        )
    },
    key="core_risk_editor"
)

with st.expander("상세 리스크 보기"):
    risk_detail_df = news_df[
        (news_df["구분"] == "리스크 확인") | (news_df["영향"] == "부정")
    ].copy()

    if risk_detail_df.empty:
        st.success("RSS 기준 뚜렷한 리스크성 뉴스가 많지 않습니다.")
    else:
        st.dataframe(
            risk_detail_df,
            use_container_width=True,
            column_config={
                "링크": st.column_config.LinkColumn("링크")
            }
        )

# =========================================================
# 5. 테마 판단
# =========================================================
st.markdown("## 5. 테마 판단")

main_theme_df = st.data_editor(
    main_theme_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "오늘 판단": st.column_config.SelectboxColumn(
            "오늘 판단",
            options=["긍정", "중립", "부정", "혼재"]
        ),
        "행동": st.column_config.SelectboxColumn(
            "행동",
            options=["유지", "소액 가능", "대기", "축소", "추격 금지"]
        )
    },
    key="main_theme_editor"
)

with st.expander("상세 테마 보기"):
    detail_theme_df = st.data_editor(
        detail_theme_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "오늘 판단": st.column_config.SelectboxColumn(
                "오늘 판단",
                options=["긍정", "중립", "부정", "혼재"]
            ),
            "행동": st.column_config.SelectboxColumn(
                "행동",
                options=["유지", "소액 가능", "대기", "축소", "추격 금지"]
            )
        },
        key="detail_theme_editor"
    )

# =========================================================
# 6. 오늘 MDD 판단 참고
# =========================================================
st.markdown("## 6. 오늘 MDD 판단 참고")

mdd_memo = st.text_area(
    "MDD 판단 메모",
    value=core_line,
    height=80
)

st.success(mdd_memo)

# =========================================================
# 7. 상세 정보 접기
# =========================================================
with st.expander("SNS·소문·찌라시"):
    st.warning("이 섹션은 확정 사실이 아닌 비공식 신호 기록용입니다.")

    sns_df = pd.DataFrame({
        "내용": [
            "데이터센터 투자 지연설",
            "특정 ETF 편입 루머",
            "SpaceX 관련 우주주 관심 증가"
        ],
        "출처": ["X/커뮤니티", "커뮤니티", "Reddit/X"],
        "신뢰도": ["낮음", "낮음", "보통"],
        "영향": ["부정", "긍정", "긍정"],
        "관련 테마": ["전력/AI 인프라", "한국장", "우주/SpaceX"],
        "확인 필요 메모": [
            "공식 보도 또는 기업 가이던스 확인 필요",
            "ETF 공시 확인 필요",
            "실제 거래량 확인 필요"
        ]
    })

    sns_df = st.data_editor(
        sns_df,
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic",
        key="sns_editor"
    )

with st.expander("다음 확인사항"):
    next_check_df = pd.DataFrame({
        "확인할 것": [
            "SOXX 회복 여부",
            "NVDA / MU 흐름",
            "VIX 하락 유지",
            "유가 안정",
            "한국 외국인 수급"
        ],
        "중요도": ["높음", "높음", "중간", "높음", "높음"],
        "관련 테마": ["반도체", "AI/메모리", "전체", "전체", "한국장"],
        "메모": [
            "반도체 반등 지속 확인",
            "AI 대장주와 메모리 동조 확인",
            "위험심리 개선 확인",
            "물가 부담 재확대 여부",
            "한국장 반등 지속성 확인"
        ]
    })

    next_check_df = st.data_editor(
        next_check_df,
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic",
        key="next_check_editor"
    )

with st.expander("CSV / 리포트 다운로드"):
    today = datetime.today().strftime("%Y-%m-%d")

    summary_df = pd.DataFrame({
        "항목": [
            "시장 분위기",
            "위험도",
            "매수 판단",
            "추격매수",
            "핵심 한 줄",
            "MDD 판단 메모"
        ],
        "내용": [
            market_mood,
            risk_level,
            buy_judgment,
            chase_buy,
            core_line,
            mdd_memo
        ]
    })

    report_markdown = f"""
# 시장 리포트 - {today}

## 1. 오늘 판단

- 시장 분위기: **{market_mood}**
- 위험도: **{risk_level}**
- 매수 판단: **{buy_judgment}**
- 추격매수: **{chase_buy}**
- 핵심 한 줄: **{core_line}**

## 2. 주요 시장 지표

{df_to_markdown(main_market_df)}

## 3. 핵심 뉴스 TOP 3

{df_to_markdown(core_news_df)}

## 4. 주요 리스크 TOP 3

{df_to_markdown(core_risk_df)}

## 5. 테마 판단

{df_to_markdown(main_theme_df)}

## 6. 오늘 MDD 판단 참고

{mdd_memo}

## 7. 주의

이 리포트는 매수·매도 판단이 아니라 현재 시장 핵심 이슈 확인용입니다.
MDD Buy Score에는 직접 반영하지 않습니다.
"""

    st.download_button(
        label="시장 리포트 Markdown 다운로드",
        data=report_markdown.encode("utf-8-sig"),
        file_name="simple_market_report.md",
        mime="text/markdown"
    )

    st.download_button(
        label="전체 수집 뉴스 CSV 다운로드",
        data=news_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="collected_market_news.csv",
        mime="text/csv"
    )

    st.download_button(
        label="오늘 판단 요약 CSV 다운로드",
        data=summary_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="today_market_summary.csv",
        mime="text/csv"
    )
