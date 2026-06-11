import streamlit as st
import pandas as pd
import yfinance as yf
import FinanceDataReader as fdr
import feedparser
from datetime import datetime, timedelta
from auth import require_login, logout_button

# =========================================================
# Market Report Page - Live Issue Scanner
# 목적:
# - 실시간 주요 시장 지표 확인
# - 한국어 RSS 뉴스 수집
# - 현재 시장 주요 이슈 자동 분류
# - 테마별 뉴스 연결
# - 매수/매도 판단 없음
# - MDD Buy Score 반영 없음
# =========================================================

st.set_page_config(page_title="시장 리포트", layout="wide")

require_login()
logout_button()

st.title("📰 시장 리포트")

st.warning(
    "이 페이지는 현재 시장 주요 이슈 확인용입니다. "
    "매수/매도 판단이나 MDD Buy Score에는 직접 반영하지 않습니다."
)

# =========================================================
# RSS Sources - 한국어 중심
# =========================================================
RSS_SOURCES = {
    "Investing KR 전체 뉴스": "https://kr.investing.com/rss/news.rss",
    "Investing KR 주식 뉴스": "https://kr.investing.com/rss/news_25.rss",
    "Investing KR 경제 뉴스": "https://kr.investing.com/rss/news_14.rss",
    "Investing KR 경제지표 뉴스": "https://kr.investing.com/rss/news_95.rss",
    "Investing KR 상품 뉴스": "https://kr.investing.com/rss/news_11.rss",
    "Investing KR IPO 뉴스": "https://kr.investing.com/rss/news_450.rss",
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

    # 일반 지수/ETF: 상승이 긍정
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
                request_headers={
                    "User-Agent": "Mozilla/5.0"
                }
            )

            for entry in feed.entries[:max_items_per_source]:
                title = getattr(entry, "title", "")
                link = getattr(entry, "link", "")
                published = getattr(entry, "published", "")

                if title:
                    rows.append({
                        "출처": source_name,
                        "뉴스 제목": title,
                        "발행": published,
                        "링크": link
                    })

        except Exception:
            continue

    if not rows:
        return pd.DataFrame(columns=["출처", "뉴스 제목", "발행", "링크"])

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["뉴스 제목"])
    return df


def classify_news(title):
    text = str(title).lower()

    issue_rules = {
        "전쟁·지정학": [
            "war", "attack", "missile", "iran", "israel", "gaza",
            "russia", "ukraine", "conflict", "geopolitical",
            "sanction", "military", "red sea", "houthi",
            "전쟁", "공격", "미사일", "이란", "이스라엘", "가자",
            "러시아", "우크라이나", "분쟁", "지정학", "제재", "군사",
            "홍해", "후티", "호르무즈"
        ],
        "유가·원자재": [
            "oil", "crude", "brent", "wti", "opec", "gas",
            "energy", "commodity", "gold", "copper",
            "유가", "원유", "브렌트", "천연가스", "원자재",
            "금", "구리", "에너지", "opec"
        ],
        "금리·물가·연준": [
            "fed", "fomc", "rate", "yield", "treasury",
            "inflation", "cpi", "ppi", "powell", "jobless",
            "payroll", "employment",
            "연준", "fomc", "금리", "국채", "수익률",
            "물가", "인플레이션", "cpi", "ppi", "파월",
            "고용", "실업수당", "비농업", "채권"
        ],
        "AI·반도체": [
            "nvidia", "ai", "artificial intelligence", "semiconductor",
            "chip", "chips", "micron", "broadcom", "amd", "intel",
            "tsmc", "memory", "hbm", "dram",
            "엔비디아", "인공지능", "ai", "반도체", "칩",
            "마이크론", "브로드컴", "amd", "인텔", "tsmc",
            "메모리", "hbm", "dram", "sk하이닉스", "삼성전자"
        ],
        "전력·AI인프라": [
            "power", "electricity", "grid", "data center", "datacenter",
            "utility", "energy infrastructure", "vertiv", "ge vernova",
            "전력", "전력망", "전기", "데이터센터", "데이터 센터",
            "전력인프라", "인프라", "버티브", "ge버노바", "냉각"
        ],
        "빅테크·나스닥": [
            "nasdaq", "apple", "microsoft", "amazon", "google",
            "alphabet", "meta", "tesla", "tech", "growth stocks",
            "나스닥", "애플", "마이크로소프트", "아마존", "구글",
            "알파벳", "메타", "테슬라", "빅테크", "성장주", "기술주"
        ],
        "한국장": [
            "korea", "kospi", "kosdaq", "won", "samsung",
            "sk hynix", "south korea",
            "한국", "코스피", "코스닥", "원화", "환율",
            "삼성전자", "sk하이닉스", "외국인", "기관"
        ],
        "중국·아시아": [
            "china", "hong kong", "asia", "japan", "taiwan",
            "yuan", "nikkei", "hang seng",
            "중국", "홍콩", "아시아", "일본", "대만",
            "위안", "니케이", "항셍"
        ],
        "우주·방산": [
            "spacex", "space", "rocket", "satellite", "defense",
            "aerospace",
            "스페이스x", "우주", "로켓", "위성", "방산", "항공우주"
        ],
        "실적·가이던스": [
            "earnings", "revenue", "profit", "forecast", "guidance",
            "outlook", "beat", "misses", "results",
            "실적", "매출", "이익", "가이던스", "전망",
            "어닝", "컨센서스", "상회", "하회"
        ],
        "금융·신용": [
            "bank", "credit", "debt", "default", "liquidity",
            "bond", "loan", "financial",
            "은행", "신용", "부채", "디폴트", "유동성",
            "채권", "대출", "금융"
        ]
    }

    theme_map = {
        "전쟁·지정학": "전체, 유가, 방산",
        "유가·원자재": "전체, 나스닥, 한국장",
        "금리·물가·연준": "나스닥, 반도체, 성장주",
        "AI·반도체": "메모리, 엔비디아, AI",
        "전력·AI인프라": "전력, AI 인프라",
        "빅테크·나스닥": "나스닥, 구글, 빅테크",
        "한국장": "한국장, 반도체",
        "중국·아시아": "한국장, 아시아 ETF",
        "우주·방산": "우주, 방산",
        "실적·가이던스": "해당 종목, 섹터",
        "금융·신용": "전체, 금융 리스크"
    }

    risk_issues = [
        "전쟁·지정학",
        "유가·원자재",
        "금리·물가·연준",
        "금융·신용"
    ]

    matched_issues = []

    for issue, keywords in issue_rules.items():
        if any(keyword in text for keyword in keywords):
            matched_issues.append(issue)

    if not matched_issues:
        matched_issues = ["기타"]

    issue = ", ".join(matched_issues)

    related_themes = []
    for item in matched_issues:
        if item in theme_map:
            related_themes.append(theme_map[item])

    related_theme = " / ".join(related_themes) if related_themes else "확인 필요"

    if any(item in risk_issues for item in matched_issues):
        risk_flag = "리스크 확인"
    else:
        risk_flag = "일반 이슈"

    negative_words = [
        "fall", "drop", "plunge", "risk", "war", "attack", "concern",
        "worry", "fear", "loss", "miss", "cut", "weak", "slowdown",
        "inflation", "higher rates",
        "하락", "급락", "위험", "전쟁", "공격", "우려", "공포",
        "손실", "부진", "둔화", "인플레이션", "금리 상승", "악화",
        "부담", "제재", "관세"
    ]

    positive_words = [
        "rise", "gain", "jump", "surge", "rally", "beat", "record",
        "optimism", "growth", "strong", "rebound", "upgrade",
        "상승", "급등", "반등", "랠리", "호조", "상회", "기록",
        "낙관", "성장", "강세", "개선", "상향"
    ]

    if any(word in text for word in negative_words):
        tone = "부정"
    elif any(word in text for word in positive_words):
        tone = "긍정"
    else:
        tone = "중립"

    return issue, related_theme, risk_flag, tone


def summarize_issues(news_df):
    if news_df.empty:
        return pd.DataFrame(columns=["이슈", "뉴스 수", "부정", "긍정", "중립", "관련 테마"])

    exploded_rows = []

    for _, row in news_df.iterrows():
        issues = [x.strip() for x in row["이슈"].split(",")]

        for issue in issues:
            exploded_rows.append({
                "이슈": issue,
                "영향": row["영향"],
                "관련 테마": row["관련 테마"]
            })

    issue_df = pd.DataFrame(exploded_rows)

    rows = []

    for issue, group in issue_df.groupby("이슈"):
        related_theme_text = " / ".join(
            sorted(set(group["관련 테마"].dropna().astype(str)))
        )

        rows.append({
            "이슈": issue,
            "뉴스 수": len(group),
            "부정": int((group["영향"] == "부정").sum()),
            "긍정": int((group["영향"] == "긍정").sum()),
            "중립": int((group["영향"] == "중립").sum()),
            "관련 테마": related_theme_text[:120]
        })

    result = pd.DataFrame(rows)
    result = result.sort_values(["뉴스 수", "부정"], ascending=False)

    return result


def make_issue_comment(issue_summary_df):
    if issue_summary_df.empty:
        return "현재 수집된 뉴스가 없어 주요 이슈를 판단할 수 없습니다."

    top = issue_summary_df.head(5)

    top_issues = top["이슈"].tolist()
    total_news = int(top["뉴스 수"].sum())
    total_negative = int(top["부정"].sum())
    total_positive = int(top["긍정"].sum())

    risk_issues = top[top["부정"] > top["긍정"]]["이슈"].tolist()
    positive_issues = top[top["긍정"] > top["부정"]]["이슈"].tolist()

    lines = []

    lines.append(f"현재 뉴스 기준 주요 이슈는 **{', '.join(top_issues)}** 입니다.")
    lines.append(f"상위 5개 이슈 관련 뉴스는 총 **{total_news}건**입니다.")

    if total_negative > total_positive:
        lines.append("부정 이슈 비중이 더 높아 시장 리스크 확인이 우선입니다.")
    elif total_positive > total_negative:
        lines.append("긍정 이슈 비중이 더 높지만, 시장 지표 확인은 필요합니다.")
    else:
        lines.append("긍정·부정 이슈가 혼재되어 방향성 확인이 필요합니다.")

    if risk_issues:
        lines.append(f"리스크성 이슈: **{', '.join(risk_issues)}**")
    if positive_issues:
        lines.append(f"긍정성 이슈: **{', '.join(positive_issues)}**")

    lines.append("이 결과는 매수·매도 판단이 아니라, MDD 분석 전 확인해야 할 시장 이슈 요약입니다.")

    return "\n\n".join(lines)


def df_to_markdown(df):
    if df.empty:
        return "데이터 없음"

    headers = list(df.columns)
    lines = []

    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for _, row in df.iterrows():
        values = [str(row[col]).replace("\n", " ") for col in headers]
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


# =========================================================
# 1. 주요 시장 지표
# =========================================================
st.markdown("## 1. 주요 시장 지표")

default_tickers = "QQQ, SPY, SOXX, IWM, EWY, ^VIX, ^TNX, USO, UUP"

ticker_text = st.text_input(
    "조회할 티커",
    value=default_tickers
)

ticker_list = [
    ticker.strip().upper()
    for ticker in ticker_text.split(",")
    if ticker.strip()
]

display_name_map = {
    "QQQ": "Nasdaq100",
    "SPY": "S&P500",
    "SOXX": "Semiconductor",
    "IWM": "Russell2000",
    "EWY": "Korea ETF",
    "^VIX": "VIX",
    "^TNX": "US 10Y Yield",
    "USO": "Oil",
    "UUP": "Dollar"
}

market_rows = []

for ticker in ticker_list:
    close, change_pct = load_yfinance_latest(ticker)

    market_rows.append({
        "표시명": display_name_map.get(ticker, ticker),
        "티커": ticker,
        "현재가": format_number(close),
        "등락률": format_pct(change_pct),
        "판단 메모": market_comment(ticker, change_pct)
    })

kospi_close, kospi_change = load_fdr_index_latest("KS11")
kosdaq_close, kosdaq_change = load_fdr_index_latest("KQ11")

market_rows.append({
    "표시명": "KOSPI",
    "티커": "KS11",
    "현재가": format_number(kospi_close),
    "등락률": format_pct(kospi_change),
    "판단 메모": market_comment("KS11", kospi_change)
})

market_rows.append({
    "표시명": "KOSDAQ",
    "티커": "KQ11",
    "현재가": format_number(kosdaq_close),
    "등락률": format_pct(kosdaq_change),
    "판단 메모": market_comment("KQ11", kosdaq_change)
})

market_df = pd.DataFrame(market_rows)
st.dataframe(market_df, use_container_width=True)


# =========================================================
# 2. 실시간 뉴스 수집
# =========================================================
st.markdown("## 2. 실시간 뉴스 수집")

col1, col2 = st.columns([2, 1])

with col1:
    selected_sources = st.multiselect(
        "뉴스 소스 선택",
        list(RSS_SOURCES.keys()),
        default=[
            "Investing KR 전체 뉴스",
            "Investing KR 주식 뉴스",
            "Investing KR 경제지표 뉴스",
            "Investing KR 상품 뉴스"
        ]
    )

with col2:
    max_items = st.number_input(
        "소스별 뉴스 개수",
        min_value=5,
        max_value=30,
        value=15,
        step=5
    )

raw_news_df = load_rss_news(selected_sources, max_items)

if raw_news_df.empty:
    st.error("뉴스를 가져오지 못했습니다. RSS 소스를 확인하세요.")
    st.stop()

classified_rows = []

for _, row in raw_news_df.iterrows():
    issue, related_theme, risk_flag, tone = classify_news(row["뉴스 제목"])

    classified_rows.append({
        "뉴스 제목": row["뉴스 제목"],
        "출처": row["출처"],
        "이슈": issue,
        "관련 테마": related_theme,
        "영향": tone,
        "구분": risk_flag,
        "발행": row["발행"],
        "링크": row["링크"]
    })

news_df = pd.DataFrame(classified_rows)

st.dataframe(
    news_df,
    use_container_width=True,
    column_config={
        "링크": st.column_config.LinkColumn("링크")
    }
)


# =========================================================
# 3. 주요 이슈 Top 10
# =========================================================
st.markdown("## 3. 주요 이슈 Top 10")

issue_summary_df = summarize_issues(news_df)

st.dataframe(
    issue_summary_df.head(10),
    use_container_width=True
)

st.markdown("### 오늘 주요 이슈 요약")
issue_comment = make_issue_comment(issue_summary_df)
st.markdown(issue_comment)


# =========================================================
# 4. 리스크성 이슈
# =========================================================
st.markdown("## 4. 리스크성 이슈")

risk_news_df = news_df[
    (news_df["구분"] == "리스크 확인") | (news_df["영향"] == "부정")
].copy()

if risk_news_df.empty:
    st.success("현재 RSS 기준 뚜렷한 리스크성 뉴스 비중은 낮습니다.")
else:
    st.dataframe(
        risk_news_df[["뉴스 제목", "출처", "이슈", "관련 테마", "영향", "링크"]],
        use_container_width=True,
        column_config={
            "링크": st.column_config.LinkColumn("링크")
        }
    )


# =========================================================
# 5. 테마별 뉴스 연결
# =========================================================
st.markdown("## 5. 테마별 뉴스 연결")

theme_issue_map = {
    "나스닥": ["빅테크·나스닥", "금리·물가·연준"],
    "메모리": ["AI·반도체"],
    "엔비디아": ["AI·반도체"],
    "전력": ["전력·AI인프라"],
    "구글": ["빅테크·나스닥"],
    "우주": ["우주·방산"],
    "한국장": ["한국장", "중국·아시아", "금리·물가·연준"]
}

theme_rows = []

for theme, issue_keys in theme_issue_map.items():
    matched = news_df[
        news_df["이슈"].apply(lambda x: any(issue_key in x for issue_key in issue_keys))
    ]

    if not matched.empty:
        issue_counts = (
            matched["이슈"]
            .str.split(", ")
            .explode()
            .value_counts()
            .head(3)
        )
        main_issues = ", ".join(issue_counts.index.tolist())
    else:
        main_issues = "-"

    theme_rows.append({
        "테마": theme,
        "관련 뉴스 수": len(matched),
        "부정 뉴스": int((matched["영향"] == "부정").sum()) if not matched.empty else 0,
        "긍정 뉴스": int((matched["영향"] == "긍정").sum()) if not matched.empty else 0,
        "주요 이슈": main_issues
    })

theme_df = pd.DataFrame(theme_rows)
theme_df = theme_df.sort_values("관련 뉴스 수", ascending=False)

st.dataframe(theme_df, use_container_width=True)


# =========================================================
# 6. 오늘 시장 리포트 출력
# =========================================================
st.markdown("## 6. 오늘 시장 리포트 출력")

today = datetime.today().strftime("%Y-%m-%d")

top_issue_md = df_to_markdown(issue_summary_df.head(10))
market_md = df_to_markdown(market_df)
theme_md = df_to_markdown(theme_df)

if not risk_news_df.empty:
    risk_md = df_to_markdown(
        risk_news_df[["뉴스 제목", "출처", "이슈", "관련 테마", "영향"]].head(15)
    )
else:
    risk_md = "리스크성 뉴스가 뚜렷하게 많지 않음"

report_markdown = f"""
# 시장 주요 이슈 리포트 - {today}

## 1. 핵심 요약

{issue_comment}

## 2. 주요 시장 지표

{market_md}

## 3. 주요 이슈 Top 10

{top_issue_md}

## 4. 리스크성 뉴스

{risk_md}

## 5. 테마별 뉴스 연결

{theme_md}

## 6. MDD 분석 참고

- 이 리포트는 매수·매도 판단이 아니다.
- 현재 시장을 움직이는 주요 이슈를 확인하기 위한 참고 화면이다.
- MDD 분석 전 QQQ, SOXX, MU/NVDA, VIX, 금리, 유가, 달러 흐름을 함께 확인한다.
- VIX 하락은 보통 위험심리 개선으로 해석한다.
- 금리·유가·달러 상승은 성장주와 한국장에 부담이 될 수 있다.
"""

st.markdown(report_markdown)

st.download_button(
    label="시장 주요 이슈 리포트 Markdown 다운로드",
    data=report_markdown.encode("utf-8-sig"),
    file_name="market_issue_report.md",
    mime="text/markdown"
)

csv_data = news_df.to_csv(index=False).encode("utf-8-sig")

st.download_button(
    label="수집 뉴스 CSV 다운로드",
    data=csv_data,
    file_name="collected_market_news.csv",
    mime="text/csv"
)
