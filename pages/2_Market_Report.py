import streamlit as st
import pandas as pd
import yfinance as yf
import FinanceDataReader as fdr
from datetime import datetime, timedelta
from auth import require_login, logout_button

# =========================================================
# 시장 리포트 페이지
# - MDD 계산 로직과 분리
# - 자동 매수판단 없음
# - 뉴스·리스크·SNS·테마 영향 수동 정리
# - CSV 다운로드 지원
# =========================================================

st.set_page_config(page_title="시장 리포트", layout="wide")

require_login()
logout_button()

st.title("📰 시장 리포트")

st.warning(
    "이 페이지는 참고용입니다. "
    "뉴스 자동 크롤링, 자동 위험점수, Buy Score 강제 연동, 자동 매수추천은 포함하지 않습니다."
)


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


def make_csv_download(df, filename, button_label):
    csv = df.to_csv(index=False).encode("utf-8-sig")

    st.download_button(
        label=button_label,
        data=csv,
        file_name=filename,
        mime="text/csv"
    )


def default_news_df():
    return pd.DataFrame({
        "뉴스 제목": ["", "", ""],
        "출처": ["Reuters", "Bloomberg", "연합뉴스"],
        "구분": ["보도", "보도", "공식"],
        "영향": ["중립", "중립", "중립"],
        "관련 테마": ["전체", "나스닥", "한국장"],
        "단기 영향": ["중립", "중립", "중립"],
        "중기 영향": ["확인 부족", "확인 부족", "확인 부족"],
        "메모": ["", "", ""]
    })


def default_risk_df():
    return pd.DataFrame({
        "리스크 제목": ["", "", ""],
        "구분": ["지정학", "유가", "금리"],
        "영향": ["중립", "중립", "중립"],
        "관련 테마": ["전체", "나스닥", "한국장"],
        "확인 상태": ["보도", "공식", "공식"],
        "메모": ["", "", ""]
    })


def default_sns_df():
    return pd.DataFrame({
        "내용": ["", "", ""],
        "출처": ["X", "커뮤니티", "Reddit"],
        "신뢰도": ["낮음", "낮음", "보통"],
        "영향": ["중립", "중립", "중립"],
        "관련 테마": ["전체", "한국장", "우주"],
        "확인 필요 메모": ["공식 확인 필요", "공시 확인 필요", "거래량 확인 필요"]
    })


def default_theme_df():
    return pd.DataFrame({
        "테마": ["나스닥", "메모리", "엔비디아", "전력", "구글", "우주", "한국장"],
        "단기 영향": ["중립", "중립", "중립", "중립", "중립", "중립", "중립"],
        "중기 영향": ["중립", "중립", "중립", "중립", "중립", "확인 부족", "중립"],
        "핵심 근거": ["", "", "", "", "", "", ""],
        "주의점": ["", "", "", "", "", "", ""]
    })


def default_mdd_ref_df():
    return pd.DataFrame({
        "항목": [
            "오늘 MDD 신호 신뢰도",
            "소액 선진입 가능 여부",
            "확인매수 필요 여부",
            "추격 매수",
            "오늘 판단 메모"
        ],
        "내용": [
            "보통",
            "조건부",
            "필요",
            "금지",
            ""
        ]
    })


def default_next_check_df():
    return pd.DataFrame({
        "확인할 것": [
            "SOXX 회복 여부",
            "NVDA 주요 가격대 유지",
            "유가 안정",
            "미국 주요 이벤트",
            "한국 외국인 수급"
        ],
        "중요도": ["높음", "높음", "높음", "중간", "높음"],
        "관련 테마": ["메모리, 반도체", "엔비디아, AI", "전체", "나스닥", "한국장"],
        "메모": ["", "", "", "", ""]
    })


# =========================================================
# 1. 오늘 시장 요약
# =========================================================
st.markdown("## 1. 오늘 시장 요약")

c1, c2, c3 = st.columns(3)

with c1:
    market_mood = st.selectbox(
        "오늘 시장 분위기",
        ["긍정", "중립", "부정"],
        index=1
    )

with c2:
    market_risk = st.selectbox(
        "시장 위험도",
        ["낮음", "보통", "높음", "매우 높음"],
        index=1
    )

with c3:
    buy_mood = st.selectbox(
        "매수 분위기",
        ["가능", "소액 가능", "대기", "금지"],
        index=2
    )

today_summary = st.text_input("오늘 핵심 한 줄", value="")
today_key_risk = st.text_input("오늘 핵심 리스크", value="")
today_positive = st.text_input("오늘 긍정 요인", value="")

summary_df = pd.DataFrame({
    "항목": [
        "오늘 시장 분위기",
        "시장 위험도",
        "매수 분위기",
        "오늘 핵심 한 줄",
        "오늘 핵심 리스크",
        "오늘 긍정 요인"
    ],
    "내용": [
        market_mood,
        market_risk,
        buy_mood,
        today_summary,
        today_key_risk,
        today_positive
    ]
})

make_csv_download(
    summary_df,
    "today_market_summary.csv",
    "오늘 시장 요약 CSV 다운로드"
)


# =========================================================
# 2. 주요 시장 지표
# =========================================================
st.markdown("## 2. 주요 시장 지표")

st.info(
    "기본값은 주요 증시·매크로 지표입니다. "
    "필요하면 티커를 직접 추가하거나 삭제하세요."
)

default_tickers = "QQQ, SPY, SOXX, IWM, EWY, ^VIX, ^TNX, USO, UUP"

ticker_text = st.text_input(
    "조회할 티커 목록",
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
        "현재가": None if close is None else round(close, 2),
        "등락률(%)": None if change_pct is None else round(change_pct, 2),
        "판단 메모": ""
    })

kospi_close, kospi_change = load_fdr_index_latest("KS11")
kosdaq_close, kosdaq_change = load_fdr_index_latest("KQ11")

market_rows.append({
    "표시명": "KOSPI",
    "티커": "KS11",
    "현재가": None if kospi_close is None else round(kospi_close, 2),
    "등락률(%)": None if kospi_change is None else round(kospi_change, 2),
    "판단 메모": ""
})

market_rows.append({
    "표시명": "KOSDAQ",
    "티커": "KQ11",
    "현재가": None if kosdaq_close is None else round(kosdaq_close, 2),
    "등락률(%)": None if kosdaq_change is None else round(kosdaq_change, 2),
    "판단 메모": ""
})

market_indicator_df = pd.DataFrame(market_rows)

edited_market_indicator_df = st.data_editor(
    market_indicator_df,
    use_container_width=True,
    num_rows="dynamic",
    key="market_indicator_editor"
)

make_csv_download(
    edited_market_indicator_df,
    "market_indicators.csv",
    "주요 시장 지표 CSV 다운로드"
)


# =========================================================
# 3. 주요 뉴스
# =========================================================
st.markdown("## 3. 주요 뉴스")

st.write("공식 뉴스와 보도 뉴스를 정리하는 영역입니다.")

news_df = st.data_editor(
    default_news_df(),
    use_container_width=True,
    num_rows="dynamic",
    column_config={
        "구분": st.column_config.SelectboxColumn(
            "구분",
            options=["공식", "보도"]
        ),
        "영향": st.column_config.SelectboxColumn(
            "영향",
            options=["긍정", "중립", "부정"]
        ),
        "단기 영향": st.column_config.SelectboxColumn(
            "단기 영향",
            options=["긍정", "중립", "부정"]
        ),
        "중기 영향": st.column_config.SelectboxColumn(
            "중기 영향",
            options=["긍정", "중립", "부정", "확인 부족"]
        )
    },
    key="news_editor"
)

make_csv_download(
    news_df,
    "market_news.csv",
    "주요 뉴스 CSV 다운로드"
)


# =========================================================
# 4. 주요 리스크
# =========================================================
st.markdown("## 4. 주요 리스크")

st.write("전쟁, 지정학, 유가, 금리, 환율, 제재, 관세, 공급망, 금융 리스크를 정리합니다.")

risk_df = st.data_editor(
    default_risk_df(),
    use_container_width=True,
    num_rows="dynamic",
    column_config={
        "구분": st.column_config.SelectboxColumn(
            "구분",
            options=[
                "지정학", "유가", "금리", "환율", "제재", "관세",
                "공급망", "금융", "사이버", "수급", "산업", "기타"
            ]
        ),
        "영향": st.column_config.SelectboxColumn(
            "영향",
            options=["긍정", "중립", "부정"]
        ),
        "확인 상태": st.column_config.SelectboxColumn(
            "확인 상태",
            options=["공식", "보도", "비공식", "찌라시", "리서치"]
        )
    },
    key="risk_editor"
)

make_csv_download(
    risk_df,
    "market_risks.csv",
    "주요 리스크 CSV 다운로드"
)


# =========================================================
# 5. SNS·소문·찌라시
# =========================================================
st.markdown("## 5. SNS·소문·찌라시")

st.warning(
    "이 섹션은 확정 사실이 아닙니다. "
    "SNS, 커뮤니티, X, Reddit, 유튜브, 텔레그램 등 비공식 신호를 참고용으로만 기록합니다."
)

sns_df = st.data_editor(
    default_sns_df(),
    use_container_width=True,
    num_rows="dynamic",
    column_config={
        "신뢰도": st.column_config.SelectboxColumn(
            "신뢰도",
            options=["낮음", "보통", "높음"]
        ),
        "영향": st.column_config.SelectboxColumn(
            "영향",
            options=["긍정", "중립", "부정"]
        )
    },
    key="sns_editor"
)

make_csv_download(
    sns_df,
    "sns_rumors.csv",
    "SNS_소문_찌라시_CSV_다운로드"
)


# =========================================================
# 6. 테마별 영향표
# =========================================================
st.markdown("## 6. 테마별 영향표")

theme_df = st.data_editor(
    default_theme_df(),
    use_container_width=True,
    num_rows="dynamic",
    column_config={
        "단기 영향": st.column_config.SelectboxColumn(
            "단기 영향",
            options=["긍정", "중립", "부정"]
        ),
        "중기 영향": st.column_config.SelectboxColumn(
            "중기 영향",
            options=["긍정", "중립", "부정", "확인 부족"]
        )
    },
    key="theme_editor"
)

make_csv_download(
    theme_df,
    "theme_impact.csv",
    "테마별 영향표 CSV 다운로드"
)


# =========================================================
# 7. 오늘 MDD 판단 참고
# =========================================================
st.markdown("## 7. 오늘 MDD 판단 참고")

st.info(
    "이 섹션은 MDD Buy Score를 자동으로 바꾸지 않습니다. "
    "오늘 MDD 신호를 어떻게 해석할지 메모하는 참고 영역입니다."
)

mdd_ref_df = st.data_editor(
    default_mdd_ref_df(),
    use_container_width=True,
    num_rows="dynamic",
    key="mdd_reference_editor"
)

make_csv_download(
    mdd_ref_df,
    "mdd_reference.csv",
    "오늘 MDD 판단 참고 CSV 다운로드"
)


# =========================================================
# 8. 다음 확인사항
# =========================================================
st.markdown("## 8. 다음 확인사항")

next_check_df = st.data_editor(
    default_next_check_df(),
    use_container_width=True,
    num_rows="dynamic",
    column_config={
        "중요도": st.column_config.SelectboxColumn(
            "중요도",
            options=["높음", "중간", "낮음"]
        )
    },
    key="next_check_editor"
)

make_csv_download(
    next_check_df,
    "next_check.csv",
    "다음 확인사항 CSV 다운로드"
)


# =========================================================
# 전체 리포트 통합 다운로드
# =========================================================
st.markdown("## 전체 리포트 통합 다운로드")

combined_report = {
    "today_summary": summary_df,
    "market_indicators": edited_market_indicator_df,
    "news": news_df,
    "risks": risk_df,
    "sns_rumors": sns_df,
    "theme_impact": theme_df,
    "mdd_reference": mdd_ref_df,
    "next_check": next_check_df
}

combined_csv_parts = []

for section_name, section_df in combined_report.items():
    combined_csv_parts.append(f"\n[{section_name}]\n")
    combined_csv_parts.append(section_df.to_csv(index=False))

combined_csv = "\n".join(combined_csv_parts).encode("utf-8-sig")

st.download_button(
    label="전체 시장 리포트 CSV 다운로드",
    data=combined_csv,
    file_name="full_market_report.csv",
    mime="text/csv"
)

st.warning(
    "저장 버튼은 CSV 다운로드 방식입니다. "
    "화면을 새로고침하면 입력 내용은 사라질 수 있으므로, 작성 후 반드시 다운로드하세요."
)
