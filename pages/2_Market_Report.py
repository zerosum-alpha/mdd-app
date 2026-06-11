import streamlit as st
import pandas as pd
import yfinance as yf
import FinanceDataReader as fdr
from datetime import datetime, timedelta
from auth import require_login, logout_button

# =========================================================
# 시장 리포트 페이지 FINAL
# 목적:
# - MDD 계산 로직과 분리
# - 주요 시장 지표 자동 조회
# - 주요 뉴스 / 리스크 / SNS / 테마 영향은 기본 리포트 형태로 제공
# - 사용자가 수정하면 아래 리포트 출력도 같이 변경
# - CSV / Markdown 다운로드 지원
# =========================================================

st.set_page_config(page_title="시장 리포트", layout="wide")

require_login()
logout_button()

st.title("📰 시장 리포트")

st.warning(
    "이 페이지는 시장 판단 참고용입니다. "
    "MDD Buy Score 계산에는 직접 반영하지 않습니다. "
    "뉴스·리스크·SNS 내용은 사용자가 직접 수정해서 당일 리포트로 관리합니다."
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


def safe_value(value):
    if pd.isna(value) or value is None:
        return "-"
    return str(value)


def format_pct(value):
    if pd.isna(value) or value is None:
        return "-"
    return f"{value:.2f}%"


def market_comment(change_pct):
    if change_pct is None or pd.isna(change_pct):
        return "데이터 확인 필요"
    if change_pct >= 1.0:
        return "강한 반등"
    if change_pct >= 0.3:
        return "상승 우위"
    if change_pct > -0.3:
        return "중립"
    if change_pct > -1.0:
        return "약세"
    return "리스크 확대"


def default_news_df():
    return pd.DataFrame({
        "뉴스 제목": [
            "미국 물가·금리 이벤트 대기",
            "반도체 섹터 반등 여부 확인",
            "AI 인프라 투자 부담 논란",
            "우주·SpaceX 관련 기대감 지속"
        ],
        "출처": [
            "경제 캘린더 / 주요 보도",
            "SOXX / MU / NVDA 흐름",
            "Reuters / 시장 보도",
            "Reuters / 시장 보도"
        ],
        "구분": [
            "공식",
            "보도",
            "보도",
            "보도"
        ],
        "영향": [
            "중립",
            "긍정",
            "부정",
            "긍정"
        ],
        "관련 테마": [
            "나스닥, 전체",
            "메모리, 엔비디아",
            "전력, AI 인프라",
            "우주"
        ],
        "단기 영향": [
            "중립",
            "긍정",
            "부정",
            "긍정"
        ],
        "중기 영향": [
            "확인 부족",
            "긍정",
            "중립",
            "확인 부족"
        ],
        "메모": [
            "CPI/PPI/FOMC 등 이벤트 전후 변동성 확대 가능성",
            "SOXX, MU, NVDA 회복 여부가 MDD 신호 신뢰도에 중요",
            "AI 수요는 유지되나 CapEx 부담 뉴스 반복 시 전력·인프라 ETF 부담",
            "이벤트성 기대는 가능하나 재료소멸과 변동성 주의"
        ]
    })


def default_risk_df():
    return pd.DataFrame({
        "리스크 제목": [
            "지정학·전쟁 리스크",
            "유가 급등 가능성",
            "미국 금리 재상승",
            "달러 강세",
            "ETF 리밸런싱 / 수급 왜곡"
        ],
        "구분": [
            "지정학",
            "유가",
            "금리",
            "환율",
            "수급"
        ],
        "영향": [
            "부정",
            "부정",
            "부정",
            "부정",
            "부정"
        ],
        "관련 테마": [
            "전체",
            "나스닥, 한국장",
            "나스닥, 반도체",
            "한국장, 해외 ETF",
            "한국장, 반도체"
        ],
        "확인 상태": [
            "보도",
            "공식",
            "공식",
            "공식",
            "리서치"
        ],
        "메모": [
            "전쟁·제재·해상 운송 리스크 확대 시 위험자산 부담",
            "유가 상승은 물가 재부담 → 성장주 할인율 부담",
            "10년물 금리 상승 시 QQQ, SOXX, AI 성장주 부담",
            "원화 약세는 한국 상장 해외 ETF에는 단기 방어, 신규 진입에는 부담",
            "리밸런싱 시 특정 대형주 수급 왜곡 가능"
        ]
    })


def default_sns_df():
    return pd.DataFrame({
        "내용": [
            "데이터센터 투자 지연설",
            "특정 AI·반도체 ETF 편입 기대",
            "SpaceX 관련 우주주 관심 증가",
            "메모리 가격 반등 기대 확산"
        ],
        "출처": [
            "X / 커뮤니티",
            "커뮤니티",
            "Reddit / X",
            "커뮤니티 / 리서치 언급"
        ],
        "신뢰도": [
            "낮음",
            "낮음",
            "보통",
            "보통"
        ],
        "영향": [
            "부정",
            "긍정",
            "긍정",
            "긍정"
        ],
        "관련 테마": [
            "전력",
            "한국장",
            "우주",
            "메모리"
        ],
        "확인 필요 메모": [
            "공식 보도 또는 기업 가이던스 확인 필요",
            "ETF 공시·구성종목 변경 확인 필요",
            "실제 거래량과 관련주 동반 상승 여부 확인",
            "MU, WDC, HBM 관련 뉴스와 가격 반응 확인"
        ]
    })


def default_theme_df():
    return pd.DataFrame({
        "테마": [
            "나스닥",
            "메모리",
            "엔비디아",
            "전력",
            "구글",
            "우주",
            "한국장"
        ],
        "단기 영향": [
            "중립",
            "긍정",
            "중립",
            "중립",
            "중립",
            "중립",
            "중립"
        ],
        "중기 영향": [
            "긍정",
            "긍정",
            "긍정",
            "긍정",
            "중립",
            "확인 부족",
            "중립"
        ],
        "핵심 근거": [
            "QQQ 회복 여부, 금리 안정 여부",
            "MU, SOXX 반등과 HBM·메모리 업황 기대",
            "AI 대장주 지위 유지, NVDA 가격 방어 여부",
            "AI 전력 수요는 유효하나 CapEx 부담 뉴스 존재",
            "AI·클라우드 노출은 있으나 단기 주도력 확인 필요",
            "SpaceX 기대감은 있으나 이벤트성 변동성 큼",
            "KOSPI·KOSDAQ·EWY·환율·외국인 수급 확인 필요"
        ],
        "주의점": [
            "금리·유가·VIX 상승 시 반등 제한",
            "급반등 후 차익실현 가능성",
            "고점권 변동성, 5일선 이탈 여부",
            "데이터센터 지연설, 금리 부담",
            "빅테크 내 상대 강도 약화 가능성",
            "재료 소멸, 관련 종목 실질성 부족",
            "외국인 매도, 리밸런싱, 환율 부담"
        ]
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
            "MDD가 깊은 종목은 소액 선진입 가능하나, 시장 지표 회복과 주요 리스크 완화 확인 필요"
        ]
    })


def default_next_check_df():
    return pd.DataFrame({
        "확인할 것": [
            "SOXX 회복 여부",
            "MU / 메모리주 반등 지속",
            "NVDA 주요 가격대 유지",
            "QQQ 5일선 회복 여부",
            "VIX 안정 여부",
            "유가 안정 여부",
            "한국 외국인 수급"
        ],
        "중요도": [
            "높음",
            "높음",
            "높음",
            "높음",
            "중간",
            "높음",
            "높음"
        ],
        "관련 테마": [
            "메모리, 반도체",
            "메모리",
            "엔비디아, AI",
            "나스닥",
            "전체",
            "전체",
            "한국장"
        ],
        "메모": [
            "반도체 반등 지속성 확인",
            "MDD 저점매수 신뢰도에 직접 영향",
            "AI 대장주 방어 여부 확인",
            "성장주 반등 지속 조건",
            "공포 완화 여부 확인",
            "물가 부담 재확대 여부 확인",
            "한국 ETF·종목 반등 지속 여부 확인"
        ]
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
        index=1
    )

today_summary = st.text_input(
    "오늘 핵심 한 줄",
    value="반도체·AI 주요 지표 회복 여부가 MDD 저점매수 신뢰도를 결정"
)

today_key_risk = st.text_input(
    "오늘 핵심 리스크",
    value="유가, 금리, 지정학, 주요 이벤트 전후 변동성"
)

today_positive = st.text_input(
    "오늘 긍정 요인",
    value="SOXX, MU, QQQ 회복 시 MDD 깊은 종목의 소액 선진입 신뢰도 상승"
)

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
        "판단 메모": market_comment(change_pct)
    })

kospi_close, kospi_change = load_fdr_index_latest("KS11")
kosdaq_close, kosdaq_change = load_fdr_index_latest("KQ11")

market_rows.append({
    "표시명": "KOSPI",
    "티커": "KS11",
    "현재가": None if kospi_close is None else round(kospi_close, 2),
    "등락률(%)": None if kospi_change is None else round(kospi_change, 2),
    "판단 메모": market_comment(kospi_change)
})

market_rows.append({
    "표시명": "KOSDAQ",
    "티커": "KQ11",
    "현재가": None if kosdaq_close is None else round(kosdaq_close, 2),
    "등락률(%)": None if kosdaq_change is None else round(kosdaq_change, 2),
    "판단 메모": market_comment(kosdaq_change)
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

st.write("아래 기본 뉴스 리포트를 당일 상황에 맞게 수정하세요.")

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
# 9. 오늘 시장 리포트 출력
# =========================================================
st.markdown("## 9. 오늘 시장 리포트 출력")

today = datetime.today().strftime("%Y-%m-%d")

market_table_md = edited_market_indicator_df.to_markdown(index=False)
news_table_md = news_df.to_markdown(index=False)
risk_table_md = risk_df.to_markdown(index=False)
sns_table_md = sns_df.to_markdown(index=False)
theme_table_md = theme_df.to_markdown(index=False)
mdd_ref_table_md = mdd_ref_df.to_markdown(index=False)
next_check_table_md = next_check_df.to_markdown(index=False)

report_markdown = f"""
# 시장 리포트 - {today}

## 1. 오늘 결론

- 시장 분위기: **{market_mood}**
- 시장 위험도: **{market_risk}**
- 매수 분위기: **{buy_mood}**
- 핵심 한 줄: **{today_summary}**
- 핵심 리스크: **{today_key_risk}**
- 긍정 요인: **{today_positive}**

## 2. 주요 시장 지표

{market_table_md}

## 3. 주요 뉴스

{news_table_md}

## 4. 주요 리스크

{risk_table_md}

## 5. SNS·소문·찌라시

주의: 이 항목은 확정 사실이 아니라 참고용 비공식 신호다.

{sns_table_md}

## 6. 테마별 영향

{theme_table_md}

## 7. 오늘 MDD 판단 참고

{mdd_ref_table_md}

## 8. 다음 확인사항

{next_check_table_md}

## 9. 최종 메모

오늘 MDD 신호는 단독 매수 신호가 아니다.  
시장 지표, 주요 뉴스, 리스크, 테마 영향과 함께 확인해야 한다.  
특히 SOXX, QQQ, MU, NVDA, VIX, 유가, 금리 흐름이 MDD 저점매수 신뢰도를 결정한다.
"""

st.markdown(report_markdown)

st.download_button(
    label="오늘 시장 리포트 Markdown 다운로드",
    data=report_markdown.encode("utf-8-sig"),
    file_name="today_market_report.md",
    mime="text/markdown"
)


# =========================================================
# 10. 전체 CSV 다운로드
# =========================================================
st.markdown("## 10. 전체 CSV 다운로드")

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
