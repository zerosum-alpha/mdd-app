import streamlit as st
from auth import require_login, logout_button

st.set_page_config(page_title="MDD 저점매수 분석기", layout="wide")

require_login()
logout_button()

st.title("📈 MDD 저점매수 분석기")

st.markdown("## 메뉴 선택")

st.write("아래 버튼을 눌러 원하는 페이지로 이동하세요.")

col1, col2 = st.columns(2)

with col1:
    st.page_link(
        "pages/1_MDD.py",
        label="📊 MDD 분석기 열기",
        icon="📊"
    )

with col2:
    st.page_link(
        "pages/2_Market_Report.py",
        label="📰 시장 리포트 열기",
        icon="📰"
    )

st.markdown("---")

st.markdown("""
## 페이지 설명

| 페이지 | 내용 |
|---|---|
| MDD 분석기 | 종목별 Current DD, Max DD, RSI, Buy Score, 차트 확인 |
| 시장 리포트 | 주요 뉴스, 시장 상황, 리스크, SNS·소문, 테마별 영향 정리 |

---

## 사용 순서

1. **MDD 분석기**에서 종목별 낙폭과 Buy Score 확인
2. **시장 리포트**에서 오늘 뉴스·리스크·테마 영향 정리
3. 두 화면을 같이 보고 저점매수·물타기 판단 보조

주의: 시장 리포트는 참고용입니다.  
MDD Buy Score 계산에는 직접 반영하지 않습니다.
""")
