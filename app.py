import streamlit as st
from auth import require_login, logout_button

st.set_page_config(page_title="MDD 저점매수 분석기", layout="wide")

require_login()
logout_button()

st.title("📈 MDD 저점매수 분석기")

st.markdown("""
## 메뉴 안내

왼쪽 사이드바에서 페이지를 선택하세요.

| 페이지 | 내용 |
|---|---|
| MDD 분석기 | 종목별 Current DD, Max DD, RSI, Buy Score, 차트 확인 |
| 시장 리포트 | 주요 뉴스, 시장 상황, 리스크, SNS·소문, 테마별 영향 정리 |

---

## 사용 순서

1. 왼쪽 사이드바에서 **MDD 분석기** 선택
2. 종목명 또는 티커 입력
3. MDD, RSI, Buy Score 확인
4. 왼쪽 사이드바에서 **시장 리포트** 선택
5. 오늘 주요 뉴스, 리스크, 테마 영향 메모
6. 두 화면을 같이 보면서 저점매수 판단 보조

주의: 시장 리포트는 참고용입니다.  
MDD Buy Score 계산에는 직접 반영하지 않습니다.
""")
