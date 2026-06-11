import streamlit as st

PASSWORD = "1234"


def require_login():
    if "login_ok" not in st.session_state:
        st.session_state["login_ok"] = False

    if st.session_state["login_ok"]:
        return

    st.title("🔒 로그인")

    pw = st.text_input("비밀번호", type="password")

    if st.button("로그인"):
        if pw == PASSWORD:
            st.session_state["login_ok"] = True
            st.rerun()
        else:
            st.error("비밀번호가 틀렸습니다.")

    st.stop()


def logout_button():
    with st.sidebar:
        if st.button("로그아웃"):
            st.session_state["login_ok"] = False
            st.rerun()
