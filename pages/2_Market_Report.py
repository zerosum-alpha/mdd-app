# -*- coding: utf-8 -*-
"""
시장 리포트 탭 - 뉴스/테마/돈의 흐름/대장주·후발주·숨은 후보
MDD 계산 로직과 분리된 독립 Streamlit page 코드입니다.
requirements.txt 변경 없음: streamlit, pandas, numpy, yfinance, feedparser 사용
"""

from __future__ import annotations

import re
import math
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import feedparser


# =========================
# 기본 설정
# =========================
st.set_page_config(page_title="시장 리포트", page_icon="📰", layout="wide")

TODAY = pd.Timestamp.today().normalize()
KST = timezone(timedelta(hours=9))


# =========================
# 기본 데이터
# =========================
DEFAULT_RSS = {
    "Reuters Markets": "https://feeds.reuters.com/reuters/marketsNews",
    "CNBC Markets": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "Yahoo Finance": "https://finance.yahoo.com/news/rssindex",
    "Investing.com News": "https://www.investing.com/rss/news_25.rss",
    "Yonhap Economy": "https://www.yna.co.kr/rss/economy.xml",
}

THEME_KEYWORDS = {
    "AI 반도체": [
        "nvidia", "gpu", "ai chip", "semiconductor", "tsmc", "broadcom", "amd",
        "asic", "hbm", "micron", "sk hynix", "samsung memory", "chip"
    ],
    "메모리/HBM": [
        "hbm", "dram", "nand", "micron", "sk hynix", "samsung", "wdc", "storage", "memory"
    ],
    "AI 서버/데이터센터": [
        "data center", "datacenter", "ai server", "hyperscaler", "cloud capex",
        "server rack", "liquid cooling", "oracle", "microsoft", "amazon", "google cloud",
        "server", "rack"
    ],
    "전력/인프라": [
        "power grid", "electricity", "transformer", "nuclear", "utility", "vrt",
        "eaton", "vertiv", "data center power", "grid bottleneck", "power", "grid"
    ],
    "구글/플랫폼 AI": [
        "google", "alphabet", "gemini", "tpu", "waymo", "youtube", "search ai", "google cloud"
    ],
    "테슬라/로봇·자율주행": [
        "tesla", "robotaxi", "fsd", "optimus", "humanoid", "autonomous",
        "energy storage", "megapack", "xai", "robot"
    ],
    "우주/SpaceX": [
        "spacex", "starlink", "rocket", "satellite", "launch", "space ipo", "rklb", "asts", "rdw", "space"
    ],
    "매크로 리스크": [
        "cpi", "ppi", "fed", "fomc", "treasury yield", "oil", "iran", "war",
        "tariff", "sanctions", "dollar", "vix", "yield", "inflation", "rate"
    ],
}

POSITIVE_WORDS = [
    "surge", "rally", "gain", "beat", "raise", "upgrade", "record", "growth", "strong",
    "partnership", "contract", "approval", "launch", "expansion", "bullish", "outperform",
    "상승", "급등", "호조", "상향", "수주", "계약", "성장", "개선", "강세", "돌파"
]
NEGATIVE_WORDS = [
    "fall", "drop", "slump", "miss", "cut", "downgrade", "weak", "loss", "delay",
    "probe", "ban", "sanction", "risk", "war", "tariff", "recall", "bearish",
    "하락", "급락", "부진", "하향", "제재", "관세", "전쟁", "리스크", "약세", "연기"
]
OFFICIAL_WORDS = ["sec", "fomc", "fed", "company announces", "press release", "earnings", "guidance", "공시", "실적", "발표"]
RUMOR_WORDS = ["rumor", "reportedly", "sources", "unconfirmed", "leak", "찌라시", "루머", "미확인", "소식통"]

DEFAULT_THEME_TICKERS = {
    "AI 반도체": "SOXX, SMH, NVDA, AVGO, AMD",
    "메모리/HBM": "MU, WDC, 000660.KS, 005930.KS",
    "AI 서버/데이터센터": "VRT, DELL, HPE, SMCI",
    "전력/인프라": "ETN, PWR, VRT, NEE",
    "구글/플랫폼 AI": "GOOGL",
    "테슬라/로봇·자율주행": "TSLA, BOTZ, ROBO",
    "우주/SpaceX": "RKLB, ASTS, RDW, ARKX, XOVR",
    "한국장": "EWY, KOSPI, KOSDAQ",
}

MARKET_TICKER_MAP = {
    "KOSPI": "^KS11",
    "KOSDAQ": "^KQ11",
    "KOSPI200": "^KS200",
}

MAIN_INDICATORS = "QQQ, SPY, SOXX, EWY, ^VIX, USO"
EXTRA_INDICATORS = "IWM, ^TNX, UUP, KOSPI, KOSDAQ, NVDA, MU, VRT, GOOGL"


# =========================
# 유틸 함수
# =========================
def normalize_ticker(ticker: str) -> str:
    """대표 표기용 티커 정규화. 실제 조회는 ticker_candidates()에서 fallback 처리."""
    t = str(ticker).strip().upper()
    return MARKET_TICKER_MAP.get(t, t)


def ticker_candidates(ticker: str) -> List[str]:
    """
    yfinance 조회 후보 생성.
    - 미국 티커: NVDA 그대로
    - 한국 지수 별칭: KOSPI -> ^KS11
    - 한국 6자리 코드: 005930 -> 005930.KS, 005930.KQ 순서로 시도
    - 사용자가 .KS/.KQ를 붙이면 그대로 우선 사용
    """
    raw = str(ticker).strip()
    if not raw:
        return []

    t = raw.upper()
    if t in MARKET_TICKER_MAP:
        return [MARKET_TICKER_MAP[t]]

    if re.fullmatch(r"\d{6}", t):
        return [f"{t}.KS", f"{t}.KQ"]

    if re.fullmatch(r"\d{6}\.(KS|KQ)", t):
        return [t]

    return [t]


def split_tickers(text: str) -> List[str]:
    out = []
    for x in str(text).replace("\n", ",").split(","):
        x = x.strip()
        if x:
            out.append(x)
    return out


def safe_pct(x: Any) -> str:
    try:
        if pd.isna(x):
            return "N/A"
        return f"{float(x):.2f}%"
    except Exception:
        return "N/A"


def safe_num(x: Any, digit: int = 2) -> str:
    try:
        if pd.isna(x):
            return "N/A"
        return f"{float(x):.{digit}f}"
    except Exception:
        return "N/A"


def calc_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def classify_theme(text: str) -> str:
    text_l = str(text).lower()
    scores = {}
    for theme, keywords in THEME_KEYWORDS.items():
        score = 0
        for kw in keywords:
            if kw.lower() in text_l:
                score += 1
        scores[theme] = score
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "기타"


def classify_impact(text: str) -> str:
    text_l = str(text).lower()
    pos = sum(1 for w in POSITIVE_WORDS if w.lower() in text_l)
    neg = sum(1 for w in NEGATIVE_WORDS if w.lower() in text_l)
    if pos > 0 and neg > 0:
        return "혼재"
    if pos > neg:
        return "긍정"
    if neg > pos:
        return "부정"
    return "중립"


def classify_reliability(title: str, source: str) -> str:
    text_l = f"{title} {source}".lower()
    if any(w.lower() in text_l for w in RUMOR_WORDS):
        return "비공식"
    if any(w.lower() in text_l for w in OFFICIAL_WORDS):
        return "공식"
    if source:
        return "보도"
    return "비공식"


def parse_entry_time(entry: Any) -> pd.Timestamp:
    for key in ["published_parsed", "updated_parsed"]:
        val = getattr(entry, key, None)
        if val:
            try:
                return pd.Timestamp(datetime(*val[:6], tzinfo=timezone.utc)).tz_convert(KST).tz_localize(None)
            except Exception:
                pass
    return pd.Timestamp.now()


# =========================
# 뉴스 로더
# =========================
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_rss_items(rss_map: Dict[str, str], max_items_per_feed: int = 15) -> pd.DataFrame:
    rows = []
    for source, url in rss_map.items():
        try:
            feed = feedparser.parse(url)
            for e in feed.entries[:max_items_per_feed]:
                title = str(getattr(e, "title", "")).strip()
                link = str(getattr(e, "link", "")).strip()
                summary = re.sub("<.*?>", " ", str(getattr(e, "summary", ""))).strip()
                dt = parse_entry_time(e)
                text = f"{title} {summary}"
                rows.append({
                    "시간": dt.strftime("%Y-%m-%d %H:%M"),
                    "제목": title,
                    "출처": source,
                    "링크": link,
                    "관련 테마": classify_theme(text),
                    "영향": classify_impact(text),
                    "신뢰도": classify_reliability(title, source),
                    "요약 메모": summary[:140] if summary else "",
                    "_dt": dt,
                })
        except Exception as exc:
            rows.append({
                "시간": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
                "제목": f"RSS 로드 실패: {url}",
                "출처": source,
                "링크": url,
                "관련 테마": "기타",
                "영향": "중립",
                "신뢰도": "비공식",
                "요약 메모": repr(exc)[:140],
                "_dt": pd.Timestamp.now(),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.drop_duplicates(subset=["제목", "링크"]).sort_values("_dt", ascending=False)
    return df.reset_index(drop=True)


def build_manual_news_df(manual_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if manual_df is None or manual_df.empty:
        return pd.DataFrame()
    for _, r in manual_df.iterrows():
        title = str(r.get("제목", "")).strip()
        if not title:
            continue
        link = str(r.get("링크", "")).strip()
        source = str(r.get("출처", "수동입력")).strip() or "수동입력"
        memo = str(r.get("메모", "")).strip()
        text = f"{title} {memo}"
        rows.append({
            "시간": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
            "제목": title,
            "출처": source,
            "링크": link,
            "관련 테마": classify_theme(text),
            "영향": classify_impact(text),
            "신뢰도": classify_reliability(title, source),
            "요약 메모": memo,
            "_dt": pd.Timestamp.now(),
        })
    return pd.DataFrame(rows)


# =========================
# 가격/수급 데이터
# =========================
@st.cache_data(ttl=900, show_spinner=False)
def load_price(ticker: str, period: str = "90d") -> Tuple[pd.DataFrame, str]:
    """여러 후보 티커를 순차 조회. 한국 6자리 코드는 .KS/.KQ를 자동 시도."""
    tried = []
    for yf_ticker in ticker_candidates(ticker):
        tried.append(yf_ticker)
        try:
            df = yf.Ticker(yf_ticker).history(period=period, interval="1d", auto_adjust=True)
            if df is None or df.empty or "Close" not in df.columns:
                continue
            df = df.copy()
            df.index = pd.to_datetime(df.index).tz_localize(None)
            return df, yf_ticker
        except Exception:
            continue
    return pd.DataFrame(), " / ".join(tried) if tried else normalize_ticker(ticker)


def calc_ticker_metrics(ticker: str, period: str = "90d") -> Dict[str, Any]:
    df, resolved_ticker = load_price(ticker, period=period)
    if df.empty or "Close" not in df.columns:
        return {
            "티커": ticker,
            "조회티커": resolved_ticker,
            "1일": np.nan,
            "3일": np.nan,
            "5일": np.nan,
            "20일": np.nan,
            "거래량비율": np.nan,
            "Current DD": np.nan,
            "RSI": np.nan,
            "20일고점근접": False,
            "현재가": np.nan,
            "상태": "데이터 없음",
        }
    close = df["Close"].dropna()
    volume = df["Volume"] if "Volume" in df.columns else pd.Series(index=df.index, dtype=float)
    cur = close.iloc[-1]

    def ret(n: int) -> float:
        if len(close) <= n:
            return np.nan
        return (cur / close.iloc[-n - 1] - 1) * 100

    vol_ratio = np.nan
    try:
        vol20 = volume.rolling(20).mean().iloc[-1]
        if vol20 and not pd.isna(vol20):
            vol_ratio = volume.iloc[-1] / vol20
    except Exception:
        pass

    rolling_high = close.cummax()
    cur_dd = (cur / rolling_high.iloc[-1] - 1) * 100
    rsi = calc_rsi(close).iloc[-1]
    high20 = close.tail(20).max()
    near_high20 = bool(cur >= high20 * 0.97)

    return {
        "티커": ticker,
        "조회티커": resolved_ticker,
        "현재가": cur,
        "1일": ret(1),
        "3일": ret(3),
        "5일": ret(5),
        "20일": ret(20),
        "거래량비율": vol_ratio,
        "Current DD": cur_dd,
        "RSI": rsi,
        "20일고점근접": near_high20,
        "상태": "OK",
    }


@st.cache_data(ttl=900, show_spinner=False)
def calc_all_metrics(theme_tickers: Dict[str, str], period: str = "90d") -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for theme, ticker_text in theme_tickers.items():
        for ticker in split_tickers(ticker_text):
            m = calc_ticker_metrics(ticker, period=period)
            m["테마"] = theme
            rows.append(m)
    detail = pd.DataFrame(rows)
    if detail.empty:
        return pd.DataFrame(), pd.DataFrame()

    theme_rows = []
    for theme, g in detail.groupby("테마"):
        ok = g[g["상태"] == "OK"].copy()
        if ok.empty:
            theme_rows.append({
                "테마": theme,
                "상태": "데이터 없음",
                "1일 평균": np.nan,
                "3일 평균": np.nan,
                "5일 평균": np.nan,
                "20일 평균": np.nan,
                "거래량비율": np.nan,
                "대표 강세 종목": "-",
                "대표 약세 종목": "-",
                "판단 메모": "티커 데이터 조회 실패",
            })
            continue

        avg1 = ok["1일"].mean()
        avg3 = ok["3일"].mean()
        avg5 = ok["5일"].mean()
        avg20 = ok["20일"].mean()
        vol = ok["거래량비율"].replace([np.inf, -np.inf], np.nan).mean()

        if pd.notna(avg5) and pd.notna(vol) and avg5 >= 5 and vol >= 1.5:
            state = "강한 유입"
        elif pd.notna(avg3) and pd.notna(vol) and avg3 > 0 and vol >= 1.2:
            state = "유입"
        elif pd.notna(avg5) and pd.notna(vol) and avg5 <= -5 and vol >= 1.5:
            state = "강한 유출"
        elif pd.notna(avg3) and pd.notna(vol) and avg3 < 0 and vol >= 1.2:
            state = "유출"
        else:
            state = "중립"

        strong = ok.sort_values("5일", ascending=False).head(1)
        weak = ok.sort_values("5일", ascending=True).head(1)
        strong_name = strong["티커"].iloc[0] if not strong.empty else "-"
        weak_name = weak["티커"].iloc[0] if not weak.empty else "-"

        memo = f"3일 {safe_pct(avg3)}, 5일 {safe_pct(avg5)}, 거래량 {safe_num(vol)}배"
        theme_rows.append({
            "테마": theme,
            "상태": state,
            "1일 평균": avg1,
            "3일 평균": avg3,
            "5일 평균": avg5,
            "20일 평균": avg20,
            "거래량비율": vol,
            "대표 강세 종목": strong_name,
            "대표 약세 종목": weak_name,
            "판단 메모": memo,
        })
    theme_df = pd.DataFrame(theme_rows)
    order = {"강한 유입": 0, "유입": 1, "중립": 2, "유출": 3, "강한 유출": 4, "데이터 없음": 5}
    theme_df["_order"] = theme_df["상태"].map(order).fillna(9)
    theme_df = theme_df.sort_values(["_order", "5일 평균"], ascending=[True, False]).drop(columns=["_order"])
    return theme_df.reset_index(drop=True), detail.reset_index(drop=True)


# =========================
# 테마 내부 후보 분류
# =========================
def pick_theme_candidates(theme_df: pd.DataFrame, detail_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if detail_df.empty:
        return pd.DataFrame()
    for theme, g in detail_df.groupby("테마"):
        ok = g[g["상태"] == "OK"].copy()
        if ok.empty:
            rows.append({"테마": theme, "대장주": "-", "후발주": "-", "숨은 후보": "-", "제외/주의 종목": "-", "이유": "데이터 없음"})
            continue

        leaders = ok[(ok["거래량비율"] >= 1.5) & (ok["20일고점근접"] == True)].sort_values("5일", ascending=False)
        if leaders.empty:
            leaders = ok.sort_values("5일", ascending=False).head(1)
        leader = ", ".join(leaders.head(2)["티커"].tolist())

        theme_state = "중립"
        if not theme_df.empty and theme in theme_df["테마"].values:
            theme_state = theme_df.loc[theme_df["테마"] == theme, "상태"].iloc[0]

        laggards = ok[
            (ok["거래량비율"] >= 1.05) &
            (ok["5일"].fillna(0) < 3) &
            (ok["Current DD"].fillna(0) <= -5) &
            (ok["RSI"].between(40, 60, inclusive="both"))
        ].sort_values("거래량비율", ascending=False)

        hidden = ok[
            (ok["거래량비율"].between(1.1, 1.5, inclusive="both")) &
            (ok["Current DD"].between(-15, -8, inclusive="both")) &
            (ok["RSI"].fillna(100) < 70)
        ].sort_values("거래량비율", ascending=False)

        caution = ok[(ok["RSI"].fillna(0) >= 72) | (ok["5일"].fillna(0) >= 10)].sort_values("5일", ascending=False)

        rows.append({
            "테마": theme,
            "대장주": leader if leader else "-",
            "후발주": ", ".join(laggards.head(2)["티커"].tolist()) if not laggards.empty else "-",
            "숨은 후보": ", ".join(hidden.head(2)["티커"].tolist()) if not hidden.empty else "-",
            "제외/주의 종목": ", ".join(caution.head(2)["티커"].tolist()) if not caution.empty else "-",
            "이유": f"테마 상태 {theme_state}. 숨은 후보는 추천이 아니라 관찰 후보.",
        })
    return pd.DataFrame(rows)


# =========================
# 뉴스 초입 가능성
# =========================
def calc_news_early_signal(news_df: pd.DataFrame, theme_df: pd.DataFrame, detail_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    now = pd.Timestamp.now()
    if news_df is None or news_df.empty:
        themes = theme_df["테마"].tolist() if not theme_df.empty else list(DEFAULT_THEME_TICKERS.keys())
        for theme in themes:
            rows.append({"테마": theme, "뉴스 초입 가능성": "낮음", "이유": "뉴스 데이터 없음", "행동": "관망"})
        return pd.DataFrame(rows)

    df = news_df.copy()
    if "_dt" not in df.columns:
        df["_dt"] = pd.Timestamp.now()
    df["_dt"] = pd.to_datetime(df["_dt"], errors="coerce").fillna(now)

    for theme in sorted(set(list(DEFAULT_THEME_TICKERS.keys()) + df["관련 테마"].dropna().tolist())):
        recent_news = df[(df["관련 테마"] == theme) & (df["_dt"] >= now - pd.Timedelta(hours=72))]
        news_count = len(recent_news)

        theme_row = theme_df[theme_df["테마"] == theme] if theme_df is not None and not theme_df.empty else pd.DataFrame()
        avg5 = theme_row["5일 평균"].iloc[0] if not theme_row.empty else np.nan
        vol = theme_row["거래량비율"].iloc[0] if not theme_row.empty else np.nan

        d = detail_df[detail_df["테마"] == theme].copy() if detail_df is not None and not detail_df.empty else pd.DataFrame()
        rsi_ok = bool(d["RSI"].dropna().mean() < 70) if not d.empty and d["RSI"].notna().any() else False
        laggard_vol = bool(((d["거래량비율"] >= 1.1) & (d["5일"].fillna(0) < 5)).any()) if not d.empty else False

        score = 0
        reasons = []
        if news_count >= 2:
            score += 1
            reasons.append(f"72시간 뉴스 {news_count}건")
        if pd.notna(avg5) and avg5 <= 5:
            score += 1
            reasons.append("5일 상승률 과열 전")
        if pd.notna(vol) and vol >= 1.1:
            score += 1
            reasons.append("거래량 증가")
        if rsi_ok:
            score += 1
            reasons.append("RSI 70 미만")
        if laggard_vol:
            score += 1
            reasons.append("후발주 거래량 증가")

        if score >= 4:
            level = "높음"
            action = "관심"
        elif score >= 3:
            level = "보통"
            action = "눌림 대기"
        elif pd.notna(avg5) and avg5 >= 8:
            level = "낮음"
            action = "추격 금지"
            reasons.append("단기 과열 가능")
        else:
            level = "낮음"
            action = "관망"

        if theme == "우주/SpaceX" and pd.notna(avg5) and avg5 <= -5:
            action = "재료소멸 주의"
            reasons.append("우주주 약세")

        rows.append({
            "테마": theme,
            "뉴스 초입 가능성": level,
            "이유": ", ".join(reasons) if reasons else "신호 부족",
            "행동": action,
        })
    return pd.DataFrame(rows)


# =========================
# UI 출력 보조
# =========================
def display_pct_table(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].apply(lambda x: safe_pct(x))
    if "거래량비율" in out.columns:
        out["거래량비율"] = out["거래량비율"].apply(lambda x: safe_num(x))
    if "RSI" in out.columns:
        out["RSI"] = out["RSI"].apply(lambda x: safe_num(x))
    if "Current DD" in out.columns:
        out["Current DD"] = out["Current DD"].apply(lambda x: safe_pct(x))
    return out


def make_top_summary(theme_flow: pd.DataFrame, early_df: pd.DataFrame) -> Dict[str, str]:
    if theme_flow.empty:
        return {"강한 유입": "-", "유입": "-", "유출": "-", "뉴스 초입 후보": "-", "추격 금지": "-"}
    strong_in = theme_flow[theme_flow["상태"] == "강한 유입"]["테마"].tolist()
    inflow = theme_flow[theme_flow["상태"] == "유입"]["테마"].tolist()
    outflow = theme_flow[theme_flow["상태"].isin(["유출", "강한 유출"])]["테마"].tolist()
    early = early_df[early_df["뉴스 초입 가능성"].isin(["높음", "보통"])] ["테마"].tolist() if early_df is not None and not early_df.empty else []
    chase_ban = early_df[early_df["행동"].isin(["추격 금지", "재료소멸 주의"])] ["테마"].tolist() if early_df is not None and not early_df.empty else []
    return {
        "강한 유입": ", ".join(strong_in) if strong_in else "-",
        "유입": ", ".join(inflow) if inflow else "-",
        "유출": ", ".join(outflow) if outflow else "-",
        "뉴스 초입 후보": ", ".join(early) if early else "-",
        "추격 금지": ", ".join(chase_ban) if chase_ban else "-",
    }




def parse_custom_theme_tickers(text: str) -> Dict[str, str]:
    """
    사용자 추가 테마 파싱.
    형식: 테마명|티커1, 티커2
    예: 국내 반도체|005930, 000660, 091990.KQ
    """
    out: Dict[str, str] = {}
    for line in str(text).splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "|" not in line:
            continue
        theme, tickers = line.split("|", 1)
        theme = theme.strip()
        tickers = tickers.strip()
        if theme and tickers:
            out[theme] = tickers
    return out

# =========================
# Streamlit 화면
# =========================
st.title("📰 시장 리포트 | 뉴스 · 테마 · 돈의 흐름")
st.caption("MDD 계산 로직과 분리된 시장 리포트 탭입니다. 뉴스는 RSS/수동 입력 기반이며, 매수 추천이 아니라 시장 판단 보조용입니다.")

with st.sidebar:
    st.header("시장 리포트 설정")
    lookback = st.selectbox("가격 데이터 기간", ["60d", "90d", "120d"], index=1)
    max_news = st.slider("RSS별 최대 뉴스 수", 5, 30, 12, 1)
    st.divider()
    run_report = st.button("시장 리포트 실행", type="primary")

# 기본 UI는 실행 버튼 없이도 한 번 계산되게 처리
if "market_report_loaded" not in st.session_state:
    st.session_state["market_report_loaded"] = False
if run_report:
    st.session_state["market_report_loaded"] = True

# 1. 오늘 시장 요약 - 입력/기본값
st.markdown("## 1. 오늘 시장 요약")
c1, c2, c3, c4 = st.columns(4)
with c1:
    manual_mood = st.selectbox("시장 분위기", ["자동", "긍정", "중립", "부정"], index=0)
with c2:
    manual_risk = st.selectbox("위험도", ["자동", "낮음", "보통", "높음", "매우 높음"], index=0)
with c3:
    manual_buy = st.selectbox("매수 판단", ["자동", "가능", "소액 가능", "대기", "금지"], index=0)
with c4:
    manual_chase = st.selectbox("추격매수", ["자동", "가능", "금지"], index=0)
core_line = st.text_input("핵심 한 줄", value="돈의 흐름과 뉴스 초입 여부를 우선 확인")

# 2. RSS 설정
with st.expander("RSS / 수동 뉴스 입력", expanded=False):
    rss_text = st.text_area(
        "추가 RSS URL 또는 기본 RSS 수정: 형식은 출처명|URL, 한 줄 하나",
        value="\n".join([f"{k}|{v}" for k, v in DEFAULT_RSS.items()]),
        height=160,
    )
    manual_news_input = st.data_editor(
        pd.DataFrame([
            {"제목": "", "링크": "", "출처": "수동입력", "메모": ""},
            {"제목": "", "링크": "", "출처": "수동입력", "메모": ""},
        ]),
        num_rows="dynamic",
        use_container_width=True,
    )

rss_map = {}
for line in rss_text.splitlines():
    if "|" in line:
        src, url = line.split("|", 1)
        if src.strip() and url.strip():
            rss_map[src.strip()] = url.strip()

# 3. 티커 설정
with st.expander("테마별 대표 티커 수정 / 한국 종목·ETF 추가", expanded=False):
    st.caption("미국 티커는 NVDA처럼 입력. 한국 종목/ETF는 005930처럼 6자리만 넣어도 .KS/.KQ를 자동 시도합니다. 직접 091990.KQ, 069500.KS처럼 넣어도 됩니다.")
    theme_tickers = {}
    for theme, default in DEFAULT_THEME_TICKERS.items():
        theme_tickers[theme] = st.text_input(theme, default)

    custom_theme_text = st.text_area(
        "추가 테마/티커 입력: 테마명|티커1, 티커2",
        value="국내 반도체|005930, 000660\n국내 ETF 관찰|069500.KS, 360750.KS",
        height=100,
    )
    custom_theme_tickers = parse_custom_theme_tickers(custom_theme_text)
    theme_tickers.update(custom_theme_tickers)

# 데이터 로드
with st.spinner("뉴스와 가격 데이터를 불러오는 중..."):
    news_df = fetch_rss_items(rss_map, max_news)
    manual_df = build_manual_news_df(manual_news_input)
    if not manual_df.empty:
        news_df = pd.concat([manual_df, news_df], ignore_index=True)
        news_df = news_df.drop_duplicates(subset=["제목", "링크"], keep="first")
    theme_flow_df, detail_df = calc_all_metrics(theme_tickers, period=lookback)
    candidate_df = pick_theme_candidates(theme_flow_df, detail_df)
    early_df = calc_news_early_signal(news_df, theme_flow_df, detail_df)

summary = make_top_summary(theme_flow_df, early_df)

# 자동 판단 카드
if manual_mood == "자동":
    if "강한 유입" in theme_flow_df.get("상태", pd.Series(dtype=str)).values or "유입" in theme_flow_df.get("상태", pd.Series(dtype=str)).values:
        mood = "긍정"
    elif "강한 유출" in theme_flow_df.get("상태", pd.Series(dtype=str)).values or "유출" in theme_flow_df.get("상태", pd.Series(dtype=str)).values:
        mood = "부정"
    else:
        mood = "중립"
else:
    mood = manual_mood

risk_score = 0
if not theme_flow_df.empty:
    if (theme_flow_df["상태"] == "강한 유출").any():
        risk_score += 2
    if (theme_flow_df["상태"] == "유출").any():
        risk_score += 1
macro_bad_news = news_df[(news_df["관련 테마"] == "매크로 리스크") & (news_df["영향"].isin(["부정", "혼재"]))] if not news_df.empty else pd.DataFrame()
if len(macro_bad_news) >= 3:
    risk_score += 1

if manual_risk == "자동":
    risk = "낮음" if risk_score == 0 else "보통" if risk_score == 1 else "높음" if risk_score == 2 else "매우 높음"
else:
    risk = manual_risk

if manual_buy == "자동":
    if risk in ["매우 높음"]:
        buy_judge = "금지"
    elif risk == "높음":
        buy_judge = "대기"
    elif summary["강한 유입"] != "-" or summary["뉴스 초입 후보"] != "-":
        buy_judge = "소액 가능"
    else:
        buy_judge = "대기"
else:
    buy_judge = manual_buy

if manual_chase == "자동":
    chase = "금지" if summary["추격 금지"] != "-" or risk in ["높음", "매우 높음"] else "가능"
else:
    chase = manual_chase

m1, m2, m3, m4 = st.columns(4)
m1.metric("시장 분위기", mood)
m2.metric("위험도", risk)
m3.metric("매수 판단", buy_judge)
m4.metric("추격매수", chase)
st.info(f"핵심 한 줄: {core_line}")

# 2. 주요 시장 지표
st.markdown("## 2. 주요 시장 지표")
main_tickers = split_tickers(MAIN_INDICATORS)
main_metric_rows = [calc_ticker_metrics(t, period=lookback) for t in main_tickers]
main_metric_df = pd.DataFrame(main_metric_rows)
show_cols = ["티커", "현재가", "1일", "3일", "5일", "20일", "거래량비율", "Current DD", "RSI"]
st.dataframe(display_pct_table(main_metric_df[show_cols], ["1일", "3일", "5일", "20일"]), use_container_width=True, hide_index=True)

with st.expander("추가 시장 지표 보기", expanded=False):
    extra_rows = [calc_ticker_metrics(t, period=lookback) for t in split_tickers(EXTRA_INDICATORS)]
    extra_df = pd.DataFrame(extra_rows)
    st.dataframe(display_pct_table(extra_df[show_cols], ["1일", "3일", "5일", "20일"]), use_container_width=True, hide_index=True)

# 최종 4개 요약
st.markdown("## 3. 돈의 흐름 요약")
st.markdown(
    f"""
- **강한 유입:** {summary['강한 유입']}
- **유입:** {summary['유입']}
- **유출:** {summary['유출']}
- **뉴스 초입 후보:** {summary['뉴스 초입 후보']}
- **추격 금지:** {summary['추격 금지']}
"""
)

# 최근 뉴스 TOP5
st.markdown("## 4. 최근 뉴스 TOP 5")
if news_df.empty:
    st.warning("뉴스를 불러오지 못했습니다. RSS URL 또는 수동 입력을 확인하세요.")
else:
    top_news = news_df.head(5)[["시간", "제목", "출처", "관련 테마", "영향", "신뢰도", "요약 메모", "링크"]]
    st.dataframe(top_news, use_container_width=True, hide_index=True)

# 돈의 흐름 테마 순위
st.markdown("## 5. 돈의 흐름 테마 순위")
flow_show_cols = ["테마", "상태", "1일 평균", "3일 평균", "5일 평균", "거래량비율", "대표 강세 종목", "대표 약세 종목", "판단 메모"]
st.dataframe(display_pct_table(theme_flow_df[flow_show_cols], ["1일 평균", "3일 평균", "5일 평균"]), use_container_width=True, hide_index=True)

# 테마별 후보
st.markdown("## 6. 테마별 대장주 / 후발주 / 숨은 후보")
st.caption("숨은 후보는 추천이 아니라 관찰 후보입니다.")
st.dataframe(candidate_df, use_container_width=True, hide_index=True)

# 뉴스 초입 가능성
st.markdown("## 7. 뉴스 초입 가능성")
st.dataframe(early_df[["테마", "뉴스 초입 가능성", "이유", "행동"]], use_container_width=True, hide_index=True)

# 주요 리스크
st.markdown("## 8. 주요 리스크")
risk_news = news_df[(news_df["관련 테마"] == "매크로 리스크") | (news_df["영향"].isin(["부정", "혼재"]))].head(5) if not news_df.empty else pd.DataFrame()
if risk_news.empty:
    st.success("상단 뉴스 기준 주요 리스크 신호는 제한적입니다.")
else:
    st.dataframe(risk_news[["시간", "제목", "출처", "관련 테마", "영향", "신뢰도", "요약 메모", "링크"]], use_container_width=True, hide_index=True)

# 상세 데이터
with st.expander("상세 뉴스 전체 보기", expanded=False):
    if not news_df.empty:
        st.dataframe(news_df.drop(columns=["_dt"], errors="ignore"), use_container_width=True, hide_index=True)
        st.download_button(
            "뉴스 CSV 다운로드",
            data=news_df.drop(columns=["_dt"], errors="ignore").to_csv(index=False).encode("utf-8-sig"),
            file_name="market_news.csv",
            mime="text/csv",
        )

with st.expander("상세 종목 데이터 보기", expanded=False):
    st.dataframe(display_pct_table(detail_df, ["1일", "3일", "5일", "20일"]), use_container_width=True, hide_index=True)
    st.download_button(
        "종목 데이터 CSV 다운로드",
        data=detail_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="theme_flow_detail.csv",
        mime="text/csv",
    )

st.caption("주의: RSS 뉴스와 가격 데이터는 지연·누락될 수 있습니다. 루머성 정보는 사실로 확정하지 말고, 공시·실적·거래대금으로 재확인하세요.")
