# -*- coding: utf-8 -*-
"""
시장 리포트 탭 v3 - 객관적 자동 스캔형
목적: 최근 뉴스 → 테마 변화 → 돈의 흐름 → 대장주/후발주/숨은 후보를 한 화면에서 확인.
MDD/PER 분석기와 분리된 독립 Streamlit page 코드.
requirements.txt 변경 없음: streamlit, pandas, numpy, yfinance, feedparser 사용.
"""

from __future__ import annotations

import re
import math
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import feedparser


# =========================
# Page setup
# =========================
st.set_page_config(page_title="시장 리포트", page_icon="📰", layout="wide")
KST = timezone(timedelta(hours=9))


# =========================
# RSS / Keywords
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
        "asic", "hbm", "micron", "sk hynix", "samsung memory", "chip", "wafer",
    ],
    "메모리/HBM": [
        "hbm", "dram", "nand", "micron", "sk hynix", "samsung", "wdc", "storage", "memory",
    ],
    "AI 서버/데이터센터": [
        "data center", "datacenter", "ai server", "hyperscaler", "cloud capex",
        "server rack", "liquid cooling", "oracle", "microsoft", "amazon", "google cloud",
        "server", "rack", "networking", "ethernet",
    ],
    "전력/인프라": [
        "power grid", "electricity", "transformer", "nuclear", "utility", "vrt",
        "eaton", "vertiv", "data center power", "grid bottleneck", "power", "grid",
    ],
    "구글/플랫폼 AI": [
        "google", "alphabet", "gemini", "tpu", "waymo", "youtube", "search ai", "google cloud",
    ],
    "테슬라/로봇·자율주행": [
        "tesla", "robotaxi", "fsd", "optimus", "humanoid", "autonomous",
        "energy storage", "megapack", "xai", "robot",
    ],
    "우주/SpaceX": [
        "spacex", "starlink", "rocket", "satellite", "launch", "space ipo", "rklb", "asts", "rdw", "space",
    ],
    "사이버보안/AI SW": [
        "cyber", "security", "hack", "cloud security", "ai agent", "software", "saas", "snowflake", "palantir",
    ],
    "매크로 리스크": [
        "cpi", "ppi", "fed", "fomc", "treasury yield", "oil", "iran", "war",
        "tariff", "sanctions", "dollar", "vix", "yield", "inflation", "rate",
    ],
}

POSITIVE_WORDS = [
    "surge", "rally", "gain", "beat", "raise", "upgrade", "record", "growth", "strong",
    "partnership", "contract", "approval", "launch", "expansion", "bullish", "outperform",
    "상승", "급등", "호조", "상향", "수주", "계약", "성장", "개선", "강세", "돌파",
]
NEGATIVE_WORDS = [
    "fall", "drop", "slump", "miss", "cut", "downgrade", "weak", "loss", "delay",
    "probe", "ban", "sanction", "risk", "war", "tariff", "recall", "bearish",
    "하락", "급락", "부진", "하향", "제재", "관세", "전쟁", "리스크", "약세", "연기",
]
OFFICIAL_WORDS = ["sec", "fomc", "fed", "company announces", "press release", "earnings", "guidance", "공시", "실적", "발표"]
RUMOR_WORDS = ["rumor", "reportedly", "sources", "unconfirmed", "leak", "찌라시", "루머", "미확인", "소식통"]


# =========================
# Objective scan universe
# =========================
# 사용자가 고르는 종목이 아니라, 앱이 이 기본 유니버스를 자동 스캔한다.
# 사용자는 expander에서 보조로 추가만 가능하다.
AUTO_UNIVERSE = {
    "AI 반도체": [
        "SOXX", "SMH", "NVDA", "AVGO", "AMD", "TSM", "ASML", "ARM", "MRVL", "QCOM", "AMAT", "LRCX", "KLAC",
    ],
    "메모리/HBM": [
        "MU", "WDC", "STX", "000660.KS", "005930.KS", "042700.KS", "095340.KQ", "039030.KQ", "089030.KQ",
    ],
    "AI 서버/데이터센터": [
        "VRT", "DELL", "HPE", "SMCI", "ORCL", "MSFT", "AMZN", "ANET", "CLS", "NTAP", "PSTG",
    ],
    "전력/인프라": [
        "ETN", "PWR", "VRT", "GEV", "NEE", "CEG", "AEP", "SO", "XLU", "010120.KS", "267260.KS",
    ],
    "구글/플랫폼 AI": [
        "GOOGL", "GOOG", "META", "MSFT", "AMZN",
    ],
    "테슬라/로봇·자율주행": [
        "TSLA", "BOTZ", "ROBO", "ISRG", "TER", "SYM", "277810.KQ", "108490.KQ", "090360.KQ",
    ],
    "우주/SpaceX": [
        "RKLB", "ASTS", "RDW", "LUNR", "PL", "IRDM", "ARKX", "BA", "LMT", "047810.KS",
    ],
    "사이버보안/AI SW": [
        "CIBR", "HACK", "CRWD", "PANW", "ZS", "FTNT", "PLTR", "SNOW", "NOW", "DDOG", "AIQ", "IGV",
    ],
    "한국장/국내 ETF": [
        "EWY", "KOSPI", "KOSDAQ", "069500.KS", "102110.KS", "360750.KS", "379810.KS", "453850.KS",
    ],
}

MARKET_TICKER_MAP = {
    "KOSPI": "^KS11",
    "KOSDAQ": "^KQ11",
    "KOSPI200": "^KS200",
}

MAIN_INDICATORS = ["QQQ", "SPY", "SOXX", "EWY", "^VIX", "USO"]
EXTRA_INDICATORS = ["IWM", "^TNX", "UUP", "KOSPI", "KOSDAQ", "NVDA", "MU", "VRT", "GOOGL"]


# =========================
# Utility
# =========================
def normalize_ticker(ticker: str) -> str:
    t = str(ticker).strip().upper()
    return MARKET_TICKER_MAP.get(t, t)


def ticker_candidates(ticker: str) -> List[str]:
    raw = str(ticker).strip()
    if not raw:
        return []
    t = raw.upper()
    if t in MARKET_TICKER_MAP:
        return [MARKET_TICKER_MAP[t]]
    if re.fullmatch(r"\d{6}", t):
        return [f"{t}.KS", f"{t}.KQ"]
    return [t]


def parse_ticker_list(text: str) -> List[str]:
    out = []
    for x in str(text).replace("\n", ",").split(","):
        x = x.strip()
        if x:
            out.append(x)
    return out


def parse_extra_universe(text: str) -> Dict[str, List[str]]:
    """형식: 테마명|티커1, 티커2"""
    out: Dict[str, List[str]] = {}
    for line in str(text).splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "|" not in line:
            continue
        theme, tickers = line.split("|", 1)
        theme = theme.strip()
        items = parse_ticker_list(tickers)
        if theme and items:
            out.setdefault(theme, [])
            out[theme].extend(items)
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
    close = pd.to_numeric(close, errors="coerce")
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
        scores[theme] = sum(1 for kw in keywords if kw.lower() in text_l)
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


def merge_universe(base: Dict[str, List[str]], extra: Dict[str, List[str]]) -> Dict[str, List[str]]:
    merged = {k: list(v) for k, v in base.items()}
    for theme, items in extra.items():
        merged.setdefault(theme, [])
        merged[theme].extend(items)
    # de-duplicate within theme preserving order
    for theme, items in merged.items():
        seen = set()
        clean = []
        for t in items:
            tu = str(t).strip().upper()
            if tu and tu not in seen:
                seen.add(tu)
                clean.append(str(t).strip())
        merged[theme] = clean
    return merged


def universe_to_rows(universe: Dict[str, List[str]]) -> pd.DataFrame:
    rows = []
    for theme, tickers in universe.items():
        for i, ticker in enumerate(tickers):
            rows.append({"테마": theme, "티커": ticker, "대표순위": i + 1})
    return pd.DataFrame(rows)


# =========================
# RSS news
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
# Price / flow metrics
# =========================
@st.cache_data(ttl=900, show_spinner=False)
def load_price_single(ticker: str, period: str = "90d") -> Tuple[pd.DataFrame, str]:
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
    df, resolved_ticker = load_price_single(ticker, period=period)
    if df.empty or "Close" not in df.columns:
        return {
            "티커": ticker, "조회티커": resolved_ticker, "현재가": np.nan,
            "1일": np.nan, "3일": np.nan, "5일": np.nan, "20일": np.nan,
            "거래량비율": np.nan, "Current DD": np.nan, "RSI": np.nan,
            "20일고점근접": False, "흐름점수": np.nan, "상태": "데이터 없음",
        }

    close = pd.to_numeric(df["Close"], errors="coerce").dropna()
    if close.empty:
        return {"티커": ticker, "조회티커": resolved_ticker, "상태": "데이터 없음"}
    volume = pd.to_numeric(df.get("Volume", pd.Series(index=df.index, dtype=float)), errors="coerce")
    cur = close.iloc[-1]

    def ret(n: int) -> float:
        if len(close) <= n:
            return np.nan
        return (cur / close.iloc[-n - 1] - 1) * 100

    r1, r3, r5, r20 = ret(1), ret(3), ret(5), ret(20)
    vol_ratio = np.nan
    try:
        vol20 = volume.rolling(20).mean().iloc[-1]
        if pd.notna(vol20) and vol20 > 0:
            vol_ratio = volume.iloc[-1] / vol20
    except Exception:
        pass

    rolling_high = close.cummax()
    cur_dd = (cur / rolling_high.iloc[-1] - 1) * 100
    rsi = calc_rsi(close).iloc[-1]
    high20 = close.tail(20).max()
    near_high20 = bool(cur >= high20 * 0.97)

    # 객관적 흐름점수: 단기 수익률 + 거래량 + 중기 추세. 과열은 후보 분류에서 별도 처리.
    vr = 1.0 if pd.isna(vol_ratio) else max(float(vol_ratio), 0.1)
    score = 0.0
    for val, weight in [(r1, 0.15), (r3, 0.30), (r5, 0.30), (r20, 0.15)]:
        if pd.notna(val):
            score += float(val) * weight
    score += math.log(vr) * 4.0

    return {
        "티커": ticker,
        "조회티커": resolved_ticker,
        "현재가": cur,
        "1일": r1,
        "3일": r3,
        "5일": r5,
        "20일": r20,
        "거래량비율": vol_ratio,
        "Current DD": cur_dd,
        "RSI": rsi,
        "20일고점근접": near_high20,
        "흐름점수": score,
        "상태": "OK",
    }


@st.cache_data(ttl=900, show_spinner=False)
def scan_universe(universe_rows: pd.DataFrame, period: str = "90d") -> pd.DataFrame:
    rows = []
    for _, r in universe_rows.iterrows():
        m = calc_ticker_metrics(str(r["티커"]), period=period)
        m["테마"] = r["테마"]
        m["대표순위"] = r.get("대표순위", 999)
        rows.append(m)
    df = pd.DataFrame(rows)
    # same ticker may belong to multiple themes; keep theme duplicate intentionally
    return df.reset_index(drop=True)


def classify_theme_flow(detail_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if detail_df.empty:
        return pd.DataFrame()

    for theme, g in detail_df.groupby("테마"):
        ok = g[g["상태"] == "OK"].copy()
        if ok.empty:
            rows.append({
                "테마": theme, "상태": "데이터 없음", "1일 평균": np.nan, "3일 평균": np.nan,
                "5일 평균": np.nan, "20일 평균": np.nan, "거래량비율": np.nan,
                "대표 강세 종목": "-", "대표 약세 종목": "-", "판단 메모": "가격 데이터 조회 실패",
                "테마점수": np.nan,
            })
            continue
        avg1 = ok["1일"].mean()
        avg3 = ok["3일"].mean()
        avg5 = ok["5일"].mean()
        avg20 = ok["20일"].mean()
        vol = ok["거래량비율"].replace([np.inf, -np.inf], np.nan).mean()
        theme_score = ok["흐름점수"].replace([np.inf, -np.inf], np.nan).mean()

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

        strong = ok.sort_values("흐름점수", ascending=False).head(1)
        weak = ok.sort_values("흐름점수", ascending=True).head(1)
        memo = f"3일 {safe_pct(avg3)}, 5일 {safe_pct(avg5)}, 거래량 {safe_num(vol)}배"
        rows.append({
            "테마": theme,
            "상태": state,
            "1일 평균": avg1,
            "3일 평균": avg3,
            "5일 평균": avg5,
            "20일 평균": avg20,
            "거래량비율": vol,
            "대표 강세 종목": strong["티커"].iloc[0] if not strong.empty else "-",
            "대표 약세 종목": weak["티커"].iloc[0] if not weak.empty else "-",
            "판단 메모": memo,
            "테마점수": theme_score,
        })
    out = pd.DataFrame(rows)
    order = {"강한 유입": 0, "유입": 1, "중립": 2, "유출": 3, "강한 유출": 4, "데이터 없음": 5}
    out["_order"] = out["상태"].map(order).fillna(9)
    out = out.sort_values(["_order", "테마점수"], ascending=[True, False]).drop(columns=["_order"])
    return out.reset_index(drop=True)


def global_stock_rankings(detail_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ok = detail_df[detail_df["상태"] == "OK"].copy()
    if ok.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    # 동일 티커가 여러 테마에 중복될 수 있으므로 티커별 최고 점수 행만 표시
    ok = ok.sort_values("흐름점수", ascending=False).drop_duplicates("티커", keep="first")

    inflow = ok[(ok["3일"].fillna(0) > 0) & (ok["거래량비율"].fillna(0) >= 1.15)].sort_values("흐름점수", ascending=False).head(15)
    outflow = ok[(ok["3일"].fillna(0) < 0) & (ok["거래량비율"].fillna(0) >= 1.15)].sort_values("흐름점수", ascending=True).head(15)
    unusual = ok[(ok["거래량비율"].fillna(0) >= 1.5)].sort_values("거래량비율", ascending=False).head(15)
    return inflow, outflow, unusual


# =========================
# Theme candidate classification
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

        theme_state = "중립"
        if not theme_df.empty and theme in theme_df["테마"].values:
            theme_state = theme_df.loc[theme_df["테마"] == theme, "상태"].iloc[0]

        # 대장주: 테마 내 흐름점수 상위 + 거래량 + 20일 고점 근처 우선
        leaders = ok[(ok["거래량비율"].fillna(0) >= 1.3) & (ok["20일고점근접"] == True)].sort_values("흐름점수", ascending=False)
        if leaders.empty:
            leaders = ok.sort_values("흐름점수", ascending=False).head(3)

        # 후발주: 테마 상태가 강할 때, 아직 수익률이 낮고 거래량이 증가하기 시작한 종목
        theme_is_hot = theme_state in ["강한 유입", "유입"]
        laggards = ok[
            theme_is_hot
            & (ok["거래량비율"].fillna(0) >= 1.05)
            & (ok["5일"].fillna(0).between(-3, 4, inclusive="both"))
            & (ok["Current DD"].fillna(0).between(-20, -3, inclusive="both"))
            & (ok["RSI"].fillna(100).between(38, 65, inclusive="both"))
        ].sort_values("거래량비율", ascending=False)

        # 숨은 후보: 추천 아님. 거래량 증가 + DD 깊음 + 과열 전
        hidden = ok[
            (ok["거래량비율"].fillna(0).between(1.1, 1.6, inclusive="both"))
            & (ok["Current DD"].fillna(0).between(-18, -8, inclusive="both"))
            & (ok["RSI"].fillna(100) < 70)
            & (ok["5일"].fillna(0) < 6)
        ].sort_values(["거래량비율", "흐름점수"], ascending=[False, False])

        caution = ok[
            (ok["RSI"].fillna(0) >= 72) | (ok["5일"].fillna(0) >= 10)
        ].sort_values("5일", ascending=False)

        rows.append({
            "테마": theme,
            "대장주": ", ".join(leaders.head(3)["티커"].tolist()) if not leaders.empty else "-",
            "후발주": ", ".join(laggards.head(3)["티커"].tolist()) if not laggards.empty else "-",
            "숨은 후보": ", ".join(hidden.head(3)["티커"].tolist()) if not hidden.empty else "-",
            "제외/주의 종목": ", ".join(caution.head(3)["티커"].tolist()) if not caution.empty else "-",
            "이유": f"테마 상태 {theme_state}. 숨은 후보는 추천이 아니라 관찰 후보.",
        })
    return pd.DataFrame(rows)


def calc_news_early_signal(news_df: pd.DataFrame, theme_df: pd.DataFrame, detail_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    now = pd.Timestamp.now()
    all_themes = list(AUTO_UNIVERSE.keys())
    if news_df is not None and not news_df.empty and "관련 테마" in news_df.columns:
        all_themes = sorted(set(all_themes + news_df["관련 테마"].dropna().tolist()))

    for theme in all_themes:
        if news_df is None or news_df.empty:
            news_count = 0
        else:
            df = news_df.copy()
            df["_dt"] = pd.to_datetime(df.get("_dt", now), errors="coerce").fillna(now)
            news_count = len(df[(df["관련 테마"] == theme) & (df["_dt"] >= now - pd.Timedelta(hours=72))])

        theme_row = theme_df[theme_df["테마"] == theme] if theme_df is not None and not theme_df.empty else pd.DataFrame()
        avg5 = theme_row["5일 평균"].iloc[0] if not theme_row.empty else np.nan
        vol = theme_row["거래량비율"].iloc[0] if not theme_row.empty else np.nan

        d = detail_df[detail_df["테마"] == theme].copy() if detail_df is not None and not detail_df.empty else pd.DataFrame()
        rsi_ok = bool(d["RSI"].dropna().mean() < 70) if not d.empty and d["RSI"].notna().any() else False
        laggard_vol = bool(((d["거래량비율"].fillna(0) >= 1.1) & (d["5일"].fillna(0) < 5)).any()) if not d.empty else False

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
            level, action = "높음", "관심"
        elif score >= 3:
            level, action = "보통", "눌림 대기"
        elif pd.notna(avg5) and avg5 >= 8:
            level, action = "낮음", "추격 금지"
            reasons.append("단기 과열 가능")
        else:
            level, action = "낮음", "관망"

        if theme == "우주/SpaceX" and pd.notna(avg5) and avg5 <= -5:
            action = "재료소멸 주의"
            reasons.append("우주주 약세")

        rows.append({"테마": theme, "뉴스 초입 가능성": level, "이유": ", ".join(reasons) if reasons else "신호 부족", "행동": action})
    out = pd.DataFrame(rows)
    order = {"높음": 0, "보통": 1, "낮음": 2}
    out["_order"] = out["뉴스 초입 가능성"].map(order).fillna(9)
    return out.sort_values(["_order", "테마"]).drop(columns=["_order"]).reset_index(drop=True)


# =========================
# Display helpers
# =========================
def display_pct_table(df: pd.DataFrame, pct_cols: List[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    for c in pct_cols:
        if c in out.columns:
            out[c] = out[c].apply(safe_pct)
    if "거래량비율" in out.columns:
        out["거래량비율"] = out["거래량비율"].apply(lambda x: safe_num(x))
    if "RSI" in out.columns:
        out["RSI"] = out["RSI"].apply(lambda x: safe_num(x))
    if "Current DD" in out.columns:
        out["Current DD"] = out["Current DD"].apply(safe_pct)
    if "흐름점수" in out.columns:
        out["흐름점수"] = out["흐름점수"].apply(lambda x: safe_num(x))
    return out


def make_top_summary(theme_flow: pd.DataFrame, early_df: pd.DataFrame) -> Dict[str, str]:
    if theme_flow is None or theme_flow.empty:
        return {"강한 유입": "-", "유입": "-", "유출": "-", "뉴스 초입 후보": "-", "추격 금지": "-"}
    strong_in = theme_flow[theme_flow["상태"] == "강한 유입"]["테마"].tolist()
    inflow = theme_flow[theme_flow["상태"] == "유입"]["테마"].tolist()
    outflow = theme_flow[theme_flow["상태"].isin(["유출", "강한 유출"])]["테마"].tolist()
    early = early_df[early_df["뉴스 초입 가능성"].isin(["높음", "보통"])]["테마"].tolist() if early_df is not None and not early_df.empty else []
    chase_ban = early_df[early_df["행동"].isin(["추격 금지", "재료소멸 주의"])]["테마"].tolist() if early_df is not None and not early_df.empty else []
    return {
        "강한 유입": ", ".join(strong_in) if strong_in else "-",
        "유입": ", ".join(inflow) if inflow else "-",
        "유출": ", ".join(outflow) if outflow else "-",
        "뉴스 초입 후보": ", ".join(early) if early else "-",
        "추격 금지": ", ".join(chase_ban) if chase_ban else "-",
    }


def top_table(df: pd.DataFrame, cols: List[str], n: int = 10) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=cols)
    use_cols = [c for c in cols if c in df.columns]
    return df.head(n)[use_cols]


# =========================
# Streamlit UI
# =========================
st.title("📰 시장 리포트 | 객관적 돈의 흐름 스캐너")
st.caption("사용자가 고른 종목이 아니라, 기본 유니버스를 자동 스캔해 뉴스·테마·거래량·수익률 기준으로 시장 흐름을 정리합니다. 매수 추천이 아닙니다.")

with st.sidebar:
    st.header("시장 리포트 설정")
    lookback = st.selectbox("가격 데이터 기간", ["60d", "90d", "120d"], index=1)
    max_news = st.slider("RSS별 최대 뉴스 수", 5, 30, 12, 1)
    include_korea = st.checkbox("한국 종목/ETF 포함", value=True)
    include_optional_extra = st.checkbox("추가 유니버스 사용", value=False)
    run_report = st.button("시장 리포트 실행", type="primary")

# 1. RSS / manual news
with st.expander("RSS / 수동 뉴스 입력", expanded=False):
    rss_text = st.text_area(
        "RSS URL: 출처명|URL, 한 줄 하나",
        value="\n".join([f"{k}|{v}" for k, v in DEFAULT_RSS.items()]),
        height=150,
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

# 2. Universe setup
base_universe = {k: v for k, v in AUTO_UNIVERSE.items()}
if not include_korea:
    for theme in list(base_universe.keys()):
        base_universe[theme] = [t for t in base_universe[theme] if not re.search(r"\.K[QS]$", t, flags=re.I) and t not in ["KOSPI", "KOSDAQ"]]

with st.expander("자동 스캔 유니버스 확인 / 보조 추가", expanded=False):
    st.info("기본값은 자동 스캔 유니버스입니다. 사용자가 종목을 골라 판단하는 구조가 아니라, 아래 전체를 스캔합니다. 추가 유니버스는 보조 확장용입니다.")
    uni_preview = universe_to_rows(base_universe)
    st.dataframe(uni_preview, use_container_width=True, hide_index=True)
    extra_text = st.text_area(
        "보조 추가 유니버스: 테마명|티커1, 티커2",
        value="국내 원전/전력|034020.KS, 051600.KS\n국내 AI SW|035720.KS, 035420.KS",
        height=100,
    )

extra_universe = parse_extra_universe(extra_text) if include_optional_extra else {}
universe = merge_universe(base_universe, extra_universe)
universe_rows = universe_to_rows(universe)

# 3. Data load
with st.spinner("뉴스와 자동 스캔 유니버스 데이터를 불러오는 중..."):
    news_df = fetch_rss_items(rss_map, max_news)
    manual_df = build_manual_news_df(manual_news_input)
    if not manual_df.empty:
        news_df = pd.concat([manual_df, news_df], ignore_index=True)
        news_df = news_df.drop_duplicates(subset=["제목", "링크"], keep="first")

    detail_df = scan_universe(universe_rows, period=lookback)
    theme_flow_df = classify_theme_flow(detail_df)
    candidate_df = pick_theme_candidates(theme_flow_df, detail_df)
    early_df = calc_news_early_signal(news_df, theme_flow_df, detail_df)
    inflow_df, outflow_df, unusual_df = global_stock_rankings(detail_df)

summary = make_top_summary(theme_flow_df, early_df)

# 4. Today summary cards
st.markdown("## 1. 오늘 시장 요약")
risk_score = 0
if not theme_flow_df.empty:
    risk_score += 2 if (theme_flow_df["상태"] == "강한 유출").any() else 0
    risk_score += 1 if (theme_flow_df["상태"] == "유출").any() else 0
macro_bad_news = news_df[(news_df["관련 테마"] == "매크로 리스크") & (news_df["영향"].isin(["부정", "혼재"]))] if not news_df.empty else pd.DataFrame()
if len(macro_bad_news) >= 3:
    risk_score += 1
risk = "낮음" if risk_score == 0 else "보통" if risk_score == 1 else "높음" if risk_score == 2 else "매우 높음"

if summary["강한 유입"] != "-" or summary["유입"] != "-":
    mood = "긍정"
elif summary["유출"] != "-":
    mood = "부정"
else:
    mood = "중립"

buy_judge = "금지" if risk == "매우 높음" else "대기" if risk == "높음" else "소액 가능" if summary["뉴스 초입 후보"] != "-" or summary["강한 유입"] != "-" else "대기"
chase = "금지" if summary["추격 금지"] != "-" or risk in ["높음", "매우 높음"] else "가능"

c1, c2, c3, c4 = st.columns(4)
c1.metric("시장 분위기", mood)
c2.metric("위험도", risk)
c3.metric("매수 판단", buy_judge)
c4.metric("추격매수", chase)

# 5. Objective flow summary
st.markdown("## 2. 돈의 흐름 요약")
st.markdown(
    f"""
- **강한 유입:** {summary['강한 유입']}
- **유입:** {summary['유입']}
- **유출:** {summary['유출']}
- **뉴스 초입 후보:** {summary['뉴스 초입 후보']}
- **추격 금지:** {summary['추격 금지']}
"""
)

r1, r2, r3 = st.columns(3)
with r1:
    st.markdown("### 수급 유입 종목")
    cols = ["테마", "티커", "1일", "3일", "5일", "거래량비율", "Current DD", "RSI", "흐름점수"]
    st.dataframe(display_pct_table(top_table(inflow_df, cols, 8), ["1일", "3일", "5일"]), use_container_width=True, hide_index=True)
with r2:
    st.markdown("### 수급 유출 종목")
    cols = ["테마", "티커", "1일", "3일", "5일", "거래량비율", "Current DD", "RSI", "흐름점수"]
    st.dataframe(display_pct_table(top_table(outflow_df, cols, 8), ["1일", "3일", "5일"]), use_container_width=True, hide_index=True)
with r3:
    st.markdown("### 거래량 급증")
    cols = ["테마", "티커", "1일", "3일", "5일", "거래량비율", "Current DD", "RSI"]
    st.dataframe(display_pct_table(top_table(unusual_df, cols, 8), ["1일", "3일", "5일"]), use_container_width=True, hide_index=True)

# 6. Market indicators
st.markdown("## 3. 주요 시장 지표")
main_rows = [calc_ticker_metrics(t, period=lookback) for t in MAIN_INDICATORS]
main_df = pd.DataFrame(main_rows)
show_cols = ["티커", "현재가", "1일", "3일", "5일", "20일", "거래량비율", "Current DD", "RSI"]
st.dataframe(display_pct_table(main_df[show_cols], ["1일", "3일", "5일", "20일"]), use_container_width=True, hide_index=True)
with st.expander("추가 시장 지표 보기", expanded=False):
    extra_rows = [calc_ticker_metrics(t, period=lookback) for t in EXTRA_INDICATORS]
    extra_df = pd.DataFrame(extra_rows)
    st.dataframe(display_pct_table(extra_df[show_cols], ["1일", "3일", "5일", "20일"]), use_container_width=True, hide_index=True)

# 7. News top 5
st.markdown("## 4. 최근 뉴스 TOP 5")
if news_df.empty:
    st.warning("뉴스를 불러오지 못했습니다. RSS URL 또는 수동 입력을 확인하세요.")
else:
    st.dataframe(news_df.head(5)[["시간", "제목", "출처", "관련 테마", "영향", "신뢰도", "요약 메모", "링크"]], use_container_width=True, hide_index=True)

# 8. Theme flow ranking
st.markdown("## 5. 돈의 흐름 테마 순위")
flow_cols = ["테마", "상태", "1일 평균", "3일 평균", "5일 평균", "거래량비율", "대표 강세 종목", "대표 약세 종목", "판단 메모"]
st.dataframe(display_pct_table(theme_flow_df[flow_cols], ["1일 평균", "3일 평균", "5일 평균"]), use_container_width=True, hide_index=True)

# 9. Leaders / laggards / hidden
st.markdown("## 6. 테마별 대장주 / 후발주 / 숨은 후보")
st.caption("자동 스캔 결과입니다. 숨은 후보는 추천이 아니라 관찰 후보입니다.")
st.dataframe(candidate_df, use_container_width=True, hide_index=True)

# 10. Early news signal
st.markdown("## 7. 뉴스 초입 가능성")
st.dataframe(early_df[["테마", "뉴스 초입 가능성", "이유", "행동"]], use_container_width=True, hide_index=True)

# 11. Risks
st.markdown("## 8. 주요 리스크")
risk_news = news_df[(news_df["관련 테마"] == "매크로 리스크") | (news_df["영향"].isin(["부정", "혼재"]))].head(5) if not news_df.empty else pd.DataFrame()
if risk_news.empty:
    st.success("상단 뉴스 기준 주요 리스크 신호는 제한적입니다.")
else:
    st.dataframe(risk_news[["시간", "제목", "출처", "관련 테마", "영향", "신뢰도", "요약 메모", "링크"]], use_container_width=True, hide_index=True)

# 12. Detail data
with st.expander("상세 뉴스 전체 보기", expanded=False):
    if not news_df.empty:
        st.dataframe(news_df.drop(columns=["_dt"], errors="ignore"), use_container_width=True, hide_index=True)
        st.download_button(
            "뉴스 CSV 다운로드",
            data=news_df.drop(columns=["_dt"], errors="ignore").to_csv(index=False).encode("utf-8-sig"),
            file_name="market_news.csv",
            mime="text/csv",
        )

with st.expander("상세 종목 데이터 전체 보기", expanded=False):
    st.dataframe(display_pct_table(detail_df, ["1일", "3일", "5일", "20일"]), use_container_width=True, hide_index=True)
    st.download_button(
        "종목 데이터 CSV 다운로드",
        data=detail_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="theme_flow_detail.csv",
        mime="text/csv",
    )

st.caption("주의: 자동 스캔은 기본 유니버스 안에서의 객관적 가격·거래량 변화입니다. 전체 상장종목 전체검색은 별도 KRX/Nasdaq screener API가 필요합니다. 루머성 정보는 사실로 확정하지 말고 공시·거래대금으로 재확인하세요.")
