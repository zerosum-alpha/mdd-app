import warnings
warnings.filterwarnings("ignore")

from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import streamlit as st

try:
    import yfinance as yf
    YF_AVAILABLE = True
    YF_IMPORT_ERROR = ""
except Exception as e:
    yf = None
    YF_AVAILABLE = False
    YF_IMPORT_ERROR = repr(e)

try:
    import FinanceDataReader as fdr
    FDR_AVAILABLE = True
    FDR_IMPORT_ERROR = ""
except Exception as e:
    fdr = None
    FDR_AVAILABLE = False
    FDR_IMPORT_ERROR = repr(e)

try:
    from pykrx import stock as pkstock
    PYKRX_AVAILABLE = True
    PYKRX_IMPORT_ERROR = ""
except Exception as e:
    pkstock = None
    PYKRX_AVAILABLE = False
    PYKRX_IMPORT_ERROR = repr(e)

try:
    from auth import require_login, logout_button
except Exception:
    def require_login():
        return None
    def logout_button():
        return None

# =========================================================
# MDD Core Dashboard - Normal Final
# 핵심 화면: Price + PER + MDD + Market Risk
# - 한국: pykrx KRX PER/PBR/EPS
# - 미국: yfinance EPS 기반 Estimated TTM P/E
# - 가짜 Price-implied P/E / 바차트 / 예측선 제거
# =========================================================

st.set_page_config(page_title="MDD 분석기", layout="wide")
require_login()
logout_button()

st.title("📈 MDD 저점매수 분석기 | Core Dashboard")
st.caption("핵심: 주가, PER, MDD, 시장위험, 이평선만 보고 저점매수/현금확보 후보를 판단합니다.")

# =========================================================
# Utils
# =========================================================
def to_ns_datetime(series_or_index):
    dt = pd.to_datetime(series_or_index, errors="coerce")
    try:
        if hasattr(dt, "dt"):
            try:
                dt = dt.dt.tz_localize(None)
            except Exception:
                dt = dt.dt.tz_convert(None)
        else:
            try:
                dt = dt.tz_localize(None)
            except Exception:
                dt = dt.tz_convert(None)
    except Exception:
        pass
    return dt.astype("datetime64[ns]")


def safe_float(x):
    try:
        if x is None or pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def fmt_num(x, digits=2):
    v = safe_float(x)
    if v is None:
        return "N/A"
    return f"{v:,.{digits}f}"


def fmt_pct(x, digits=2):
    v = safe_float(x)
    if v is None:
        return "N/A"
    return f"{v * 100:.{digits}f}%"


def is_korean_text(text):
    return any("가" <= ch <= "힣" for ch in str(text))


KR_FALLBACK_MAP = {
    "삼성전자": "005930",
    "삼성전자우": "005935",
    "SK하이닉스": "000660",
    "sk하이닉스": "000660",
    "현대차": "005380",
    "기아": "000270",
    "NAVER": "035420",
    "네이버": "035420",
    "카카오": "035720",
    "LG에너지솔루션": "373220",
    "삼성SDI": "006400",
    "삼성바이오로직스": "207940",
    "셀트리온": "068270",
    "POSCO홀딩스": "005490",
    "포스코홀딩스": "005490",
    "한화에어로스페이스": "012450",
    "두산에너빌리티": "034020",
}

# =========================================================
# Ticker search
# =========================================================
@st.cache_data(ttl=86400)
def get_krx_listing():
    if not FDR_AVAILABLE:
        return pd.DataFrame()
    try:
        df = fdr.StockListing("KRX")
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.copy()
        df["Code"] = df["Code"].astype(str).str.zfill(6)
        df["Name"] = df["Name"].astype(str).str.strip()
        return df[["Code", "Name"]]
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=86400)
def get_kr_market_type_map():
    # 한국 종목이 KOSPI/KOSDAQ 중 어디인지 대략 판단. 실패하면 KOSPI 기준 사용.
    result = {}
    if not PYKRX_AVAILABLE:
        return result
    try:
        today = datetime.today()
        for offset in range(0, 10):
            d = (today - timedelta(days=offset)).strftime("%Y%m%d")
            try:
                kospi = pkstock.get_market_ticker_list(d, market="KOSPI")
                kosdaq = pkstock.get_market_ticker_list(d, market="KOSDAQ")
                for code in kospi:
                    result[str(code).zfill(6)] = "KOSPI"
                for code in kosdaq:
                    result[str(code).zfill(6)] = "KOSDAQ"
                if result:
                    return result
            except Exception:
                continue
    except Exception:
        return result
    return result


def find_ticker(query):
    q = str(query).strip()
    if q == "":
        return None, None, None

    if q.isdigit() and len(q) == 6:
        return "KR", q, q

    if q in KR_FALLBACK_MAP:
        code = KR_FALLBACK_MAP[q]
        return "KR", code, q

    listing = get_krx_listing()
    if not listing.empty:
        exact = listing[listing["Name"] == q]
        if not exact.empty:
            return "KR", exact.iloc[0]["Code"], exact.iloc[0]["Name"]

        partial = listing[listing["Name"].str.contains(q, case=False, na=False)]
        if not partial.empty:
            return "KR", partial.iloc[0]["Code"], partial.iloc[0]["Name"]

    if is_korean_text(q):
        return None, None, None

    return "US", q.upper(), q.upper()

# =========================================================
# Data loading
# =========================================================
@st.cache_data(ttl=3600)
def load_price_data(market, ticker, start_date):
    start = pd.to_datetime(start_date).strftime("%Y-%m-%d")
    end = datetime.today().strftime("%Y-%m-%d")

    if market == "KR":
        if FDR_AVAILABLE:
            try:
                df = fdr.DataReader(ticker, start)
                if df is not None and not df.empty:
                    df = df.copy()
                    df.index = to_ns_datetime(df.index)
                    if "Volume" not in df.columns:
                        df["Volume"] = 0
                    return df[[c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]]
            except Exception:
                pass

        if PYKRX_AVAILABLE:
            try:
                s = pd.to_datetime(start_date).strftime("%Y%m%d")
                e = datetime.today().strftime("%Y%m%d")
                df = pkstock.get_market_ohlcv(s, e, ticker)
                if df is not None and not df.empty:
                    rename_map = {"시가": "Open", "고가": "High", "저가": "Low", "종가": "Close", "거래량": "Volume"}
                    df = df.rename(columns=rename_map)
                    df.index = to_ns_datetime(df.index)
                    return df[[c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]]
            except Exception:
                pass
        return pd.DataFrame()

    if market == "US" and YF_AVAILABLE:
        try:
            df = yf.Ticker(ticker).history(start=start, auto_adjust=True)
            if df is not None and not df.empty:
                df = df.copy()
                df.index = to_ns_datetime(df.index)
                if "Volume" not in df.columns:
                    df["Volume"] = 0
                return df[[c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]]
        except Exception:
            return pd.DataFrame()

    return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_yf_info(ticker):
    empty = {
        "trailing_pe": None,
        "forward_pe": None,
        "price_to_sales": None,
        "peg_ratio": None,
        "market_cap": None,
        "status": "N/A"
    }
    if not YF_AVAILABLE:
        empty["status"] = f"yfinance import 실패: {YF_IMPORT_ERROR}"
        return empty
    try:
        info = yf.Ticker(ticker).info
        return {
            "trailing_pe": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "price_to_sales": info.get("priceToSalesTrailing12Months"),
            "peg_ratio": info.get("pegRatio"),
            "market_cap": info.get("marketCap"),
            "status": "OK"
        }
    except Exception as e:
        empty["status"] = repr(e)
        return empty


@st.cache_data(ttl=3600)
def load_us_vix(start_date):
    if not YF_AVAILABLE:
        return pd.DataFrame(), "yfinance 없음"
    try:
        start = pd.to_datetime(start_date).strftime("%Y-%m-%d")
        vix = yf.Ticker("^VIX").history(start=start, auto_adjust=True)
        if vix is None or vix.empty:
            return pd.DataFrame(), "VIX empty"
        vix = vix.copy()
        vix.index = to_ns_datetime(vix.index)
        return vix[["Close"]].rename(columns={"Close": "VIX"}), "OK"
    except Exception as e:
        return pd.DataFrame(), repr(e)


@st.cache_data(ttl=3600)
def load_kr_market_index(start_date, kr_market_type):
    # 한국 종목은 미국 VIX 대신 KOSPI/KOSDAQ의 MDD를 시장위험으로 사용
    symbol = "KS11" if kr_market_type == "KOSPI" else "KQ11"
    name = "KOSPI" if kr_market_type == "KOSPI" else "KOSDAQ"
    start = pd.to_datetime(start_date).strftime("%Y-%m-%d")

    if FDR_AVAILABLE:
        try:
            idx = fdr.DataReader(symbol, start)
            if idx is not None and not idx.empty:
                idx = idx.copy()
                idx.index = to_ns_datetime(idx.index)
                idx["Market_Peak"] = idx["Close"].cummax()
                idx["Market_DD"] = idx["Close"] / idx["Market_Peak"] - 1
                return idx[["Close", "Market_DD"]].rename(columns={"Close": name}), f"OK: {name}"
        except Exception as e:
            return pd.DataFrame(), f"{name} FDR error: {repr(e)}"
    return pd.DataFrame(), f"{name} 데이터 없음"

# =========================================================
# Indicators
# =========================================================
def calculate_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calculate_indicators(df):
    out = df.copy()
    out["Peak"] = out["Close"].cummax()
    out["Current_Drawdown"] = out["Close"] / out["Peak"] - 1
    out["Max_Drawdown"] = out["Current_Drawdown"].cummin()
    out["Recovery_To_Peak"] = 1 / (1 + out["Current_Drawdown"]) - 1
    out["MA20"] = out["Close"].rolling(20).mean()
    out["MA60"] = out["Close"].rolling(60).mean()
    out["MA200"] = out["Close"].rolling(200).mean()
    out["RSI"] = calculate_rsi(out["Close"])
    out["Volume_MA20"] = out["Volume"].rolling(20).mean()
    out["Volume_Ratio"] = out["Volume"] / out["Volume_MA20"].replace(0, np.nan)
    out["Return"] = out["Close"].pct_change()
    out["Low20"] = out["Close"].rolling(20).min()
    out["Prev_Low20"] = out["Close"].shift(1).rolling(20).min()
    return out


def classify_profile(asset_type):
    profiles = {
        "일반 주식/ETF": {"watch": -0.08, "buy1": -0.12, "deep": -0.15, "risk": -0.20},
        "나스닥형 ETF": {"watch": -0.06, "buy1": -0.08, "deep": -0.12, "risk": -0.15},
        "반도체/메모리": {"watch": -0.08, "buy1": -0.12, "deep": -0.15, "risk": -0.20},
        "전력/AI인프라": {"watch": -0.08, "buy1": -0.10, "deep": -0.15, "risk": -0.18},
        "우주/고변동": {"watch": -0.12, "buy1": -0.18, "deep": -0.25, "risk": -0.30},
    }
    return profiles.get(asset_type, profiles["일반 주식/ETF"])


def build_trade_signals(df, profile, min_gap_days=25):
    sig = df.copy()
    hard_stop = (
        (sig["Current_Drawdown"] <= profile["risk"]) &
        (sig["Close"] < sig["Prev_Low20"]) &
        (sig["Return"] < 0)
    )

    raw_buy = (
        (sig["Current_Drawdown"] <= profile["buy1"]) &
        (sig["RSI"] <= 42) &
        (~hard_stop.fillna(False))
    )

    raw_cash = (
        ((sig["Current_Drawdown"] >= -0.03) & (sig["RSI"] >= 68)) |
        ((sig["Close"] < sig["MA20"]) & (sig["RSI"] >= 65) & (sig["Current_Drawdown"] > -0.08))
    )

    buy_points = []
    cash_points = []
    last_buy = None
    last_cash = None

    for dt, flag in raw_buy.items():
        if bool(flag):
            if last_buy is None or (dt - last_buy).days >= min_gap_days:
                buy_points.append(dt)
                last_buy = dt

    for dt, flag in raw_cash.items():
        if bool(flag):
            if last_cash is None or (dt - last_cash).days >= min_gap_days:
                cash_points.append(dt)
                last_cash = dt

    result = pd.DataFrame(index=df.index)
    result["Buy_Display"] = np.nan
    result["Cash_Display"] = np.nan
    result.loc[buy_points, "Buy_Display"] = df.loc[buy_points, "Close"]
    result.loc[cash_points, "Cash_Display"] = df.loc[cash_points, "Close"]
    return result

# =========================================================
# PER data
# =========================================================
@st.cache_data(ttl=3600)
def load_kr_per_series(ticker, start_date, end_date):
    if not PYKRX_AVAILABLE:
        return pd.DataFrame(), f"pykrx import 실패: {PYKRX_IMPORT_ERROR}"

    ticker = str(ticker).strip().zfill(6)
    start = pd.to_datetime(start_date).strftime("%Y%m%d")
    end = pd.to_datetime(end_date).strftime("%Y%m%d")
    errors = []

    attempts = [
        ("daily", lambda: pkstock.get_market_fundamental(start, end, ticker)),
        ("monthly", lambda: pkstock.get_market_fundamental(start, end, ticker, freq="m")),
    ]

    for name, loader in attempts:
        try:
            df = loader()
            if df is None or df.empty:
                errors.append(f"{name}: empty")
                continue
            df = df.copy()
            df.index = to_ns_datetime(df.index)
            keep = [c for c in ["PER", "PBR", "EPS", "BPS"] if c in df.columns]
            if "PER" not in keep:
                errors.append(f"{name}: PER 컬럼 없음 columns={df.columns.tolist()}")
                continue
            out = df[keep].copy()
            for col in keep:
                out[col] = pd.to_numeric(out[col], errors="coerce")
            out = out.replace([np.inf, -np.inf], np.nan)
            out = out[(out["PER"] > 0) & (out["PER"] < 300)].dropna(subset=["PER"])
            if not out.empty:
                return out, f"OK: pykrx {name} fundamental"
            errors.append(f"{name}: PER 유효값 없음")
        except Exception as e:
            errors.append(f"{name}: {repr(e)}")

    # 최종 보조: 최근 단일 날짜 전체 시장 fundamental에서 현재 PER만 확인
    try:
        for offset in range(0, 15):
            d = (pd.to_datetime(end_date) - pd.Timedelta(days=offset)).strftime("%Y%m%d")
            try:
                one = pkstock.get_market_fundamental(d, market="ALL")
                if one is not None and not one.empty:
                    one = one.copy()
                    one.index = one.index.astype(str).str.zfill(6)
                    if ticker in one.index and "PER" in one.columns:
                        val = pd.to_numeric(one.loc[ticker, "PER"], errors="coerce")
                        if pd.notna(val) and 0 < val < 300:
                            out = pd.DataFrame({"PER": [float(val)]}, index=[pd.to_datetime(d)])
                            return out, f"OK: pykrx latest market fundamental only {d}"
            except Exception as e:
                errors.append(f"latest {d}: {repr(e)}")
    except Exception as e:
        errors.append(f"latest fallback fatal: {repr(e)}")

    return pd.DataFrame(), " / ".join(errors)


@st.cache_data(ttl=3600)
def load_us_ttm_pe_series(ticker, price_df):
    if not YF_AVAILABLE:
        return pd.DataFrame(), f"yfinance import 실패: {YF_IMPORT_ERROR}"

    try:
        tk = yf.Ticker(ticker)
        eps_q = None
        source = ""

        # 1) reported EPS from earnings dates: 날짜가 발표일이라 일별 merge에 가장 자연스러움
        try:
            ed = tk.get_earnings_dates(limit=40)
            if ed is not None and not ed.empty and "Reported EPS" in ed.columns:
                eps = pd.to_numeric(ed["Reported EPS"], errors="coerce").dropna()
                eps.index = to_ns_datetime(eps.index)
                eps = eps.sort_index()
                if len(eps) >= 4:
                    eps_q = eps
                    source = "yfinance earnings_dates Reported EPS"
        except Exception:
            pass

        # 2) financial statement EPS rows
        if eps_q is None or len(eps_q) < 4:
            frames = []
            for attr in ["quarterly_income_stmt", "quarterly_financials"]:
                try:
                    stmt = getattr(tk, attr, None)
                    if stmt is not None and isinstance(stmt, pd.DataFrame) and not stmt.empty:
                        frames.append(stmt.copy())
                except Exception:
                    pass
            for stmt in frames:
                try:
                    stmt.columns = to_ns_datetime(stmt.columns)
                except Exception:
                    pass
                for row_name in ["Diluted EPS", "Basic EPS", "Normalized EPS", "Basic Average Shares"]:
                    if row_name in stmt.index and "EPS" in row_name:
                        eps = pd.to_numeric(stmt.loc[row_name], errors="coerce").dropna().sort_index()
                        if len(eps) >= 4:
                            eps_q = eps
                            source = f"yfinance financials {row_name}"
                            break
                if eps_q is not None and len(eps_q) >= 4:
                    # financial statement dates are period end; apply 45-day reporting lag
                    eps_q.index = to_ns_datetime(eps_q.index) + pd.Timedelta(days=45)
                    break

        if eps_q is None or len(eps_q) < 4:
            return pd.DataFrame(), "EPS 분기 데이터 부족"

        eps_q = eps_q.sort_index()
        eps_ttm = eps_q.rolling(4).sum().dropna()
        eps_ttm = eps_ttm[eps_ttm > 0]
        if eps_ttm.empty:
            return pd.DataFrame(), "EPS_TTM 유효값 없음"

        eps_df = pd.DataFrame({"Date": to_ns_datetime(eps_ttm.index), "EPS_TTM": eps_ttm.values})
        px = price_df[["Close"]].copy().reset_index()
        px.columns = ["Date", "Close"]
        px["Date"] = to_ns_datetime(px["Date"])
        eps_df["Date"] = to_ns_datetime(eps_df["Date"])

        px = px.sort_values("Date")
        eps_df = eps_df.sort_values("Date")
        merged = pd.merge_asof(px, eps_df, on="Date", direction="backward")
        merged["PER"] = merged["Close"] / merged["EPS_TTM"]
        merged["PER"] = pd.to_numeric(merged["PER"], errors="coerce")
        merged = merged.replace([np.inf, -np.inf], np.nan)
        merged = merged[(merged["PER"] > 0) & (merged["PER"] < 300)].dropna(subset=["PER"])

        if merged.empty:
            return pd.DataFrame(), "TTM P/E 유효값 없음"

        result = merged.set_index("Date")[["PER", "EPS_TTM"]]
        result.index = to_ns_datetime(result.index)
        return result, f"OK: {source}"

    except Exception as e:
        return pd.DataFrame(), repr(e)

# =========================================================
# Dashboard analysis text
# =========================================================
def get_current_valuation_snapshot(market, valuation, per_df):
    if market == "KR":
        current_pe = per_df["PER"].dropna().iloc[-1] if per_df is not None and not per_df.empty and "PER" in per_df.columns else None
        return {
            "pe_label": "KRX P/E",
            "pe": current_pe,
            "forward_pe": None,
            "ps": None,
            "peg": None,
        }
    return {
        "pe_label": "TTM P/E",
        "pe": valuation.get("trailing_pe"),
        "forward_pe": valuation.get("forward_pe"),
        "ps": valuation.get("price_to_sales"),
        "peg": valuation.get("peg_ratio"),
    }


def make_simple_action(latest, profile, per_df, market_risk_value=None):
    dd = latest["Current_Drawdown"]
    rsi = latest["RSI"]
    close = latest["Close"]
    ma20 = latest["MA20"]
    ma200 = latest["MA200"]

    if pd.notna(latest.get("Prev_Low20", np.nan)) and dd <= profile["risk"] and close < latest["Prev_Low20"]:
        return "매수 금지", "저점 이탈 + 깊은 MDD 구간입니다. 추세 훼손 확인이 우선입니다."

    if dd <= profile["buy1"] and pd.notna(rsi) and rsi <= 42:
        if pd.notna(ma20) and close >= ma20:
            return "1차 소액 가능", "MDD와 RSI는 저점매수 후보이며 MA20 회복으로 반등 확인이 일부 있습니다."
        return "소액만 / 확인 대기", "MDD와 RSI는 매력적이나 MA20 회복 전입니다. 선진입은 소액만 적합합니다."

    if dd > -0.05 and pd.notna(rsi) and rsi >= 65:
        return "추격 금지 / 현금확보 검토", "고점권 또는 과열 구간입니다. 신규 추격보다 현금확보 후보입니다."

    if pd.notna(ma20) and close < ma20 and dd > profile["watch"]:
        return "대기", "낙폭은 깊지 않은데 단기 추세가 약합니다. 가격 매력 부족 구간입니다."

    return "관망", "극단적인 저점매수 또는 과열 신호가 아닙니다. MDD와 MA20 회복 여부를 계속 확인하세요."


def make_chart_comment(df, per_df, market, market_risk_df):
    latest = df.iloc[-1]
    parts = []

    close = latest["Close"]
    ma20 = latest["MA20"]
    ma200 = latest["MA200"]
    dd = latest["Current_Drawdown"]
    rsi = latest["RSI"]

    if pd.notna(ma20):
        if close >= ma20:
            parts.append("가격: MA20 위라 단기 반등 흐름은 유지 중입니다.")
        else:
            parts.append("가격: MA20 아래라 확인매수보다 대기/소액 접근이 우선입니다.")

    if pd.notna(ma200):
        if close >= ma200:
            parts.append("장기추세: MA200 위라 장기 추세 훼손은 제한적입니다.")
        else:
            parts.append("장기추세: MA200 아래라 추세 훼손 위험을 확인해야 합니다.")

    parts.append(f"MDD: 현재 {dd * 100:.2f}%로 {'저점매수 관심 구간' if dd <= -0.12 else '일반 조정 구간'}입니다.")

    if pd.notna(rsi):
        if rsi <= 30:
            parts.append(f"RSI: {rsi:.1f}로 강한 과매도권입니다.")
        elif rsi <= 42:
            parts.append(f"RSI: {rsi:.1f}로 약한 과매도/눌림 구간입니다.")
        elif rsi >= 70:
            parts.append(f"RSI: {rsi:.1f}로 과열권입니다.")
        else:
            parts.append(f"RSI: {rsi:.1f}로 중립권입니다.")

    if per_df is not None and not per_df.empty and "PER" in per_df.columns and len(per_df.dropna()) >= 10:
        joined = df[["Close"]].join(per_df[["PER"]], how="left")
        joined["PER"] = joined["PER"].ffill()
        joined = joined.dropna(subset=["Close", "PER"])
        if len(joined) >= 60:
            recent = joined.iloc[-1]
            past = joined.iloc[-60]
            price_chg = recent["Close"] / past["Close"] - 1
            per_chg = recent["PER"] / past["PER"] - 1
            if price_chg > 0 and per_chg < 0:
                parts.append(f"PER: 최근 60거래일 주가 +{price_chg*100:.1f}%, PER {per_chg*100:.1f}%입니다. 이익 개선이 주가 상승을 정당화하는 흐름입니다.")
            elif price_chg > 0 and per_chg > 0:
                parts.append(f"PER: 최근 60거래일 주가 +{price_chg*100:.1f}%, PER +{per_chg*100:.1f}%입니다. 밸류 부담이 같이 커졌습니다.")
            elif price_chg < 0 and per_chg < 0:
                parts.append(f"PER: 최근 60거래일 주가 {price_chg*100:.1f}%, PER {per_chg*100:.1f}%입니다. 밸류 부담은 완화됐지만 업황 확인이 필요합니다.")
            else:
                parts.append(f"PER: 최근 60거래일 주가 {price_chg*100:.1f}%, PER +{per_chg*100:.1f}%입니다. 주가 하락 중 이익 둔화 가능성을 확인해야 합니다.")
    else:
        parts.append("PER: 시계열 데이터가 부족해 현재 PER 카드만 참고해야 합니다.")

    if market == "US" and market_risk_df is not None and not market_risk_df.empty and "VIX" in market_risk_df.columns:
        vix = market_risk_df["VIX"].dropna().iloc[-1]
        parts.append(f"시장위험: VIX {vix:.1f}입니다. 25 이상이면 공포성 눌림 여부를 별도 확인하세요.")
    elif market == "KR" and market_risk_df is not None and not market_risk_df.empty and "Market_DD" in market_risk_df.columns:
        mdd = market_risk_df["Market_DD"].dropna().iloc[-1]
        parts.append(f"시장위험: 한국 지수 MDD {mdd*100:.2f}%입니다. 지수 자체가 약하면 개별주 반등 신뢰가 낮아집니다.")

    return parts

# =========================================================
# Plot
# =========================================================
def plot_core_dashboard(df, per_df, market_risk_df, signal_df, title, market, valuation):
    chart = df[["Close", "MA20", "MA60", "MA200", "Current_Drawdown"]].copy()
    chart = chart.rename(columns={"Close": "Price"})
    chart.index = to_ns_datetime(chart.index)

    if per_df is not None and not per_df.empty and "PER" in per_df.columns:
        p = per_df[["PER"]].copy()
        p.index = to_ns_datetime(p.index)
        chart = chart.join(p, how="left")
        chart["PER"] = chart["PER"].ffill()
    else:
        chart["PER"] = np.nan

    if signal_df is not None and not signal_df.empty:
        chart = chart.join(signal_df[["Buy_Display", "Cash_Display"]], how="left")
    else:
        chart["Buy_Display"] = np.nan
        chart["Cash_Display"] = np.nan

    chart = chart.sort_index()

    fig, (ax1, ax3) = plt.subplots(
        2, 1,
        figsize=(15, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]}
    )

    # 상단: Price + MA
    ax1.plot(chart.index, chart["Price"], label="Price", linewidth=1.8)
    ax1.plot(chart.index, chart["MA20"], label="MA20", linewidth=1.1)
    ax1.plot(chart.index, chart["MA60"], label="MA60", linewidth=1.1)
    ax1.plot(chart.index, chart["MA200"], label="MA200", linewidth=1.1)
    ax1.set_ylabel("Price")
    ax1.grid(True, linestyle=":", alpha=0.5)

    if "Buy_Display" in chart.columns:
        buy_y = chart["Buy_Display"] * 0.97
        ax1.scatter(chart.index, buy_y, marker="^", s=90, label="BUY candidate", zorder=5)
    if "Cash_Display" in chart.columns:
        cash_y = chart["Cash_Display"] * 1.03
        ax1.scatter(chart.index, cash_y, marker="v", s=90, label="Cash / overheat", zorder=5)

    # 오른쪽 축: PER
    ax2 = ax1.twinx()
    per_available = not chart["PER"].dropna().empty
    if per_available:
        ax2.plot(chart.index, chart["PER"], label="P/E", linewidth=1.5)
        avg_pe = chart["PER"].dropna().mean()
        ax2.axhline(avg_pe, linestyle="--", linewidth=1.0, alpha=0.6, label="P/E avg")

    if market == "US":
        fpe = safe_float(valuation.get("forward_pe"))
        tpe = safe_float(valuation.get("trailing_pe"))
        if fpe is not None:
            ax2.axhline(fpe, linestyle="-.", linewidth=1.0, alpha=0.7, label="Current forward P/E")
        if tpe is not None:
            ax2.axhline(tpe, linestyle=":", linewidth=1.0, alpha=0.7, label="Current trailing P/E")

    ax2.set_ylabel("P/E")

    ax1.set_title(title, fontsize=14, fontweight="bold")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

    # 하단: MDD + Market risk
    ax3.plot(chart.index, chart["Current_Drawdown"] * 100, label="Current DD", linewidth=1.4)
    for y, label in [(-8, "Watch -8%"), (-12, "Buy zone -12%"), (-15, "Deep -15%"), (-20, "Risk -20%")]:
        ax3.axhline(y, linestyle="--", linewidth=0.9, alpha=0.5, label=label)
    ax3.set_ylabel("MDD (%)")
    ax3.grid(True, linestyle=":", alpha=0.5)

    if market == "US" and market_risk_df is not None and not market_risk_df.empty and "VIX" in market_risk_df.columns:
        risk = market_risk_df.copy()
        risk.index = to_ns_datetime(risk.index)
        ax4 = ax3.twinx()
        ax4.plot(risk.index, risk["VIX"], linestyle="--", linewidth=1.1, alpha=0.7, label="VIX")
        ax4.set_ylabel("VIX")
        ax4.set_ylim(0, max(45, float(risk["VIX"].dropna().max()) * 1.15) if not risk["VIX"].dropna().empty else 45)
        lines3, labels3 = ax3.get_legend_handles_labels()
        lines4, labels4 = ax4.get_legend_handles_labels()
        ax3.legend(lines3 + lines4, labels3 + labels4, loc="lower left", fontsize=8)
    elif market == "KR" and market_risk_df is not None and not market_risk_df.empty and "Market_DD" in market_risk_df.columns:
        risk = market_risk_df.copy()
        risk.index = to_ns_datetime(risk.index)
        ax3.plot(risk.index, risk["Market_DD"] * 100, linestyle="--", linewidth=1.1, alpha=0.8, label="Korea market DD")
        ax3.legend(loc="lower left", fontsize=8)
    else:
        ax3.legend(loc="lower left", fontsize=8)

    plt.tight_layout()
    st.pyplot(fig)

# =========================================================
# UI
# =========================================================
with st.expander("데이터 모듈 상태", expanded=False):
    st.write({
        "yfinance": YF_AVAILABLE,
        "FinanceDataReader": FDR_AVAILABLE,
        "pykrx": PYKRX_AVAILABLE,
        "yfinance_error": YF_IMPORT_ERROR,
        "FDR_error": FDR_IMPORT_ERROR,
        "pykrx_error": PYKRX_IMPORT_ERROR,
    })

col1, col2, col3 = st.columns(3)
with col1:
    user_input = st.text_input("종목명 / 종목코드 / 미국 티커", value="NVDA")
with col2:
    start_date = st.date_input("기준 시작일", value=pd.to_datetime("2024-01-01"))
with col3:
    asset_type = st.selectbox(
        "종목 유형",
        ["일반 주식/ETF", "나스닥형 ETF", "반도체/메모리", "전력/AI인프라", "우주/고변동"],
        index=2,
    )

run = st.button("분석 실행")

if run:
    market, ticker, display_name = find_ticker(user_input)
    if ticker is None:
        st.error("종목을 찾지 못했습니다. 한국 종목은 종목명 또는 6자리 코드, 미국 종목은 티커를 입력하세요.")
        st.stop()

    profile = classify_profile(asset_type)

    with st.spinner("가격 / PER / 시장위험 데이터 분석 중..."):
        price_df = load_price_data(market, ticker, start_date)
        if price_df.empty:
            st.error("가격 데이터를 가져오지 못했습니다.")
            st.stop()

        df = calculate_indicators(price_df)
        signal_df = build_trade_signals(df, profile)

        valuation = load_yf_info(ticker) if market == "US" else {
            "trailing_pe": None,
            "forward_pe": None,
            "price_to_sales": None,
            "peg_ratio": None,
            "market_cap": None,
            "status": "KR"
        }

        if market == "KR":
            kr_type_map = get_kr_market_type_map()
            kr_market_type = kr_type_map.get(ticker, "KOSPI")
            per_df, per_status = load_kr_per_series(ticker, start_date, datetime.today())
            market_risk_df, risk_status = load_kr_market_index(start_date, kr_market_type)
        else:
            per_df, per_status = load_us_ttm_pe_series(ticker, df)
            market_risk_df, risk_status = load_us_vix(start_date)

        latest = df.iloc[-1]
        current_val = get_current_valuation_snapshot(market, valuation, per_df)
        action, action_reason = make_simple_action(latest, profile, per_df)

    st.subheader(f"분석 대상: {display_name} / {ticker} / {market}")

    # 핵심 카드
    st.markdown("## 1. 핵심 판단")
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("최종 행동", action)
    k2.metric("Current DD", f"{latest['Current_Drawdown']*100:.2f}%")
    k3.metric("Max DD", f"{df['Max_Drawdown'].min()*100:.2f}%")
    k4.metric("RSI", fmt_num(latest["RSI"]))
    ma20_state = "위" if pd.notna(latest["MA20"]) and latest["Close"] >= latest["MA20"] else "아래"
    k5.metric("MA20 상태", ma20_state)
    k6.metric("Vol Ratio", fmt_num(latest["Volume_Ratio"]))
    st.info(action_reason)

    # 현재 밸류 카드
    st.markdown("## 2. 현재 Valuation")
    v1, v2, v3, v4 = st.columns(4)
    v1.metric(current_val["pe_label"], fmt_num(current_val["pe"]))
    v2.metric("Forward P/E", fmt_num(current_val["forward_pe"]))
    v3.metric("P/S", fmt_num(current_val["ps"]))
    v4.metric("PEG", fmt_num(current_val["peg"]))

    if per_df.empty:
        st.warning(f"PER 시계열 없음: {per_status}")
    else:
        st.caption(f"PER data status: {per_status}")

    st.caption(f"Market risk status: {risk_status}")

    # 핵심 차트
    st.markdown("## 3. Price + PER + MDD + Market Risk")
    if market == "KR":
        title = f"{ticker} Price + KRX P/E + MDD + Korea Market Risk"
    else:
        title = f"{ticker} Price + Estimated TTM P/E + MDD + VIX"
    plot_core_dashboard(df, per_df, market_risk_df, signal_df, title, market, valuation)

    # 자동 해석
    st.markdown("## 4. 차트 해석")
    comments = make_chart_comment(df, per_df, market, market_risk_df)
    for c in comments:
        st.write(f"- {c}")

    # Debug / raw data
    with st.expander("PER 원자료 / 디버그"):
        st.write("PER status:", per_status)
        st.write("Market risk status:", risk_status)
        if not per_df.empty:
            st.dataframe(per_df.tail(30), use_container_width=True)
        else:
            st.info("PER 원자료가 비어 있습니다.")

        if market == "KR" and PYKRX_AVAILABLE:
            st.markdown("### pykrx 직접 검증: 삼성전자 005930")
            try:
                test = pkstock.get_market_fundamental("20240102", "20240110", "005930")
                st.dataframe(test, use_container_width=True)
            except Exception as e:
                st.code(repr(e))

    with st.expander("최근 20거래일 데이터"):
        view = df[["Close", "Current_Drawdown", "Max_Drawdown", "RSI", "MA20", "MA60", "MA200", "Volume_Ratio"]].tail(20).copy()
        view["Current_Drawdown"] = view["Current_Drawdown"] * 100
        view["Max_Drawdown"] = view["Max_Drawdown"] * 100
        st.dataframe(view, use_container_width=True)

    st.warning(
        "주의: 이 도구는 매수/매도 자동 추천기가 아닙니다. "
        "PER은 한국은 KRX PER, 미국은 yfinance 기반 Estimated TTM P/E입니다. "
        "미국 Forward P/E 과거 시계열은 무료 데이터로 안정적으로 제공되지 않아 현재 기준선으로만 표시합니다."
    )
