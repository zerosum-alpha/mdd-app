import warnings
warnings.filterwarnings("ignore")

from datetime import datetime, timedelta
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# =========================================================
# MDD Core Dashboard - Stable Final
# 핵심만 표시:
# - Price + MA20/60/200
# - P/E, Current valuation
# - MDD
# - Market risk: US=VIX, KR=KOSPI/KOSDAQ drawdown
# - BUY / CASH markers
# =========================================================

# ---------- Optional imports ----------
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
    import requests
    REQUESTS_AVAILABLE = True
    REQUESTS_IMPORT_ERROR = ""
except Exception as e:
    requests = None
    REQUESTS_AVAILABLE = False
    REQUESTS_IMPORT_ERROR = repr(e)

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
    BS4_IMPORT_ERROR = ""
except Exception as e:
    BeautifulSoup = None
    BS4_AVAILABLE = False
    BS4_IMPORT_ERROR = repr(e)

try:
    from auth import require_login, logout_button
except Exception:
    def require_login():
        return None
    def logout_button():
        return None

# ---------- Page ----------
st.set_page_config(page_title="MDD 분석기", layout="wide")
require_login()
logout_button()

st.title("📈 MDD 저점매수 분석기 | Core Stable")
st.caption("주가 / PER / MDD / 시장위험 / 이평선만 봅니다.")

# =========================================================
# Utility
# =========================================================
def normalize_dt_index(idx):
    out = pd.to_datetime(idx, errors="coerce")
    try:
        out = out.tz_localize(None)
    except Exception:
        try:
            out = out.tz_convert(None)
        except Exception:
            pass
    return pd.DatetimeIndex(out).astype("datetime64[ns]")


def yyyymmdd(x):
    return pd.to_datetime(x).strftime("%Y%m%d")


def safe_float(x):
    try:
        if x is None or pd.isna(x):
            return None
        return float(str(x).replace(",", ""))
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
    return f"{v:.{digits}f}%"


def is_korean_text(text):
    return any("가" <= ch <= "힣" for ch in str(text))


def compact_status(text, max_len=160):
    text = str(text)
    if len(text) <= max_len:
        return text
    return text[:max_len] + " ..."


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
    "엘지에너지솔루션": "373220",
    "삼성SDI": "006400",
    "삼성바이오로직스": "207940",
    "셀트리온": "068270",
    "POSCO홀딩스": "005490",
    "포스코홀딩스": "005490",
    "한화에어로스페이스": "012450",
    "두산에너빌리티": "034020",
}

# =========================================================
# Ticker lookup
# =========================================================
@st.cache_data(ttl=86400)
def get_krx_stock_list():
    if not FDR_AVAILABLE:
        return pd.DataFrame()
    try:
        df = fdr.StockListing("KRX")
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.copy()
        if "Code" in df.columns:
            df["Code"] = df["Code"].astype(str).str.zfill(6)
        if "Name" in df.columns:
            df["Name"] = df["Name"].astype(str).str.strip()
        return df
    except Exception:
        return pd.DataFrame()


def find_ticker(query):
    q = str(query).strip()
    if not q:
        return None, None, None

    if q.isdigit() and len(q) == 6:
        return "KR", q, q

    if q in KR_FALLBACK_MAP:
        return "KR", KR_FALLBACK_MAP[q], q

    stock_list = get_krx_stock_list()
    if not stock_list.empty and {"Name", "Code"}.issubset(stock_list.columns):
        exact = stock_list[stock_list["Name"] == q]
        if not exact.empty:
            return "KR", exact.iloc[0]["Code"], exact.iloc[0]["Name"]
        partial = stock_list[stock_list["Name"].str.contains(q, case=False, na=False)]
        if not partial.empty:
            return "KR", partial.iloc[0]["Code"], partial.iloc[0]["Name"]

    if is_korean_text(q):
        return None, None, None

    return "US", q.upper(), q.upper()

# =========================================================
# Price data
# =========================================================
@st.cache_data(ttl=1800)
def load_price_data(market, ticker, start_date):
    start = pd.to_datetime(start_date).strftime("%Y-%m-%d")
    try:
        if market == "KR":
            if not FDR_AVAILABLE:
                return pd.DataFrame(), f"FinanceDataReader import 실패: {FDR_IMPORT_ERROR}"
            df = fdr.DataReader(str(ticker).zfill(6), start)
            if df is None or df.empty:
                return pd.DataFrame(), "FinanceDataReader 가격 데이터 empty"
            df = df.copy()
            df.index = normalize_dt_index(df.index)
            return df, "OK"

        if market == "US":
            if not YF_AVAILABLE:
                return pd.DataFrame(), f"yfinance import 실패: {YF_IMPORT_ERROR}"
            df = yf.Ticker(ticker).history(start=start, auto_adjust=True)
            if df is None or df.empty:
                return pd.DataFrame(), "yfinance 가격 데이터 empty"
            df = df.copy()
            df.index = normalize_dt_index(df.index)
            return df, "OK"
    except Exception as e:
        return pd.DataFrame(), repr(e)
    return pd.DataFrame(), "unknown market"


@st.cache_data(ttl=1800)
def load_yf_close(ticker, start_date):
    if not YF_AVAILABLE:
        return pd.DataFrame()
    try:
        df = yf.Ticker(ticker).history(start=pd.to_datetime(start_date).strftime("%Y-%m-%d"), auto_adjust=True)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.copy()
        df.index = normalize_dt_index(df.index)
        return df[["Close"]].rename(columns={"Close": ticker})
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=1800)
def load_fdr_close(symbol, start_date):
    if not FDR_AVAILABLE:
        return pd.DataFrame()
    try:
        df = fdr.DataReader(symbol, pd.to_datetime(start_date).strftime("%Y-%m-%d"))
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.copy()
        df.index = normalize_dt_index(df.index)
        return df[["Close"]].rename(columns={"Close": symbol})
    except Exception:
        return pd.DataFrame()

# =========================================================
# Indicators
# =========================================================
def calculate_rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calculate_indicators(df):
    df = df.copy()
    if "Volume" not in df.columns:
        df["Volume"] = 0
    df["Peak"] = df["Close"].cummax()
    df["Current_Drawdown"] = df["Close"] / df["Peak"] - 1
    df["Max_Drawdown"] = df["Current_Drawdown"].cummin()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA60"] = df["Close"].rolling(60).mean()
    df["MA200"] = df["Close"].rolling(200).mean()
    df["RSI"] = calculate_rsi(df["Close"], 14)
    df["Volume_MA20"] = df["Volume"].rolling(20).mean()
    df["Volume_Ratio"] = df["Volume"] / df["Volume_MA20"].replace(0, np.nan)
    return df


def de_dupe_signal(mask, min_gap=20):
    out = pd.Series(False, index=mask.index)
    last_i = -10**9
    vals = mask.fillna(False).values
    for i, v in enumerate(vals):
        if v and i - last_i >= min_gap:
            out.iloc[i] = True
            last_i = i
    return out


def build_signals(df):
    sig = pd.DataFrame(index=df.index)
    buy_raw = (df["Current_Drawdown"] <= -0.12) & (df["RSI"] <= 42)
    cash_raw = ((df["Current_Drawdown"] >= -0.03) & (df["RSI"] >= 68)) | (
        (df["Close"] > df["MA20"] * 1.08) & (df["RSI"] >= 65)
    )
    sig["Buy_Display"] = df["Close"].where(de_dupe_signal(buy_raw, min_gap=25))
    sig["Cash_Display"] = df["Close"].where(de_dupe_signal(cash_raw, min_gap=35))
    return sig

# =========================================================
# Valuation - Current
# =========================================================
@st.cache_data(ttl=3600)
def load_us_current_valuation(ticker):
    data = {"trailing_pe": None, "forward_pe": None, "price_to_sales": None, "peg_ratio": None}
    if not YF_AVAILABLE:
        return data, f"yfinance import 실패: {YF_IMPORT_ERROR}"
    try:
        info = yf.Ticker(ticker).info
        data["trailing_pe"] = info.get("trailingPE")
        data["forward_pe"] = info.get("forwardPE")
        data["price_to_sales"] = info.get("priceToSalesTrailing12Months")
        data["peg_ratio"] = info.get("pegRatio")
        return data, "OK"
    except Exception as e:
        return data, repr(e)


@st.cache_data(ttl=3600)
def load_kr_naver_current_valuation(code):
    """Fallback: Naver current valuation snapshot only. Not historical."""
    data = {"trailing_pe": None, "forward_pe": None, "price_to_sales": None, "peg_ratio": None, "eps": None}
    if not REQUESTS_AVAILABLE:
        return data, f"requests import 실패: {REQUESTS_IMPORT_ERROR}"

    url = f"https://finance.naver.com/item/main.naver?code={str(code).zfill(6)}"
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=8)
        r.raise_for_status()
        html = r.text

        def _extract_id(id_name):
            # Naver often uses ids like _per, _eps, _pbr
            m = re.search(rf'id=["\']{re.escape(id_name)}["\'][^>]*>\s*([^<]+)\s*<', html)
            if m:
                return safe_float(m.group(1))
            return None

        data["trailing_pe"] = _extract_id("_per")
        data["eps"] = _extract_id("_eps")
        # keep PBR inside status only; main cards are P/E, Fwd P/E, P/S, PEG
        pbr = _extract_id("_pbr")
        status = "OK: Naver current snapshot"
        if pbr is not None:
            status += f" / PBR={pbr:.2f}"
        if data["trailing_pe"] is None:
            status = "Naver current PER not found"
        return data, status
    except Exception as e:
        return data, f"Naver current valuation error: {repr(e)}"

# =========================================================
# KR historical P/E with robust pykrx loaders
# =========================================================
FUND_ALIASES = {
    "BPS": ["BPS", "bps", "주당순자산", "주당순자산가치"],
    "PER": ["PER", "per", "P/E", "PE", "주가수익비율"],
    "PBR": ["PBR", "pbr", "P/B", "PB", "주가순자산비율"],
    "EPS": ["EPS", "eps", "주당순이익"],
    "DIV": ["DIV", "div", "배당수익률"],
    "DPS": ["DPS", "dps", "주당배당금"],
}


def flatten_columns(df):
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = ["_".join([str(x) for x in tup if str(x) != ""]).strip() for tup in out.columns]
    else:
        out.columns = [str(c).strip() for c in out.columns]
    return out


def canonical_fundamental_df(raw):
    if raw is None or not isinstance(raw, pd.DataFrame) or raw.empty:
        return pd.DataFrame()

    candidates = []
    candidates.append(flatten_columns(raw))
    try:
        candidates.append(flatten_columns(raw.T))
    except Exception:
        pass

    for cand in candidates:
        if cand is None or cand.empty:
            continue

        rename = {}
        for canonical, aliases in FUND_ALIASES.items():
            for c in cand.columns:
                cs = str(c).strip()
                if cs in aliases or cs.upper() == canonical:
                    rename[c] = canonical
                    break
        temp = cand.rename(columns=rename)
        keep = [c for c in ["BPS", "PER", "PBR", "EPS", "DIV", "DPS"] if c in temp.columns]
        if "PER" in keep or "EPS" in keep:
            out = temp[keep].copy()
            for c in keep:
                out[c] = pd.to_numeric(out[c], errors="coerce")
            return out.dropna(how="all")

    return pd.DataFrame()


def extract_ticker_row(raw, ticker):
    if raw is None or not isinstance(raw, pd.DataFrame) or raw.empty:
        return pd.DataFrame()
    ticker = str(ticker).zfill(6)
    raw = raw.copy()
    idx = raw.index.astype(str).str.zfill(6)
    if ticker in set(idx):
        return raw.loc[idx == ticker]
    for col in ["티커", "Ticker", "Code", "종목코드"]:
        if col in raw.columns:
            mask = raw[col].astype(str).str.zfill(6) == ticker
            if mask.any():
                return raw.loc[mask]
    return pd.DataFrame()


def clean_per_out(out, price_df=None):
    if out is None or out.empty:
        return pd.DataFrame()
    out = out.copy()
    # date index
    try:
        out.index = normalize_dt_index(out.index)
    except Exception:
        pass
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    # If PER is missing but EPS exists, calculate Price/EPS after join later.
    if "PER" in out.columns:
        out["PER"] = pd.to_numeric(out["PER"], errors="coerce")
        out = out[(out["PER"] > 0) & (out["PER"] < 500)]
    return out.dropna(how="all")


@st.cache_data(ttl=3600)
def load_kr_per_series(code, start_date, end_date):
    if not PYKRX_AVAILABLE:
        return pd.DataFrame(), f"pykrx import 실패: {PYKRX_IMPORT_ERROR}", PYKRX_IMPORT_ERROR

    code = str(code).strip().zfill(6)
    start = yyyymmdd(start_date)
    end = yyyymmdd(end_date)
    debug = []

    # 1. Official period APIs. Support multiple versions.
    calls = []
    if hasattr(pkstock, "get_market_fundamental"):
        calls.extend([
            ("period get_market_fundamental", lambda: pkstock.get_market_fundamental(start, end, code)),
            ("period get_market_fundamental freq=d", lambda: pkstock.get_market_fundamental(start, end, code, freq="d")),
            ("period get_market_fundamental freq=m", lambda: pkstock.get_market_fundamental(start, end, code, freq="m")),
        ])
    if hasattr(pkstock, "get_market_fundamental_by_date"):
        calls.append(("period get_market_fundamental_by_date", lambda: pkstock.get_market_fundamental_by_date(start, end, code)))

    for name, fn in calls:
        try:
            raw = fn()
            debug.append(f"{name}: shape={getattr(raw, 'shape', None)} cols={list(getattr(raw, 'columns', []))[:6]}")
            out = canonical_fundamental_df(raw)
            if not out.empty:
                # preserve date index when possible
                try:
                    if len(out) == len(raw):
                        out.index = normalize_dt_index(raw.index)
                except Exception:
                    pass
                out = clean_per_out(out)
                if not out.empty and ("PER" in out.columns or "EPS" in out.columns):
                    return out.sort_index(), f"OK: {name}", " / ".join(debug[-6:])
        except Exception as e:
            debug.append(f"{name}: {type(e).__name__}: {repr(e)[:180]}")

    # 2. Market-wide sample fallback: KOSPI/KOSDAQ/KONEX by ticker.
    try:
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        # month-end + recent business days: enough for chart without over-calling KRX
        try:
            month_ends = list(pd.date_range(start=start_dt, end=end_dt, freq="ME"))
        except Exception:
            month_ends = list(pd.date_range(start=start_dt, end=end_dt, freq="M"))
        recent_bdays = list(pd.bdate_range(end=end_dt, periods=min(45, max(1, len(pd.bdate_range(start_dt, end_dt))))))
        samples = sorted(set(month_ends + recent_bdays))
    except Exception:
        samples = []

    rows = []
    markets = ["KOSPI", "KOSDAQ", "KONEX", "ALL"]

    for d in samples:
        got = False
        # if holiday, search back 7 days
        for back in range(0, 8):
            ds = (pd.to_datetime(d) - pd.Timedelta(days=back)).strftime("%Y%m%d")
            for mkt in markets:
                if not hasattr(pkstock, "get_market_fundamental_by_ticker"):
                    continue
                try:
                    raw = pkstock.get_market_fundamental_by_ticker(ds, market=mkt)
                    row = extract_ticker_row(raw, code)
                    if row.empty:
                        continue
                    out = canonical_fundamental_df(row)
                    out = clean_per_out(out)
                    if not out.empty:
                        out = out.iloc[[0]].copy()
                        out.index = pd.DatetimeIndex([pd.to_datetime(ds)]).astype("datetime64[ns]")
                        rows.append(out)
                        got = True
                        if len(debug) < 12:
                            debug.append(f"OK sample {ds} {mkt}")
                        break
                except Exception as e:
                    if len(debug) < 12:
                        debug.append(f"sample {ds} {mkt}: {type(e).__name__}: {repr(e)[:120]}")
            if got:
                break

    if rows:
        out = pd.concat(rows).sort_index()
        out = out[~out.index.duplicated(keep="last")]
        out = clean_per_out(out)
        if not out.empty and ("PER" in out.columns or "EPS" in out.columns):
            return out, "OK: sampled get_market_fundamental_by_ticker", " / ".join(debug[-10:])

    return pd.DataFrame(), "PER 데이터 없음: pykrx 기간조회/샘플조회 모두 실패", " / ".join(debug[-12:])

# =========================================================
# US Estimated TTM P/E
# =========================================================
def find_row_by_keywords(df, keywords):
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    for idx in df.index:
        s = str(idx).lower().replace(" ", "")
        if all(k.lower().replace(" ", "") in s for k in keywords):
            return idx
    return None


@st.cache_data(ttl=3600)
def load_us_ttm_pe_series(ticker, price_df):
    if not YF_AVAILABLE:
        return pd.DataFrame(), f"yfinance import 실패: {YF_IMPORT_ERROR}"
    try:
        tk = yf.Ticker(ticker)
        frames = []
        for attr in ["quarterly_income_stmt", "quarterly_financials"]:
            obj = getattr(tk, attr, None)
            if obj is not None and isinstance(obj, pd.DataFrame) and not obj.empty:
                frames.append(obj.copy())

        eps_q = None
        status = []
        for stmt in frames:
            stmt.columns = pd.to_datetime(stmt.columns, errors="coerce")
            for keys in [["diluted", "eps"], ["basic", "eps"], ["normalized", "eps"]]:
                row = find_row_by_keywords(stmt, keys)
                if row is not None:
                    eps_q = pd.to_numeric(stmt.loc[row], errors="coerce").dropna().sort_index()
                    status.append(f"EPS row={row}")
                    break
            if eps_q is not None and len(eps_q) >= 4:
                break

            ni_row = find_row_by_keywords(stmt, ["net", "income"])
            sh_row = find_row_by_keywords(stmt, ["diluted", "average", "shares"])
            if ni_row is not None and sh_row is not None:
                ni = pd.to_numeric(stmt.loc[ni_row], errors="coerce")
                sh = pd.to_numeric(stmt.loc[sh_row], errors="coerce")
                eps_q = (ni / sh).dropna().sort_index()
                status.append(f"EPS=NetIncome/Shares {ni_row}/{sh_row}")
                if len(eps_q) >= 4:
                    break

        if eps_q is None or len(eps_q) < 4:
            return pd.DataFrame(), "분기 EPS 데이터 부족"

        eps_ttm = eps_q.rolling(4).sum().dropna()
        if eps_ttm.empty:
            return pd.DataFrame(), "EPS TTM 계산 불가"

        eps_df = pd.DataFrame({
            "Date": normalize_dt_index(pd.to_datetime(eps_ttm.index) + pd.Timedelta(days=45)),
            "EPS_TTM": eps_ttm.values,
        }).dropna().sort_values("Date")

        daily = price_df[["Close"]].copy().reset_index()
        daily.columns = ["Date", "Close"]
        daily["Date"] = normalize_dt_index(daily["Date"])
        daily = daily.dropna().sort_values("Date")

        merged = pd.merge_asof(daily, eps_df, on="Date", direction="backward")
        merged["PER"] = merged["Close"] / merged["EPS_TTM"]
        merged["PER"] = pd.to_numeric(merged["PER"], errors="coerce")
        merged = merged[(merged["PER"] > 0) & (merged["PER"] < 500)]
        if merged.empty:
            return pd.DataFrame(), "PER 계산 결과 empty"
        out = merged.set_index("Date")[["PER", "EPS_TTM"]]
        return out, "OK: yfinance EPS TTM / " + " / ".join(status[:2])
    except Exception as e:
        return pd.DataFrame(), repr(e)

# =========================================================
# Market Risk
# =========================================================
def make_market_risk_series(market, start_date, asset_type):
    if market == "US":
        vix = load_yf_close("^VIX", start_date)
        if vix.empty:
            return pd.DataFrame(), "VIX 없음", "VIX"
        return vix.rename(columns={"^VIX": "Risk"}), "OK: VIX", "VIX"

    symbol = "KQ11" if "코스닥" in str(asset_type) else "KS11"
    idx = load_fdr_close(symbol, start_date)
    if idx.empty:
        return pd.DataFrame(), f"{symbol} 지수 없음", "KR Index DD"
    risk = pd.DataFrame(index=idx.index)
    risk["Risk"] = (idx.iloc[:, 0] / idx.iloc[:, 0].cummax() - 1) * 100
    return risk, f"OK: {symbol} DD", "KR Index DD(%)"

# =========================================================
# Plot and comment
# =========================================================
def merge_chart_df(df, per_df, risk_df, signal_df):
    chart = df[["Close", "MA20", "MA60", "MA200", "Current_Drawdown"]].copy()
    chart = chart.rename(columns={"Close": "Price"})
    chart.index = normalize_dt_index(chart.index)

    if per_df is not None and not per_df.empty:
        p = per_df.copy()
        p.index = normalize_dt_index(p.index)
        # If PER missing but EPS exists, calculate after joining price.
        if "PER" in p.columns:
            chart = chart.join(p[["PER"]], how="left")
            chart["PER"] = chart["PER"].ffill()
        elif "EPS" in p.columns:
            chart = chart.join(p[["EPS"]], how="left")
            chart["EPS"] = chart["EPS"].ffill()
            chart["PER"] = chart["Price"] / chart["EPS"].replace(0, np.nan)
        else:
            chart["PER"] = np.nan
    else:
        chart["PER"] = np.nan

    if risk_df is not None and not risk_df.empty:
        r = risk_df[["Risk"]].copy()
        r.index = normalize_dt_index(r.index)
        chart = chart.join(r, how="left")
        chart["Risk"] = chart["Risk"].ffill()
    else:
        chart["Risk"] = np.nan

    if signal_df is not None and not signal_df.empty:
        s = signal_df.copy()
        s.index = normalize_dt_index(s.index)
        chart = chart.join(s[["Buy_Display", "Cash_Display"]], how="left")

    return chart


def plot_core_dashboard(df, per_df, risk_df, signal_df, title, risk_label):
    chart = merge_chart_df(df, per_df, risk_df, signal_df)

    fig, (ax1, ax3) = plt.subplots(
        2,
        1,
        figsize=(15, 8.2),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]}
    )
    fig.subplots_adjust(left=0.075, right=0.89, top=0.92, bottom=0.09, hspace=0.08)

    # Upper price axis
    ax1.plot(chart.index, chart["Price"], color="#0057B8", linewidth=2.1, label="Price")
    ax1.plot(chart.index, chart["MA20"], color="#FF8C00", linewidth=1.35, label="MA20")
    ax1.plot(chart.index, chart["MA60"], color="#2CA02C", linewidth=1.35, label="MA60")
    ax1.plot(chart.index, chart["MA200"], color="#7E57C2", linewidth=1.35, label="MA200")
    ax1.set_ylabel("Price", color="#0057B8")
    ax1.tick_params(axis="y", labelcolor="#0057B8")
    ax1.grid(True, linestyle=":", alpha=0.42)

    if "Buy_Display" in chart.columns:
        ax1.scatter(chart.index, chart["Buy_Display"] * 0.97, color="#008000", marker="^", s=88, zorder=5, label="BUY candidate")
    if "Cash_Display" in chart.columns:
        ax1.scatter(chart.index, chart["Cash_Display"] * 1.03, color="#E60000", marker="v", s=88, zorder=5, label="Cash / overheat")

    # Upper PER axis
    ax2 = ax1.twinx()
    if chart["PER"].dropna().empty:
        ax2.text(0.02, 0.05, "P/E series unavailable", transform=ax2.transAxes, color="#D62728", fontsize=10)
    else:
        ax2.plot(chart.index, chart["PER"], color="#D62728", linewidth=1.75, label="P/E")
        per_avg = chart["PER"].dropna().mean()
        ax2.axhline(per_avg, color="#D62728", linestyle="--", linewidth=1.0, alpha=0.45, label="P/E avg")
    ax2.set_ylabel("P/E", color="#D62728")
    ax2.tick_params(axis="y", labelcolor="#D62728")

    ax1.set_title(title, fontsize=13.5, fontweight="bold")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8.2, framealpha=0.88)

    # Lower MDD axis
    dd_pct = chart["Current_Drawdown"] * 100
    ax3.plot(chart.index, dd_pct, color="#8B0000", linewidth=1.75, label="Current DD")
    for level, lab in [(-8, "Watch -8%"), (-12, "Buy zone -12%"), (-15, "Deep -15%"), (-20, "Risk -20%")]:
        ax3.axhline(level, color="#1E90FF", linestyle="--", linewidth=0.95, alpha=0.42, label=lab)
    ax3.set_ylabel("MDD (%)", color="#8B0000")
    ax3.tick_params(axis="y", labelcolor="#8B0000")
    ax3.grid(True, linestyle=":", alpha=0.42)

    ax4 = ax3.twinx()
    if not chart["Risk"].dropna().empty:
        ax4.plot(chart.index, chart["Risk"], color="#00A6A6", linestyle="--", linewidth=1.25, alpha=0.85, label=risk_label)
    ax4.set_ylabel(risk_label, color="#00A6A6")
    ax4.tick_params(axis="y", labelcolor="#00A6A6")

    lines3, labels3 = ax3.get_legend_handles_labels()
    lines4, labels4 = ax4.get_legend_handles_labels()
    ax3.legend(lines3 + lines4, labels3 + labels4, loc="lower left", fontsize=7.3, ncol=2, framealpha=0.88)

    st.pyplot(fig, clear_figure=True, use_container_width=True)


def make_comment(latest, per_df, risk_df, risk_label):
    msgs = []
    close = latest.get("Close", np.nan)
    ma20 = latest.get("MA20", np.nan)
    ma200 = latest.get("MA200", np.nan)
    dd = latest.get("Current_Drawdown", np.nan) * 100
    rsi = latest.get("RSI", np.nan)

    if pd.notna(ma20):
        msgs.append("가격: MA20 위라 단기 추세 유지." if close >= ma20 else "가격: MA20 아래라 반등 확인 전 추격 금지.")
    if pd.notna(ma200):
        msgs.append("장기추세: MA200 위라 장기 추세 훼손은 제한적." if close >= ma200 else "장기추세: MA200 아래라 추세 훼손 우선 확인.")

    if pd.notna(dd):
        if dd <= -15:
            msgs.append(f"MDD: {dd:.1f}%로 깊은 조정 구간.")
        elif dd <= -12:
            msgs.append(f"MDD: {dd:.1f}%로 1차 관심 구간.")
        elif dd <= -8:
            msgs.append(f"MDD: {dd:.1f}%로 관심 초입.")
        else:
            msgs.append(f"MDD: {dd:.1f}%로 낙폭 매력은 제한적.")

    if pd.notna(rsi):
        if rsi <= 35:
            msgs.append(f"RSI: {rsi:.1f}, 과매도권.")
        elif rsi >= 68:
            msgs.append(f"RSI: {rsi:.1f}, 과열 주의.")
        else:
            msgs.append(f"RSI: {rsi:.1f}, 중립권.")

    if per_df is not None and not per_df.empty and "PER" in per_df.columns:
        s = per_df["PER"].dropna()
        if len(s) >= 30:
            recent = s.iloc[-1]
            past = s.iloc[max(0, len(s) - 60)]
            per_chg = (recent / past - 1) * 100 if past != 0 else np.nan
            if pd.notna(per_chg):
                if per_chg < -5:
                    msgs.append(f"PER: 최근 약 60개 관측치 기준 {per_chg:.1f}% 하락, 밸류 부담 완화.")
                elif per_chg > 5:
                    msgs.append(f"PER: 최근 약 60개 관측치 기준 {per_chg:.1f}% 상승, 밸류 부담 확대.")
                else:
                    msgs.append(f"PER: 최근 변화 {per_chg:.1f}%, 큰 변화 없음.")
    else:
        msgs.append("PER: 시계열 없음. 현재 PER 카드만 참고.")

    if risk_df is not None and not risk_df.empty and "Risk" in risk_df.columns and not risk_df["Risk"].dropna().empty:
        rv = risk_df["Risk"].dropna().iloc[-1]
        if risk_label == "VIX":
            if rv >= 25:
                msgs.append(f"시장위험: VIX {rv:.1f}, 공포 구간.")
            elif rv <= 15:
                msgs.append(f"시장위험: VIX {rv:.1f}, 공포 낮음.")
            else:
                msgs.append(f"시장위험: VIX {rv:.1f}, 보통.")
        else:
            msgs.append(f"시장위험: 한국 지수 MDD {rv:.1f}%.")

    if pd.notna(dd) and dd <= -12 and pd.notna(rsi) and rsi <= 42 and pd.notna(ma20) and close >= ma20:
        final = "최종: 1차 소액 후보. DD가 깊고 MA20 회복이 동반됨."
    elif pd.notna(dd) and dd <= -12 and pd.notna(rsi) and rsi <= 42:
        final = "최종: 관심/소액 후보. MA20 회복 전에는 대기 또는 아주 소액만."
    elif pd.notna(dd) and dd > -8 and pd.notna(rsi) and rsi >= 65:
        final = "최종: 추격 금지. 낙폭 얕고 과열 신호."
    else:
        final = "최종: 판단 보류. 가격·PER·MDD 조합이 강한 진입 신호는 아님."

    return final, msgs

# =========================================================
# Inputs
# =========================================================
col_a, col_b, col_c = st.columns(3)
with col_a:
    user_input = st.text_input("종목명 / 종목코드 / 미국 티커", value="삼성전자")
with col_b:
    start_date = st.date_input("기준 시작일", pd.to_datetime("2024-01-01"))
with col_c:
    asset_type = st.selectbox(
        "종목 유형",
        ["일반 주식/ETF", "나스닥형 ETF", "반도체/메모리 ETF", "전력/인프라 ETF", "우주/소형 테마", "코스닥/중소형"],
        index=0,
    )

run = st.button("분석 실행")

# =========================================================
# Main
# =========================================================
if run:
    market, ticker, display_name = find_ticker(user_input)
    if ticker is None:
        st.error("종목을 찾을 수 없습니다. 한국 종목은 종목명 또는 6자리 코드로 입력하세요. 예: 삼성전자 / 005930")
        st.stop()

    df_raw, price_status = load_price_data(market, ticker, start_date)
    if df_raw.empty:
        st.error(f"가격 데이터를 가져오지 못했습니다. 원인: {price_status}")
        st.stop()

    df = calculate_indicators(df_raw)
    latest = df.iloc[-1]
    last_price_date = df.index.max()

    if market == "KR":
        per_df, per_status, per_debug = load_kr_per_series(ticker, start_date, last_price_date)
        naver_val, naver_status = load_kr_naver_current_valuation(ticker)
        if not per_df.empty and "PER" in per_df.columns and not per_df["PER"].dropna().empty:
            current_pe = per_df["PER"].dropna().iloc[-1]
        else:
            current_pe = naver_val.get("trailing_pe")
        valuation = {
            "trailing_pe": current_pe,
            "forward_pe": None,
            "price_to_sales": None,
            "peg_ratio": None,
        }
        valuation_status = f"{per_status} / {naver_status}"
    else:
        valuation, valuation_status = load_us_current_valuation(ticker)
        per_df, per_status = load_us_ttm_pe_series(ticker, df)
        per_debug = per_status

    risk_df, risk_status, risk_label = make_market_risk_series(market, start_date, asset_type)
    signal_df = build_signals(df)

    st.markdown(f"## 분석 대상: {display_name} / {ticker} / {market}")

    c1, c2, c3, c4, c5, c6 = st.columns([1.4, 1, 1, 1, 0.8, 0.9])
    c1.metric("현재가", fmt_num(latest["Close"]))
    c2.metric("Current DD", fmt_pct(latest["Current_Drawdown"] * 100))
    c3.metric("Max DD", fmt_pct(df["Max_Drawdown"].min() * 100))
    c4.metric("RSI", fmt_num(latest["RSI"]))
    ma20_status = "위" if pd.notna(latest["MA20"]) and latest["Close"] >= latest["MA20"] else "아래"
    c5.metric("MA20", ma20_status)
    c6.metric("Vol Ratio", fmt_num(latest["Volume_Ratio"]))

    st.markdown("## 1. 현재 Valuation")
    v1, v2, v3, v4 = st.columns(4)
    v1.metric("TTM / KRX P/E", fmt_num(valuation.get("trailing_pe")))
    v2.metric("Forward P/E", fmt_num(valuation.get("forward_pe")))
    v3.metric("P/S", fmt_num(valuation.get("price_to_sales")))
    v4.metric("PEG", fmt_num(valuation.get("peg_ratio")))

    if per_df.empty:
        st.warning(f"PER 시계열 없음: {compact_status(per_status)}")

    st.markdown("## 2. 핵심 차트")
    st.info("주가, PER, MDD, 시장위험만 봅니다. 주가 상승 중 PER 하락은 이익 개선이 주가 상승을 정당화하는 흐름일 수 있습니다.")

    plot_core_dashboard(
        df,
        per_df,
        risk_df,
        signal_df,
        f"{ticker} Price + P/E + MDD + {risk_label}",
        risk_label,
    )

    final, comments = make_comment(latest, per_df, risk_df, risk_label)
    st.markdown("## 3. 자동 해석")
    if "추격 금지" in final:
        st.error(final)
    elif "소액" in final or "후보" in final:
        st.warning(final)
    else:
        st.info(final)
    for m in comments:
        st.write(f"- {m}")

    with st.expander("PER 원자료 / 상태"):
        st.write(f"PER status: {per_status}")
        st.write(f"Valuation status: {valuation_status}")
        st.write(f"PER debug: {per_debug}")
        st.write(f"Risk status: {risk_status}")
        if market == "KR":
            st.write(f"PYKRX_AVAILABLE: {PYKRX_AVAILABLE}")
            if not PYKRX_AVAILABLE:
                st.code(PYKRX_IMPORT_ERROR)
        if per_df is not None and not per_df.empty:
            st.dataframe(per_df.tail(40), use_container_width=True)
        else:
            st.write("PER DataFrame empty")

    with st.expander("최근 20거래일 데이터"):
        show = df[["Close", "Current_Drawdown", "RSI", "MA20", "MA60", "MA200", "Volume_Ratio"]].tail(20).copy()
        show["Current_Drawdown"] = show["Current_Drawdown"] * 100
        st.dataframe(show, use_container_width=True)
