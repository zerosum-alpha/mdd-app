import warnings
warnings.filterwarnings("ignore")

from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

try:
    import yfinance as yf
    YF_OK = True
    YF_ERR = ""
except Exception as e:
    yf = None
    YF_OK = False
    YF_ERR = repr(e)

try:
    import FinanceDataReader as fdr
    FDR_OK = True
    FDR_ERR = ""
except Exception as e:
    fdr = None
    FDR_OK = False
    FDR_ERR = repr(e)

try:
    from pykrx import stock as pkstock
    PYKRX_OK = True
    PYKRX_ERR = ""
except Exception as e:
    pkstock = None
    PYKRX_OK = False
    PYKRX_ERR = repr(e)

try:
    import requests
    REQ_OK = True
except Exception as e:
    requests = None
    REQ_OK = False

try:
    from bs4 import BeautifulSoup
    BS4_OK = True
except Exception:
    BeautifulSoup = None
    BS4_OK = False

try:
    from auth import require_login, logout_button
except Exception:
    def require_login():
        return None
    def logout_button():
        return None

# =========================================================
# Page
# =========================================================
st.set_page_config(page_title="MDD 핵심 분석기", layout="wide")
require_login()
logout_button()

st.title("📈 MDD 저점매수 분석기 | Core Practical")
st.caption("주가 / PER / MDD / 시장위험 / 이평선만 표시합니다. 미국·한국 PER은 Actual과 Proxy를 구분 표시합니다.")

# =========================================================
# Utilities
# =========================================================
def to_dt_index(idx):
    out = pd.to_datetime(idx, errors="coerce")
    try:
        out = out.tz_localize(None)
    except Exception:
        try:
            out = out.tz_convert(None)
        except Exception:
            pass
    return pd.DatetimeIndex(out).astype("datetime64[ns]")


def ymd(x):
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


def fmt_pct_ratio(x, digits=2):
    v = safe_float(x)
    if v is None:
        return "N/A"
    return f"{v * 100:.{digits}f}%"


def is_korean(text):
    return any("가" <= ch <= "힣" for ch in str(text))


def run_with_timeout(fn, timeout_sec=8):
    """Prevent pykrx/remote calls from freezing Streamlit."""
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(fn)
    try:
        result = future.result(timeout=timeout_sec)
        executor.shutdown(wait=False, cancel_futures=True)
        return result, "OK"
    except TimeoutError:
        executor.shutdown(wait=False, cancel_futures=True)
        return None, f"timeout {timeout_sec}s"
    except Exception as e:
        executor.shutdown(wait=False, cancel_futures=True)
        return None, repr(e)


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
def kr_stock_list():
    if not FDR_OK:
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


def find_ticker(q):
    q = str(q).strip()
    if not q:
        return None, None, None
    if q.isdigit() and len(q) == 6:
        return "KR", q, q
    if q in KR_FALLBACK_MAP:
        return "KR", KR_FALLBACK_MAP[q], q

    sl = kr_stock_list()
    if not sl.empty and {"Name", "Code"}.issubset(sl.columns):
        exact = sl[sl["Name"] == q]
        if not exact.empty:
            return "KR", exact.iloc[0]["Code"], exact.iloc[0]["Name"]
        partial = sl[sl["Name"].str.contains(q, case=False, na=False)]
        if not partial.empty:
            return "KR", partial.iloc[0]["Code"], partial.iloc[0]["Name"]

    if is_korean(q):
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
            if not FDR_OK:
                return pd.DataFrame(), f"FinanceDataReader import 실패: {FDR_ERR}"
            df = fdr.DataReader(str(ticker).zfill(6), start)
        else:
            if not YF_OK:
                return pd.DataFrame(), f"yfinance import 실패: {YF_ERR}"
            df = yf.Ticker(ticker).history(start=start, auto_adjust=True)
        if df is None or df.empty:
            return pd.DataFrame(), "가격 데이터 empty"
        df = df.copy()
        df.index = to_dt_index(df.index)
        if "Volume" not in df.columns:
            df["Volume"] = 0
        return df[[c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]], "OK"
    except Exception as e:
        return pd.DataFrame(), repr(e)


@st.cache_data(ttl=1800)
def load_us_close(ticker, start_date):
    if not YF_OK:
        return pd.DataFrame()
    try:
        df = yf.Ticker(ticker).history(start=pd.to_datetime(start_date).strftime("%Y-%m-%d"), auto_adjust=True)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.copy()
        df.index = to_dt_index(df.index)
        return df[["Close"]].rename(columns={"Close": ticker})
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=1800)
def load_fdr_close(symbol, start_date):
    if not FDR_OK:
        return pd.DataFrame()
    try:
        df = fdr.DataReader(symbol, pd.to_datetime(start_date).strftime("%Y-%m-%d"))
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.copy()
        df.index = to_dt_index(df.index)
        return df[["Close"]].rename(columns={"Close": symbol})
    except Exception:
        return pd.DataFrame()

# =========================================================
# Indicators
# =========================================================
def calc_rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def add_indicators(df):
    out = df.copy()
    out["Peak"] = out["Close"].cummax()
    out["Current_Drawdown"] = out["Close"] / out["Peak"] - 1
    out["Max_Drawdown"] = out["Current_Drawdown"].cummin()
    out["MA20"] = out["Close"].rolling(20).mean()
    out["MA60"] = out["Close"].rolling(60).mean()
    out["MA200"] = out["Close"].rolling(200).mean()
    out["RSI"] = calc_rsi(out["Close"])
    out["Volume_MA20"] = out["Volume"].rolling(20).mean()
    out["Volume_Ratio"] = out["Volume"] / out["Volume_MA20"].replace(0, np.nan)
    return out


def spaced_signal(mask, min_gap=25):
    out = pd.Series(False, index=mask.index)
    last = -999999
    vals = mask.fillna(False).values
    for i, v in enumerate(vals):
        if v and i - last >= min_gap:
            out.iloc[i] = True
            last = i
    return out


def build_signals(df):
    buy = (df["Current_Drawdown"] <= -0.12) & (df["RSI"] <= 42)
    cash = ((df["Current_Drawdown"] >= -0.03) & (df["RSI"] >= 68)) | ((df["Close"] > df["MA20"] * 1.08) & (df["RSI"] >= 65))
    sig = pd.DataFrame(index=df.index)
    sig["Buy"] = df["Close"].where(spaced_signal(buy, 30))
    sig["Cash"] = df["Close"].where(spaced_signal(cash, 45))
    return sig

# =========================================================
# Valuation
# =========================================================
@st.cache_data(ttl=3600)
def us_current_valuation(ticker):
    data = {"ttm_pe": None, "fwd_pe": None, "ps": None, "peg": None}
    if not YF_OK:
        return data, f"yfinance import 실패: {YF_ERR}"
    try:
        info = yf.Ticker(ticker).info
        data["ttm_pe"] = info.get("trailingPE")
        data["fwd_pe"] = info.get("forwardPE")
        data["ps"] = info.get("priceToSalesTrailing12Months")
        data["peg"] = info.get("pegRatio")
        return data, "OK"
    except Exception as e:
        return data, repr(e)


@st.cache_data(ttl=3600)
def naver_current_per(code):
    """Naver current PER/EPS fallback. This is current snapshot only, not historical series."""
    data = {"ttm_pe": None, "fwd_pe": None, "ps": None, "peg": None, "eps": None, "pbr": None}
    if not REQ_OK:
        return data, "requests 없음"
    try:
        code = str(code).zfill(6)
        url = f"https://finance.naver.com/item/main.naver?code={code}"
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=6)
        r.raise_for_status()
        html = r.text

        def parse_num_text(t):
            if t is None:
                return None
            t = str(t).strip().replace(",", "")
            t = re.sub(r"[^0-9.\-]", "", t)
            return safe_float(t)

        def pick_by_id(_id):
            # Regex first
            m = re.search(rf'id=["\']{re.escape(_id)}["\'][^>]*>\s*([^<]+)\s*<', html)
            if m:
                v = parse_num_text(m.group(1))
                if v is not None:
                    return v
            # BS4 fallback
            if BS4_OK:
                soup = BeautifulSoup(html, "html.parser")
                tag = soup.select_one(f"#{_id}")
                if tag:
                    return parse_num_text(tag.get_text(" "))
            return None

        data["ttm_pe"] = pick_by_id("_per")
        data["eps"] = pick_by_id("_eps")
        data["pbr"] = pick_by_id("_pbr")

        # Fallback regex near labels if ids fail
        if data["ttm_pe"] is None:
            m = re.search(r"PER[^0-9\-]*([0-9][0-9,\.\-]*)\s*배", html)
            if m:
                data["ttm_pe"] = parse_num_text(m.group(1))
        if data["eps"] is None:
            m = re.search(r"EPS[^0-9\-]*([0-9][0-9,\.\-]*)\s*원", html)
            if m:
                data["eps"] = parse_num_text(m.group(1))

        status_parts = []
        if data["ttm_pe"] is not None:
            status_parts.append(f"Naver current PER {data['ttm_pe']:.2f}")
        if data["eps"] is not None:
            status_parts.append(f"EPS {data['eps']:.0f}")
        if data["pbr"] is not None:
            status_parts.append(f"PBR {data['pbr']:.2f}")

        if not status_parts:
            return data, "Naver current valuation 없음"
        return data, "OK: " + " / ".join(status_parts)
    except Exception as e:
        return data, f"Naver current valuation error: {repr(e)}"


def canonical_fundamental(raw):
    if raw is None or not isinstance(raw, pd.DataFrame) or raw.empty:
        return pd.DataFrame()
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(map(str, c)).strip() for c in df.columns]
    else:
        df.columns = [str(c).strip().upper() for c in df.columns]
    cols = {}
    for c in df.columns:
        cc = str(c).upper().strip()
        if cc in ["PER", "PBR", "EPS", "BPS", "DIV", "DPS"]:
            cols[c] = cc
    df = df.rename(columns=cols)
    keep = [c for c in ["PER", "PBR", "EPS", "BPS", "DIV", "DPS"] if c in df.columns]
    if not keep:
        return pd.DataFrame()
    out = df[keep].copy()
    for c in keep:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.dropna(how="all")


@st.cache_data(ttl=3600, show_spinner=False)
def kr_per_series_nohang(code, start_date, end_date, timeout_sec=7):
    if not PYKRX_OK:
        return pd.DataFrame(), f"pykrx import 실패: {PYKRX_ERR}"

    code = str(code).zfill(6)
    start = ymd(start_date)
    end = ymd(end_date)

    def fetch_period():
        errors = []
        for label, kwargs in [
            ("daily", {}),
            ("monthly", {"freq": "m"}),
        ]:
            try:
                raw = pkstock.get_market_fundamental(start, end, code, **kwargs)
                out = canonical_fundamental(raw)
                if not out.empty:
                    out.index = to_dt_index(raw.index)
                    if "PER" in out.columns:
                        out = out[(out["PER"] > 0) & (out["PER"] < 500)]
                    if not out.empty:
                        return out.sort_index(), f"OK: pykrx {label}"
                errors.append(f"{label}: empty/no usable cols")
            except Exception as e:
                errors.append(f"{label}: {type(e).__name__} {str(e)[:120]}")
        return pd.DataFrame(), " / ".join(errors)

    result, status = run_with_timeout(fetch_period, timeout_sec=timeout_sec)
    if result is not None:
        return result

    return pd.DataFrame(), f"pykrx 조회 제한: {status}"


@st.cache_data(ttl=3600, show_spinner=False)
def us_ttm_pe_series(ticker, price_df):
    if not YF_OK:
        return pd.DataFrame(), f"yfinance import 실패: {YF_ERR}"
    try:
        tk = yf.Ticker(ticker)
        eps = None
        source = ""

        # 1) Earnings dates: often extends longer than financial statements
        try:
            ed = tk.get_earnings_dates(limit=40)
            if ed is not None and not ed.empty and "Reported EPS" in ed.columns:
                s = pd.to_numeric(ed["Reported EPS"], errors="coerce").dropna()
                s.index = to_dt_index(s.index)
                s = s.sort_index()
                if len(s) >= 4:
                    eps = s
                    source = "Reported EPS"
        except Exception:
            pass

        # 2) Financial statements fallback
        if eps is None or len(eps) < 4:
            for attr in ["quarterly_income_stmt", "quarterly_financials"]:
                stmt = getattr(tk, attr, None)
                if stmt is None or not isinstance(stmt, pd.DataFrame) or stmt.empty:
                    continue
                stmt = stmt.copy()
                stmt.columns = pd.to_datetime(stmt.columns, errors="coerce")
                found = None
                for idx in stmt.index:
                    t = str(idx).lower().replace(" ", "")
                    if ("diluted" in t and "eps" in t) or ("basic" in t and "eps" in t):
                        found = idx
                        break
                if found is not None:
                    s = pd.to_numeric(stmt.loc[found], errors="coerce").dropna().sort_index()
                    if len(s) >= 4:
                        eps = s
                        source = str(found)
                        break

        if eps is None or len(eps) < 4:
            return pd.DataFrame(), "분기 EPS 데이터 부족"

        eps_ttm = eps.rolling(4).sum().dropna()
        if eps_ttm.empty:
            return pd.DataFrame(), "EPS TTM 계산 불가"

        eps_df = pd.DataFrame({
            "Date": to_dt_index(pd.to_datetime(eps_ttm.index) + pd.Timedelta(days=1)),
            "EPS_TTM": eps_ttm.values,
        }).sort_values("Date")

        daily = price_df[["Close"]].reset_index()
        daily.columns = ["Date", "Close"]
        daily["Date"] = to_dt_index(daily["Date"])
        daily = daily.sort_values("Date")
        merged = pd.merge_asof(daily, eps_df, on="Date", direction="backward")
        merged["PER"] = merged["Close"] / merged["EPS_TTM"].replace(0, np.nan)
        merged = merged[(merged["PER"] > 0) & (merged["PER"] < 500)].dropna(subset=["PER"])
        if merged.empty:
            return pd.DataFrame(), "PER 계산 결과 empty"
        out = merged.set_index("Date")[["PER", "EPS_TTM"]]
        return out, f"OK: US Estimated TTM P/E ({source})"
    except Exception as e:
        return pd.DataFrame(), repr(e)



def build_per_proxy_from_current(price_df, current_price, current_pe=None, current_eps=None, label="proxy"):
    """Build full-period P/E proxy using current EPS or EPS implied by current P/E.
    This is not actual historical P/E; used only to keep a full reference line visible.
    """
    if price_df is None or price_df.empty:
        return pd.DataFrame(), "proxy 실패: price empty"
    eps_ref = safe_float(current_eps)
    pe_ref = safe_float(current_pe)
    px = safe_float(current_price)
    if (eps_ref is None or eps_ref <= 0) and pe_ref is not None and pe_ref > 0 and px is not None and px > 0:
        eps_ref = px / pe_ref
    if eps_ref is None or eps_ref <= 0:
        return pd.DataFrame(), "proxy 실패: current EPS/PER 없음"
    out = price_df[["Close"]].copy()
    out["PER_PROXY"] = out["Close"] / eps_ref
    out["EPS_PROXY"] = eps_ref
    out = out[(out["PER_PROXY"] > 0) & (out["PER_PROXY"] < 500)]
    return out[["PER_PROXY", "EPS_PROXY"]], f"OK: {label} P/E proxy using EPS {eps_ref:.4f}"


def normalize_time_index(df):
    """Return DataFrame with a clean datetime64[ns] index.
    Handles both index-based Date and a real 'Date' column.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "Date" in out.columns:
        out["Date"] = to_dt_index(out["Date"])
        out = out.set_index("Date")
    else:
        out.index = to_dt_index(out.index)
    out = out[~out.index.isna()]
    out = out.sort_index()
    return out


def merge_actual_and_proxy_per(actual_df, proxy_df):
    frames = []
    if actual_df is not None and not actual_df.empty:
        a = normalize_time_index(actual_df)
        if not a.empty:
            frames.append(a)
    if proxy_df is not None and not proxy_df.empty:
        p = normalize_time_index(proxy_df)
        if not p.empty:
            frames.append(p)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, axis=1)
    out = out.loc[:, ~out.columns.duplicated()]
    return out.sort_index()

# =========================================================
# Market risk
# =========================================================
def market_risk_series(market, ticker, start_date):
    if market == "US":
        vix = load_us_close("^VIX", start_date)
        if not vix.empty:
            return vix.rename(columns={"^VIX": "Risk"}), "VIX"
        return pd.DataFrame(), "VIX 없음"

    # KR: use KOSPI drawdown as default market risk
    kospi = load_fdr_close("KS11", start_date)
    if kospi.empty:
        return pd.DataFrame(), "KOSPI 위험지표 없음"
    s = kospi.iloc[:, 0]
    dd = s / s.cummax() - 1
    return pd.DataFrame({"Risk": dd}, index=kospi.index), "KOSPI DD"

# =========================================================
# Chart and comment
# =========================================================
def plot_core_chart(df, per_df, risk_df, risk_label, ticker):
    chart = df[["Close", "MA20", "MA60", "MA200", "Current_Drawdown"]].copy()
    chart = chart.rename(columns={"Close": "Price", "Current_Drawdown": "DD"})
    chart.index = to_dt_index(chart.index)

    # P/E input can be actual PER or full-view PER_PROXY.
    if per_df is not None and not per_df.empty:
        p = normalize_time_index(per_df)
        join_cols = [c for c in ["PER", "PER_PROXY", "EPS", "EPS_TTM", "EPS_PROXY"] if c in p.columns]
        if join_cols:
            chart = chart.join(p[join_cols], how="left")

    # Forward-fill sparse or monthly PER values onto daily price dates.
    for col in ["PER", "PER_PROXY", "EPS", "EPS_TTM", "EPS_PROXY"]:
        if col in chart.columns:
            chart[col] = pd.to_numeric(chart[col], errors="coerce").ffill()

    # If only EPS exists, calculate PER from daily price.
    if "PER" not in chart.columns and "EPS" in chart.columns:
        chart["PER"] = chart["Price"] / chart["EPS"].replace(0, np.nan)

    if "PER" not in chart.columns:
        chart["PER"] = np.nan
    if "PER_PROXY" not in chart.columns:
        chart["PER_PROXY"] = np.nan

    # Explicit display line: actual first, proxy second.
    chart["PER_DISPLAY"] = chart["PER"]
    proxy_mask = chart["PER_DISPLAY"].isna() & chart["PER_PROXY"].notna()
    chart.loc[proxy_mask, "PER_DISPLAY"] = chart.loc[proxy_mask, "PER_PROXY"]

    if risk_df is not None and not risk_df.empty:
        r = normalize_time_index(risk_df)
        if "Risk" in r.columns:
            chart = chart.join(r[["Risk"]], how="left")
            chart["Risk"] = pd.to_numeric(chart["Risk"], errors="coerce").ffill()
        else:
            chart["Risk"] = np.nan
    else:
        chart["Risk"] = np.nan

    sig = build_signals(df)
    chart = chart.join(sig, how="left")

    fig, (ax1, ax3) = plt.subplots(
        2, 1, figsize=(18.8, 10.8), sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1]}
    )

    # Upper: Price, MA, PER
    ax1.plot(chart.index, chart["Price"], color="#0047AB", linewidth=2.6, label="Price")
    ax1.plot(chart.index, chart["MA20"], color="#FF8C00", linewidth=1.55, label="MA20")
    ax1.plot(chart.index, chart["MA60"], color="#228B22", linewidth=1.55, label="MA60")
    ax1.plot(chart.index, chart["MA200"], color="#7B2CBF", linewidth=1.55, label="MA200")

    if chart["Buy"].notna().any():
        ax1.scatter(chart.index, chart["Buy"] * 0.975, color="#00A000", marker="^", s=100, label="BUY candidate", zorder=5)
    if chart["Cash"].notna().any():
        ax1.scatter(chart.index, chart["Cash"] * 1.025, color="#E60000", marker="v", s=100, label="Cash / overheat", zorder=5)

    ax1.set_ylabel("Price", color="#0047AB")
    ax1.tick_params(axis="y", labelcolor="#0047AB")
    ax1.grid(True, linestyle=":", alpha=0.32)

    ax2 = ax1.twinx()
    has_actual = chart["PER"].dropna().shape[0] > 0
    has_proxy = chart["PER_PROXY"].dropna().shape[0] > 0

    # Actual PER: solid red. Proxy: dashed red. If only proxy exists, draw it thick enough to be visible.
    if has_actual:
        ax2.plot(chart.index, chart["PER"], color="#D62728", linewidth=2.2, linestyle="-", label="P/E actual")
        per_avg = chart["PER"].dropna().mean()
        ax2.axhline(per_avg, color="#D62728", linewidth=1.1, linestyle="--", alpha=0.42, label="P/E actual avg")
        if has_proxy:
            ax2.plot(chart.index, chart["PER_PROXY"], color="#D62728", linewidth=1.35, linestyle=":", alpha=0.55, label="P/E proxy")
    elif has_proxy:
        ax2.plot(chart.index, chart["PER_PROXY"], color="#D62728", linewidth=2.2, linestyle="--", label="P/E proxy(current EPS)")
        per_avg = chart["PER_PROXY"].dropna().mean()
        ax2.axhline(per_avg, color="#D62728", linewidth=1.1, linestyle=":", alpha=0.45, label="P/E proxy avg")
    else:
        ax2.text(0.99, 0.95, "P/E line: N/A", transform=ax2.transAxes, ha="right", va="top", color="#D62728")

    ax2.set_ylabel("P/E", color="#D62728")
    ax2.tick_params(axis="y", labelcolor="#D62728")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
    ax1.set_title(f"{ticker} Price + P/E + MDD + Market Risk", fontweight="bold")

    # Lower: MDD + Risk
    ax3.plot(chart.index, chart["DD"] * 100, color="#8B0000", linewidth=1.75, label="Current DD")
    for level, label in [(-8, "Watch -8%"), (-12, "Buy zone -12%"), (-15, "Deep -15%"), (-20, "Risk -20%")]:
        ax3.axhline(level, color="#5DADE2", linestyle="--", linewidth=0.95, alpha=0.58, label=label)
    ax3.set_ylabel("MDD (%)", color="#8B0000")
    ax3.tick_params(axis="y", labelcolor="#8B0000")
    ax3.grid(True, linestyle=":", alpha=0.32)

    ax4 = ax3.twinx()
    if chart["Risk"].notna().any():
        if risk_label == "VIX":
            ax4.plot(chart.index, chart["Risk"], color="#00A6A6", linewidth=1.35, linestyle="--", alpha=0.88, label="VIX")
            ax4.set_ylabel("VIX", color="#00A6A6")
        else:
            ax4.plot(chart.index, chart["Risk"] * 100, color="#00A6A6", linewidth=1.35, linestyle="--", alpha=0.88, label=risk_label)
            ax4.set_ylabel(risk_label + " (%)", color="#00A6A6")
        ax4.tick_params(axis="y", labelcolor="#00A6A6")

    lines3, labels3 = ax3.get_legend_handles_labels()
    lines4, labels4 = ax4.get_legend_handles_labels()
    ax3.legend(lines3 + lines4, labels3 + labels4, loc="lower left", fontsize=7)

    fig.subplots_adjust(left=0.06, right=0.94, top=0.93, bottom=0.09, hspace=0.08)
    return fig, chart

def make_comment(df, per_df, risk_label, risk_df):
    latest = df.iloc[-1]
    msg = []
    dd = latest["Current_Drawdown"]
    rsi = latest["RSI"]
    close = latest["Close"]
    ma20 = latest["MA20"]
    ma200 = latest["MA200"]

    if close < ma20:
        msg.append("가격: MA20 아래입니다. 반등 확인 전에는 추격보다 대기/소액이 적합합니다.")
    else:
        msg.append("가격: MA20 위입니다. 단기 추세는 유지 중입니다.")
    if pd.notna(ma200):
        if close >= ma200:
            msg.append("장기추세: MA200 위라 장기 추세 훼손은 제한적입니다.")
        else:
            msg.append("장기추세: MA200 아래라 추세 훼손을 우선 확인해야 합니다.")

    if dd <= -0.15:
        msg.append(f"MDD: {dd*100:.1f}%로 깊은 조정권입니다.")
    elif dd <= -0.12:
        msg.append(f"MDD: {dd*100:.1f}%로 1차 관심 구간입니다.")
    elif dd <= -0.08:
        msg.append(f"MDD: {dd*100:.1f}%로 관찰 구간입니다.")
    else:
        msg.append(f"MDD: {dd*100:.1f}%로 낙폭 매력은 크지 않습니다.")

    if pd.notna(rsi):
        if rsi <= 35:
            msg.append(f"RSI: {rsi:.1f}. 과매도권입니다.")
        elif rsi >= 68:
            msg.append(f"RSI: {rsi:.1f}. 단기 과열권입니다.")
        else:
            msg.append(f"RSI: {rsi:.1f}. 중립권입니다.")

    if per_df is not None and not per_df.empty and "PER" in per_df.columns and per_df["PER"].dropna().shape[0] >= 20:
        p = per_df["PER"].dropna()
        n = min(60, len(p)-1)
        if n > 0:
            chg = p.iloc[-1] / p.iloc[-n] - 1
            if chg < -0.10:
                msg.append(f"PER: 최근 기준 {chg*100:.1f}% 하락. 밸류 부담이 낮아진 흐름입니다.")
            elif chg > 0.10:
                msg.append(f"PER: 최근 기준 {chg*100:.1f}% 상승. 밸류 부담 확대 구간입니다.")
            else:
                msg.append(f"PER: 최근 기준 {chg*100:.1f}% 변화. 큰 방향성은 약합니다.")
    else:
        msg.append("PER: 시계열은 제한적입니다. 현재 PER 카드와 가격/MDD를 우선 보세요.")

    if risk_df is not None and not risk_df.empty:
        rv = risk_df["Risk"].dropna().iloc[-1]
        if risk_label == "VIX":
            if rv >= 25:
                msg.append(f"시장위험: VIX {rv:.1f}. 공포 구간입니다.")
            elif rv <= 15:
                msg.append(f"시장위험: VIX {rv:.1f}. 공포는 낮습니다.")
            else:
                msg.append(f"시장위험: VIX {rv:.1f}. 보통 수준입니다.")
        else:
            msg.append(f"시장위험: {risk_label} {rv*100:.1f}%. 한국 지수 낙폭을 참고하세요.")

    if dd <= -0.12 and pd.notna(rsi) and rsi <= 42 and close >= ma200:
        final = "최종: 1차 눌림 후보입니다. 단, MA20 회복 전에는 소액/분할 기준입니다."
    elif dd > -0.08 and pd.notna(rsi) and rsi >= 65:
        final = "최종: 추격 금지 구간입니다. 현금확보 또는 대기 우선입니다."
    elif close < ma20 and dd <= -0.12:
        final = "최종: 낙폭은 있지만 반등 확인이 부족합니다. 대기 또는 소액만 적합합니다."
    else:
        final = "최종: 강한 진입 신호는 아닙니다. 가격·PER·MDD 조합을 더 확인하세요."

    return final, msg

# =========================================================
# Inputs
# =========================================================
c1, c2, c3 = st.columns(3)
with c1:
    user_input = st.text_input("종목명 / 종목코드 / 미국 티커", value="삼성전자")
with c2:
    start_date = st.date_input("기준 시작일", pd.to_datetime("2025-01-01"))
with c3:
    asset_type = st.selectbox("종목 유형", ["일반 주식/ETF", "나스닥형 ETF", "반도체/메모리 ETF", "전력/인프라 ETF", "우주/소형 테마"])

run = st.button("분석 실행")

if run:
    market, ticker, display_name = find_ticker(user_input)
    if ticker is None:
        st.error("종목을 찾지 못했습니다. 예: 삼성전자, 005930, NVDA")
        st.stop()

    price_df, price_status = load_price_data(market, ticker, start_date)
    if price_df.empty:
        st.error(f"가격 데이터를 가져오지 못했습니다: {price_status}")
        st.stop()

    df = add_indicators(price_df)
    latest = df.iloc[-1]
    last_price_date = df.index.max()

    # Valuation and P/E series
    if market == "US":
        val, val_status = us_current_valuation(ticker)
        actual_per_df, actual_per_status = us_ttm_pe_series(ticker, df)
        proxy_pe = val.get("fwd_pe") or val.get("ttm_pe")
        proxy_df, proxy_status = build_per_proxy_from_current(
            df, latest["Close"], current_pe=proxy_pe, current_eps=None, label="US current EPS-implied"
        )
        per_df = merge_actual_and_proxy_per(actual_per_df, proxy_df)
        per_status = f"Actual: {actual_per_status} / Full-view proxy: {proxy_status}"
    else:
        val, val_status = naver_current_per(ticker)
        actual_per_df, actual_per_status = kr_per_series_nohang(ticker, start_date, last_price_date, timeout_sec=7)
        # If KRX PER series returns EPS only, compute actual-style PER from that EPS.
        if not actual_per_df.empty and "PER" not in actual_per_df.columns and "EPS" in actual_per_df.columns:
            tmp = df[["Close"]].join(actual_per_df[["EPS"]], how="left")
            tmp["EPS"] = tmp["EPS"].ffill()
            tmp["PER"] = tmp["Close"] / tmp["EPS"].replace(0, np.nan)
            actual_per_df = tmp[["PER", "EPS"]].dropna()
        proxy_df, proxy_status = build_per_proxy_from_current(
            df, latest["Close"], current_pe=val.get("ttm_pe"), current_eps=val.get("eps"), label="KR current EPS-implied"
        )
        per_df = merge_actual_and_proxy_per(actual_per_df, proxy_df)
        per_status = f"Actual: {actual_per_status} / Full-view proxy: {proxy_status}"

    risk_df, risk_label = market_risk_series(market, ticker, start_date)

    st.subheader(f"분석 대상: {display_name} / {ticker} / {market}")

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("현재가", fmt_num(latest["Close"]))
    m2.metric("Current DD", fmt_pct_ratio(latest["Current_Drawdown"]))
    m3.metric("Max DD", fmt_pct_ratio(df["Max_Drawdown"].min()))
    m4.metric("RSI", fmt_num(latest["RSI"]))
    m5.metric("MA20", "위" if latest["Close"] >= latest["MA20"] else "아래")
    m6.metric("Vol Ratio", fmt_num(latest["Volume_Ratio"]))

    st.markdown("## 1. 현재 Valuation")
    v1, v2, v3, v4 = st.columns(4)
    v1.metric("TTM / KRX P/E", fmt_num(val.get("ttm_pe")))
    v2.metric("Forward P/E", fmt_num(val.get("fwd_pe")))
    v3.metric("P/S", fmt_num(val.get("ps")))
    v4.metric("PEG", fmt_num(val.get("peg")))

    has_actual_per = per_df is not None and not per_df.empty and "PER" in per_df.columns and not per_df["PER"].dropna().empty
    has_proxy_per = per_df is not None and not per_df.empty and "PER_PROXY" in per_df.columns and not per_df["PER_PROXY"].dropna().empty
    if has_actual_per:
        st.success(f"PER 시계열: {per_status}")
    elif has_proxy_per:
        st.warning(f"실제 PER 시계열은 제한적입니다. 전체 기간은 P/E proxy로 표시합니다. {per_status}")
    else:
        st.warning(f"PER 시계열 없음: {per_status}")

    st.markdown("## 2. 핵심 차트")
    st.info("차트는 주가·PER·MDD·시장위험만 봅니다. 실선 P/E는 실제 계산 가능한 구간, 점선 P/E proxy는 현재 EPS/PER로 역산한 전체기간 참고선입니다.")
    fig, chart_df = plot_core_chart(df, per_df, risk_df, risk_label, ticker)
    st.pyplot(fig)

    st.markdown("## 3. 자동 해석")
    final, comments = make_comment(df, per_df, risk_label, risk_df)
    if "추격 금지" in final or "대기" in final:
        st.warning(final)
    elif "후보" in final:
        st.success(final)
    else:
        st.info(final)
    for msg in comments:
        st.write(f"- {msg}")

    with st.expander("PER 원자료 / 상태"):
        st.write(f"Current valuation status: {val_status}")
        st.write(f"PER status: {per_status}")
        if per_df is not None and not per_df.empty:
            st.dataframe(per_df.tail(30), use_container_width=True)
        else:
            st.write("PER DataFrame empty")

    with st.expander("최근 20거래일"):
        show = df[["Close", "Current_Drawdown", "RSI", "MA20", "MA60", "MA200", "Volume_Ratio"]].tail(20).copy()
        show["Current_Drawdown"] = show["Current_Drawdown"] * 100
        st.dataframe(show, use_container_width=True)
