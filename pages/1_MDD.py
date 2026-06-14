import re
import math
import concurrent.futures
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Optional login helpers
try:
    from auth import require_login, logout_button
except Exception:
    def require_login():
        return None
    def logout_button():
        return None

# Optional data libraries
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
    from bs4 import BeautifulSoup
    REQ_OK = True
except Exception:
    requests = None
    BeautifulSoup = None
    REQ_OK = False


# =========================================================
# Page setup
# =========================================================
st.set_page_config(page_title="MDD 저점매수 분석기", layout="wide")
require_login()
logout_button()

st.title("📈 MDD 저점매수 분석기 | Clean Final")
st.caption("주가 / PER / MDD / 시장위험 / 이평선만 봅니다. 가짜 PER·바차트·예측선 제거.")


# =========================================================
# Utility
# =========================================================
def safe_float(x):
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
        v = float(x)
        if math.isfinite(v):
            return v
        return None
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


def ymd(x):
    return pd.to_datetime(x).strftime("%Y%m%d")


def to_dt_index(x):
    idx = pd.to_datetime(x, errors="coerce")
    try:
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)
    except Exception:
        pass
    return pd.DatetimeIndex(idx).tz_localize(None) if getattr(idx, "tz", None) is not None else pd.DatetimeIndex(idx).astype("datetime64[ns]")


def normalize_index(df):
    df = df.copy()
    df.index = to_dt_index(df.index)
    df = df[~df.index.isna()]
    df = df.sort_index()
    return df


def run_with_timeout(func, timeout_sec=8):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(func)
        return fut.result(timeout=timeout_sec)


# =========================================================
# Stock search
# =========================================================
@st.cache_data(ttl=86400, show_spinner=False)
def get_krx_stock_list():
    if not FDR_OK:
        return pd.DataFrame()
    try:
        df = fdr.StockListing("KRX")
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.copy()
        df["Code"] = df["Code"].astype(str).str.zfill(6)
        df["Name"] = df["Name"].astype(str).str.strip()
        return df
    except Exception:
        return pd.DataFrame()


KR_FALLBACK = {
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


def find_ticker(query):
    q = str(query).strip()
    if not q:
        return None, None, None, None
    if q.isdigit() and len(q) == 6:
        return "KR", q, q, None
    if q in KR_FALLBACK:
        return "KR", KR_FALLBACK[q], q, None

    stock_list = get_krx_stock_list()
    if not stock_list.empty:
        exact = stock_list[stock_list["Name"] == q]
        if not exact.empty:
            row = exact.iloc[0]
            return "KR", row["Code"], row["Name"], row.get("Market", None)
        partial = stock_list[stock_list["Name"].str.contains(q, case=False, na=False)]
        if not partial.empty:
            row = partial.iloc[0]
            return "KR", row["Code"], row["Name"], row.get("Market", None)

    if any("가" <= ch <= "힣" for ch in q):
        return None, None, None, None
    return "US", q.upper(), q.upper(), None


# =========================================================
# Price data
# =========================================================
@st.cache_data(ttl=1800, show_spinner=False)
def load_price_data(market, ticker, start_date):
    start = pd.to_datetime(start_date).strftime("%Y-%m-%d")
    try:
        if market == "KR":
            if not FDR_OK:
                return pd.DataFrame(), f"FinanceDataReader import 실패: {FDR_ERR}"
            df = fdr.DataReader(str(ticker).zfill(6), start)
            if df is None or df.empty:
                return pd.DataFrame(), "한국 가격 데이터 empty"
            df = normalize_index(df)
            return df, "OK"

        if market == "US":
            if not YF_OK:
                return pd.DataFrame(), f"yfinance import 실패: {YF_ERR}"
            df = yf.Ticker(ticker).history(start=start, auto_adjust=True)
            if df is None or df.empty:
                return pd.DataFrame(), "미국 가격 데이터 empty"
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df = normalize_index(df)
            return df, "OK"
    except Exception as e:
        return pd.DataFrame(), repr(e)
    return pd.DataFrame(), "unknown"


@st.cache_data(ttl=1800, show_spinner=False)
def load_us_close(ticker, start_date):
    if not YF_OK:
        return pd.DataFrame()
    try:
        df = yf.Ticker(ticker).history(start=pd.to_datetime(start_date).strftime("%Y-%m-%d"), auto_adjust=True)
        if df is None or df.empty:
            return pd.DataFrame()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        out = df[["Close"]].rename(columns={"Close": ticker})
        return normalize_index(out)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=1800, show_spinner=False)
def load_kr_index_close(index_name, start_date):
    if not FDR_OK:
        return pd.DataFrame()
    try:
        symbol = "KS11" if index_name == "KOSPI" else "KQ11"
        df = fdr.DataReader(symbol, pd.to_datetime(start_date).strftime("%Y-%m-%d"))
        if df is None or df.empty:
            return pd.DataFrame()
        df = normalize_index(df)
        return df[["Close"]].rename(columns={"Close": index_name})
    except Exception:
        return pd.DataFrame()


# =========================================================
# Indicators / judgement
# =========================================================
def calc_rsi(close, period=14):
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
        df["Volume"] = np.nan
    df["Peak"] = df["Close"].cummax()
    df["Current_Drawdown"] = df["Close"] / df["Peak"] - 1
    df["Max_Drawdown"] = df["Current_Drawdown"].cummin()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA60"] = df["Close"].rolling(60).mean()
    df["MA200"] = df["Close"].rolling(200).mean()
    df["RSI"] = calc_rsi(df["Close"])
    df["Volume_MA20"] = df["Volume"].rolling(20).mean()
    df["Volume_Ratio"] = df["Volume"] / df["Volume_MA20"].replace(0, np.nan)
    return df


def get_type_profile(asset_type):
    profiles = {
        "일반 주식/ETF": {"watch": -0.08, "buy1": -0.12, "buy2": -0.15, "risk": -0.20},
        "나스닥형 ETF": {"watch": -0.06, "buy1": -0.08, "buy2": -0.12, "risk": -0.15},
        "반도체/메모리 ETF": {"watch": -0.10, "buy1": -0.12, "buy2": -0.15, "risk": -0.20},
        "전력/인프라 ETF": {"watch": -0.08, "buy1": -0.10, "buy2": -0.15, "risk": -0.18},
        "우주/소형 테마": {"watch": -0.15, "buy1": -0.20, "buy2": -0.25, "risk": -0.30},
    }
    return profiles.get(asset_type, profiles["일반 주식/ETF"])


def make_signals(df, profile, min_gap=18):
    sig = df.copy()
    sig["Buy_raw"] = (
        (sig["Current_Drawdown"] <= profile["buy1"]) &
        (sig["RSI"] <= 42) &
        ((sig["Close"] >= sig["MA200"]) | sig["MA200"].isna())
    )
    sig["Cash_raw"] = (
        ((sig["Current_Drawdown"] >= -0.025) & (sig["RSI"] >= 67)) |
        ((sig["Close"] < sig["MA20"]) & (sig["RSI"] >= 65) & (sig["Current_Drawdown"] > -0.08))
    )
    buy = []
    cash = []
    last_b = -9999
    last_c = -9999
    for i, (_, row) in enumerate(sig.iterrows()):
        if bool(row["Buy_raw"]) and i - last_b >= min_gap:
            buy.append(row["Close"])
            last_b = i
        else:
            buy.append(np.nan)
        if bool(row["Cash_raw"]) and i - last_c >= min_gap:
            cash.append(row["Close"])
            last_c = i
        else:
            cash.append(np.nan)
    out = pd.DataFrame(index=df.index)
    out["Buy_Display"] = buy
    out["Cash_Display"] = cash
    return out


def final_action(latest, profile):
    dd = latest["Current_Drawdown"]
    rsi = latest["RSI"]
    close = latest["Close"]
    ma20 = latest["MA20"]
    if dd <= profile["risk"] and close < ma20:
        return "매수 금지 / 추세 확인"
    if dd <= profile["buy1"] and rsi <= 42:
        if close >= ma20:
            return "1차 소액 가능"
        return "1차 후보지만 MA20 확인 필요"
    if dd > -0.05 and rsi >= 65:
        return "추격 금지 / 현금확보 후보"
    return "대기"


# =========================================================
# Current valuation
# =========================================================
@st.cache_data(ttl=3600, show_spinner=False)
def us_current_valuation(ticker):
    data = {"ttm_pe": None, "fwd_pe": None, "ps": None, "peg": None, "eps": None, "pbr": None}
    if not YF_OK:
        return data, f"yfinance import 실패: {YF_ERR}"
    try:
        info = yf.Ticker(ticker).info
        data["ttm_pe"] = safe_float(info.get("trailingPE"))
        data["fwd_pe"] = safe_float(info.get("forwardPE"))
        data["ps"] = safe_float(info.get("priceToSalesTrailing12Months"))
        data["peg"] = safe_float(info.get("pegRatio"))
        price = safe_float(info.get("currentPrice") or info.get("regularMarketPrice"))
        if price and data["ttm_pe"]:
            data["eps"] = price / data["ttm_pe"]
        return data, "OK: yfinance current valuation"
    except Exception as e:
        return data, repr(e)


@st.cache_data(ttl=3600, show_spinner=False)
def naver_current_valuation(code):
    data = {"ttm_pe": None, "fwd_pe": None, "ps": None, "peg": None, "eps": None, "pbr": None}
    if not REQ_OK:
        return data, "requests/bs4 import 실패"
    try:
        code = str(code).zfill(6)
        url = f"https://finance.naver.com/item/main.naver?code={code}"
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
        r.raise_for_status()
        html = r.text
        soup = BeautifulSoup(html, "html.parser") if BeautifulSoup else None

        def parse_num(txt):
            if txt is None:
                return None
            txt = str(txt).replace(",", "").strip()
            txt = re.sub(r"[^0-9.\-]", "", txt)
            return safe_float(txt)

        def by_id(_id):
            if soup:
                tag = soup.select_one(f"#{_id}")
                if tag:
                    return parse_num(tag.get_text(" "))
            m = re.search(rf'id=["\']{re.escape(_id)}["\'][^>]*>\s*([^<]+)\s*<', html)
            return parse_num(m.group(1)) if m else None

        data["ttm_pe"] = by_id("_per")
        data["eps"] = by_id("_eps")
        data["pbr"] = by_id("_pbr")

        text = soup.get_text(" ", strip=True) if soup else html
        if data["ttm_pe"] is None:
            m = re.search(r"PER\s*([0-9,\.\-]+)\s*배", text)
            if m:
                data["ttm_pe"] = parse_num(m.group(1))
        if data["eps"] is None:
            m = re.search(r"EPS\s*([0-9,\.\-]+)\s*원", text)
            if m:
                data["eps"] = parse_num(m.group(1))
        if data["pbr"] is None:
            m = re.search(r"PBR\s*([0-9,\.\-]+)\s*배", text)
            if m:
                data["pbr"] = parse_num(m.group(1))

        parts = []
        if data["ttm_pe"] is not None:
            parts.append(f"Naver PER {data['ttm_pe']:.2f}")
        if data["eps"] is not None:
            parts.append(f"EPS {data['eps']:.0f}")
        if data["pbr"] is not None:
            parts.append(f"PBR {data['pbr']:.2f}")
        return data, "OK: " + " / ".join(parts) if parts else "Naver valuation 없음"
    except Exception as e:
        return data, repr(e)


# =========================================================
# PER series
# =========================================================
def clean_fundamental_df(raw):
    if raw is None or not isinstance(raw, pd.DataFrame) or raw.empty:
        return pd.DataFrame()
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([str(x) for x in c]).upper().strip() for c in df.columns]
    else:
        df.columns = [str(c).upper().strip() for c in df.columns]
    out_cols = {}
    for c in df.columns:
        key = c.replace(" ", "").upper()
        for t in ["PER", "PBR", "EPS", "BPS", "DIV", "DPS"]:
            if key == t or key.endswith("_" + t):
                out_cols[c] = t
    if not out_cols:
        return pd.DataFrame()
    out = df.rename(columns=out_cols)[list(out_cols.values())].copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.dropna(how="all")


@st.cache_data(ttl=3600, show_spinner=False)
def kr_per_series(code, start_date, end_date, current_val=None):
    """
    Returns actual KRX PER time-series if available.
    If only current PER exists from Naver, do not create fake time-series; return empty series + current value in current_val.
    """
    code = str(code).zfill(6)
    errors = []
    if PYKRX_OK:
        start = ymd(start_date)
        end = ymd(end_date)

        # Period query: official call. Timeout prevents Streamlit hang.
        for label, kwargs in [("daily", {}), ("monthly", {"freq": "m"})]:
            try:
                raw = run_with_timeout(lambda: pkstock.get_market_fundamental(start, end, code, **kwargs), timeout_sec=6)
                out = clean_fundamental_df(raw)
                if not out.empty:
                    out.index = to_dt_index(raw.index)
                    if "PER" in out.columns:
                        out = out[(out["PER"] > 0) & (out["PER"] < 500)].dropna(subset=["PER"])
                    if not out.empty and "PER" in out.columns:
                        return out.sort_index(), f"OK: pykrx {label} PER series"
                errors.append(f"{label}: empty/no PER")
            except concurrent.futures.TimeoutError:
                errors.append(f"{label}: timeout")
            except Exception as e:
                errors.append(f"{label}: {type(e).__name__} {str(e)[:80]}")

        # By ticker fallback: only latest row, not a time-series.
        for back in range(1, 8):
            d = pd.to_datetime(end_date) - pd.Timedelta(days=back)
            ds = ymd(d)
            for market in ["KOSPI", "KOSDAQ", "KONEX"]:
                try:
                    raw = run_with_timeout(lambda: pkstock.get_market_fundamental_by_ticker(ds, market=market), timeout_sec=4)
                    if raw is None or raw.empty:
                        continue
                    tmp = raw.copy()
                    tmp.index = tmp.index.astype(str).str.zfill(6)
                    if code in tmp.index:
                        row = clean_fundamental_df(tmp.loc[[code]])
                        if not row.empty and "PER" in row.columns:
                            row.index = pd.DatetimeIndex([pd.to_datetime(ds)]).astype("datetime64[ns]")
                            return row, f"OK: pykrx latest only {market} {ds}"
                except concurrent.futures.TimeoutError:
                    errors.append(f"by_ticker {ds}/{market}: timeout")
                except Exception as e:
                    errors.append(f"by_ticker {ds}/{market}: {type(e).__name__}")
    else:
        errors.append(f"pykrx import 실패: {PYKRX_ERR}")

    # No actual series. Keep empty; current PER card/reference will still display from Naver.
    return pd.DataFrame(), "PER 시계열 없음: " + " / ".join(errors[:5])


@st.cache_data(ttl=3600, show_spinner=False)
def us_estimated_ttm_pe_series(ticker, price_df):
    """Existing sane method: actual estimated TTM P/E only. No proxy line."""
    if not YF_OK:
        return pd.DataFrame(), f"yfinance import 실패: {YF_ERR}"
    try:
        tk = yf.Ticker(ticker)
        eps_q = None
        source = ""

        # 1) Use earnings dates first. This usually gives the longer/cleaner series that was closer to previous output.
        try:
            ed = tk.get_earnings_dates(limit=100)
            if ed is not None and not ed.empty and "Reported EPS" in ed.columns:
                s = pd.to_numeric(ed["Reported EPS"], errors="coerce").dropna()
                s.index = to_dt_index(s.index)
                s = s.sort_index()
                if len(s) >= 4:
                    eps_q = s
                    source = "Reported EPS"
        except Exception:
            pass

        # 2) Financial statement fallback
        if eps_q is None or len(eps_q) < 4:
            for attr in ["quarterly_income_stmt", "quarterly_financials"]:
                stmt = getattr(tk, attr, None)
                if stmt is None or not isinstance(stmt, pd.DataFrame) or stmt.empty:
                    continue
                stmt = stmt.copy()
                stmt.columns = pd.to_datetime(stmt.columns, errors="coerce")
                found = None
                for idx in stmt.index:
                    name = str(idx).lower().replace(" ", "")
                    if ("diluted" in name and "eps" in name) or ("basiceps" in name) or ("basic" in name and "eps" in name):
                        found = idx
                        break
                if found is not None:
                    s = pd.to_numeric(stmt.loc[found], errors="coerce").dropna().sort_index()
                    if len(s) >= 4:
                        eps_q = s
                        source = str(found)
                        break

        if eps_q is None or len(eps_q) < 4:
            return pd.DataFrame(), "US EPS 데이터 부족"

        eps_q = eps_q.sort_index()
        eps_ttm = eps_q.rolling(4).sum().dropna()
        if eps_ttm.empty:
            return pd.DataFrame(), "EPS TTM 계산 불가"

        # Use earnings date + 1 day, not quarter-end +45d, to preserve longer visible history.
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


# =========================================================
# Market risk
# =========================================================
def market_risk_series(market, ticker, start_date, kr_market_hint=None):
    if market == "US":
        vix = load_us_close("^VIX", start_date)
        if not vix.empty:
            return vix.rename(columns={"^VIX": "Risk"}), "VIX"
        return pd.DataFrame(), "VIX"

    idx_name = "KOSDAQ" if str(kr_market_hint).upper() == "KOSDAQ" else "KOSPI"
    idx = load_kr_index_close(idx_name, start_date)
    if not idx.empty:
        risk = idx.copy()
        risk["Risk"] = risk[idx_name] / risk[idx_name].cummax() - 1
        return risk[["Risk"]], f"{idx_name} DD"
    return pd.DataFrame(), f"{idx_name} DD"


# =========================================================
# Chart and comment
# =========================================================
def make_chart_df(df, per_df, risk_df):
    chart = df[["Close", "MA20", "MA60", "MA200", "Current_Drawdown"]].copy()
    chart = chart.rename(columns={"Close": "Price", "Current_Drawdown": "DD"})
    chart.index = to_dt_index(chart.index)
    sig = make_signals(df, get_type_profile("일반 주식/ETF"))
    sig.index = to_dt_index(sig.index)
    chart = chart.join(sig, how="left")

    chart["PER"] = np.nan
    if per_df is not None and not per_df.empty and "PER" in per_df.columns:
        p = per_df.copy()
        p.index = to_dt_index(p.index)
        if p["PER"].dropna().shape[0] >= 2:
            chart = chart.drop(columns=["PER"]).join(p[["PER"]], how="left")
            chart["PER"] = chart["PER"].ffill()
        # if one point only, don't fake a series. handled as reference line from current_val.

    if risk_df is not None and not risk_df.empty:
        r = risk_df.copy()
        r.index = to_dt_index(r.index)
        chart = chart.join(r[["Risk"]], how="left")
        chart["Risk"] = chart["Risk"].ffill()
    else:
        chart["Risk"] = np.nan
    return chart


def plot_core_chart(df, per_df, risk_df, risk_label, ticker, current_val, profile):
    chart = make_chart_df(df, per_df, risk_df)
    sig = make_signals(df, profile)
    chart["Buy_Display"] = sig["Buy_Display"]
    chart["Cash_Display"] = sig["Cash_Display"]

    plt.rcParams.update({
        "axes.titlesize": 14,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    })

    fig = plt.figure(figsize=(18.2, 9.4), dpi=120)
    gs = fig.add_gridspec(5, 1, height_ratios=[3.5, 0.08, 1.45, 0.02, 0.01])
    ax1 = fig.add_subplot(gs[0])
    ax3 = fig.add_subplot(gs[2], sharex=ax1)

    # Price axis
    ax1.plot(chart.index, chart["Price"], color="#0057B8", linewidth=2.2, label="Price")
    ax1.plot(chart.index, chart["MA20"], color="#FF8C00", linewidth=1.55, label="MA20")
    ax1.plot(chart.index, chart["MA60"], color="#2E8B57", linewidth=1.6, label="MA60")
    ax1.plot(chart.index, chart["MA200"], color="#7B61FF", linewidth=1.75, label="MA200")

    if chart["Buy_Display"].notna().any():
        ax1.scatter(chart.index, chart["Buy_Display"] * 0.97, color="#008000", marker="^", s=95, zorder=6, label="BUY candidate")
    if chart["Cash_Display"].notna().any():
        ax1.scatter(chart.index, chart["Cash_Display"] * 1.03, color="#FF0000", marker="v", s=95, zorder=6, label="Cash / overheat")

    ax1.set_ylabel("Price", color="#0057B8")
    ax1.tick_params(axis="y", labelcolor="#0057B8")
    ax1.grid(True, linestyle=":", alpha=0.35)

    # PER axis
    ax2 = ax1.twinx()
    if chart["PER"].dropna().shape[0] >= 2:
        ax2.plot(chart.index, chart["PER"], color="#D62728", linewidth=2.05, label="P/E")
        per_avg = chart["PER"].dropna().mean()
        ax2.axhline(per_avg, color="#D62728", linewidth=1.0, linestyle="--", alpha=0.35, label="P/E avg")

    cur_ttm = safe_float(current_val.get("ttm_pe"))
    cur_fwd = safe_float(current_val.get("fwd_pe"))
    if cur_ttm is not None and cur_ttm > 0:
        ax2.axhline(cur_ttm, color="#D62728", linewidth=1.2, linestyle="-.", alpha=0.85, label="Current TTM/KRX P/E")
    if cur_fwd is not None and cur_fwd > 0:
        ax2.axhline(cur_fwd, color="#111111", linewidth=1.2, linestyle=":", alpha=0.85, label="Current forward P/E")

    if chart["PER"].dropna().shape[0] < 2 and cur_ttm is None and cur_fwd is None:
        ax2.text(0.99, 0.94, "P/E: N/A", transform=ax2.transAxes, ha="right", va="top", color="#D62728")

    ax2.set_ylabel("P/E", color="#D62728")
    ax2.tick_params(axis="y", labelcolor="#D62728")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", ncol=2, framealpha=0.88)
    ax1.set_title(f"{ticker} Price + P/E + MDD + Market Risk", fontweight="bold")

    # MDD / Risk
    ax3.plot(chart.index, chart["DD"] * 100, color="#8B0000", linewidth=1.8, label="Current DD")
    for level, label in [(-8, "Watch -8%"), (-12, "Buy zone -12%"), (-15, "Deep -15%"), (-20, "Risk -20%")]:
        ax3.axhline(level, color="#4682B4", linestyle="--", linewidth=0.95, alpha=0.58, label=label)
    ax3.set_ylabel("MDD (%)", color="#8B0000")
    ax3.tick_params(axis="y", labelcolor="#8B0000")
    ax3.grid(True, linestyle=":", alpha=0.35)

    ax4 = ax3.twinx()
    if chart["Risk"].notna().any():
        if risk_label == "VIX":
            ax4.plot(chart.index, chart["Risk"], color="#00A6A6", linewidth=1.25, linestyle="--", alpha=0.9, label="VIX")
            ax4.set_ylabel("VIX", color="#00A6A6")
        else:
            ax4.plot(chart.index, chart["Risk"] * 100, color="#00A6A6", linewidth=1.25, linestyle="--", alpha=0.9, label=risk_label)
            ax4.set_ylabel(risk_label + " (%)", color="#00A6A6")
        ax4.tick_params(axis="y", labelcolor="#00A6A6")

    lines3, labels3 = ax3.get_legend_handles_labels()
    lines4, labels4 = ax4.get_legend_handles_labels()
    ax3.legend(lines3 + lines4, labels3 + labels4, loc="lower left", ncol=3, framealpha=0.88)

    fig.subplots_adjust(left=0.055, right=0.945, top=0.925, bottom=0.085, hspace=0.11)
    return fig, chart


def make_comment(df, per_df, risk_label, risk_df, current_val):
    latest = df.iloc[-1]
    dd = latest["Current_Drawdown"]
    rsi = latest["RSI"]
    close = latest["Close"]
    ma20 = latest["MA20"]
    ma200 = latest["MA200"]
    msg = []

    msg.append("가격: MA20 위라 단기 추세는 유지 중입니다." if close >= ma20 else "가격: MA20 아래입니다. 반등 확인 전에는 추격보다 대기/소액 기준입니다.")
    if pd.notna(ma200):
        msg.append("장기추세: MA200 위라 장기 추세 훼손은 제한적입니다." if close >= ma200 else "장기추세: MA200 아래입니다. 추세 훼손 가능성을 먼저 봐야 합니다.")
    if dd <= -0.15:
        msg.append(f"MDD: {dd*100:.1f}%로 깊은 조정 구간입니다.")
    elif dd <= -0.12:
        msg.append(f"MDD: {dd*100:.1f}%로 1차 관심 구간입니다.")
    elif dd <= -0.08:
        msg.append(f"MDD: {dd*100:.1f}%로 관찰 구간입니다.")
    else:
        msg.append(f"MDD: {dd*100:.1f}%로 낙폭은 아직 깊지 않습니다.")

    if per_df is not None and not per_df.empty and "PER" in per_df.columns and per_df["PER"].dropna().shape[0] >= 20:
        joined = df[["Close"]].join(per_df[["PER"]], how="left")
        joined["PER"] = joined["PER"].ffill()
        joined = joined.dropna()
        if len(joined) >= 40:
            recent = joined.tail(min(60, len(joined)))
            price_chg = recent["Close"].iloc[-1] / recent["Close"].iloc[0] - 1
            per_chg = recent["PER"].iloc[-1] / recent["PER"].iloc[0] - 1
            msg.append(f"PER: 최근 구간 주가 {price_chg*100:.1f}%, PER {per_chg*100:.1f}% 변화입니다.")
            if price_chg > 0 and per_chg < 0:
                msg.append("해석: 주가 상승에도 PER이 낮아져 실적 개선이 주가 상승을 정당화하는 흐름입니다.")
            elif price_chg > 0 and per_chg > 0:
                msg.append("해석: 주가와 PER이 같이 올라 밸류 부담 확대 가능성이 있습니다.")
            elif price_chg < 0 and per_chg > 0:
                msg.append("해석: 주가는 빠졌지만 PER이 올라 이익 악화 가능성을 확인해야 합니다.")
            else:
                msg.append("해석: 주가와 PER이 같이 낮아져 밸류 부담은 완화됐지만 업황 확인이 필요합니다.")
    else:
        cur = safe_float(current_val.get("ttm_pe"))
        if cur is not None:
            msg.append(f"PER: 시계열은 없지만 현재 PER {cur:.2f}배 기준선은 차트에 표시했습니다.")
        else:
            msg.append("PER: 시계열과 현재 PER 모두 부족합니다. MDD·이평선 중심으로 판단하세요.")

    if risk_df is not None and not risk_df.empty:
        risk_val = safe_float(risk_df["Risk"].dropna().iloc[-1]) if risk_df["Risk"].dropna().shape[0] else None
        if risk_label == "VIX" and risk_val is not None:
            msg.append(f"시장위험: VIX {risk_val:.1f}입니다.")
        elif risk_val is not None:
            msg.append(f"시장위험: {risk_label} {risk_val*100:.1f}입니다.")

    if dd <= -0.12 and rsi <= 42 and close >= ma20:
        final = "최종: 1차 소액 가능. 단, 분할 기준입니다."
    elif dd <= -0.12 and rsi <= 42:
        final = "최종: 1차 후보지만 MA20 회복 확인 전에는 소액 또는 대기입니다."
    elif dd > -0.05 and rsi >= 65:
        final = "최종: 추격 금지. 일부 현금확보 후보입니다."
    else:
        final = "최종: 대기. 가격·PER·MDD 방향을 추가 확인하세요."
    return final, msg


# =========================================================
# UI
# =========================================================
col1, col2, col3 = st.columns(3)
with col1:
    user_input = st.text_input("종목명 / 종목코드 / 미국 티커", value="NVDA")
with col2:
    start_date = st.date_input("기준 시작일", pd.to_datetime("2024-01-01"))
with col3:
    asset_type = st.selectbox("종목 유형", ["일반 주식/ETF", "나스닥형 ETF", "반도체/메모리 ETF", "전력/인프라 ETF", "우주/소형 테마"], index=0)

run = st.button("분석 실행")

if run:
    market, ticker, display_name, kr_market_hint = find_ticker(user_input)
    if ticker is None:
        st.error("종목을 찾을 수 없습니다. 한국 종목은 종목명 또는 6자리 코드, 미국 종목은 티커를 입력하세요.")
        st.stop()

    price_df, price_status = load_price_data(market, ticker, start_date)
    if price_df.empty:
        st.error(f"가격 데이터를 가져오지 못했습니다. 원인: {price_status}")
        st.stop()

    df = calculate_indicators(price_df)
    latest = df.iloc[-1]
    profile = get_type_profile(asset_type)

    if market == "US":
        current_val, current_val_status = us_current_valuation(ticker)
        per_df, per_status = us_estimated_ttm_pe_series(ticker, df)
    else:
        current_val, current_val_status = naver_current_valuation(ticker)
        per_df, per_status = kr_per_series(ticker, start_date, df.index[-1], current_val=current_val)
        # If latest-only pykrx row is available, use it for current card when Naver is missing.
        if (current_val.get("ttm_pe") is None) and (not per_df.empty) and ("PER" in per_df.columns) and per_df["PER"].dropna().shape[0] > 0:
            current_val["ttm_pe"] = safe_float(per_df["PER"].dropna().iloc[-1])
        if (current_val.get("pbr") is None) and (not per_df.empty) and ("PBR" in per_df.columns) and per_df["PBR"].dropna().shape[0] > 0:
            current_val["pbr"] = safe_float(per_df["PBR"].dropna().iloc[-1])
        if (current_val.get("eps") is None) and (not per_df.empty) and ("EPS" in per_df.columns) and per_df["EPS"].dropna().shape[0] > 0:
            current_val["eps"] = safe_float(per_df["EPS"].dropna().iloc[-1])

    risk_df, risk_label = market_risk_series(market, ticker, start_date, kr_market_hint)
    action = final_action(latest, profile)

    st.subheader(f"분석 대상: {display_name} / {ticker} / {market}")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("현재가", fmt_num(latest["Close"]))
    c2.metric("Current DD", f"{latest['Current_Drawdown']*100:.2f}%")
    c3.metric("Max DD", f"{df['Max_Drawdown'].min()*100:.2f}%")
    c4.metric("RSI", fmt_num(latest["RSI"]))
    c5.metric("MA20", "위" if latest["Close"] >= latest["MA20"] else "아래")
    c6.metric("Vol Ratio", fmt_num(latest["Volume_Ratio"]))

    if "금지" in action:
        st.error(f"최종 행동: {action}")
    elif "가능" in action:
        st.success(f"최종 행동: {action}")
    else:
        st.info(f"최종 행동: {action}")

    st.markdown("## 1. 현재 Valuation")
    v1, v2, v3, v4 = st.columns(4)
    v1.metric("TTM / KRX P/E", fmt_num(current_val.get("ttm_pe")))
    v2.metric("Forward P/E", fmt_num(current_val.get("fwd_pe")))
    v3.metric("P/S", fmt_num(current_val.get("ps")))
    v4.metric("PEG", fmt_num(current_val.get("peg")))

    # Current PER is displayed even if time-series is unavailable.
    if per_df.empty:
        st.warning(f"PER 시계열 없음. 현재 PER 기준선만 사용합니다. 상태: {per_status}")
    else:
        st.success(f"PER 상태: {per_status}")

    st.markdown("## 2. 핵심 차트")
    fig, chart_df = plot_core_chart(df, per_df, risk_df, risk_label, ticker, current_val, profile)
    st.pyplot(fig, use_container_width=True)

    st.markdown("## 3. 자동 해석")
    final_msg, msg_list = make_comment(df, per_df, risk_label, risk_df, current_val)
    st.info(final_msg)
    for m in msg_list:
        st.write(f"- {m}")

    with st.expander("PER 원자료 / 상태"):
        st.write(f"Current valuation status: {current_val_status}")
        st.write(f"PER status: {per_status}")
        if not per_df.empty:
            st.dataframe(per_df.tail(30), use_container_width=True)
        else:
            st.write("PER DataFrame empty")
        st.write("pykrx available:", PYKRX_OK)
        if not PYKRX_OK:
            st.code(PYKRX_ERR)

    with st.expander("최근 20거래일"):
        view = df[["Close", "Current_Drawdown", "Max_Drawdown", "RSI", "MA20", "MA60", "MA200", "Volume_Ratio"]].tail(20).copy()
        view["Current_Drawdown"] = view["Current_Drawdown"] * 100
        view["Max_Drawdown"] = view["Max_Drawdown"] * 100
        st.dataframe(view, use_container_width=True)
