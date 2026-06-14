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

st.title("📈 MDD 저점매수 분석기 | Core No-Hang")
st.caption("주가 / PER / MDD / 시장위험 / 이평선만 표시합니다. PER 조회가 실패해도 앱은 멈추지 않습니다.")

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
    data = {"ttm_pe": None, "fwd_pe": None, "ps": None, "peg": None, "eps": None}
    if not REQ_OK:
        return data, "requests 없음"
    try:
        url = f"https://finance.naver.com/item/main.naver?code={str(code).zfill(6)}"
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
        r.raise_for_status()
        html = r.text

        def pick(_id):
            m = re.search(rf'id=["\']{re.escape(_id)}["\'][^>]*>\s*([^<]+)\s*<', html)
            return safe_float(m.group(1)) if m else None

        data["ttm_pe"] = pick("_per")
        data["eps"] = pick("_eps")
        pbr = pick("_pbr")
        status = "OK: Naver current PER" if data["ttm_pe"] is not None else "Naver current PER 없음"
        if pbr is not None:
            status += f" / PBR {pbr:.2f}"
        return data, status
    except Exception as e:
        return data, f"Naver PER error: {repr(e)}"


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
    if per_df is not None and not per_df.empty:
        p = per_df.copy()
        p.index = to_dt_index(p.index)
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
        r = risk_df.copy()
        r.index = to_dt_index(r.index)
        chart = chart.join(r[["Risk"]], how="left")
        chart["Risk"] = chart["Risk"].ffill()
    else:
        chart["Risk"] = np.nan

    sig = build_signals(df)
    chart = chart.join(sig, how="left")

    fig, (ax1, ax3) = plt.subplots(
        2, 1, figsize=(16, 9), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]}
    )

    # Upper: Price, MA, PER
    ax1.plot(chart.index, chart["Price"], color="#0057B8", linewidth=2.2, label="Price")
    ax1.plot(chart.index, chart["MA20"], color="#FF8C00", linewidth=1.35, label="MA20")
    ax1.plot(chart.index, chart["MA60"], color="#2CA02C", linewidth=1.35, label="MA60")
    ax1.plot(chart.index, chart["MA200"], color="#7B2CBF", linewidth=1.35, label="MA200")

    if chart["Buy"].notna().any():
        ax1.scatter(chart.index, chart["Buy"] * 0.975, color="#008000", marker="^", s=95, label="BUY candidate", zorder=5)
    if chart["Cash"].notna().any():
        ax1.scatter(chart.index, chart["Cash"] * 1.025, color="#FF0000", marker="v", s=95, label="Cash / overheat", zorder=5)

    ax1.set_ylabel("Price", color="#0057B8")
    ax1.tick_params(axis="y", labelcolor="#0057B8")
    ax1.grid(True, linestyle=":", alpha=0.35)

    ax2 = ax1.twinx()
    if chart["PER"].dropna().empty:
        ax2.text(0.99, 0.95, "P/E line: N/A", transform=ax2.transAxes, ha="right", va="top", color="#D62728")
    else:
        ax2.plot(chart.index, chart["PER"], color="#D62728", linewidth=1.7, label="P/E")
        per_avg = chart["PER"].dropna().mean()
        ax2.axhline(per_avg, color="#D62728", linewidth=1.0, linestyle="--", alpha=0.35, label="P/E avg")
    ax2.set_ylabel("P/E", color="#D62728")
    ax2.tick_params(axis="y", labelcolor="#D62728")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
    ax1.set_title(f"{ticker} Price + P/E + MDD + Market Risk", fontweight="bold")

    # Lower: MDD + Risk
    ax3.plot(chart.index, chart["DD"] * 100, color="#8B0000", linewidth=1.5, label="Current DD")
    for level, label in [(-8, "Watch -8%"), (-12, "Buy zone -12%"), (-15, "Deep -15%"), (-20, "Risk -20%")]:
        ax3.axhline(level, color="#5DADE2", linestyle="--", linewidth=0.9, alpha=0.55, label=label)
    ax3.set_ylabel("MDD (%)", color="#8B0000")
    ax3.tick_params(axis="y", labelcolor="#8B0000")
    ax3.grid(True, linestyle=":", alpha=0.35)

    ax4 = ax3.twinx()
    if chart["Risk"].notna().any():
        if risk_label == "VIX":
            ax4.plot(chart.index, chart["Risk"], color="#00A6A6", linewidth=1.1, linestyle="--", alpha=0.85, label="VIX")
            ax4.set_ylabel("VIX", color="#00A6A6")
        else:
            ax4.plot(chart.index, chart["Risk"] * 100, color="#00A6A6", linewidth=1.1, linestyle="--", alpha=0.85, label=risk_label)
            ax4.set_ylabel(risk_label + " (%)", color="#00A6A6")
        ax4.tick_params(axis="y", labelcolor="#00A6A6")

    lines3, labels3 = ax3.get_legend_handles_labels()
    lines4, labels4 = ax4.get_legend_handles_labels()
    ax3.legend(lines3 + lines4, labels3 + labels4, loc="lower left", fontsize=7)

    fig.subplots_adjust(left=0.07, right=0.90, top=0.92, bottom=0.10, hspace=0.08)
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
        per_df, per_status = us_ttm_pe_series(ticker, df)
    else:
        val, val_status = naver_current_per(ticker)
        per_df, per_status = kr_per_series_nohang(ticker, start_date, last_price_date, timeout_sec=7)
        # If KRX PER series fails but EPS is present, plot no series; current card still from Naver.
        if not per_df.empty and "PER" not in per_df.columns and "EPS" in per_df.columns:
            tmp = df[["Close"]].join(per_df[["EPS"]], how="left")
            tmp["EPS"] = tmp["EPS"].ffill()
            tmp["PER"] = tmp["Close"] / tmp["EPS"].replace(0, np.nan)
            per_df = tmp[["PER", "EPS"]].dropna()

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

    if per_df is None or per_df.empty or "PER" not in per_df.columns or per_df["PER"].dropna().empty:
        st.warning(f"PER 시계열 없음: {per_status}")
    else:
        st.success(f"PER 시계열: {per_status}")

    st.markdown("## 2. 핵심 차트")
    st.info("차트는 주가·PER·MDD·시장위험만 봅니다. PER 조회가 실패해도 주가/MDD 차트는 계속 표시됩니다.")
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
