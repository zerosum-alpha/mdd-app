import warnings
warnings.filterwarnings("ignore")

from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import re
import zipfile
import io
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


def render_chart(fig):
    """Render matplotlib chart safely.
    Do not change calculation or plot logic; only handle Streamlit versions.
    """
    try:
        st.pyplot(fig, use_container_width=True)
    except TypeError:
        st.pyplot(fig)
    finally:
        try:
            plt.close(fig)
        except Exception:
            pass


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


def merge_actual_and_proxy_per(actual_df, proxy_df):
    frames = []
    if actual_df is not None and not actual_df.empty:
        a = actual_df.copy()
        a.index = to_dt_index(a.index)
        frames.append(a)
    if proxy_df is not None and not proxy_df.empty:
        p = proxy_df.copy()
        p.index = to_dt_index(p.index)
        frames.append(p)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, axis=1)
    out = out.loc[:, ~out.columns.duplicated()]
    return out.sort_index()


# =========================================================
# DART-based KR actual TTM PER
# =========================================================
def clean_amount(x):
    try:
        if x is None or pd.isna(x):
            return None
        s = str(x).replace(",", "").replace(" ", "").strip()
        if s in ["", "-", "—"]:
            return None
        # Korean negative format may be (123)
        if s.startswith("(") and s.endswith(")"):
            s = "-" + s[1:-1]
        return float(s)
    except Exception:
        return None


@st.cache_data(ttl=86400, show_spinner=False)
def dart_corp_code_map(api_key):
    """Return stock_code -> corp_code mapping from OpenDART corpCode.xml."""
    if not api_key or not REQ_OK:
        return {}, "DART key 또는 requests 없음"
    try:
        url = "https://opendart.fss.or.kr/api/corpCode.xml"
        r = requests.get(url, params={"crtfc_key": api_key}, timeout=12)
        if r.status_code != 200:
            return {}, f"corpCode HTTP {r.status_code}"
        z = zipfile.ZipFile(io.BytesIO(r.content))
        xml_name = z.namelist()[0]
        root = ET.fromstring(z.read(xml_name))
        mp = {}
        for item in root.findall("list"):
            corp_code = (item.findtext("corp_code") or "").strip()
            stock_code = (item.findtext("stock_code") or "").strip()
            if corp_code and stock_code:
                mp[stock_code.zfill(6)] = corp_code
        if not mp:
            return {}, "corpCode mapping empty"
        return mp, "OK"
    except Exception as e:
        return {}, f"corpCode error: {repr(e)}"


def _pick_eps_from_dart_list(items):
    """Pick EPS from DART account rows. Prefer basic EPS, then diluted EPS."""
    if not items:
        return None, "no list"
    rows = []
    for it in items:
        acc_id = str(it.get("account_id", ""))
        acc_nm = str(it.get("account_nm", ""))
        sj_nm = str(it.get("sj_nm", ""))
        amount = clean_amount(it.get("thstrm_amount"))
        if amount is None:
            continue
        key = (acc_id + " " + acc_nm).lower()
        score = 0
        # Korean labels vary; keep broad but prefer exact EPS rows.
        if "주당" in acc_nm and "이익" in acc_nm:
            score += 10
        if "기본" in acc_nm:
            score += 5
        if "희석" in acc_nm:
            score += 3
        if "basic" in key and "earnings" in key and "share" in key:
            score += 8
        if "diluted" in key and "earnings" in key and "share" in key:
            score += 6
        if "eps" in key:
            score += 4
        # Exclude continuing-operation-only EPS when possible.
        if "계속" in acc_nm or "continuing" in key:
            score -= 3
        if score > 0:
            rows.append((score, amount, acc_id, acc_nm, sj_nm))
    if not rows:
        return None, "EPS row not found"
    rows.sort(key=lambda x: x[0], reverse=True)
    score, amount, acc_id, acc_nm, sj_nm = rows[0]
    return amount, f"{acc_nm} / {acc_id}"


@st.cache_data(ttl=86400, show_spinner=False)
def dart_kr_ttm_pe_series(code, price_df, start_date, end_date, api_key):
    """Build actual-style daily PER for KR stocks using DART quarterly cumulative EPS.

    This is the correct way to see whether price rises while P/E falls when KRX PER series is unavailable.
    It requires a DART API key. It is still historical trailing P/E, not 12M forward consensus P/E.
    """
    if not api_key:
        return pd.DataFrame(), "DART key 없음: 실제 EPS 기반 한국 PER 계산 불가"
    code = str(code).zfill(6)
    mp, mp_status = dart_corp_code_map(api_key)
    corp_code = mp.get(code)
    if not corp_code:
        return pd.DataFrame(), f"DART corp_code 없음: {mp_status}"

    # Need at least previous fiscal year to calculate rolling 4 quarters.
    start_y = pd.to_datetime(start_date).year - 1
    end_y = pd.to_datetime(end_date).year
    report_map = {
        "11013": ("Q1", 1, "05-15"),
        "11012": ("H1", 2, "08-15"),
        "11014": ("Q3", 3, "11-15"),
        "11011": ("A", 4, "03-31"),
    }
    cum_rows = []
    errors = []
    for year in range(start_y, end_y + 1):
        for reprt_code, (label, q, md) in report_map.items():
            try:
                url = "https://opendart.fss.or.kr/api/fnlttSinglAcntAll.json"
                params = {
                    "crtfc_key": api_key,
                    "corp_code": corp_code,
                    "bsns_year": str(year),
                    "reprt_code": reprt_code,
                    "fs_div": "CFS",
                }
                res = requests.get(url, params=params, timeout=10).json()
                if res.get("status") != "000":
                    # Try separate financial statements only if consolidated unavailable.
                    params["fs_div"] = "OFS"
                    res = requests.get(url, params=params, timeout=10).json()
                if res.get("status") != "000":
                    errors.append(f"{year}-{label}:{res.get('status')} {res.get('message','')[:40]}")
                    continue
                eps, eps_src = _pick_eps_from_dart_list(res.get("list", []))
                if eps is None:
                    errors.append(f"{year}-{label}:{eps_src}")
                    continue
                # Report availability date approximation; sufficient for chart alignment.
                if label == "A":
                    avail_date = pd.Timestamp(year=year + 1, month=3, day=31)
                else:
                    month, day = map(int, md.split("-"))
                    avail_date = pd.Timestamp(year=year, month=month, day=day)
                cum_rows.append({
                    "year": year,
                    "q": q,
                    "label": label,
                    "Date": avail_date,
                    "EPS_CUM": eps,
                    "EPS_SOURCE": eps_src,
                })
            except Exception as e:
                errors.append(f"{year}-{label}:{type(e).__name__} {str(e)[:50]}")

    if len(cum_rows) < 4:
        return pd.DataFrame(), "DART EPS 데이터 부족: " + " / ".join(errors[:5])

    cum = pd.DataFrame(cum_rows).sort_values(["year", "q"])
    q_rows = []
    for year, g in cum.groupby("year"):
        g = g.set_index("q").sort_index()
        prev_cum = 0.0
        for q in [1, 2, 3, 4]:
            if q not in g.index:
                continue
            eps_cum = safe_float(g.loc[q, "EPS_CUM"])
            if eps_cum is None:
                continue
            q_eps = eps_cum - prev_cum
            prev_cum = eps_cum
            q_rows.append({
                "Date": g.loc[q, "Date"],
                "EPS_Q": q_eps,
                "EPS_CUM": eps_cum,
                "EPS_SOURCE": g.loc[q, "EPS_SOURCE"],
            })
    if len(q_rows) < 4:
        return pd.DataFrame(), "DART quarterly EPS 계산 부족"

    qdf = pd.DataFrame(q_rows).sort_values("Date")
    qdf["EPS_TTM"] = qdf["EPS_Q"].rolling(4).sum()
    qdf = qdf.dropna(subset=["EPS_TTM"])
    qdf = qdf[qdf["EPS_TTM"] > 0]
    if qdf.empty:
        return pd.DataFrame(), "DART EPS_TTM empty"

    daily = price_df[["Close"]].reset_index()
    daily.columns = ["Date", "Close"]
    daily["Date"] = to_dt_index(daily["Date"]).normalize()
    qdf["Date"] = to_dt_index(qdf["Date"]).normalize()
    merged = pd.merge_asof(daily.sort_values("Date"), qdf[["Date", "EPS_TTM"]].sort_values("Date"), on="Date", direction="backward")
    merged["PER"] = merged["Close"] / merged["EPS_TTM"].replace(0, np.nan)
    merged = merged[(merged["PER"] > 0) & (merged["PER"] < 500)].dropna(subset=["PER"])
    if merged.empty:
        return pd.DataFrame(), "DART PER 계산 결과 empty"
    out = merged.set_index("Date")[["PER", "EPS_TTM"]]
    return out, "OK: DART actual TTM P/E"

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
    """Core chart with robust PER plotting.

    Fix point:
    - If PER_PROXY exists, it is not just shown in the raw table. It is merged into the chart by date and plotted.
    - Date matching uses normalized dates + merge_asof, so Date column / index / monthly sparse data all work.
    - Actual PER and Proxy PER are clearly separated.
    """

    # ---- Base price chart ----
    chart = df[["Close", "MA20", "MA60", "MA200", "Current_Drawdown"]].copy()
    chart = chart.rename(columns={"Close": "Price", "Current_Drawdown": "DD"})
    chart.index = to_dt_index(chart.index).normalize()
    chart = chart[~chart.index.duplicated(keep="last")].sort_index()
    chart["_Date"] = chart.index

    def _merge_series_by_date(base, source, col):
        """Merge one numeric column from source into base by nearest previous date."""
        if source is None or source.empty or col not in source.columns:
            return base
        p = source.copy()
        if "Date" in p.columns:
            p["_Date"] = to_dt_index(p["Date"]).normalize()
        else:
            p["_Date"] = to_dt_index(p.index).normalize()
        p[col] = pd.to_numeric(p[col], errors="coerce")
        p = p[["_Date", col]].dropna().sort_values("_Date")
        p = p.drop_duplicates("_Date", keep="last")
        if p.empty:
            return base
        left = base[["_Date"]].sort_values("_Date")
        merged = pd.merge_asof(left, p, on="_Date", direction="backward")
        base[col] = merged[col].to_numpy()
        return base

    # ---- Merge PER data robustly ----
    if per_df is not None and not per_df.empty:
        p = per_df.copy()
        if "Date" in p.columns:
            p["_Date"] = to_dt_index(p["Date"]).normalize()
        else:
            p["_Date"] = to_dt_index(p.index).normalize()
        for c in ["PER", "PER_PROXY", "EPS", "EPS_TTM", "EPS_PROXY"]:
            if c in p.columns:
                p[c] = pd.to_numeric(p[c], errors="coerce")
        for col in ["PER", "PER_PROXY", "EPS", "EPS_TTM", "EPS_PROXY"]:
            if col in p.columns:
                chart = _merge_series_by_date(chart, p, col)

    for col in ["PER", "PER_PROXY", "EPS", "EPS_TTM", "EPS_PROXY"]:
        if col not in chart.columns:
            chart[col] = np.nan
        else:
            chart[col] = pd.to_numeric(chart[col], errors="coerce")

    # If EPS exists but no PER, calculate PER from price / EPS.
    if chart["PER"].dropna().empty:
        eps_col = None
        if chart["EPS"].dropna().any():
            eps_col = "EPS"
        elif chart["EPS_TTM"].dropna().any():
            eps_col = "EPS_TTM"
        if eps_col:
            chart[eps_col] = chart[eps_col].ffill()
            chart["PER"] = chart["Price"] / chart[eps_col].replace(0, np.nan)

    # Important: if actual PER is absent but PER_PROXY exists, make proxy the visible main PER line.
    has_actual_per = not chart["PER"].dropna().empty
    has_proxy_per = not chart["PER_PROXY"].dropna().empty

    # ---- Merge market risk robustly ----
    if risk_df is not None and not risk_df.empty and "Risk" in risk_df.columns:
        r = risk_df.copy()
        if "Date" in r.columns:
            r["_Date"] = to_dt_index(r["Date"]).normalize()
        else:
            r["_Date"] = to_dt_index(r.index).normalize()
        r["Risk"] = pd.to_numeric(r["Risk"], errors="coerce")
        r = r[["_Date", "Risk"]].dropna().sort_values("_Date").drop_duplicates("_Date", keep="last")
        if not r.empty:
            merged = pd.merge_asof(chart[["_Date"]].sort_values("_Date"), r, on="_Date", direction="backward")
            chart["Risk"] = merged["Risk"].to_numpy()
    if "Risk" not in chart.columns:
        chart["Risk"] = np.nan

    # ---- Signals ----
    sig = build_signals(df).copy()
    if sig is not None and not sig.empty:
        sig.index = to_dt_index(sig.index).normalize()
        sig = sig[~sig.index.duplicated(keep="last")]
        chart = chart.join(sig[[c for c in ["Buy", "Cash"] if c in sig.columns]], how="left")

    # ---- Draw chart ----
    fig, (ax1, ax3) = plt.subplots(
        2, 1, figsize=(19, 10.5), sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.1]}
    )

    # Price and moving averages
    ax1.plot(chart.index, chart["Price"], color="#0057B8", linewidth=2.4, label="Price")
    ax1.plot(chart.index, chart["MA20"], color="#FF8C00", linewidth=1.35, label="MA20")
    ax1.plot(chart.index, chart["MA60"], color="#228B22", linewidth=1.35, label="MA60")
    ax1.plot(chart.index, chart["MA200"], color="#7B2CBF", linewidth=1.35, label="MA200")

    if "Buy" in chart.columns and chart["Buy"].notna().any():
        ax1.scatter(chart.index, chart["Buy"] * 0.975, color="#008000", marker="^", s=90, label="BUY candidate", zorder=5)
    if "Cash" in chart.columns and chart["Cash"].notna().any():
        ax1.scatter(chart.index, chart["Cash"] * 1.025, color="#FF0000", marker="v", s=90, label="Cash / overheat", zorder=5)

    ax1.set_ylabel("Price", color="#0057B8")
    ax1.tick_params(axis="y", labelcolor="#0057B8")
    ax1.grid(True, linestyle=":", alpha=0.35)

    # PER axis
    ax2 = ax1.twinx()
    per_lines = []
    if has_actual_per:
        if has_proxy_per:
            ln = ax2.plot(chart.index, chart["PER_PROXY"], color="#D62728", linewidth=1.15, linestyle=":", alpha=0.6, label="P/E proxy(current EPS)")
            per_lines.append(chart["PER_PROXY"])
        ax2.plot(chart.index, chart["PER"], color="#D62728", linewidth=2.25, linestyle="-", label="P/E actual")
        per_lines.append(chart["PER"])
        per_avg = chart["PER"].dropna().mean()
        if pd.notna(per_avg):
            ax2.axhline(per_avg, color="#D62728", linewidth=1.0, linestyle="--", alpha=0.35, label="P/E actual avg")
    elif has_proxy_per:
        # Main fix for KR: proxy is promoted to visible P/E line.
        ax2.plot(chart.index, chart["PER_PROXY"], color="#D62728", linewidth=2.25, linestyle="-", alpha=0.98, label="P/E proxy(current EPS)")
        per_lines.append(chart["PER_PROXY"])
        proxy_avg = chart["PER_PROXY"].dropna().mean()
        if pd.notna(proxy_avg):
            ax2.axhline(proxy_avg, color="#D62728", linewidth=1.0, linestyle="--", alpha=0.35, label="P/E proxy avg")
    else:
        ax2.text(0.99, 0.95, "P/E line: N/A", transform=ax2.transAxes, ha="right", va="top", color="#D62728")

    # Keep PER axis readable; do not let outliers flatten the visible line.
    if per_lines:
        vals = pd.concat([x.dropna() for x in per_lines if x is not None and not x.dropna().empty])
        vals = vals[(vals > 0) & (vals < 500)]
        if not vals.empty:
            q05, q95 = vals.quantile([0.05, 0.95])
            span = max(q95 - q05, 1)
            ax2.set_ylim(max(0, q05 - span * 0.35), q95 + span * 0.35)

    ax2.set_ylabel("P/E", color="#D62728")
    ax2.tick_params(axis="y", labelcolor="#D62728")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
    ax1.set_title(f"{ticker} Price + P/E + MDD + Market Risk", fontweight="bold")

    # Lower: MDD + market risk
    ax3.plot(chart.index, chart["DD"] * 100, color="#8B0000", linewidth=1.7, label="Current DD")
    for level, label in [(-8, "Watch -8%"), (-12, "Buy zone -12%"), (-15, "Deep -15%"), (-20, "Risk -20%")]:
        ax3.axhline(level, color="#5DADE2", linestyle="--", linewidth=0.9, alpha=0.55, label=label)
    ax3.set_ylabel("MDD (%)", color="#8B0000")
    ax3.tick_params(axis="y", labelcolor="#8B0000")
    ax3.grid(True, linestyle=":", alpha=0.35)

    ax4 = ax3.twinx()
    if chart["Risk"].notna().any():
        if risk_label == "VIX":
            ax4.plot(chart.index, chart["Risk"], color="#00A6A6", linewidth=1.2, linestyle="--", alpha=0.85, label="VIX")
            ax4.set_ylabel("VIX", color="#00A6A6")
        else:
            ax4.plot(chart.index, chart["Risk"] * 100, color="#00A6A6", linewidth=1.2, linestyle="--", alpha=0.85, label=risk_label)
            ax4.set_ylabel(risk_label + " (%)", color="#00A6A6")
        ax4.tick_params(axis="y", labelcolor="#00A6A6")

    lines3, labels3 = ax3.get_legend_handles_labels()
    lines4, labels4 = ax4.get_legend_handles_labels()
    ax3.legend(lines3 + lines4, labels3 + labels4, loc="lower left", fontsize=7)

    # Wider right margin prevents clipping.
    fig.subplots_adjust(left=0.06, right=0.90, top=0.93, bottom=0.08, hspace=0.08)
    return fig, chart


def build_per_display_chart_df(price_df, per_df):
    """Create a dedicated PER display frame.

    Important fix:
    - Do not choose only one of PER or PER_PROXY.
    - If actual PER is partial, show it as a solid line and show proxy as a dotted full-period line.
    - This prevents the PER check chart from looking broken or cut off.
    """
    if price_df is None or price_df.empty:
        return pd.DataFrame(), "가격 데이터 없음"

    base = price_df[["Close"]].copy().rename(columns={"Close": "Price"})
    base.index = to_dt_index(base.index).normalize()
    base = base[~base.index.duplicated(keep="last")].sort_index()
    base["_Date"] = base.index
    base["PER_ACTUAL"] = np.nan
    base["PER_PROXY_VIEW"] = np.nan

    if per_df is None or per_df.empty:
        return base, "PER 원자료 없음"

    p = per_df.copy()
    if "Date" in p.columns:
        p["_Date"] = to_dt_index(p["Date"]).normalize()
    else:
        p["_Date"] = to_dt_index(p.index).normalize()

    for col in ["PER", "PER_PROXY", "EPS", "EPS_TTM", "EPS_PROXY"]:
        if col in p.columns:
            p[col] = pd.to_numeric(p[col], errors="coerce")

    def _merge_col(target_col, source_col):
        if source_col not in p.columns:
            return
        src = p[["_Date", source_col]].dropna().sort_values("_Date")
        src = src.drop_duplicates("_Date", keep="last")
        if src.empty:
            return
        merged = pd.merge_asof(
            base[["_Date"]].sort_values("_Date"),
            src,
            on="_Date",
            direction="backward"
        )
        base[target_col] = pd.to_numeric(merged[source_col].to_numpy(), errors="coerce")

    _merge_col("PER_ACTUAL", "PER")
    _merge_col("PER_PROXY_VIEW", "PER_PROXY")

    has_actual = base["PER_ACTUAL"].dropna().shape[0] > 0
    has_proxy = base["PER_PROXY_VIEW"].dropna().shape[0] > 0

    if has_actual and has_proxy:
        label = "P/E actual + proxy"
    elif has_actual:
        label = "P/E actual only"
    elif has_proxy:
        label = "P/E proxy only(current EPS)"
    else:
        label = f"PER/PER_PROXY 없음. columns={list(per_df.columns)}"

    return base, label


def plot_dedicated_per_chart(price_df, per_df, ticker):
    """Dedicated Price + PER chart.

    Solid red = actual EPS-based P/E where available.
    Dotted red = current-EPS proxy, full-period reference only.
    """
    chart, label = build_per_display_chart_df(price_df, per_df)
    if chart.empty:
        st.warning(f"PER 전용 차트 없음: {label}")
        return

    has_actual = chart["PER_ACTUAL"].dropna().shape[0] > 0
    has_proxy = chart["PER_PROXY_VIEW"].dropna().shape[0] > 0

    if not has_actual and not has_proxy:
        st.warning(f"PER 전용 차트 없음: {label}")
        return

    fig, ax1 = plt.subplots(figsize=(14.5, 5.2))
    ax1.plot(chart.index, chart["Price"], color="#0057B8", linewidth=2.1, label="Price")
    ax1.set_ylabel("Price", color="#0057B8")
    ax1.tick_params(axis="y", labelcolor="#0057B8")
    ax1.grid(True, linestyle=":", alpha=0.35)

    ax2 = ax1.twinx()
    per_lines = []

    if has_proxy:
        ax2.plot(
            chart.index,
            chart["PER_PROXY_VIEW"],
            color="#D62728",
            linewidth=1.25,
            linestyle=":",
            alpha=0.55,
            label="P/E proxy(current EPS)"
        )
        per_lines.append(chart["PER_PROXY_VIEW"])

    if has_actual:
        ax2.plot(
            chart.index,
            chart["PER_ACTUAL"],
            color="#D62728",
            linewidth=2.35,
            linestyle="-",
            alpha=0.98,
            label="P/E actual"
        )
        per_lines.append(chart["PER_ACTUAL"])
        per_avg = chart["PER_ACTUAL"].dropna().mean()
        if pd.notna(per_avg):
            ax2.axhline(per_avg, color="#D62728", linestyle="--", linewidth=1.0, alpha=0.35, label="P/E actual avg")
    elif has_proxy:
        proxy_avg = chart["PER_PROXY_VIEW"].dropna().mean()
        if pd.notna(proxy_avg):
            ax2.axhline(proxy_avg, color="#D62728", linestyle="--", linewidth=1.0, alpha=0.35, label="P/E proxy avg")

    vals = pd.concat([s.dropna() for s in per_lines if s is not None and not s.dropna().empty]) if per_lines else pd.Series(dtype=float)
    vals = vals[(vals > 0) & (vals < 500)]
    if not vals.empty:
        q05, q95 = vals.quantile([0.05, 0.95])
        span = max(float(q95 - q05), 1.0)
        ax2.set_ylim(max(0, q05 - span * 0.35), q95 + span * 0.35)

    ax2.set_ylabel("P/E", color="#D62728")
    ax2.tick_params(axis="y", labelcolor="#D62728")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
    ax1.set_title(f"{ticker} Price + P/E display check", fontweight="bold")
    fig.subplots_adjust(left=0.07, right=0.90, top=0.90, bottom=0.15)
    render_chart(fig)

    st.caption(
        f"PER 전용 차트 기준: {label}. "
        "빨간 실선은 실제 EPS 기반 PER, 빨간 점선은 현재 EPS 기준 proxy입니다. "
        "실선이 중간부터 시작하면 그 이전 구간은 실제 EPS_TTM 원자료가 없는 것입니다."
    )



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

    per_col = None
    if per_df is not None and not per_df.empty:
        if "PER" in per_df.columns and per_df["PER"].dropna().shape[0] >= 20:
            per_col = "PER"
        elif "PER_PROXY" in per_df.columns and per_df["PER_PROXY"].dropna().shape[0] >= 20:
            per_col = "PER_PROXY"

    if per_col:
        p = pd.to_numeric(per_df[per_col], errors="coerce").dropna()
        n = min(60, len(p)-1)
        if n > 0:
            chg = p.iloc[-1] / p.iloc[-n] - 1
            label = "PER" if per_col == "PER" else "PER proxy"
            if chg < -0.10:
                msg.append(f"{label}: 최근 기준 {chg*100:.1f}% 하락. 밸류 부담이 낮아진 흐름입니다.")
            elif chg > 0.10:
                msg.append(f"{label}: 최근 기준 {chg*100:.1f}% 상승. 밸류 부담 확대 구간입니다.")
            else:
                msg.append(f"{label}: 최근 기준 {chg*100:.1f}% 변화. 큰 방향성은 약합니다.")
    else:
        msg.append("PER: 실제 시계열은 제한적입니다. 현재 PER 카드와 가격/MDD를 우선 보세요.")

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

try:
    dart_secret = st.secrets.get("DART_API_KEY", "")
except Exception:
    dart_secret = ""
with st.expander("한국 실제 PER 설정", expanded=False):
    st.caption("한국 종목에서 pykrx PER 시계열이 안 잡힐 때, DART 분기 EPS로 TTM P/E를 계산합니다. Naver 현재 EPS proxy와 달리 EPS가 분기별로 변합니다.")
    dart_key_input = st.text_input("DART API Key", value=dart_secret, type="password")

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

        # If pykrx returns EPS without PER, calculate PER from that historical EPS.
        if not actual_per_df.empty and "PER" not in actual_per_df.columns and "EPS" in actual_per_df.columns:
            tmp = df[["Close"]].join(actual_per_df[["EPS"]], how="left")
            tmp["EPS"] = tmp["EPS"].ffill()
            tmp["PER"] = tmp["Close"] / tmp["EPS"].replace(0, np.nan)
            actual_per_df = tmp[["PER", "EPS"]].dropna()

        # If pykrx actual PER is unavailable, use DART quarterly EPS TTM.
        dart_status = "DART not used"
        if actual_per_df.empty or "PER" not in actual_per_df.columns or actual_per_df["PER"].dropna().empty:
            dart_per_df, dart_status = dart_kr_ttm_pe_series(ticker, df, start_date, last_price_date, dart_key_input)
            if dart_per_df is not None and not dart_per_df.empty:
                actual_per_df = dart_per_df
                actual_per_status = dart_status

        # Current EPS proxy is only a fallback reference, not actual valuation trend.
        proxy_df, proxy_status = build_per_proxy_from_current(
            df, latest["Close"], current_pe=val.get("ttm_pe"), current_eps=val.get("eps"), label="KR current EPS-implied"
        )
        per_df = merge_actual_and_proxy_per(actual_per_df, proxy_df)
        per_status = f"Actual: {actual_per_status} / DART: {dart_status} / Proxy: {proxy_status}"

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
    st.info("차트는 주가·PER·MDD·시장위험만 봅니다. 미국은 기존처럼 actual P/E + full-view proxy를 함께 표시하고, 한국은 실제 PER이 없으면 현재 EPS 기반 P/E proxy를 빨간 실선으로 표시합니다.")
    fig, chart_df = plot_core_chart(df, per_df, risk_df, risk_label, ticker)
    render_chart(fig)

    st.markdown("### PER 전용 확인 차트")
    plot_dedicated_per_chart(df, per_df, ticker)

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
