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
st.set_page_config(page_title="MDD 저점매수 분석기 FINAL", layout="wide")
require_login()
logout_button()

st.title("📈 MDD 저점매수 분석기 | Trading Final")
st.caption("기본 종목 없음 · 기준 시작일 2024/01/01 · 자산유형별 MDD 기준 · 국내상장 해외 ETF 기초자산 분류 · 주가/PER/MDD/시장위험/이평선/매매 체크포인트 중심")

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

# Market index aliases. These are analyzed as indices, not individual stocks.
# Valuation/PER is intentionally not calculated unless reliable index PER data is supplied separately.
INDEX_ALIAS_MAP = {
    "KOSPI": ("KR_INDEX", "KS11", "KOSPI"),
    "코스피": ("KR_INDEX", "KS11", "KOSPI"),
    "KS11": ("KR_INDEX", "KS11", "KOSPI"),
    "KOSDAQ": ("KR_INDEX", "KQ11", "KOSDAQ"),
    "코스닥": ("KR_INDEX", "KQ11", "KOSDAQ"),
    "KQ11": ("KR_INDEX", "KQ11", "KOSDAQ"),
    "NASDAQ": ("US_INDEX", "^IXIC", "NASDAQ Composite"),
    "나스닥": ("US_INDEX", "^IXIC", "NASDAQ Composite"),
    "^IXIC": ("US_INDEX", "^IXIC", "NASDAQ Composite"),
    "S&P500": ("US_INDEX", "^GSPC", "S&P 500"),
    "SP500": ("US_INDEX", "^GSPC", "S&P 500"),
    "SNP500": ("US_INDEX", "^GSPC", "S&P 500"),
    "에스앤피": ("US_INDEX", "^GSPC", "S&P 500"),
    "^GSPC": ("US_INDEX", "^GSPC", "S&P 500"),
    "DOW": ("US_INDEX", "^DJI", "Dow Jones"),
    "다우": ("US_INDEX", "^DJI", "Dow Jones"),
    "^DJI": ("US_INDEX", "^DJI", "Dow Jones"),
    "RUSSELL2000": ("US_INDEX", "^RUT", "Russell 2000"),
    "러셀2000": ("US_INDEX", "^RUT", "Russell 2000"),
    "^RUT": ("US_INDEX", "^RUT", "Russell 2000"),
    "SOX": ("US_INDEX", "^SOX", "PHLX Semiconductor Index"),
    "필라델피아반도체": ("US_INDEX", "^SOX", "PHLX Semiconductor Index"),
    "반도체지수": ("US_INDEX", "^SOX", "PHLX Semiconductor Index"),
    "^SOX": ("US_INDEX", "^SOX", "PHLX Semiconductor Index"),
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


def normalize_index_alias(q):
    raw = str(q).strip()
    compact = raw.upper().replace(" ", "").replace("-", "")
    candidates = [raw, raw.upper(), compact]
    for key in candidates:
        if key in INDEX_ALIAS_MAP:
            return INDEX_ALIAS_MAP[key]
    return None


def find_ticker(q):
    q = str(q).strip()
    if not q:
        return None, None, None

    idx = normalize_index_alias(q)
    if idx is not None:
        return idx

    if q.isdigit() and len(q) == 6:
        sl = kr_stock_list()
        if not sl.empty and {"Code", "Name"}.issubset(sl.columns):
            hit = sl[sl["Code"].astype(str).str.zfill(6) == q]
            if not hit.empty:
                return "KR", q, hit.iloc[0]["Name"]
        return "KR", q, q

    # New KRX ETF codes may include letters, e.g. 0174B0, 0051A0, 0020H0.
    q_up = q.upper()
    if re.fullmatch(r"[0-9A-Z]{6}", q_up) and any(ch.isdigit() for ch in q_up):
        sl = kr_stock_list()
        if not sl.empty and {"Code", "Name"}.issubset(sl.columns):
            hit = sl[sl["Code"].astype(str).str.upper() == q_up]
            if not hit.empty:
                return "KR", q_up, hit.iloc[0]["Name"]
        # If it starts with a digit, treat as KR code fallback rather than a US ticker.
        if q_up[0].isdigit():
            return "KR", q_up, q_up
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

    idx = normalize_index_alias(q.upper())
    if idx is not None:
        return idx
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
        elif market == "KR_INDEX":
            if not FDR_OK:
                return pd.DataFrame(), f"FinanceDataReader import 실패: {FDR_ERR}"
            df = fdr.DataReader(str(ticker), start)
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


def build_signals(df, buy_dd=-0.12):
    # Buy marker uses the asset-specific 1차 관심 MDD threshold.
    # This is a chart marker only; final action is decided by asset rule + correction score.
    buy = (df["Current_Drawdown"] <= buy_dd) & (df["RSI"] <= 42)
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
def market_risk_series(market, ticker, start_date, asset_rule_key=None):
    # 국내상장 해외 ETF라도 자산 룰이 US_*이면 VIX를 시장위험 기준으로 사용한다.
    if str(asset_rule_key).startswith("US_") or market in ["US", "US_INDEX"]:
        vix = load_us_close("^VIX", start_date)
        if not vix.empty:
            return vix.rename(columns={"^VIX": "Risk"}), "VIX"
        return pd.DataFrame(), "VIX 없음"

    # KR stock: use KOSPI drawdown. KR index: use its own drawdown when possible.
    symbol = "KQ11" if str(ticker).upper() == "KQ11" else "KS11"
    label = "KOSDAQ DD" if symbol == "KQ11" else "KOSPI DD"
    idx = load_fdr_close(symbol, start_date)
    if idx.empty:
        return pd.DataFrame(), f"{label} 위험지표 없음"
    s = idx.iloc[:, 0]
    dd = s / s.cummax() - 1
    return pd.DataFrame({"Risk": dd}, index=idx.index), label

# =========================================================
# Chart and comment
# =========================================================
def plot_core_chart(df, per_df, risk_df, risk_label, ticker, asset_rule_key=None):
    """Core chart with full PER visibility.

    Rule:
    - P/E actual is plotted where actual EPS/TTM data exists.
    - P/E proxy(current EPS) is plotted across the full price range when available.
    - If actual is partial, proxy is still visible; it is not suppressed.
    """
    chart = df[["Close", "MA20", "MA60", "MA200", "Current_Drawdown"]].copy()
    chart = chart.rename(columns={"Close": "Price", "Current_Drawdown": "DD"})
    chart.index = to_dt_index(chart.index).normalize()
    chart = chart[~chart.index.duplicated(keep="last")].sort_index()
    chart["_Date"] = chart.index

    def _merge_col(base, source, col):
        if source is None or source.empty or col not in source.columns:
            return base
        s = source.copy()
        if "Date" in s.columns:
            s["_Date"] = to_dt_index(s["Date"]).normalize()
        else:
            s["_Date"] = to_dt_index(s.index).normalize()
        s[col] = pd.to_numeric(s[col], errors="coerce")
        s = s[["_Date", col]].dropna().sort_values("_Date")
        s = s.drop_duplicates("_Date", keep="last")
        if s.empty:
            return base
        merged = pd.merge_asof(
            base[["_Date"]].sort_values("_Date"),
            s,
            on="_Date",
            direction="backward"
        )
        base[col] = pd.to_numeric(merged[col].to_numpy(), errors="coerce")
        return base

    if per_df is not None and not per_df.empty:
        p = per_df.copy()
        if "Date" in p.columns:
            p["_Date"] = to_dt_index(p["Date"]).normalize()
        else:
            p["_Date"] = to_dt_index(p.index).normalize()
        for col in ["PER", "PER_PROXY", "EPS", "EPS_TTM", "EPS_PROXY"]:
            if col in p.columns:
                p[col] = pd.to_numeric(p[col], errors="coerce")
                chart = _merge_col(chart, p, col)

    for col in ["PER", "PER_PROXY", "EPS", "EPS_TTM", "EPS_PROXY"]:
        if col not in chart.columns:
            chart[col] = np.nan
        chart[col] = pd.to_numeric(chart[col], errors="coerce")

    if chart["PER"].dropna().empty:
        eps_col = None
        if chart["EPS"].dropna().shape[0] > 0:
            eps_col = "EPS"
        elif chart["EPS_TTM"].dropna().shape[0] > 0:
            eps_col = "EPS_TTM"
        if eps_col:
            chart[eps_col] = chart[eps_col].ffill()
            chart["PER"] = chart["Price"] / chart[eps_col].replace(0, np.nan)

    chart["PER"] = chart["PER"].where((chart["PER"] > 0) & (chart["PER"] < 500))
    chart["PER_PROXY"] = chart["PER_PROXY"].where((chart["PER_PROXY"] > 0) & (chart["PER_PROXY"] < 500))

    has_actual_per = chart["PER"].dropna().shape[0] > 0
    has_proxy_per = chart["PER_PROXY"].dropna().shape[0] > 0
    chart["PER_DISPLAY"] = chart["PER"].combine_first(chart["PER_PROXY"])
    has_per_display = chart["PER_DISPLAY"].dropna().shape[0] > 0

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
            chart["Risk"] = pd.to_numeric(merged["Risk"].to_numpy(), errors="coerce")
    if "Risk" not in chart.columns:
        chart["Risk"] = np.nan

    buy_dd = -0.12
    if asset_rule_key in ASSET_RULES:
        try:
            buy_dd = ASSET_RULES[asset_rule_key]["mdd_levels"][1] / 100.0
        except Exception:
            buy_dd = -0.12
    sig = build_signals(df, buy_dd=buy_dd).copy()
    if sig is not None and not sig.empty:
        sig.index = to_dt_index(sig.index).normalize()
        sig = sig[~sig.index.duplicated(keep="last")]
        chart = chart.join(sig[[c for c in ["Buy", "Cash"] if c in sig.columns]], how="left")

    fig, (ax1, ax3) = plt.subplots(
        2, 1, figsize=(17.0, 9.6), sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.1]}
    )

    ax1.plot(chart.index, chart["Price"], color="#0057B8", linewidth=2.35, label="Price")
    ax1.plot(chart.index, chart["MA20"], color="#FF8C00", linewidth=1.35, label="MA20")
    ax1.plot(chart.index, chart["MA60"], color="#228B22", linewidth=1.35, label="MA60")
    ax1.plot(chart.index, chart["MA200"], color="#7B2CBF", linewidth=1.35, label="MA200")

    if "Buy" in chart.columns and chart["Buy"].notna().any():
        ax1.scatter(chart.index, chart["Buy"] * 0.975, color="#008000", marker="^", s=74, label="BUY candidate", zorder=5)
    if "Cash" in chart.columns and chart["Cash"].notna().any():
        ax1.scatter(chart.index, chart["Cash"] * 1.025, color="#FF0000", marker="v", s=74, label="Cash / overheat", zorder=5)

    ax1.set_ylabel("Price", color="#0057B8")
    ax1.tick_params(axis="y", labelcolor="#0057B8")
    ax1.grid(True, linestyle=":", alpha=0.35)

    ax2 = ax1.twinx()
    per_lines = []

    # Make proxy visible whenever it exists. Do not suppress it because actual exists.
    if has_proxy_per:
        ax2.plot(chart.index, chart["PER_PROXY"], color="#D62728", linewidth=1.85, linestyle="--", alpha=0.78, label="P/E proxy(current EPS)")
        per_lines.append(chart["PER_PROXY"])
    if has_actual_per:
        ax2.plot(chart.index, chart["PER"], color="#D62728", linewidth=2.55, linestyle="-", alpha=0.98, label="P/E actual")
        per_lines.append(chart["PER"])
        per_avg = chart["PER"].dropna().mean()
        if pd.notna(per_avg):
            ax2.axhline(per_avg, color="#D62728", linewidth=1.0, linestyle=":", alpha=0.45, label="P/E actual avg")
    elif has_proxy_per:
        proxy_avg = chart["PER_PROXY"].dropna().mean()
        if pd.notna(proxy_avg):
            ax2.axhline(proxy_avg, color="#D62728", linewidth=1.0, linestyle=":", alpha=0.45, label="P/E proxy avg")
    elif has_per_display:
        ax2.plot(chart.index, chart["PER_DISPLAY"], color="#D62728", linewidth=2.25, linestyle="-", alpha=0.98, label="P/E display")
        per_lines.append(chart["PER_DISPLAY"])
    else:
        ax2.text(0.99, 0.95, "P/E line: N/A", transform=ax2.transAxes, ha="right", va="top", color="#D62728")

    if per_lines:
        vals = pd.concat([s.dropna() for s in per_lines if s is not None and not s.dropna().empty])
        vals = vals[(vals > 0) & (vals < 500)]
        if not vals.empty:
            q03, q97 = vals.quantile([0.03, 0.97])
            span = max(float(q97 - q03), 1.0)
            ax2.set_ylim(max(0, q03 - span * 0.25), q97 + span * 0.25)

    ax2.set_ylabel("P/E", color="#D62728")
    ax2.tick_params(axis="y", labelcolor="#D62728")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=7)
    ax1.set_title(f"{ticker} Price + P/E + MDD + Market Risk", fontweight="bold")

    ax3.plot(chart.index, chart["DD"] * 100, color="#8B0000", linewidth=1.6, label="Current DD")
    for y, name in [(-8, "Watch -8%"), (-12, "Buy zone -12%"), (-15, "Deep -15%"), (-20, "Risk -20%")]:
        ax3.axhline(y, color="#9ED0FF", linestyle="--", linewidth=0.85, alpha=0.8, label=name)
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

    fig.subplots_adjust(left=0.065, right=0.885, top=0.93, bottom=0.085, hspace=0.08)
    return fig, chart


def build_per_display_chart_df(price_df, per_df):
    """Create dedicated Price + P/E display frame.

    Returns a frame where actual P/E and proxy P/E are both merged across the full
    selected price range. Proxy is not suppressed when actual exists.
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
    base["PER_ACTUAL"] = base["PER_ACTUAL"].where((base["PER_ACTUAL"] > 0) & (base["PER_ACTUAL"] < 500))
    base["PER_PROXY_VIEW"] = base["PER_PROXY_VIEW"].where((base["PER_PROXY_VIEW"] > 0) & (base["PER_PROXY_VIEW"] < 500))
    base["PER_DISPLAY"] = base["PER_ACTUAL"].combine_first(base["PER_PROXY_VIEW"])

    has_actual = base["PER_ACTUAL"].dropna().shape[0] > 0
    has_proxy = base["PER_PROXY_VIEW"].dropna().shape[0] > 0

    if has_actual and has_proxy:
        label = f"P/E actual {base['PER_ACTUAL'].dropna().shape[0]}pts + proxy {base['PER_PROXY_VIEW'].dropna().shape[0]}pts"
    elif has_actual:
        label = f"P/E actual only {base['PER_ACTUAL'].dropna().shape[0]}pts"
    elif has_proxy:
        label = f"P/E proxy only {base['PER_PROXY_VIEW'].dropna().shape[0]}pts"
    else:
        label = f"PER/PER_PROXY 없음. columns={list(per_df.columns)}"

    return base, label


def plot_dedicated_per_chart(price_df, per_df, ticker):
    """Dedicated Price + P/E chart.

    This chart is intentionally simple for debugging:
    - Blue = price
    - Red solid = actual P/E when available
    - Red dashed = proxy P/E across the full period when available
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
            linewidth=1.95,
            linestyle="--",
            alpha=0.82,
            label="P/E proxy(current EPS)"
        )
        per_lines.append(chart["PER_PROXY_VIEW"])
    if has_actual:
        ax2.plot(
            chart.index,
            chart["PER_ACTUAL"],
            color="#D62728",
            linewidth=2.65,
            linestyle="-",
            alpha=0.98,
            label="P/E actual"
        )
        per_lines.append(chart["PER_ACTUAL"])
        per_avg = chart["PER_ACTUAL"].dropna().mean()
        if pd.notna(per_avg):
            ax2.axhline(per_avg, color="#D62728", linestyle=":", linewidth=1.0, alpha=0.45, label="P/E actual avg")
    else:
        proxy_avg = chart["PER_PROXY_VIEW"].dropna().mean()
        if pd.notna(proxy_avg):
            ax2.axhline(proxy_avg, color="#D62728", linestyle=":", linewidth=1.0, alpha=0.45, label="P/E proxy avg")

    vals = pd.concat([s.dropna() for s in per_lines if s is not None and not s.dropna().empty]) if per_lines else pd.Series(dtype=float)
    vals = vals[(vals > 0) & (vals < 500)]
    if not vals.empty:
        q03, q97 = vals.quantile([0.03, 0.97])
        span = max(float(q97 - q03), 1.0)
        ax2.set_ylim(max(0, q03 - span * 0.25), q97 + span * 0.25)

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
        "실선이 중간부터 시작해도 점선은 전체 선택 기간에 보여야 정상입니다."
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




def classify_valuation_metric(kind, value):
    """Return compact valuation interpretation.
    This is a reference filter only; it does not affect MDD/Buy Score logic.
    """
    v = safe_float(value)
    if v is None:
        return "N/A", "데이터 없음", "판단 불가"

    kind = str(kind).upper()
    if kind in ["FORWARD_PE", "FWD_PE"]:
        if v <= 0:
            return "해석 제외", "예상PER 0 이하", "이익 추정치 확인 필요"
        if v <= 15:
            return "낮음", "15배 이하", "밸류 부담 낮음"
        if v <= 30:
            return "보통", "15~30배", "성장주 기준 무난"
        if v <= 50:
            return "부담", "30~50배", "성장 기대 반영, 추격 주의"
        return "고평가", "50배 초과", "고평가·추격 주의"

    if kind in ["TTM_PE", "KRX_PE", "ACTUAL_PE", "PER", "PROXY_PE"]:
        if v <= 0:
            return "해석 제외", "PER 0 이하", "적자·일회성 이익 가능성 확인"
        if v <= 10:
            return "낮음", "10배 이하", "저평가 가능. 단, 경기민감주는 이익 피크 여부 확인"
        if v <= 20:
            return "양호~보통", "10~20배", "일반주 기준 부담 제한적"
        if v <= 35:
            return "성장 반영", "20~35배", "성장 기대 반영. 실적 상향 필요"
        if v <= 50:
            return "부담", "35~50배", "밸류 부담 확대, 눌림 확인 우선"
        return "고평가", "50배 초과", "고평가·추격 주의"

    if kind in ["PS", "P/S"]:
        if v <= 0:
            return "해석 제외", "P/S 0 이하", "매출 데이터 확인 필요"
        if v <= 3:
            return "낮음", "3배 이하", "매출 대비 부담 낮음"
        if v <= 10:
            return "보통~성장", "3~10배", "성장주 일반 구간"
        if v <= 30:
            return "고성장 반영", "10~30배", "고성장 기대 반영, 실적 확인 필요"
        return "과열 가능", "30배 초과", "매출 대비 과열 가능성, 추격 주의"

    if kind in ["PEG"]:
        if v <= 0:
            return "해석 제외", "PEG 0 이하", "성장률 추정치 확인 필요"
        if v <= 1:
            return "양호", "1 이하", "성장 대비 밸류 양호"
        if v <= 2:
            return "보통", "1~2", "성장 대비 보통"
        return "부담", "2 초과", "성장 대비 밸류 부담"

    return "N/A", "기준 없음", "해석 기준 없음"


def latest_per_from_series(per_df, col):
    if per_df is None or per_df.empty or col not in per_df.columns:
        return None
    try:
        s = pd.to_numeric(per_df[col], errors="coerce").dropna()
        s = s[(s > 0) & (s < 500)]
        if s.empty:
            return None
        return safe_float(s.iloc[-1])
    except Exception:
        return None


def build_valuation_guide_table(val, per_df=None, market=""):
    """Readable guide: what level is cheap/normal/expensive.
    Does not change calculations; display only.
    """
    if market in ["US_INDEX", "KR_INDEX"]:
        return pd.DataFrame()

    rows = []
    metrics = [
        ("Forward P/E", "FORWARD_PE", val.get("fwd_pe"), "미국 개별주 예상 이익 기준. 한국 종목은 없을 수 있음"),
        ("TTM / KRX P/E", "TTM_PE", val.get("ttm_pe"), "현재 이익 기준. 경기민감주는 낮아도 이익 피크면 함정 가능"),
        ("P/S", "PS", val.get("ps"), "매출 대비 가격. 성장주 비교에 보조 사용"),
        ("PEG", "PEG", val.get("peg"), "성장률 대비 PER. 1 이하가 가장 양호한 편"),
    ]

    actual_latest = latest_per_from_series(per_df, "PER")
    proxy_latest = latest_per_from_series(per_df, "PER_PROXY")
    if actual_latest is not None:
        metrics.append(("Actual PER 최신", "ACTUAL_PE", actual_latest, "실제 EPS 변화가 반영된 시계열 PER. 가장 우선"))
    if proxy_latest is not None:
        metrics.append(("Proxy PER 최신", "PROXY_PE", proxy_latest, "현재 EPS 고정 기준. 저평가 판단에는 단독 사용 금지"))

    seen = set()
    for name, kind, value, note in metrics:
        key = (name, fmt_num(value, 4))
        if key in seen:
            continue
        seen.add(key)
        grade, good_range, comment = classify_valuation_metric(kind, value)
        rows.append({
            "항목": name,
            "현재값": value,
            "판정": grade,
            "좋은 기준/해석 구간": good_range,
            "해석": comment,
            "주의": note,
        })
    return pd.DataFrame(rows)


def format_valuation_guide_table(df):
    out = df.copy()
    if "현재값" in out.columns:
        out["현재값"] = out["현재값"].apply(lambda x: fmt_num(x, 2))
    return out

# =========================================================
# Trading helper tables
# =========================================================

# =========================================================
# Asset-type MDD rules and correction score
# =========================================================
ASSET_RULES = {
    "US_SP500": {
        "label": "미국 S&P500 대표지수",
        "mdd_levels": [-5, -7, -10, -15, -20],
        "actions": ["관찰", "대기", "1차 매수", "2차 매수", "적극 매수", "위기 확인"],
        "memo": "대표지수는 회복력이 높아 테마 ETF보다 얕은 MDD에서도 분할 관심 가능"
    },
    "US_NASDAQ100": {
        "label": "미국 나스닥100 대표지수",
        "mdd_levels": [-5, -8, -12, -18, -25],
        "actions": ["관찰", "대기", "1차 매수", "2차 매수", "적극 매수", "위기 확인"],
        "memo": "성장주 비중이 높아 S&P500보다 한 단계 깊은 조정 기준 적용"
    },
    "US_AI_THEME": {
        "label": "미국 AI·성장 테마 ETF",
        "mdd_levels": [-7, -10, -12, -15, -20],
        "actions": ["관찰", "대기", "1차 매수", "2차 매수", "적극 매수", "과매도 확인"],
        "memo": "AI 테마는 주도주 유지와 EPS 상향 여부를 같이 확인"
    },
    "US_SEMICONDUCTOR": {
        "label": "미국 반도체 ETF/종목군",
        "mdd_levels": [-8, -12, -15, -20, -25],
        "actions": ["관찰", "대기", "1차 매수", "2차 매수", "적극 매수", "과매도 확인"],
        "memo": "SOXX/SMH와 NVDA·AVGO·AMD·MU 흐름 확인 필요"
    },
    "US_MEMORY_HBM": {
        "label": "미국/글로벌 메모리·HBM 종목군",
        "mdd_levels": [-10, -15, -20, -25, -30],
        "actions": ["관찰", "대기", "1차 매수", "2차 매수", "적극 매수", "사이클 점검"],
        "memo": "메모리/HBM은 사이클 변동성이 커서 -15% 이상부터 의미 있는 1차권"
    },
    "US_POWER_INFRA": {
        "label": "미국 전력·AI 인프라 ETF",
        "mdd_levels": [-7, -10, -12, -18, -20],
        "actions": ["관찰", "대기", "1차 매수", "2차 매수", "적극 매수", "수주/실적 점검"],
        "memo": "데이터센터 전력·냉각·변압기 수요 지속 여부 확인"
    },
    "US_VALUECHAIN": {
        "label": "미국 빅테크 밸류체인 ETF/종목군",
        "mdd_levels": [-8, -12, -15, -20, -25],
        "actions": ["관찰", "대기", "1차 매수", "2차 매수", "적극 매수", "본주 흐름 확인"],
        "memo": "엔비디아·구글·브로드컴·테슬라 등 본주 흐름 확인 필요"
    },
    "KR_KOSPI": {
        "label": "한국 KOSPI 지수/대형 ETF",
        "mdd_levels": [-5, -8, -12, -18, -25],
        "actions": ["관찰", "대기", "조건부 1차", "2차 매수", "적극 매수", "환율/외국인 확인"],
        "memo": "한국 대형주는 외국인 현물·선물, 원/달러, 연기금 매도를 함께 확인"
    },
    "KR_KOSDAQ": {
        "label": "한국 KOSDAQ 지수/성장 ETF",
        "mdd_levels": [-7, -12, -18, -25, -35],
        "actions": ["관찰", "대기", "조건부 1차", "2차 매수", "적극 매수", "거래대금 확인"],
        "memo": "코스닥은 변동성이 커서 거래대금 회복 전까지 보수적으로 판단"
    },
    "KR_SEMICON_LARGE": {
        "label": "한국 반도체 대형 ETF/종목군",
        "mdd_levels": [-5, -8, -10, -15, -20],
        "actions": ["관찰", "대기", "조건부 1차", "2차 매수", "적극 매수", "외국인 흡수 확인"],
        "memo": "국내 반도체 대형주는 -10% 전후부터 조건부 1차권. 외국인 흡수와 SOXX/메모리 본주 흐름 확인"
    },
    "KR_SEMICON_EQUIP": {
        "label": "한국 반도체 소부장·장비 ETF/종목군",
        "mdd_levels": [-8, -12, -15, -20, -25],
        "actions": ["관찰", "대기", "조건부 1차", "2차 매수", "적극 매수", "거래대금 확인"],
        "memo": "소부장·장비는 대형 반도체보다 변동성이 커서 -12~-15% 이상과 거래대금 유지가 필요"
    },
    "KR_BATTERY": {
        "label": "한국 2차전지 ETF/종목군",
        "mdd_levels": [-10, -15, -25, -35, -45],
        "actions": ["관찰", "대기", "조건부 1차", "2차 매수", "적극 매수", "실적 턴 확인"],
        "memo": "2차전지는 사이클·실적 변동성이 커서 -15% 이상부터 1차권. 실적 턴 확인 전 추격 금지"
    },
    "KR_POWER": {
        "label": "한국 전력기기 ETF/종목군",
        "mdd_levels": [-7, -12, -18, -25, -35],
        "actions": ["관찰", "대기", "조건부 1차", "2차 매수", "적극 매수", "수주/실적 확인"],
        "memo": "전력기기는 수주·실적 유지가 전제. -12% 이상부터 조건부 접근"
    },
    "KR_BIO": {
        "label": "한국 바이오 ETF/종목군",
        "mdd_levels": [-10, -15, -25, -35, -45],
        "actions": ["관찰", "대기", "조건부 1차", "2차 매수", "적극 매수", "임상/이벤트 확인"],
        "memo": "바이오는 이벤트 리스크가 커서 MDD만으로 매수 금지. 임상·허가·수급 확인 필요"
    },
    "KR_GROWTH": {
        "label": "한국 코스닥 성장 ETF/종목군",
        "mdd_levels": [-7, -15, -20, -25, -30],
        "actions": ["관찰", "대기", "조건부 1차", "2차 매수", "적극 매수", "위험선호 회복 확인"],
        "memo": "코스닥 성장주는 위험선호 회복과 거래대금 증가가 전제"
    },
    "KR_THEME_HIGH_BETA": {
        "label": "한국 고변동 테마 ETF/종목군",
        "mdd_levels": [-10, -15, -20, -25, -35],
        "actions": ["관찰", "대기", "조건부 1차", "2차 매수", "적극 매수", "테마 생존 확인"],
        "memo": "로봇·우주 등 고변동 테마는 더 깊은 MDD와 거래대금 회복 필요"
    },
    "US_INDIVIDUAL": {
        "label": "미국 개별주",
        "mdd_levels": [-8, -12, -18, -25, -35],
        "actions": ["관찰", "대기", "1차 매수", "2차 매수", "적극 매수", "실적/가이던스 확인"],
        "memo": "개별주는 ETF보다 보수적으로 판단. 실적·가이던스·밸류에이션 확인 필요"
    },
    "KR_INDIVIDUAL": {
        "label": "한국 개별주",
        "mdd_levels": [-8, -12, -18, -25, -35],
        "actions": ["관찰", "대기", "조건부 1차", "2차 매수", "적극 매수", "공시/수급 확인"],
        "memo": "한국 개별주는 공시·수급·거래대금 리스크가 ETF보다 크므로 조건부 판단"
    },
}

SP500_TICKERS = {"SPY", "VOO", "IVV", "SPLG"}
NASDAQ_TICKERS = {"QQQ", "QQQM", "TQQQ", "QLD"}
AI_THEME_TICKERS = {"AIQ", "BOTZ", "ROBO", "ARKQ", "IRBO", "CIBR"}
SEMICONDUCTOR_TICKERS = {"SOXX", "SMH", "SOXL", "XSD", "NVDA", "AVGO", "AMD", "TSM", "ASML", "MRVL", "ARM"}
MEMORY_HBM_TICKERS = {"MU", "WDC", "STX", "NXPI", "LRCX", "AMAT"}
POWER_INFRA_TICKERS = {"VRT", "ETN", "PWR", "GEV", "NEE", "CEG", "GNRC", "EME"}
VALUECHAIN_TICKERS = {"GOOGL", "GOOG", "TSLA", "MSFT", "AMZN", "ORCL", "DELL", "HPE", "SMCI"}

KR_KOSPI_ETF_CODES = {"069500", "102110", "152100", "278530", "226490", "360750", "379810", "367380", "381180"}
KR_KOSDAQ_ETF_CODES = {"229200", "233740", "251340", "364960", "376250"}
KR_SEMICON_LARGE_CODES = {"005930", "000660", "091160", "395160"}
KR_SEMICON_EQUIP_CODES = {"042700", "039030", "058470", "108320", "240810", "036930", "403870", "471990", "396500", "487130"}
KR_BATTERY_CODES = {"373220", "006400", "066970", "247540"}
KR_POWER_CODES = {"010120", "267260", "272210", "103590", "491010"}
KR_BIO_CODES = {"068270", "207940"}
KR_GROWTH_CODES = {"229200", "233740", "251340", "364960", "376250"}
KR_HIGH_BETA_CODES = {"277810", "108490"}

# 국내상장 해외 ETF는 상장시장이 KR이어도 기초자산 기준 MDD 룰을 우선 적용한다.
KR_LISTED_US_NASDAQ_CODES = {"379810", "360750", "367380", "381180"}
KR_LISTED_US_AI_CODES = {"465580"}
KR_LISTED_US_MEMORY_CODES = {"0174B0"}
KR_LISTED_US_POWER_CODES = {"491010"}
KR_LISTED_US_VALUECHAIN_CODES = {"483320", "483340", "457480", "0051A0"}


def asset_rule_options():
    return ["AUTO"] + list(ASSET_RULES.keys())


def asset_rule_label(key):
    if key == "AUTO":
        return "자동 분류"
    return ASSET_RULES.get(key, {}).get("label", str(key))


def normalize_asset_code(ticker):
    t = str(ticker).upper().strip()
    if t.endswith(".KS") or t.endswith(".KQ"):
        return t.split(".")[0]
    return t


def auto_classify_asset(market, ticker, display_name=""):
    t = str(ticker).upper().strip()
    code = normalize_asset_code(ticker)
    name = str(display_name).upper()
    name_kr = str(display_name)

    if market == "US_INDEX":
        if t in ["^GSPC", "SPY", "VOO"]:
            return "US_SP500", "지수/대표 ETF 매핑"
        if t in ["^IXIC", "^NDX", "QQQ", "QQQM"]:
            return "US_NASDAQ100", "지수/대표 ETF 매핑"
        if t in ["^SOX"]:
            return "US_SEMICONDUCTOR", "반도체 지수 매핑"
        return "US_SP500", "미국 지수 기본값"

    if market == "KR_INDEX":
        if t == "KS11":
            return "KR_KOSPI", "KOSPI 지수 매핑"
        if t == "KQ11":
            return "KR_KOSDAQ", "KOSDAQ 지수 매핑"
        return "KR_KOSPI", "한국 지수 기본값"

    if market == "US":
        if t in SP500_TICKERS:
            return "US_SP500", "S&P500 대표 ETF"
        if t in NASDAQ_TICKERS:
            return "US_NASDAQ100", "나스닥100 대표 ETF"
        if t in SEMICONDUCTOR_TICKERS:
            return "US_SEMICONDUCTOR", "미국 반도체 유니버스"
        if t in MEMORY_HBM_TICKERS:
            return "US_MEMORY_HBM", "메모리/HBM 유니버스"
        if t in POWER_INFRA_TICKERS:
            return "US_POWER_INFRA", "전력/AI 인프라 유니버스"
        if t in VALUECHAIN_TICKERS:
            return "US_VALUECHAIN", "빅테크 밸류체인 유니버스"
        if t in AI_THEME_TICKERS or any(k in name for k in ["AI", "ROBOT", "ROBO", "BOTZ", "CLOUD", "BIGTECH", "GROWTH"]):
            return "US_AI_THEME", "AI/성장 테마명 매핑"
        return "US_INDIVIDUAL", "미국 개별주 기본값"

    # KR stocks and ETFs. 국내상장 해외 ETF는 먼저 기초자산 기준으로 분류한다.
    if code in KR_LISTED_US_NASDAQ_CODES or any(k in name_kr for k in ["미국나스닥", "나스닥100", "NASDAQ100", "나스닥성장"]):
        return "US_NASDAQ100", "국내상장 미국 나스닥/성장 ETF: 기초자산 기준"
    if code in KR_LISTED_US_MEMORY_CODES or any(k in name_kr for k in ["글로벌AI메모리", "글로벌 AI 메모리", "HBM", "글로벌메모리", "메모리반도체"]):
        return "US_MEMORY_HBM", "국내상장 글로벌 메모리/HBM ETF: 기초자산 기준"
    if code in KR_LISTED_US_POWER_CODES or any(k in name_kr for k in ["글로벌AI전력", "글로벌 AI 전력", "AI전력인프라", "AI 전력인프라"]):
        return "US_POWER_INFRA", "국내상장 글로벌 전력·AI 인프라 ETF: 기초자산 기준"
    if code in KR_LISTED_US_VALUECHAIN_CODES or any(k in name_kr for k in ["엔비디아밸류체인", "구글밸류체인", "테슬라밸류체인", "브로드컴밸류체인", "밸류체인액티브"]):
        return "US_VALUECHAIN", "국내상장 미국 밸류체인 ETF: 기초자산 기준"
    if code in KR_LISTED_US_AI_CODES or any(k in name_kr for k in ["글로벌AI인공지능", "글로벌 AI", "미국빅테크", "빅테크TOP"]):
        return "US_AI_THEME", "국내상장 미국/글로벌 AI ETF: 기초자산 기준"

    if code in KR_KOSPI_ETF_CODES or any(k in name_kr for k in ["KOSPI", "코스피", "200", "대형"]):
        return "KR_KOSPI", "한국 대형 ETF/이름 매핑"
    if code in KR_KOSDAQ_ETF_CODES or any(k in name_kr for k in ["KOSDAQ", "코스닥"]):
        return "KR_KOSDAQ", "한국 코스닥 ETF/이름 매핑"
    if code in KR_SEMICON_LARGE_CODES or any(k in name_kr for k in ["삼성전자", "하이닉스", "반도체대형", "반도체 대형"]):
        return "KR_SEMICON_LARGE", "한국 반도체 대형 매핑"
    if code in KR_SEMICON_EQUIP_CODES or any(k in name_kr for k in ["소부장", "장비", "AI인프라", "반도체핵심장비"]):
        return "KR_SEMICON_EQUIP", "한국 반도체 소부장/장비 매핑"
    if code in KR_BATTERY_CODES or any(k in name_kr for k in ["2차전지", "배터리", "양극재"]):
        return "KR_BATTERY", "한국 2차전지 매핑"
    if code in KR_POWER_CODES or any(k in name_kr for k in ["전력기기", "전력", "변압기", "전선"]):
        return "KR_POWER", "한국 전력기기 매핑"
    if code in KR_BIO_CODES or any(k in name_kr for k in ["바이오", "헬스케어", "임상"]):
        return "KR_BIO", "한국 바이오 매핑"
    if code in KR_GROWTH_CODES or any(k in name_kr for k in ["성장", "코스닥성장"]):
        return "KR_GROWTH", "한국 성장 ETF 매핑"
    if code in KR_HIGH_BETA_CODES or any(k in name_kr for k in ["로봇", "우주"]):
        return "KR_THEME_HIGH_BETA", "한국 고변동 테마 매핑"
    return "KR_INDIVIDUAL", "한국 개별주 기본값"


def get_mdd_action(mdd_pct, asset_type):
    """mdd_pct is percentage value, e.g. -11.5."""
    rule = ASSET_RULES.get(asset_type, ASSET_RULES["US_SP500"])
    levels = rule["mdd_levels"]
    actions = rule["actions"]
    mdd_pct = safe_float(mdd_pct)
    if mdd_pct is None:
        stage = "판단 보류"
    elif mdd_pct > levels[0]:
        stage = actions[0]
    elif mdd_pct > levels[1]:
        stage = actions[1]
    elif mdd_pct > levels[2]:
        stage = actions[2]
    elif mdd_pct > levels[3]:
        stage = actions[3]
    elif mdd_pct > levels[4]:
        stage = actions[4]
    else:
        stage = actions[5]
    return {"asset_label": rule["label"], "mdd_pct": mdd_pct, "stage": stage, "levels": levels, "rule": rule}


def us_signal_score(
    rate_stable=False, vix_down=False, eps_stable=False, index_reclaim_20d=False, leaders_strong=False,
    rate_spike=False, vix_spike=False, eps_down=False, leaders_break_20d=False,
):
    score = 0
    score += 1 if rate_stable else 0
    score += 1 if vix_down else 0
    score += 1 if eps_stable else 0
    score += 1 if index_reclaim_20d else 0
    score += 1 if leaders_strong else 0
    score -= 2 if rate_spike else 0
    score -= 1 if vix_spike else 0
    score -= 2 if eps_down else 0
    score -= 2 if leaders_break_20d else 0
    return int(score)


def kr_signal_score(
    foreign_spot_buy=False, foreign_futures_buy=False, fx_stable=False, volume_up=False, leaders_close_strong=False,
    pension_heavy_sell=False, foreign_double_sell=False, fx_spike=False, volume_down=False, upper_tail_repeat=False,
):
    score = 0
    score += 2 if foreign_spot_buy else 0
    score += 1 if foreign_futures_buy else 0
    score += 1 if fx_stable else 0
    score += 1 if volume_up else 0
    score += 1 if leaders_close_strong else 0
    score -= 2 if pension_heavy_sell else 0
    score -= 3 if foreign_double_sell else 0
    score -= 2 if fx_spike else 0
    score -= 1 if volume_down else 0
    score -= 1 if upper_tail_repeat else 0
    return int(score)


def final_buy_decision(stage, signal_score):
    if stage in ["관찰"]:
        return "관망"
    if stage in ["대기"]:
        return "소액 관심" if signal_score >= 3 else "대기"
    if stage in ["1차 매수", "조건부 1차"]:
        if signal_score >= 2:
            return "1차 매수 가능"
        if signal_score >= 0:
            return "대기"
        return "매수 금지"
    if stage in ["2차 매수"]:
        return "2차 매수 가능" if signal_score >= 1 else "분할 대기"
    if stage in ["적극 매수"]:
        return "적극 분할매수" if signal_score >= 0 else "위험 확인 후 분할"
    return "판단 보류"


def latest_safe(series, default=None):
    try:
        s = pd.to_numeric(series, errors="coerce").dropna()
        if s.empty:
            return default
        return float(s.iloc[-1])
    except Exception:
        return default


@st.cache_data(ttl=1800)
def load_usdkrw_series(start_date):
    if YF_OK:
        try:
            df = yf.Ticker("KRW=X").history(start=pd.to_datetime(start_date).strftime("%Y-%m-%d"), auto_adjust=True)
            if df is not None and not df.empty and "Close" in df.columns:
                df = df.copy()
                df.index = to_dt_index(df.index)
                return df[["Close"]].rename(columns={"Close": "USDKRW"}), "OK:yfinance KRW=X"
        except Exception as e:
            err1 = repr(e)
    else:
        err1 = YF_ERR
    if FDR_OK:
        try:
            df = fdr.DataReader("USD/KRW", pd.to_datetime(start_date).strftime("%Y-%m-%d"))
            if df is not None and not df.empty and "Close" in df.columns:
                df = df.copy()
                df.index = to_dt_index(df.index)
                return df[["Close"]].rename(columns={"Close": "USDKRW"}), "OK:FDR USD/KRW"
        except Exception as e:
            return pd.DataFrame(), f"USD/KRW 조회 실패: {err1} / {repr(e)}"
    return pd.DataFrame(), f"USD/KRW 조회 실패: {err1}"


def detect_upper_tail_repeat(df, lookback=5):
    try:
        x = df.tail(lookback).copy()
        rng = (x["High"] - x["Low"]).replace(0, np.nan)
        upper = (x["High"] - x["Close"]) / rng
        return bool((upper >= 0.45).sum() >= 3)
    except Exception:
        return False


def compute_auto_correction_flags(df, risk_df, risk_label, market, start_date):
    latest = df.iloc[-1]
    close = safe_float(latest.get("Close"))
    ma20 = safe_float(latest.get("MA20"))
    rsi = safe_float(latest.get("RSI"))
    volr = safe_float(latest.get("Volume_Ratio"))
    day_ret = None
    if len(df) >= 2 and safe_float(df["Close"].iloc[-2]) not in (None, 0):
        day_ret = close / safe_float(df["Close"].iloc[-2]) - 1

    vix_latest = latest_safe(risk_df["Risk"]) if risk_df is not None and not risk_df.empty and "Risk" in risk_df.columns else None
    vix_prev = None
    if risk_df is not None and not risk_df.empty and "Risk" in risk_df.columns and len(risk_df.dropna()) >= 6:
        try:
            vix_prev = float(pd.to_numeric(risk_df["Risk"], errors="coerce").dropna().iloc[-6])
        except Exception:
            pass

    index_reclaim_20d = close is not None and ma20 is not None and close >= ma20
    leaders_strong = bool(index_reclaim_20d and (day_ret is not None and day_ret > 0) and (rsi is None or rsi < 70))
    leaders_break_20d = close is not None and ma20 is not None and close < ma20
    vix_down = bool(risk_label == "VIX" and vix_latest is not None and vix_prev is not None and vix_latest < vix_prev)
    vix_spike = bool(risk_label == "VIX" and vix_latest is not None and vix_latest >= 25)

    kr_fx_stable = False
    kr_fx_spike = False
    fx_note = "USD/KRW 미조회"
    if str(market).startswith("KR"):
        fx_df, fx_status = load_usdkrw_series(start_date)
        fx_note = fx_status
        if not fx_df.empty and len(fx_df) >= 6:
            fx_now = latest_safe(fx_df["USDKRW"])
            fx_5 = safe_float(fx_df["USDKRW"].iloc[-6])
            if fx_now is not None and fx_5 not in (None, 0):
                fx_chg = fx_now / fx_5 - 1
                kr_fx_stable = fx_chg <= 0.005
                kr_fx_spike = fx_chg >= 0.015
                fx_note = f"USD/KRW 5일 변화 {fx_chg*100:.1f}%"

    return {
        "rate_stable": False,
        "vix_down": vix_down,
        "eps_stable": False,
        "index_reclaim_20d": bool(index_reclaim_20d),
        "leaders_strong": bool(leaders_strong),
        "rate_spike": False,
        "vix_spike": vix_spike,
        "eps_down": False,
        "leaders_break_20d": bool(leaders_break_20d),
        "foreign_spot_buy": False,
        "foreign_futures_buy": False,
        "fx_stable": bool(kr_fx_stable),
        "volume_up": bool(volr is not None and volr >= 1.15),
        "leaders_close_strong": bool(index_reclaim_20d and day_ret is not None and day_ret > 0),
        "pension_heavy_sell": False,
        "foreign_double_sell": False,
        "fx_spike": bool(kr_fx_spike),
        "volume_down": bool(volr is not None and volr <= 0.80),
        "upper_tail_repeat": detect_upper_tail_repeat(df),
        "fx_note": fx_note,
    }


def correction_score_details(asset_rule_key, market, flags):
    rows = []
    if str(asset_rule_key).startswith("KR_"):
        mapping = [
            ("외국인 현물 순매수", "foreign_spot_buy", 2),
            ("외국인 선물 순매수", "foreign_futures_buy", 1),
            ("원/달러 안정 또는 하락", "fx_stable", 1),
            ("거래대금 증가", "volume_up", 1),
            ("주도주/가격 종가 강세", "leaders_close_strong", 1),
            ("연기금 대규모 순매도", "pension_heavy_sell", -2),
            ("외국인 현물·선물 동반 매도", "foreign_double_sell", -3),
            ("원/달러 급등", "fx_spike", -2),
            ("거래대금 감소", "volume_down", -1),
            ("장중 윗꼬리 반복", "upper_tail_repeat", -1),
        ]
    else:
        mapping = [
            ("금리 안정 또는 하락", "rate_stable", 1),
            ("VIX 하락 전환", "vix_down", 1),
            ("EPS 전망 유지/상향", "eps_stable", 1),
            ("지수/종목 MA20 회복", "index_reclaim_20d", 1),
            ("주도주 강세", "leaders_strong", 1),
            ("금리 급등", "rate_spike", -2),
            ("VIX 재급등", "vix_spike", -1),
            ("EPS 하향", "eps_down", -2),
            ("주도주 20일선 이탈", "leaders_break_20d", -2),
        ]
    score = 0
    for label, key, pts in mapping:
        on = bool(flags.get(key, False))
        applied = pts if on else 0
        score += applied
        rows.append({"조건": label, "충족": "Y" if on else "-", "점수": applied})
    return int(score), pd.DataFrame(rows)


def build_asset_mdd_rule_table(asset_rule_key):
    rule = ASSET_RULES.get(asset_rule_key, ASSET_RULES["US_SP500"])
    levels = rule["mdd_levels"]
    actions = rule["actions"]
    ranges = [
        f"0 ~ {levels[0]}%",
        f"{levels[0]} ~ {levels[1]}%",
        f"{levels[1]} ~ {levels[2]}%",
        f"{levels[2]} ~ {levels[3]}%",
        f"{levels[3]} ~ {levels[4]}%",
        f"{levels[4]}% 이하",
    ]
    meanings = [
        "정상 눌림 / 초기 관찰",
        "약한 조정 / 대기",
        "1차 매수권",
        "2차 매수권",
        "적극 매수권",
        "위기성 구간 / 매크로 확인",
    ]
    return pd.DataFrame({"MDD 구간": ranges, "단계": actions, "의미": meanings})


def make_mdd_comment(asset_label, mdd_pct, stage, signal_score, final_decision, rule_memo=""):
    return (
        f"현재 자산 유형: {asset_label}\n"
        f"현재 MDD: {mdd_pct:.1f}%\n"
        f"MDD 기준 단계: {stage}\n"
        f"보정 점수: {signal_score:+d}점\n"
        f"최종 판단: {final_decision}\n"
        f"기준 메모: {rule_memo}"
    )

def _price_at_drawdown_from_peak(peak_price, dd_level):
    v = safe_float(peak_price)
    if v is None:
        return None
    return v * (1 + dd_level)


def build_trading_snapshot(df, per_df=None, risk_df=None, risk_label=""):
    """Compact trading decision data. This does not change MDD/PER logic."""
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else latest
    close = safe_float(latest.get("Close"))
    peak = safe_float(latest.get("Peak"))
    dd = safe_float(latest.get("Current_Drawdown"))
    rsi = safe_float(latest.get("RSI"))
    volr = safe_float(latest.get("Volume_Ratio"))
    ma20 = safe_float(latest.get("MA20"))
    ma60 = safe_float(latest.get("MA60"))
    ma200 = safe_float(latest.get("MA200"))
    day_chg = None
    if safe_float(prev.get("Close")) not in (None, 0):
        day_chg = close / safe_float(prev.get("Close")) - 1

    ret_5 = ret_20 = None
    try:
        if len(df) >= 6:
            ret_5 = close / float(df["Close"].iloc[-6]) - 1
        if len(df) >= 21:
            ret_20 = close / float(df["Close"].iloc[-21]) - 1
    except Exception:
        pass

    actual_per = None
    proxy_per = None
    per_basis = "N/A"
    per_change_60 = None
    if per_df is not None and not per_df.empty:
        pwork = per_df.copy()
        if "PER" in pwork.columns and pd.to_numeric(pwork["PER"], errors="coerce").dropna().shape[0] >= 5:
            s = pd.to_numeric(pwork["PER"], errors="coerce").dropna()
            actual_per = safe_float(s.iloc[-1])
            per_basis = "Actual"
            n = min(60, len(s)-1)
            if n > 0 and safe_float(s.iloc[-n]) not in (None, 0):
                per_change_60 = actual_per / float(s.iloc[-n]) - 1
        if "PER_PROXY" in pwork.columns and pd.to_numeric(pwork["PER_PROXY"], errors="coerce").dropna().shape[0] >= 5:
            sp = pd.to_numeric(pwork["PER_PROXY"], errors="coerce").dropna()
            proxy_per = safe_float(sp.iloc[-1])
            if per_basis == "N/A":
                per_basis = "Proxy"
                n = min(60, len(sp)-1)
                if n > 0 and safe_float(sp.iloc[-n]) not in (None, 0):
                    per_change_60 = proxy_per / float(sp.iloc[-n]) - 1

    risk_latest = None
    if risk_df is not None and not risk_df.empty and "Risk" in risk_df.columns:
        try:
            risk_latest = safe_float(pd.to_numeric(risk_df["Risk"], errors="coerce").dropna().iloc[-1])
        except Exception:
            risk_latest = None

    # Decision: reference only, not buy recommendation.
    if close is None:
        action = "판단 보류"
        memo = "가격 데이터 없음"
    elif dd is not None and dd <= -0.12 and rsi is not None and rsi <= 45 and close >= (ma200 or -np.inf):
        action = "1차 소액 가능"
        memo = "MDD가 깊고 RSI 과열이 아님. 단, MA20 회복 전에는 분할/소액 기준."
    elif dd is not None and dd <= -0.12 and close < (ma20 or np.inf):
        action = "눌림 대기"
        memo = "낙폭은 있으나 MA20 아래. 반등 확인 전 추격 금지."
    elif rsi is not None and rsi >= 68 and dd is not None and dd > -0.08:
        action = "추격 금지"
        memo = "낙폭은 얕고 RSI 과열권. 신규 진입 손익비 불리."
    elif close < (ma60 or np.inf) and close < (ma20 or np.inf):
        action = "대기"
        memo = "단기·중기 추세가 모두 약함. 가격 안정 확인 필요."
    else:
        action = "관찰"
        memo = "강한 신호는 아님. 가격·PER·MDD 조합 확인 필요."

    return {
        "close": close,
        "peak": peak,
        "dd": dd,
        "rsi": rsi,
        "volr": volr,
        "ma20": ma20,
        "ma60": ma60,
        "ma200": ma200,
        "day_chg": day_chg,
        "ret_5": ret_5,
        "ret_20": ret_20,
        "actual_per": actual_per,
        "proxy_per": proxy_per,
        "per_basis": per_basis,
        "per_change_60": per_change_60,
        "risk_latest": risk_latest,
        "risk_label": risk_label,
        "action": action,
        "memo": memo,
    }


def build_mdd_level_table(df):
    latest = df.iloc[-1]
    peak = safe_float(latest.get("Peak"))
    close = safe_float(latest.get("Close"))
    rows = []
    for name, dd_level, meaning in [
        ("관찰 -8%", -0.08, "급락 전 초기 관심선"),
        ("1차 -12%", -0.12, "저점매수 후보 시작선"),
        ("깊은 눌림 -15%", -0.15, "분할매수 후보선"),
        ("위험 -20%", -0.20, "추세 훼손 확인선"),
    ]:
        price = _price_at_drawdown_from_peak(peak, dd_level)
        gap = None if close in (None, 0) or price is None else price / close - 1
        rows.append({
            "구분": name,
            "가격": price,
            "현재가 대비": gap,
            "의미": meaning,
        })
    return pd.DataFrame(rows)


def build_trade_level_table(df):
    latest = df.iloc[-1]
    close = safe_float(latest.get("Close"))
    ma20 = safe_float(latest.get("MA20"))
    ma60 = safe_float(latest.get("MA60"))
    ma200 = safe_float(latest.get("MA200"))
    peak = safe_float(latest.get("Peak"))
    dd12 = _price_at_drawdown_from_peak(peak, -0.12)
    dd15 = _price_at_drawdown_from_peak(peak, -0.15)
    rows = [
        {"항목": "현재가", "가격": close, "판단": "기준 가격"},
        {"항목": "MA20", "가격": ma20, "판단": "단기 반등/추세 확인선"},
        {"항목": "MA60", "가격": ma60, "판단": "중기 추세 확인선"},
        {"항목": "MA200", "가격": ma200, "판단": "장기 추세 훼손 여부"},
        {"항목": "MDD -12%", "가격": dd12, "판단": "1차 눌림 후보"},
        {"항목": "MDD -15%", "가격": dd15, "판단": "깊은 눌림 후보"},
    ]
    out = pd.DataFrame(rows)
    if close not in (None, 0):
        out["현재가 대비"] = out["가격"].apply(lambda x: None if safe_float(x) is None else safe_float(x) / close - 1)
    else:
        out["현재가 대비"] = None
    return out


def build_per_reference_table(df, per_df):
    """PER reference table for quick trading context. No chart/band logic."""
    rows = []
    latest_price = safe_float(df["Close"].iloc[-1]) if df is not None and not df.empty else None
    if per_df is None or per_df.empty:
        return pd.DataFrame()
    p = per_df.copy()
    for col, label in [("PER", "Actual PER"), ("PER_PROXY", "Proxy PER")]:
        if col not in p.columns:
            continue
        s = pd.to_numeric(p[col], errors="coerce").dropna()
        s = s[(s > 0) & (s < 500)]
        if s.empty:
            continue
        latest = safe_float(s.iloc[-1])
        avg = safe_float(s.tail(min(252, len(s))).mean())
        low = safe_float(s.tail(min(252, len(s))).quantile(0.2))
        high = safe_float(s.tail(min(252, len(s))).quantile(0.8))
        change60 = None
        n = min(60, len(s)-1)
        if n > 0 and safe_float(s.iloc[-n]) not in (None, 0):
            change60 = latest / float(s.iloc[-n]) - 1
        kind = "ACTUAL_PE" if col == "PER" else "PROXY_PE"
        grade, good_range, comment = classify_valuation_metric(kind, latest)
        rows.append({
            "구분": label,
            "현재": latest,
            "최근 1년 평균": avg,
            "하위 20%": low,
            "상위 20%": high,
            "60거래일 변화": change60,
            "현재 위치": grade,
            "좋은 기준": good_range,
            "해석": comment,
            "판단": "실제 판단용" if col == "PER" else "참고용",
        })
    return pd.DataFrame(rows)


def format_trade_tables(df):
    out = df.copy()
    for col in ["가격"]:
        if col in out.columns:
            out[col] = out[col].apply(lambda x: fmt_num(x, 0))
    for col in ["현재가 대비", "60거래일 변화"]:
        if col in out.columns:
            out[col] = out[col].apply(lambda x: fmt_pct_ratio(x, 1))
    for col in ["현재", "최근 1년 평균", "하위 20%", "상위 20%"]:
        if col in out.columns:
            out[col] = out[col].apply(lambda x: fmt_num(x, 2))
    return out


def render_trading_dashboard(df, per_df, risk_df, risk_label, market, ticker, display_name, asset_rule_key, auto_rule_reason, start_date):
    snap = build_trading_snapshot(df, per_df, risk_df, risk_label)
    st.markdown("## 4. 매매 체크포인트")

    current_mdd_pct = (snap["dd"] or 0) * 100
    mdd_info = get_mdd_action(current_mdd_pct, asset_rule_key)
    auto_flags = compute_auto_correction_flags(df, risk_df, risk_label, market, start_date)

    with st.expander("보정 조건 직접 확인 / 수정", expanded=False):
        st.caption("자동으로 확인 가능한 것은 기본 반영했습니다. 외국인·연기금·EPS·금리처럼 앱이 직접 확인하지 못한 항목은 여기서 수동 보정하세요.")
        if str(asset_rule_key).startswith("KR_"):
            cc1, cc2 = st.columns(2)
            auto_flags["foreign_spot_buy"] = cc1.checkbox("외국인 현물 순매수", value=auto_flags.get("foreign_spot_buy", False))
            auto_flags["foreign_futures_buy"] = cc1.checkbox("외국인 선물 순매수", value=auto_flags.get("foreign_futures_buy", False))
            auto_flags["pension_heavy_sell"] = cc2.checkbox("연기금 대규모 순매도", value=auto_flags.get("pension_heavy_sell", False))
            auto_flags["foreign_double_sell"] = cc2.checkbox("외국인 현물·선물 동반 매도", value=auto_flags.get("foreign_double_sell", False))
            st.caption(auto_flags.get("fx_note", ""))
        else:
            cc1, cc2 = st.columns(2)
            auto_flags["rate_stable"] = cc1.checkbox("금리 안정 또는 하락", value=auto_flags.get("rate_stable", False))
            auto_flags["eps_stable"] = cc1.checkbox("EPS 전망 유지/상향", value=auto_flags.get("eps_stable", False))
            auto_flags["rate_spike"] = cc2.checkbox("금리 급등", value=auto_flags.get("rate_spike", False))
            auto_flags["eps_down"] = cc2.checkbox("EPS 하향", value=auto_flags.get("eps_down", False))

    signal_score, score_df = correction_score_details(asset_rule_key, market, auto_flags)
    final_decision = final_buy_decision(mdd_info["stage"], signal_score)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("최종 판단", final_decision)
    c2.metric("MDD 단계", mdd_info["stage"])
    c3.metric("보정 점수", f"{signal_score:+d}점")
    c4.metric("RSI", fmt_num(snap["rsi"]))
    c5.metric("거래량비율", fmt_num(snap["volr"]))

    if "금지" in final_decision or "보류" in final_decision:
        st.warning(make_mdd_comment(mdd_info["asset_label"], current_mdd_pct, mdd_info["stage"], signal_score, final_decision, mdd_info["rule"].get("memo", "")))
    elif "가능" in final_decision or "분할" in final_decision:
        st.success(make_mdd_comment(mdd_info["asset_label"], current_mdd_pct, mdd_info["stage"], signal_score, final_decision, mdd_info["rule"].get("memo", "")))
    else:
        st.info(make_mdd_comment(mdd_info["asset_label"], current_mdd_pct, mdd_info["stage"], signal_score, final_decision, mdd_info["rule"].get("memo", "")))

    st.caption(f"자산 유형 분류: {asset_rule_label(asset_rule_key)} / 근거: {auto_rule_reason}")

    st.markdown("### 자산 유형별 MDD 기준표")
    st.dataframe(build_asset_mdd_rule_table(asset_rule_key), use_container_width=True, hide_index=True)

    with st.expander("보정 점수 상세"):
        st.dataframe(score_df, use_container_width=True, hide_index=True)
        st.write("- 미국: VIX, MA20 회복, 주도주 흐름, 금리/EPS 수동 보정을 반영합니다.")
        st.write("- 한국: 원/달러, 거래대금, 윗꼬리, 외국인/연기금 수동 보정을 반영합니다.")

    st.markdown("### 기존 가격·MDD·PER 체크")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("기존 행동 판단", snap["action"])
    c2.metric("현재 MDD", fmt_pct_ratio(snap["dd"]))
    c3.metric("RSI", fmt_num(snap["rsi"]))
    c4.metric("거래량비율", fmt_num(snap["volr"]))
    st.info(snap["memo"])

    st.markdown("### 가격 기준선")
    st.dataframe(format_trade_tables(build_trade_level_table(df)), use_container_width=True, hide_index=True)

    st.markdown("### MDD 기준 매수·관망 레벨")
    st.dataframe(format_trade_tables(build_mdd_level_table(df)), use_container_width=True, hide_index=True)

    per_ref = build_per_reference_table(df, per_df)
    if not per_ref.empty:
        st.markdown("### PER 참고표")
        st.dataframe(format_trade_tables(per_ref), use_container_width=True, hide_index=True)
        if snap["per_basis"] == "Proxy":
            st.caption("주의: Proxy PER은 현재 EPS 고정 기준입니다. 실제 이익 변화가 반영된 PER 판단은 Actual PER을 우선하세요.")

    with st.expander("매매 기준 해석"):
        st.write("- MA20 회복 전에는 추격보다 대기/소액이 기본입니다.")
        st.write("- MDD -12% 이하는 1차 관심, -15% 이하는 깊은 눌림 후보입니다. 단, MA200 이탈 시 손절·관망 기준을 우선합니다.")
        st.write("- PER actual 하락 + 주가 상승은 실적/EPS 개선이 가격 상승을 정당화하는 구간일 수 있습니다.")
        st.write("- PER proxy는 전체 흐름 참고용입니다. 실제 저평가 판단에는 단독 사용하지 않습니다.")

# =========================================================
# Inputs
# =========================================================
if "ticker_input" not in st.session_state:
    st.session_state["ticker_input"] = ""

def _clear_ticker_input():
    st.session_state["ticker_input"] = ""

c1, c2, c3 = st.columns(3)
with c1:
    user_input = st.text_input(
        "종목명 / 종목코드 / 미국 티커",
        key="ticker_input",
        placeholder="예: NVDA, 005930, 0174B0, 삼성전자, KOSPI, NASDAQ, SOX",
        help="개별주/ETF는 물론 KOSPI·KOSDAQ·NASDAQ·S&P500·SOX 같은 지수도 입력 가능합니다."
    )
    st.button("입력 초기화", on_click=_clear_ticker_input)
with c2:
    start_date = st.date_input("기준 시작일", pd.to_datetime("2024-01-01"))
with c3:
    asset_rule_choice = st.selectbox("자산 유형 보정", asset_rule_options(), format_func=asset_rule_label, help="기본은 자동 분류입니다. 잘못 잡히면 직접 선택하세요.")

try:
    dart_secret = st.secrets.get("DART_API_KEY", "")
except Exception:
    dart_secret = ""
with st.expander("한국 실제 PER 설정", expanded=False):
    st.caption("한국 종목에서 pykrx PER 시계열이 안 잡힐 때, DART 분기 EPS로 TTM P/E를 계산합니다. Naver 현재 EPS proxy와 달리 EPS가 분기별로 변합니다.")
    dart_key_input = st.text_input("DART API Key", value=dart_secret, type="password")

with st.expander("지원하는 시장지수 입력 예시", expanded=False):
    st.write("- 한국: KOSPI / 코스피 / KS11, KOSDAQ / 코스닥 / KQ11")
    st.write("- 미국: NASDAQ / 나스닥 / ^IXIC, S&P500 / SP500 / ^GSPC, SOX / 필라델피아반도체 / ^SOX, DOW / ^DJI, Russell2000 / ^RUT")

with st.expander("자산 유형별 MDD 기준 설명", expanded=False):
    st.write("- MDD 매수 기준은 자산 유형별로 다르게 적용합니다.")
    st.write("- 미국 대표지수는 -7~-10%부터 1차 분할 관심이 가능하지만, AI·반도체·메모리·전력 같은 테마 ETF는 -12~-15% 이상 조정과 주도주 확인이 필요합니다.")
    st.write("- 한국 자산은 MDD만으로 판단하지 않습니다. 외국인 현물·선물, 원/달러, 거래대금, 연기금 매도 여부를 함께 확인합니다.")
    st.write("- 최종 판단은 MDD 구간 + 시장 보정 점수로 산출합니다. Buy Score 계산 자체는 변경하지 않습니다.")

run = st.button("분석 실행")

if run:
    if not str(user_input).strip():
        st.error("종목명/코드/티커를 입력하세요. 예: NVDA, 005930, 0174B0, 삼성전자")
        st.stop()

    market, ticker, display_name = find_ticker(user_input)
    if ticker is None:
        st.error("종목을 찾지 못했습니다. 예: 삼성전자, 005930, 0174B0, NVDA")
        st.stop()

    auto_rule_key, auto_rule_reason = auto_classify_asset(market, ticker, display_name)
    selected_asset_rule_key = auto_rule_key if asset_rule_choice == "AUTO" else asset_rule_choice
    if asset_rule_choice != "AUTO":
        auto_rule_reason = "사용자 직접 선택"

    price_df, price_status = load_price_data(market, ticker, start_date)
    if price_df.empty:
        st.error(f"가격 데이터를 가져오지 못했습니다: {price_status}")
        st.stop()

    df = add_indicators(price_df)
    latest = df.iloc[-1]
    last_price_date = df.index.max()

    # Valuation and P/E series
    if market in ["US_INDEX", "KR_INDEX"]:
        val = {"ttm_pe": None, "fwd_pe": None, "ps": None, "peg": None, "market_cap": None, "ev_ebitda": None}
        val_status = "시장지수: 개별주식 PER/EPS가 아니므로 현재 Valuation 카드는 N/A로 표시합니다. 지수는 가격·MDD·이평선·RSI·시장위험 중심으로 판단하세요."
        per_df = pd.DataFrame()
        per_status = "시장지수 PER 시계열은 미사용. KOSPI/NASDAQ/S&P500 지수 자체는 가격·MDD·MA·RSI 기준으로 분석합니다."
    elif market == "US":
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

    risk_df, risk_label = market_risk_series(market, ticker, start_date, selected_asset_rule_key)

    st.subheader(f"분석 대상: {display_name} / {ticker} / {market}")
    if market in ["US_INDEX", "KR_INDEX"]:
        st.info("시장지수 분석 모드입니다. 개별 종목처럼 PER·EPS·DART 밸류에이션을 보지 않고, 지수 가격 위치·MDD·이평선·RSI·VIX/KOSPI DD로 시장 진입 위험을 판단합니다.")

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
    if market in ["US_INDEX", "KR_INDEX"]:
        st.caption("지수 Valuation은 별도 데이터가 필요합니다. 이 앱에서는 지수 자체의 저점매수 여부를 가격/MDD/MA/RSI/시장위험으로 판단합니다.")
    else:
        guide_df = build_valuation_guide_table(val, per_df, market)
        if not guide_df.empty:
            st.markdown("### 밸류 기준 판단")
            st.dataframe(format_valuation_guide_table(guide_df), use_container_width=True, hide_index=True)
            with st.expander("PER은 얼마가 좋은가? 기준표"):
                st.write("- **Forward P/E**: 15배 이하 낮음, 15~30배 보통, 30~50배 부담, 50배 초과 고평가·추격 주의")
                st.write("- **TTM/KRX/Actual P/E**: 10배 이하 낮음, 10~20배 양호~보통, 20~35배 성장 반영, 35~50배 부담, 50배 초과 고평가")
                st.write("- **P/S**: 3배 이하 낮음, 3~10배 보통~성장주, 10~30배 고성장 기대 반영, 30배 초과 과열 가능성")
                st.write("- **PEG**: 1 이하 양호, 1~2 보통, 2 초과 성장 대비 밸류 부담")
                st.warning("낮은 PER이 항상 매수 신호는 아닙니다. 경기민감주·반도체·화학·조선은 이익 피크 구간에서 PER이 낮게 보일 수 있습니다. MDD, EPS 방향, 가격 추세와 함께 보세요.")

    has_actual_per = per_df is not None and not per_df.empty and "PER" in per_df.columns and not per_df["PER"].dropna().empty
    has_proxy_per = per_df is not None and not per_df.empty and "PER_PROXY" in per_df.columns and not per_df["PER_PROXY"].dropna().empty
    if market in ["US_INDEX", "KR_INDEX"]:
        st.info(per_status)
    elif has_actual_per:
        st.success(f"PER 시계열: {per_status}")
    elif has_proxy_per:
        st.warning(f"실제 PER 시계열은 제한적입니다. 전체 기간은 P/E proxy로 표시합니다. {per_status}")
    else:
        st.warning(f"PER 시계열 없음: {per_status}")

    st.markdown("## 2. 핵심 차트")
    if market in ["US_INDEX", "KR_INDEX"]:
        st.info("지수 차트는 가격·MDD·시장위험·이평선을 봅니다. PER선은 지수 분석에서 제외합니다.")
    else:
        st.info("차트는 주가·PER·MDD·시장위험만 봅니다. 빨간 실선은 actual P/E, 빨간 점선은 proxy P/E입니다. 실제 판단은 actual P/E를 우선합니다.")
    fig, chart_df = plot_core_chart(df, per_df, risk_df, risk_label, ticker, selected_asset_rule_key)
    render_chart(fig)

    if market not in ["US_INDEX", "KR_INDEX"]:
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

    render_trading_dashboard(df, per_df, risk_df, risk_label, market, ticker, display_name, selected_asset_rule_key, auto_rule_reason, start_date)

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
