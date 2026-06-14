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
# MDD Core Dashboard - Working Core Final
# 핵심만 표시: Price / PER / MDD / Market Risk / MA / RSI / Volume
# =========================================================
st.set_page_config(page_title="MDD 분석기", layout="wide")
require_login()
logout_button()

st.title("📈 MDD 저점매수 분석기 | Core")
st.caption("핵심 차트: 주가 + PER + MDD + 시장위험. 바차트/예측선/가짜 PER 제거.")

# =========================================================
# Utils
# =========================================================
def to_ns_datetime(x):
    dt = pd.to_datetime(x, errors="coerce")
    try:
        if hasattr(dt, "dt"):
            try:
                dt = dt.dt.tz_localize(None)
            except Exception:
                try:
                    dt = dt.dt.tz_convert(None)
                except Exception:
                    pass
        else:
            try:
                dt = dt.tz_localize(None)
            except Exception:
                try:
                    dt = dt.tz_convert(None)
                except Exception:
                    pass
    except Exception:
        pass
    try:
        return dt.astype("datetime64[ns]")
    except Exception:
        return pd.to_datetime(dt, errors="coerce")


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
# Ticker search
# =========================================================
@st.cache_data(ttl=86400)
def get_krx_listing():
    if not FDR_AVAILABLE:
        return pd.DataFrame()
    try:
        listing = fdr.StockListing("KRX")
        if listing is None or listing.empty:
            return pd.DataFrame()
        listing = listing.copy()
        listing["Code"] = listing["Code"].astype(str).str.zfill(6)
        listing["Name"] = listing["Name"].astype(str).str.strip()
        return listing[["Code", "Name"]]
    except Exception:
        return pd.DataFrame()


def find_ticker(query):
    q = str(query).strip()
    if q == "":
        return None, None, None
    if q.isdigit() and len(q) == 6:
        return "KR", q, q
    if q in KR_FALLBACK_MAP:
        return "KR", KR_FALLBACK_MAP[q], q

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
# Price data
# =========================================================
@st.cache_data(ttl=3600)
def load_price_data(market, ticker, start_date):
    start = pd.to_datetime(start_date).strftime("%Y-%m-%d")
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
                    df = df.rename(columns={"시가": "Open", "고가": "High", "저가": "Low", "종가": "Close", "거래량": "Volume"})
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


def get_profile(asset_type):
    profiles = {
        "일반 주식/ETF": {"watch": -0.08, "buy": -0.12, "deep": -0.15, "risk": -0.20},
        "나스닥형 ETF": {"watch": -0.06, "buy": -0.08, "deep": -0.12, "risk": -0.15},
        "반도체/메모리": {"watch": -0.08, "buy": -0.12, "deep": -0.15, "risk": -0.20},
        "전력/AI인프라": {"watch": -0.08, "buy": -0.10, "deep": -0.15, "risk": -0.18},
        "우주/고변동": {"watch": -0.12, "buy": -0.18, "deep": -0.25, "risk": -0.30},
    }
    return profiles.get(asset_type, profiles["일반 주식/ETF"])

# =========================================================
# yfinance valuation
# =========================================================
@st.cache_data(ttl=3600)
def load_yf_info(ticker):
    empty = {"trailing_pe": None, "forward_pe": None, "price_to_sales": None, "peg_ratio": None, "status": "N/A"}
    if not YF_AVAILABLE:
        empty["status"] = YF_IMPORT_ERROR
        return empty
    try:
        info = yf.Ticker(ticker).info
        return {
            "trailing_pe": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "price_to_sales": info.get("priceToSalesTrailing12Months"),
            "peg_ratio": info.get("pegRatio"),
            "status": "OK",
        }
    except Exception as e:
        empty["status"] = repr(e)
        return empty

# =========================================================
# PER data helpers
# =========================================================
def normalize_fundamental_df(raw):
    """
    pykrx 버전/함수별 반환 형태가 달라도 PER/PBR/EPS/BPS를 최대한 추출.
    지원 형태:
    - index=date, columns=PER/PBR/EPS/BPS
    - index=ticker, columns=PER/PBR/EPS/BPS
    - 한국어 컬럼 일부
    - 컬럼/인덱스가 뒤집힌 형태는 transpose 시도
    """
    if raw is None or not isinstance(raw, pd.DataFrame) or raw.empty:
        return pd.DataFrame()

    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([str(x) for x in col if str(x) != ""]).strip() for col in df.columns]

    rename_map = {}
    for col in df.columns:
        c = str(col).strip()
        cu = c.upper()
        if cu in ["PER", "PBR", "EPS", "BPS", "DIV", "DPS"]:
            rename_map[col] = cu
        elif c in ["주가수익비율", "PER(배)", "P/E"]:
            rename_map[col] = "PER"
        elif c in ["주가순자산비율", "PBR(배)", "P/B"]:
            rename_map[col] = "PBR"
        elif c in ["주당순이익"]:
            rename_map[col] = "EPS"
        elif c in ["주당순자산"]:
            rename_map[col] = "BPS"
    df = df.rename(columns=rename_map)

    wanted = [c for c in ["PER", "PBR", "EPS", "BPS", "DIV", "DPS"] if c in df.columns]
    if "PER" in wanted or "EPS" in wanted:
        out = df[wanted].copy()
    else:
        # 혹시 행/열이 바뀐 형태면 transpose 시도
        tdf = df.T.copy()
        rename_map2 = {}
        for col in tdf.columns:
            c = str(col).strip().upper()
            if c in ["PER", "PBR", "EPS", "BPS", "DIV", "DPS"]:
                rename_map2[col] = c
        tdf = tdf.rename(columns=rename_map2)
        wanted2 = [c for c in ["PER", "PBR", "EPS", "BPS", "DIV", "DPS"] if c in tdf.columns]
        if "PER" in wanted2 or "EPS" in wanted2:
            out = tdf[wanted2].copy()
        else:
            return pd.DataFrame()

    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


@st.cache_data(ttl=3600)
def load_kr_per_series(ticker, start_date, price_df):
    if not PYKRX_AVAILABLE:
        return pd.DataFrame(), f"pykrx import 실패: {PYKRX_IMPORT_ERROR}", None

    ticker = str(ticker).strip().zfill(6)
    if price_df is None or price_df.empty:
        return pd.DataFrame(), "가격 데이터 없음", None

    # 중요: 오늘 날짜가 KRX 데이터보다 미래일 수 있으므로 가격 데이터의 마지막 날짜를 종료일로 사용
    start = pd.to_datetime(start_date).strftime("%Y%m%d")
    end_dt = pd.to_datetime(price_df.index.max())
    end = end_dt.strftime("%Y%m%d")

    errors = []

    attempts = [
        ("get_market_fundamental daily", lambda: pkstock.get_market_fundamental(start, end, ticker)),
        ("get_market_fundamental monthly", lambda: pkstock.get_market_fundamental(start, end, ticker, freq="m")),
    ]

    # 버전에 따라 존재할 수 있는 함수도 안전하게 시도
    if hasattr(pkstock, "get_market_fundamental_by_date"):
        attempts.append(("get_market_fundamental_by_date daily", lambda: pkstock.get_market_fundamental_by_date(start, end, ticker)))
        attempts.append(("get_market_fundamental_by_date monthly", lambda: pkstock.get_market_fundamental_by_date(start, end, ticker, freq="m")))

    for name, loader in attempts:
        try:
            raw = loader()
            if raw is None or raw.empty:
                errors.append(f"{name}: empty")
                continue
            out = normalize_fundamental_df(raw)
            if out.empty:
                errors.append(f"{name}: PER/EPS 추출 실패 columns={list(raw.columns)[:10]}")
                continue
            out.index = to_ns_datetime(out.index)
            if "PER" in out.columns:
                out = out[(out["PER"] > 0) & (out["PER"] < 300)].dropna(subset=["PER"])
            elif "EPS" in out.columns:
                # PER이 없고 EPS가 있으면 가격으로 계산
                px = price_df[["Close"]].copy()
                px.index = to_ns_datetime(px.index)
                merged = px.join(out[["EPS"]], how="left")
                merged["EPS"] = merged["EPS"].ffill()
                merged["PER"] = merged["Close"] / merged["EPS"]
                out = merged[["PER", "EPS"]].replace([np.inf, -np.inf], np.nan)
                out = out[(out["PER"] > 0) & (out["PER"] < 300)].dropna(subset=["PER"])
            if not out.empty and "PER" in out.columns:
                return out, f"OK: {name}", safe_float(out["PER"].dropna().iloc[-1])
            errors.append(f"{name}: 유효 PER 없음")
        except Exception as e:
            errors.append(f"{name}: {repr(e)}")

    # 최종 fallback: 최근 단일일자 전체시장 fundamental에서 EPS/PER를 찾아 현재 EPS로 일별 PER 생성
    fallback_errors = []
    try:
        recent_dates = list(pd.to_datetime(price_df.index).sort_values()[-80:])[::-1]
        for dt in recent_dates:
            d = pd.to_datetime(dt).strftime("%Y%m%d")
            loaders = []
            if hasattr(pkstock, "get_market_fundamental_by_ticker"):
                loaders.append(("by_ticker", lambda dd=d: pkstock.get_market_fundamental_by_ticker(dd, market="ALL")))
            loaders.append(("fundamental_date", lambda dd=d: pkstock.get_market_fundamental(dd, market="ALL")))
            for lname, loader in loaders:
                try:
                    raw = loader()
                    if raw is None or raw.empty:
                        continue
                    raw = raw.copy()
                    raw.index = raw.index.astype(str).str.zfill(6)
                    if ticker not in raw.index:
                        continue
                    out = normalize_fundamental_df(raw.loc[[ticker]])
                    if out.empty:
                        continue
                    cur_per = safe_float(out["PER"].iloc[0]) if "PER" in out.columns else None
                    cur_eps = safe_float(out["EPS"].iloc[0]) if "EPS" in out.columns else None
                    if cur_eps is not None and cur_eps > 0:
                        px = price_df[["Close"]].copy()
                        px.index = to_ns_datetime(px.index)
                        calc = pd.DataFrame(index=px.index)
                        calc["EPS"] = cur_eps
                        calc["PER"] = px["Close"] / cur_eps
                        calc = calc[(calc["PER"] > 0) & (calc["PER"] < 300)].dropna(subset=["PER"])
                        return calc, f"OK: latest EPS fallback {d} ({lname})", cur_per
                    if cur_per is not None and 0 < cur_per < 300:
                        one = pd.DataFrame({"PER": [cur_per]}, index=[pd.to_datetime(d)])
                        return one, f"OK: latest PER only {d} ({lname})", cur_per
                except Exception as e:
                    fallback_errors.append(f"{d} {lname}: {repr(e)}")
    except Exception as e:
        fallback_errors.append(f"fallback fatal: {repr(e)}")

    all_errors = errors + fallback_errors[-5:]
    return pd.DataFrame(), " / ".join(all_errors[-8:]), None


@st.cache_data(ttl=3600)
def load_us_ttm_pe_series(ticker, price_df):
    if not YF_AVAILABLE:
        return pd.DataFrame(), f"yfinance import 실패: {YF_IMPORT_ERROR}"
    try:
        tk = yf.Ticker(ticker)
        eps_q = None
        source = ""

        try:
            ed = tk.get_earnings_dates(limit=40)
            if ed is not None and not ed.empty and "Reported EPS" in ed.columns:
                eps = pd.to_numeric(ed["Reported EPS"], errors="coerce").dropna()
                eps.index = to_ns_datetime(eps.index)
                eps = eps.sort_index()
                if len(eps) >= 4:
                    eps_q = eps
                    source = "earnings_dates Reported EPS"
        except Exception:
            pass

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
                for row_name in ["Diluted EPS", "Basic EPS", "Normalized EPS"]:
                    if row_name in stmt.index:
                        eps = pd.to_numeric(stmt.loc[row_name], errors="coerce").dropna().sort_index()
                        if len(eps) >= 4:
                            eps_q = eps
                            eps_q.index = to_ns_datetime(eps_q.index) + pd.Timedelta(days=45)
                            source = f"financials {row_name}"
                            break
                if eps_q is not None and len(eps_q) >= 4:
                    break

        if eps_q is None or len(eps_q) < 4:
            return pd.DataFrame(), "EPS 분기 데이터 부족"

        eps_ttm = eps_q.sort_index().rolling(4).sum().dropna()
        eps_ttm = eps_ttm[eps_ttm > 0]
        if eps_ttm.empty:
            return pd.DataFrame(), "EPS_TTM 유효값 없음"

        eps_df = pd.DataFrame({"Date": to_ns_datetime(eps_ttm.index), "EPS_TTM": eps_ttm.values}).sort_values("Date")
        px = price_df[["Close"]].copy().reset_index()
        px.columns = ["Date", "Close"]
        px["Date"] = to_ns_datetime(px["Date"])
        px = px.sort_values("Date")
        merged = pd.merge_asof(px, eps_df, on="Date", direction="backward")
        merged["PER"] = merged["Close"] / merged["EPS_TTM"]
        merged = merged.replace([np.inf, -np.inf], np.nan)
        merged = merged[(merged["PER"] > 0) & (merged["PER"] < 300)].dropna(subset=["PER"])
        if merged.empty:
            return pd.DataFrame(), "TTM P/E 유효값 없음"
        out = merged.set_index("Date")[["PER", "EPS_TTM"]]
        out.index = to_ns_datetime(out.index)
        return out, f"OK: {source}"
    except Exception as e:
        return pd.DataFrame(), repr(e)

# =========================================================
# Market risk
# =========================================================
@st.cache_data(ttl=3600)
def load_us_vix(start_date):
    if not YF_AVAILABLE:
        return pd.DataFrame(), YF_IMPORT_ERROR
    try:
        start = pd.to_datetime(start_date).strftime("%Y-%m-%d")
        vix = yf.Ticker("^VIX").history(start=start, auto_adjust=True)
        if vix is None or vix.empty:
            return pd.DataFrame(), "VIX empty"
        vix = vix.copy()
        vix.index = to_ns_datetime(vix.index)
        return vix[["Close"]].rename(columns={"Close": "VIX"}), "OK: ^VIX"
    except Exception as e:
        return pd.DataFrame(), repr(e)


@st.cache_data(ttl=3600)
def load_kr_market_index(start_date, ticker):
    symbol = "KS11"
    label = "KOSPI"
    if PYKRX_AVAILABLE:
        try:
            end = datetime.today()
            for offset in range(0, 15):
                d = (end - timedelta(days=offset)).strftime("%Y%m%d")
                try:
                    kosdaq = pkstock.get_market_ticker_list(d, market="KOSDAQ")
                    if str(ticker).zfill(6) in [str(x).zfill(6) for x in kosdaq]:
                        symbol = "KQ11"
                        label = "KOSDAQ"
                    break
                except Exception:
                    continue
        except Exception:
            pass
    if not FDR_AVAILABLE:
        return pd.DataFrame(), "FDR 없음"
    try:
        idx = fdr.DataReader(symbol, pd.to_datetime(start_date).strftime("%Y-%m-%d"))
        if idx is None or idx.empty:
            return pd.DataFrame(), f"{label} empty"
        idx = idx.copy()
        idx.index = to_ns_datetime(idx.index)
        idx["Market_Peak"] = idx["Close"].cummax()
        idx["Market_DD"] = idx["Close"] / idx["Market_Peak"] - 1
        return idx[["Close", "Market_DD"]].rename(columns={"Close": label}), f"OK: {label}"
    except Exception as e:
        return pd.DataFrame(), repr(e)

# =========================================================
# Signals / action / comment
# =========================================================
def build_trade_signals(df, profile, min_gap_days=30):
    hard_stop = (df["Current_Drawdown"] <= profile["risk"]) & (df["Close"] < df["Prev_Low20"]) & (df["Return"] < 0)
    raw_buy = (df["Current_Drawdown"] <= profile["buy"]) & (df["RSI"] <= 42) & (~hard_stop.fillna(False))
    raw_cash = ((df["Current_Drawdown"] >= -0.03) & (df["RSI"] >= 68)) | ((df["Close"] < df["MA20"]) & (df["RSI"] >= 65) & (df["Current_Drawdown"] > -0.08))

    result = pd.DataFrame(index=df.index)
    result["Buy_Display"] = np.nan
    result["Cash_Display"] = np.nan
    last_buy = None
    last_cash = None
    for dt, flag in raw_buy.items():
        if bool(flag) and (last_buy is None or (dt - last_buy).days >= min_gap_days):
            result.loc[dt, "Buy_Display"] = df.loc[dt, "Close"]
            last_buy = dt
    for dt, flag in raw_cash.items():
        if bool(flag) and (last_cash is None or (dt - last_cash).days >= min_gap_days):
            result.loc[dt, "Cash_Display"] = df.loc[dt, "Close"]
            last_cash = dt
    return result


def make_action(latest, profile):
    dd = latest["Current_Drawdown"]
    rsi = latest["RSI"]
    close = latest["Close"]
    ma20 = latest["MA20"]
    prev_low20 = latest.get("Prev_Low20", np.nan)

    if pd.notna(prev_low20) and dd <= profile["risk"] and close < prev_low20:
        return "매수 금지", "깊은 MDD + 저점 이탈입니다. 반등 확인 전 신규매수 금지."
    if dd <= profile["buy"] and pd.notna(rsi) and rsi <= 42:
        if pd.notna(ma20) and close >= ma20:
            return "1차 소액 가능", "MDD와 RSI는 매력적이고 MA20 회복도 확인됩니다."
        return "소액만 / 확인 대기", "MDD와 RSI는 매력적이나 MA20 미회복입니다. 선진입은 소액만."
    if dd > -0.05 and pd.notna(rsi) and rsi >= 65:
        return "추격 금지", "고점권/과열 구간입니다. 신규 추격보다 현금확보 후보입니다."
    return "관망", "극단적 저점/과열은 아닙니다. MDD와 MA20 회복 여부를 확인하세요."


def make_comment(df, per_df, market_risk_df, market):
    latest = df.iloc[-1]
    comments = []
    close = latest["Close"]
    ma20 = latest["MA20"]
    ma200 = latest["MA200"]
    dd = latest["Current_Drawdown"]
    rsi = latest["RSI"]

    comments.append("가격: MA20 위로 회복하면 확인매수 신뢰가 높아지고, MA20 아래면 소액/대기가 우선입니다." if pd.notna(ma20) and close < ma20 else "가격: MA20 위라 단기 반등 흐름은 유지 중입니다.")
    comments.append("장기추세: MA200 위라 장기 추세 훼손은 제한적입니다." if pd.notna(ma200) and close >= ma200 else "장기추세: MA200 아래면 추세 훼손 가능성을 확인해야 합니다.")
    comments.append(f"MDD: 현재 {dd*100:.2f}%입니다. {'저점매수 관심 구간입니다.' if dd <= -0.12 else '아직 깊은 저점 구간은 아닙니다.'}")
    if pd.notna(rsi):
        if rsi <= 30:
            comments.append(f"RSI: {rsi:.1f}로 강한 과매도권입니다.")
        elif rsi <= 42:
            comments.append(f"RSI: {rsi:.1f}로 눌림 후보 구간입니다.")
        elif rsi >= 70:
            comments.append(f"RSI: {rsi:.1f}로 과열권입니다.")
        else:
            comments.append(f"RSI: {rsi:.1f}로 중립권입니다.")

    if per_df is not None and not per_df.empty and "PER" in per_df.columns:
        joined = df[["Close"]].join(per_df[["PER"]], how="left")
        joined["PER"] = joined["PER"].ffill()
        joined = joined.dropna(subset=["Close", "PER"])
        if len(joined) >= 60:
            recent = joined.iloc[-1]
            past = joined.iloc[-60]
            price_chg = recent["Close"] / past["Close"] - 1
            per_chg = recent["PER"] / past["PER"] - 1
            if price_chg > 0 and per_chg < 0:
                comments.append(f"PER: 최근 60거래일 주가 +{price_chg*100:.1f}%, PER {per_chg*100:.1f}%입니다. 실적 개선이 주가 상승을 정당화하는 흐름입니다.")
            elif price_chg > 0 and per_chg > 0:
                comments.append(f"PER: 최근 60거래일 주가 +{price_chg*100:.1f}%, PER +{per_chg*100:.1f}%입니다. 밸류 부담도 같이 커졌습니다.")
            elif price_chg < 0 and per_chg < 0:
                comments.append(f"PER: 최근 60거래일 주가 {price_chg*100:.1f}%, PER {per_chg*100:.1f}%입니다. 밸류 부담은 완화됐지만 업황 확인이 필요합니다.")
            else:
                comments.append(f"PER: 최근 60거래일 주가 {price_chg*100:.1f}%, PER +{per_chg*100:.1f}%입니다. 주가 하락 중 이익 둔화 가능성을 확인해야 합니다.")
    else:
        comments.append("PER: 시계열 데이터가 없어 현재 PER 카드만 참고하세요.")

    if market == "US" and market_risk_df is not None and not market_risk_df.empty and "VIX" in market_risk_df.columns:
        vix = market_risk_df["VIX"].dropna().iloc[-1]
        comments.append(f"시장위험: VIX {vix:.1f}. 25 이상이면 공포성 눌림 여부를 확인하세요.")
    elif market == "KR" and market_risk_df is not None and not market_risk_df.empty and "Market_DD" in market_risk_df.columns:
        mdd = market_risk_df["Market_DD"].dropna().iloc[-1]
        comments.append(f"시장위험: 한국 지수 MDD {mdd*100:.2f}%. 지수 약세가 크면 개별주 반등 신뢰가 낮아집니다.")
    return comments

# =========================================================
# Plot
# =========================================================
def plot_core_chart(df, per_df, market_risk_df, signal_df, title, market, valuation):
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

    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(15, 9), sharex=True, gridspec_kw={"height_ratios": [3, 1]})

    # 상단 Price + MA
    ax1.plot(chart.index, chart["Price"], label="Price", color="#0057B8", linewidth=2.0)
    ax1.plot(chart.index, chart["MA20"], label="MA20", color="#FF8C00", linewidth=1.2)
    ax1.plot(chart.index, chart["MA60"], label="MA60", color="#2CA02C", linewidth=1.2)
    ax1.plot(chart.index, chart["MA200"], label="MA200", color="#6F42C1", linewidth=1.2)
    ax1.set_ylabel("Price", color="#0057B8", fontweight="bold")
    ax1.tick_params(axis="y", labelcolor="#0057B8")
    ax1.grid(True, linestyle=":", alpha=0.45)

    if "Buy_Display" in chart.columns:
        ax1.scatter(chart.index, chart["Buy_Display"] * 0.97, marker="^", s=90, color="#00A000", label="BUY candidate", zorder=5)
    if "Cash_Display" in chart.columns:
        ax1.scatter(chart.index, chart["Cash_Display"] * 1.03, marker="v", s=90, color="#E60000", label="Cash / overheat", zorder=5)

    # 오른쪽 PER
    ax2 = ax1.twinx()
    if not chart["PER"].dropna().empty:
        ax2.plot(chart.index, chart["PER"], label="P/E", color="#D62728", linewidth=1.8)
        avg_pe = chart["PER"].dropna().mean()
        ax2.axhline(avg_pe, label="P/E avg", color="#D62728", linestyle="--", linewidth=1.0, alpha=0.45)

    if market == "US":
        fpe = safe_float(valuation.get("forward_pe"))
        tpe = safe_float(valuation.get("trailing_pe"))
        if fpe is not None:
            ax2.axhline(fpe, label="Current forward P/E", color="#111111", linestyle="-.", linewidth=1.1)
        if tpe is not None:
            ax2.axhline(tpe, label="Current trailing P/E", color="#FF1493", linestyle=":", linewidth=1.1)

    ax2.set_ylabel("P/E", color="#D62728", fontweight="bold")
    ax2.tick_params(axis="y", labelcolor="#D62728")

    ax1.set_title(title, fontsize=14, fontweight="bold")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

    # 하단 MDD + risk
    ax3.plot(chart.index, chart["Current_Drawdown"] * 100, label="Current DD", color="#8B0000", linewidth=1.5)
    for y, label in [(-8, "Watch -8%"), (-12, "Buy zone -12%"), (-15, "Deep -15%"), (-20, "Risk -20%")]:
        ax3.axhline(y, color="#4682B4", linestyle="--", linewidth=0.9, alpha=0.45, label=label)
    ax3.set_ylabel("MDD (%)", color="#8B0000", fontweight="bold")
    ax3.tick_params(axis="y", labelcolor="#8B0000")
    ax3.grid(True, linestyle=":", alpha=0.45)

    if market == "US" and market_risk_df is not None and not market_risk_df.empty and "VIX" in market_risk_df.columns:
        risk = market_risk_df.copy()
        risk.index = to_ns_datetime(risk.index)
        ax4 = ax3.twinx()
        ax4.plot(risk.index, risk["VIX"], label="VIX", color="#008B8B", linestyle="--", linewidth=1.2, alpha=0.8)
        ax4.set_ylabel("VIX", color="#008B8B", fontweight="bold")
        ax4.tick_params(axis="y", labelcolor="#008B8B")
        if not risk["VIX"].dropna().empty:
            ax4.set_ylim(0, max(45, float(risk["VIX"].dropna().max()) * 1.15))
        l3, lb3 = ax3.get_legend_handles_labels()
        l4, lb4 = ax4.get_legend_handles_labels()
        ax3.legend(l3 + l4, lb3 + lb4, loc="lower left", fontsize=8)
    elif market == "KR" and market_risk_df is not None and not market_risk_df.empty and "Market_DD" in market_risk_df.columns:
        risk = market_risk_df.copy()
        risk.index = to_ns_datetime(risk.index)
        ax3.plot(risk.index, risk["Market_DD"] * 100, label="Korea market DD", color="#008B8B", linestyle="--", linewidth=1.2)
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
    asset_type = st.selectbox("종목 유형", ["일반 주식/ETF", "나스닥형 ETF", "반도체/메모리", "전력/AI인프라", "우주/고변동"], index=2)

run = st.button("분석 실행")

if run:
    market, ticker, display_name = find_ticker(user_input)
    if ticker is None:
        st.error("종목을 찾지 못했습니다. 한국 종목은 종목명 또는 6자리 코드, 미국 종목은 티커를 입력하세요.")
        st.stop()

    profile = get_profile(asset_type)
    with st.spinner("가격 / PER / MDD / 시장위험 데이터 분석 중..."):
        price_df = load_price_data(market, ticker, start_date)
        if price_df.empty:
            st.error("가격 데이터를 가져오지 못했습니다.")
            st.stop()

        df = calculate_indicators(price_df)
        signal_df = build_trade_signals(df, profile)

        if market == "KR":
            per_df, per_status, kr_current_pe = load_kr_per_series(ticker, start_date, price_df)
            valuation = {"trailing_pe": kr_current_pe, "forward_pe": None, "price_to_sales": None, "peg_ratio": None, "status": "KR"}
            market_risk_df, risk_status = load_kr_market_index(start_date, ticker)
        else:
            valuation = load_yf_info(ticker)
            per_df, per_status = load_us_ttm_pe_series(ticker, df)
            market_risk_df, risk_status = load_us_vix(start_date)
            kr_current_pe = None

        latest = df.iloc[-1]
        action, action_reason = make_action(latest, profile)

    st.subheader(f"분석 대상: {display_name} / {ticker} / {market}")

    st.markdown("## 1. 핵심 판단")
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("최종 행동", action)
    k2.metric("Current DD", f"{latest['Current_Drawdown']*100:.2f}%")
    k3.metric("Max DD", f"{df['Max_Drawdown'].min()*100:.2f}%")
    k4.metric("RSI", fmt_num(latest["RSI"]))
    ma20_state = "위" if pd.notna(latest["MA20"]) and latest["Close"] >= latest["MA20"] else "아래"
    k5.metric("MA20", ma20_state)
    k6.metric("Vol Ratio", fmt_num(latest["Volume_Ratio"]))
    st.info(action_reason)

    st.markdown("## 2. 현재 Valuation")
    v1, v2, v3, v4 = st.columns(4)
    pe_label = "KRX P/E" if market == "KR" else "TTM P/E"
    v1.metric(pe_label, fmt_num(valuation.get("trailing_pe")))
    v2.metric("Forward P/E", fmt_num(valuation.get("forward_pe")))
    v3.metric("P/S", fmt_num(valuation.get("price_to_sales")))
    v4.metric("PEG", fmt_num(valuation.get("peg_ratio")))

    if per_df.empty:
        st.warning(f"PER 시계열 없음: {per_status}")
    else:
        st.caption(f"PER data status: {per_status}")
    st.caption(f"Market risk status: {risk_status}")

    st.markdown("## 3. Price + PER + MDD + Market Risk")
    title = f"{ticker} Price + {'KRX P/E' if market == 'KR' else 'Estimated TTM P/E'} + MDD + {'Korea Market Risk' if market == 'KR' else 'VIX'}"
    plot_core_chart(df, per_df, market_risk_df, signal_df, title, market, valuation)

    st.markdown("## 4. 차트 해석")
    for c in make_comment(df, per_df, market_risk_df, market):
        st.write(f"- {c}")

    with st.expander("PER 원자료 / 디버그"):
        st.write("PER status:", per_status)
        st.write("Market risk status:", risk_status)
        if not per_df.empty:
            st.dataframe(per_df.tail(60), use_container_width=True)
        else:
            st.info("PER 원자료가 비어 있습니다.")
        if market == "KR" and PYKRX_AVAILABLE:
            st.markdown("### pykrx 직접 검증")
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
        "주의: PER은 한국은 KRX/pykrx 기반, 미국은 yfinance 분기 EPS 기반 Estimated TTM P/E입니다. "
        "미국 Forward P/E 과거 시계열은 무료 데이터로 안정적으로 제공되지 않아 현재 기준선으로만 표시합니다."
    )
