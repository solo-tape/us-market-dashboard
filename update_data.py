import io
import json
import os
import re
from datetime import datetime, timezone

import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup

FRED_API_KEY = os.environ.get("FRED_API_KEY", "")
DATA_FILE = "data.json"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/128.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


def load_existing():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def fetch_yf_history(ticker, period="1y"):
    df = yf.Ticker(ticker).history(period=period)
    if df.empty:
        return pd.DataFrame()
    df.index = df.index.strftime("%m/%d")
    return df


def fetch_fred_series(series_id, limit=15):
    if not FRED_API_KEY:
        return []
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "sort_order": "desc",
        "limit": limit,
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        obs = [
            (o["date"][5:].replace("-", "/"), float(o["value"]))
            for o in r.json().get("observations", [])
            if o["value"] != "."
        ]
        return list(reversed(obs))
    except Exception as e:
        print(f"FRED {series_id} fetch failed: {e}")
        return []


def fetch_cnn_fear_greed():
    """抓取 CNN 恐慌贪婪指数，返回纯数字（如 24.9）"""
    url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
    headers = {**HEADERS, "Referer": "https://edition.cnn.com/"}
    try:
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        score = r.json().get("fear_and_greed", {}).get("score")
        if score is not None:
            return round(float(score), 1)
    except Exception as e:
        print(f"CNN Fear & Greed fetch failed: {e}")
    return None


def fetch_s5fi():
    """计算标普 500 站上 50 日均线的股票占比，返回纯数字（如 64.67）"""
    try:
        wiki_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        r = requests.get(wiki_url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        tables = pd.read_html(io.StringIO(r.text))
        tickers = [t.replace(".", "-") for t in tables[0]["Symbol"].tolist()]

        df = yf.download(
            tickers,
            period="4mo",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        closes = df["Close"] if "Close" in df else df
        closes = closes.dropna(axis=1, how="all")
        ma50 = closes.rolling(50).mean()
        latest_close = closes.iloc[-1]
        latest_ma50 = ma50.iloc[-1]

        valid = latest_close.notna() & latest_ma50.notna()
        if valid.sum() > 400:
            pct = (latest_close[valid] > latest_ma50[valid]).mean() * 100
            return round(float(pct), 2)
    except Exception as e:
        print(f"S5FI calculation failed: {e}")
    return None


def fetch_aaii(limit=8):
    """抓取 AAII 周度情绪，过滤底部汇总行，返回最近 8 周折线图数组"""
    xls_url = "https://www.aaii.com/files/surveys/sentiment.xls"
    try:
        r = requests.get(xls_url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        df = pd.read_excel(io.BytesIO(r.content), skiprows=3)
        df = df.iloc[:, :4].copy()
        df.columns = ["date", "bull", "neutral", "bear"]

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        for col in ["bull", "neutral", "bear"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["date", "bull", "neutral", "bear"])
        df = df.sort_values("date").tail(limit)

        if not df.empty:
            # 兼容小数(0.371)与百分比(37.1)
            scale = 100.0 if df["bull"].max() <= 1.0 else 1.0
            return {
                "dates": df["date"].dt.strftime("%m/%d").tolist(),
                "bull": [round(float(x) * scale, 1) for x in df["bull"].tolist()],
                "bear": [round(float(x) * scale, 1) for x in df["bear"].tolist()],
                "neutral": [round(float(x) * scale, 1) for x in df["neutral"].tolist()],
            }
    except Exception as e:
        print(f"AAII XLS fetch failed: {e}")
    return None


def fetch_naaim():
    """抓取 NAAIM 机构股票仓位，返回纯数字"""
    url = "https://www.naaim.org/programs/naaim-exposure-index/"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        tables = pd.read_html(io.StringIO(r.text))
        for tbl in tables:
            for col in tbl.columns:
                if "naaim" in str(col).lower() or "number" in str(col).lower():
                    val = pd.to_numeric(tbl[col], errors="coerce").dropna()
                    if not val.empty:
                        return round(float(val.iloc[0]), 2)
        soup = BeautifulSoup(r.text, "lxml")
        text = soup.get_text(" ", strip=True)
        m = re.search(r"NAAIM Exposure Index.*?(\d{1,3}\.\d{1,2})", text)
        if m:
            return round(float(m.group(1)), 2)
    except Exception as e:
        print(f"NAAIM fetch failed: {e}")
    return None


def fetch_put_call():
    """抓取 CBOE Put/Call 比率，返回纯数字"""
    url = "https://ycharts.com/indicators/cboe_equity_put_call_ratio"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")
        text = soup.get_text(" ", strip=True)
        m = re.search(r"(\d\.\d{2})\s+for\s+[A-Z][a-z]{2}\s+\d{1,2},\s+\d{4}", text)
        if m:
            return round(float(m.group(1)), 2)
    except Exception as e:
        print(f"Put/Call fetch failed: {e}")
    return None


def main():
    data = load_existing()
    # 清理旧版错误格式，防止残留导致前端报错
    for k in ["cnn", "s5fi", "naaim", "put_call"]:
        if isinstance(data.get(k), dict):
            data.pop(k, None)
    if isinstance(data.get("aaii"), dict) and "bull" not in data["aaii"]:
        data.pop("aaii", None)

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    data["updated_at"] = now_str

    # 1. 标普 500 及 50/200 日均线
    spx = fetch_yf_history("^GSPC", period="1y")
    if not spx.empty:
        spx["MA50"] = spx["Close"].rolling(50).mean()
        spx["MA200"] = spx["Close"].rolling(200).mean()
        recent = spx.tail(15)
        latest = spx.iloc[-1]
        data["spx"] = {
            "dates": recent.index.tolist(),
            "close": [round(x, 2) for x in recent["Close"].tolist()],
            "ma50": round(float(latest["MA50"]), 2) if pd.notna(latest["MA50"]) else None,
            "ma200": round(float(latest["MA200"]), 2) if pd.notna(latest["MA200"]) else None,
            "latest": round(float(latest["Close"]), 2),
        }

    # 2. 费城半导体 SOX 及 SOX/SPX 比值
    sox = fetch_yf_history("^SOX", period="6mo")
    if not sox.empty and not spx.empty:
        recent_sox = sox.tail(15)
        sox_latest = float(sox["Close"].iloc[-1])
        spx_latest = float(spx["Close"].iloc[-1])
        data["sox"] = {
            "dates": recent_sox.index.tolist(),
            "close": [round(x, 2) for x in recent_sox["Close"].tolist()],
            "latest": round(sox_latest, 2),
            "sox_spx_ratio": round(sox_latest / spx_latest, 3),
        }

    # 3. CBOE VIX 期限结构
    vix_map = {"VIX1D": "^VIX1D", "VIX9D": "^VIX9D", "VIX": "^VIX", "VIX3M": "^VIX3M"}
    vix_vals = {}
    for label, sym in vix_map.items():
        df = fetch_yf_history(sym, period="5d")
        if not df.empty:
            vix_vals[label] = round(float(df["Close"].iloc[-1]), 2)
    if "VIX" in vix_vals and "VIX3M" in vix_vals and vix_vals["VIX3M"] > 0:
        vix_vals["ratio"] = round(vix_vals["VIX"] / vix_vals["VIX3M"], 2)
    if vix_vals:
        data["vix_term"] = vix_vals

    # 4. FRED 高收益债利差（HY OAS）与美债利率
    hy = fetch_fred_series("BAMLH0A0HYM2", limit=15)
    if hy:
        data["hy_oas"] = {
            "dates": [x[0] for x in hy],
            "values": [round(x[1], 2) for x in hy],
            "latest": round(hy[-1][1], 2),
        }

    y10 = fetch_fred_series("DGS10", limit=15)
    y2 = fetch_fred_series("DGS2", limit=15)
    tips10 = fetch_fred_series("DFII10", limit=5)
    if y10 and y2:
        data["rates"] = {
            "dates": [x[0] for x in y10],
            "y10": [round(x[1], 2) for x in y10],
            "y2": [round(x[1], 2) for x in y2[-len(y10):]],
            "spread_2s10s": round(y10[-1][1] - y2[-1][1], 2),
            "tips10": round(tips10[-1][1], 2) if tips10 else None,
        }

    # 5. CNN 恐慌贪婪指数（纯数字）
    cnn = fetch_cnn_fear_greed()
    if cnn is not None:
        data["cnn"] = cnn

    # 6. 标普 500 站上 50 日线占比（纯数字）
    s5fi = fetch_s5fi()
    if s5fi is not None:
        data["s5fi"] = s5fi

    # 7. AAII 散户情绪（8 周数组）
    aaii = fetch_aaii(limit=8)
    if aaii is not None:
        data["aaii"] = aaii

    # 8. NAAIM 机构仓位（纯数字）
    naaim = fetch_naaim()
    if naaim is not None:
        data["naaim"] = naaim

    # 9. CBOE Put/Call 比率（纯数字）
    put_call = fetch_put_call()
    if put_call is not None:
        data["put_call"] = put_call

    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Updated {DATA_FILE} at {now_str}")


if __name__ == "__main__":
    main()
