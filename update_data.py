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
    """抓取 CNN 恐慌贪婪指数官方 JSON 接口"""
    url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
    headers = {**HEADERS, "Referer": "https://edition.cnn.com/"}
    try:
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        fg = r.json().get("fear_and_greed", {})
        score = fg.get("score")
        rating = fg.get("rating", "")
        if score is not None:
            return {
                "score": round(float(score), 1),
                "rating": rating,
                "previous_close": round(float(fg.get("previous_close", score)), 1),
            }
    except Exception as e:
        print(f"CNN Fear & Greed fetch failed: {e}")
    return None


def fetch_s5fi():
    """直接抓取标普 500 成分股并计算站上 50 日均线的股票占比 ($S5FI)"""
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
            return {
                "latest": round(float(pct), 2),
                "sample_size": int(valid.sum()),
            }
    except Exception as e:
        print(f"S5FI calculation failed: {e}")
    return None


def fetch_aaii():
    """抓取 AAII 散户情绪调查（优先官方 XLS，失败则解析网页）"""
    xls_url = "https://www.aaii.com/files/surveys/sentiment.xls"
    try:
        r = requests.get(xls_url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        df = pd.read_excel(io.BytesIO(r.content), skiprows=3)
        df = df.dropna(subset=[df.columns[1], df.columns[2], df.columns[3]])
        # 取最新有效行
        for i in range(len(df) - 1, -1, -1):
            row = df.iloc[i]
            try:
                bull = float(row.iloc[1])
                neut = float(row.iloc[2])
                bear = float(row.iloc[3])
                # 兼容小数(0.45)与百分比(45.0)
                if bull <= 1.0 and bear <= 1.0:
                    bull, neut, bear = bull * 100, neut * 100, bear * 100
                if 0 < bull + neut + bear <= 105:
                    return {
                        "bullish": round(bull, 1),
                        "neutral": round(neut, 1),
                        "bearish": round(bear, 1),
                    }
            except (ValueError, TypeError):
                continue
    except Exception as e:
        print(f"AAII XLS fetch failed: {e}")
    return None


def fetch_naaim():
    """抓取 NAAIM 机构股票仓位指数"""
    url = "https://www.naaim.org/programs/naaim-exposure-index/"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")
        # 查找页面中的最新表格数值
        tables = pd.read_html(io.StringIO(r.text))
        for tbl in tables:
            for col in tbl.columns:
                if "naaim" in str(col).lower() or "number" in str(col).lower():
                    val = pd.to_numeric(tbl[col], errors="coerce").dropna()
                    if not val.empty:
                        return {"latest": round(float(val.iloc[0]), 2)}
        # 兜底：正则匹配页面中的最新读数
        text = soup.get_text(" ", strip=True)
        m = re.search(r"NAAIM Exposure Index.*?(\d{1,3}\.\d{1,2})", text)
        if m:
            return {"latest": round(float(m.group(1)), 2)}
    except Exception as e:
        print(f"NAAIM fetch failed: {e}")
    return None


def fetch_put_call():
    """抓取 CBOE 全市场 Put/Call 比率（备用 YCharts 公开页面）"""
    url = "https://ycharts.com/indicators/cboe_equity_put_call_ratio"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")
        text = soup.get_text(" ", strip=True)
        m = re.search(r"(\d\.\d{2})\s+for\s+[A-Z][a-z]{2}\s+\d{1,2},\s+\d{4}", text)
        if m:
            return {"latest": round(float(m.group(1)), 2)}
    except Exception as e:
        print(f"Put/Call fetch failed: {e}")
    return None


def main():
    data = load_existing()
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

    # 5. 新增：CNN 恐慌贪婪指数
    cnn = fetch_cnn_fear_greed()
    if cnn:
        data["cnn"] = cnn

    # 6. 新增：标普 500 站上 50 日线占比 ($S5FI)
    s5fi = fetch_s5fi()
    if s5fi:
        data["s5fi"] = s5fi

    # 7. 新增：AAII 散户看多/看空比
    aaii = fetch_aaii()
    if aaii:
        data["aaii"] = aaii

    # 8. 新增：NAAIM 机构仓位
    naaim = fetch_naaim()
    if naaim:
        data["naaim"] = naaim

    # 9. 新增：CBOE Put/Call 比率
    put_call = fetch_put_call()
    if put_call:
        data["put_call"] = put_call

    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Updated {DATA_FILE} at {now_str}")


if __name__ == "__main__":
    main()
