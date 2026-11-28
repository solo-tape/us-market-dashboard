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
    url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
    headers = {**HEADERS, "Referer": "https://edition.cnn.com/"}
    try:
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        score = r.json().get("fear_and_greed", {}).get("score")
        if score is not None:
            return round(float(score), 1)
    except Exception as e:
        print(f"CNN fetch failed: {e}")
    return None


def fetch_s5fi():
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
        print(f"S5FI failed: {e}")
    return None


def fetch_aaii(limit=8):
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
            scale = 100.0 if df["bull"].max() <= 1.0 else 1.0
            return {
                "dates": df["date"].dt.strftime("%m/%d").tolist(),
                "bull": [round(float(x) * scale, 1) for x in df["bull"].tolist()],
                "bear": [round(float(x) * scale, 1) for x in df["bear"].tolist()],
                "neutral": [round(float(x) * scale, 1) for x in df["neutral"].tolist()],
            }
    except Exception as e:
        print(f"AAII fetch failed: {e}")
    return None


def fetch_naaim():
    url = "https://www.naaim.org/programs/naaim-exposure-index/"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        try:
            tables = pd.read_html(io.StringIO(r.text))
            for tbl in tables:
                for col in tbl.columns:
                    if "naaim" in str(col).lower() or "number" in str(col).lower():
                        val = pd.to_numeric(tbl[col], errors="coerce").dropna()
                        if not val.empty:
                            return round(float(val.iloc[0]), 2)
        except ValueError:
            pass

        soup = BeautifulSoup(r.text, "lxml")
        text = soup.get_text(" ", strip=True)
        for pat in [
            r"NAAIM\s+Exposure\s+Index[^0-9]{0,40}(\d{1,3}\.\d{1,2})",
            r"Current\s+Value[^0-9]{0,20}(\d{1,3}\.\d{1,2})",
        ]:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                return round(float(m.group(1)), 2)
    except Exception as e:
        print(f"NAAIM fetch failed: {e}")
    return None


def fetch_put_call():
    """优先拉取 CBOE 公开 Equity Put/Call CSV，失败再尝试 YCharts"""
    cboe_urls = [
        "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/equitypc.csv",
        "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/totalpc.csv",
    ]
    for url in cboe_urls:
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text), skiprows=2)
            ratio_cols = [c for c in df.columns if "p/c" in str(c).lower() or "ratio" in str(c).lower()]
            if ratio_cols:
                vals = pd.to_numeric(df[ratio_cols[-1]], errors="coerce").dropna()
                if len(vals) >= 10:
                    return round(float(vals.tail(10).mean()), 2)
                if not vals.empty:
                    return round(float(vals.iloc[-1]), 2)
        except Exception as e:
            print(f"CBOE CSV ({url}) failed: {e}")

    try:
        r = requests.get("https://ycharts.com/indicators/cboe_equity_put_call_ratio", headers=HEADERS, timeout=15)
        r.raise_for_status()
        text = BeautifulSoup(r.text, "lxml").get_text(" ", strip=True)
        m = re.search(r"(\d\.\d{2})\s+for\s+[A-Z][a-z]{2}\s+\d{1,2},\s+\d{4}", text)
        if m:
            return round(float(m.group(1)), 2)
    except Exception as e:
        print(f"YCharts Put/Call failed: {e}")
    return None


def fetch_zq_implied_rate():
    """抓取 CBOT 30天联邦基金利率期货 (ZQ=F) 隐含利率"""
    try:
        df = yf.Ticker("ZQ=F").history(period="5d")
        if not df.empty:
            px = float(df["Close"].iloc[-1])
            if 85.0 < px < 100.0:
                return round(100.0 - px, 2)
    except Exception as e:
        print(f"ZQ=F fetch failed: {e}")
    return None


def main():
    data = load_existing()
    # 清理旧的估算占位字段
    data.pop("cds", None)
    data.pop("fedwatch", None)

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

    # 2. 费城半导体 SOX 及真实 SOX/SPX 50 日均值
    sox = fetch_yf_history("^SOX", period="1y")
    if not sox.empty and not spx.empty:
        sox_aligned = sox["Close"]
        spx_aligned = spx["Close"].reindex(sox_aligned.index).ffill()
        ratio_series = sox_aligned / spx_aligned
        ratio_50ma = ratio_series.rolling(50).mean().iloc[-1]

        recent_sox = sox.tail(15)
        sox_latest = float(sox["Close"].iloc[-1])
        spx_latest = float(spx["Close"].iloc[-1])
        data["sox"] = {
            "dates": recent_sox.index.tolist(),
            "close": [round(x, 2) for x in recent_sox["Close"].tolist()],
            "latest": round(sox_latest, 2),
            "sox_spx_ratio": round(sox_latest / spx_latest, 3),
            "sox_spx_50ma": round(float(ratio_50ma), 3) if pd.notna(ratio_50ma) else None,
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

    # 4. FRED 高收益债利差（HY OAS）+ 真实分级信用利差（IG OAS / CCC OAS）
    hy = fetch_fred_series("BAMLH0A0HYM2", limit=15)
    if hy:
        data["hy_oas"] = {
            "dates": [x[0] for x in hy],
            "values": [round(x[1], 2) for x in hy],
            "latest": round(hy[-1][1], 2),
        }

    ig = fetch_fred_series("BAMLC0A0CM", limit=5)
    ccc = fetch_fred_series("BAMLH0A3HYC", limit=5)
    if ig or ccc:
        credit_detail = data.get("credit_detail", {})
        if ig:
            credit_detail["ig_oas_bps"] = int(round(ig[-1][1] * 100))
        if ccc:
            credit_detail["ccc_oas_bps"] = int(round(ccc[-1][1] * 100))
        data["credit_detail"] = credit_detail

    # 5. FRED 美债利率 + 真实政策利率路径（EFFR vs 联邦基金期货/2Y）
    y10 = fetch_fred_series("DGS10", limit=15)
    y2 = fetch_fred_series("DGS2", limit=15)
    tips10 = fetch_fred_series("DFII10", limit=5)
    dff = fetch_fred_series("DFF", limit=5)
    if y10 and y2:
        y10_val, y2_val = y10[-1][1], y2[-1][1]
        data["rates"] = {
            "dates": [x[0] for x in y10],
            "y10": [round(x[1], 2) for x in y10],
            "y2": [round(x[1], 2) for x in y2[-len(y10):]],
            "spread_2s10s": round(y10_val - y2_val, 2),
            "tips10": round(tips10[-1][1], 2) if tips10 else None,
        }
        if dff:
            effr = round(dff[-1][1], 2)
            zq_rate = fetch_zq_implied_rate()
            market_rate = zq_rate if zq_rate is not None else round(y2_val, 2)
            diff_bps = int(round((market_rate - effr) * 100))
            data["fed_policy"] = {
                "effr": effr,
                "implied_rate": market_rate,
                "source": "ZQ期货" if zq_rate is not None else "2Y美债",
                "diff_bps": diff_bps,
            }

    # 6. 情绪与广度指标
    cnn = fetch_cnn_fear_greed()
    if cnn is not None:
        data["cnn"] = cnn

    s5fi = fetch_s5fi()
    if s5fi is not None:
        data["s5fi"] = s5fi

    aaii = fetch_aaii(limit=8)
    if aaii is not None:
        data["aaii"] = aaii

    naaim = fetch_naaim()
    if naaim is not None:
        data["naaim"] = naaim

    put_call = fetch_put_call()
    if put_call is not None:
        data["put_call"] = put_call

    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Updated {DATA_FILE} at {now_str}")


if __name__ == "__main__":
    main()
