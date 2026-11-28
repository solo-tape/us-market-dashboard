import io
import json
import os
import re
from datetime import datetime, timedelta, timezone
from urllib.parse import urljoin

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


def fetch_yf_raw(ticker, period="1y"):
    """保留完整日期索引（YYYY-MM-DD），避免跨年同月日重复导致对齐出错"""
    df = yf.Ticker(ticker).history(period=period)
    if df.empty:
        return pd.DataFrame()
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df = df[~df.index.duplicated(keep="last")]
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


def _aaii_rows_from_xls():
    """AAII 官方 XLS：日期为发布日（周四），对应调查周截至前一日（周三）。
    返回 [(week_ending, bull, neutral, bear), ...]"""
    xls_url = "https://www.aaii.com/files/surveys/sentiment.xls"
    r = requests.get(xls_url, headers=HEADERS, timeout=15)
    r.raise_for_status()
    df = pd.read_excel(io.BytesIO(r.content), skiprows=3)
    df = df.iloc[:, :4].copy()
    df.columns = ["date", "bull", "neutral", "bear"]

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ["bull", "neutral", "bear"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["date", "bull", "neutral", "bear"])
    df = df.sort_values("date")

    rows = []
    if not df.empty:
        scale = 100.0 if df["bull"].max() <= 1.0 else 1.0
        for _, row in df.iterrows():
            week_ending = row["date"] - pd.Timedelta(days=1)
            rows.append((
                week_ending,
                round(float(row["bull"]) * scale, 1),
                round(float(row["neutral"]) * scale, 1),
                round(float(row["bear"]) * scale, 1),
            ))
    return rows


def _aaii_rows_from_web():
    """AAII 官网 sentimentsurvey 页的 Recent weekly results 表通常比 XLS 早更新一周，
    用它补 XLS 尚未收录的最新周。表格列序为 Bullish / Neutral / Bearish。"""
    url = "https://www.aaii.com/sentimentsurvey"
    r = requests.get(url, headers=HEADERS, timeout=15)
    r.raise_for_status()
    text = BeautifulSoup(r.text, "html.parser").get_text(" ")
    rows = []
    seen = set()
    for m in re.finditer(
        r"(\d{1,2})/(\d{1,2})/(\d{4})\s+([\d.]+)%\s+([\d.]+)%\s+([\d.]+)%",
        text,
    ):
        month, day, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            week_ending = pd.Timestamp(year=year, month=month, day=day)
        except ValueError:
            continue
        if week_ending in seen:
            continue
        seen.add(week_ending)
        rows.append((
            week_ending,
            round(float(m.group(4)), 1),
            round(float(m.group(5)), 1),
            round(float(m.group(6)), 1),
        ))
    return rows


def fetch_aaii(limit=8):
    merged = {}
    try:
        for row in _aaii_rows_from_xls():
            merged[row[0]] = row
    except Exception as e:
        print(f"AAII XLS fetch failed: {e}")
    try:
        for row in _aaii_rows_from_web():
            merged[row[0]] = row
    except Exception as e:
        print(f"AAII web fetch failed: {e}")

    if not merged:
        return None
    rows = [merged[k] for k in sorted(merged)][-limit:]
    return {
        # 展示日期沿用 XLS 口径：发布日 = 调查周截止日 + 1 天
        "dates": [(r[0] + pd.Timedelta(days=1)).strftime("%m/%d") for r in rows],
        "bull": [r[1] for r in rows],
        "neutral": [r[2] for r in rows],
        "bear": [r[3] for r in rows],
    }


def fetch_naaim():
    """换源抓取 NAAIM 机构仓位，走 Jina Reader 提取纯文本，匹配最新周度读数"""
    candidates = [
        "https://r.jina.ai/https://www.investing.com/economic-calendar/naaim-exposure-index-1906",
        "https://r.jina.ai/https://en.macromicro.me/charts/41621/naaim-exposure-index",
        "https://r.jina.ai/https://www.barchart.com/stocks/quotes/$NAAIM/overview",
    ]
    for url in candidates:
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            if r.status_code != 200 or not r.text:
                continue
            text = r.text
            # 匹配 NAAIM 仓位常见数值区间（-100 到 200 之间的小数）
            for pat in [
                r"NAAIM\s+Exposure\s+Index[\s\S]{0,120}?(-?\d{1,3}\.\d{1,2})",
                r"Latest[\s\S]{0,60}?(-?\d{1,3}\.\d{1,2})\s*%?",
            ]:
                m = re.search(pat, text, re.IGNORECASE)
                if m:
                    val = round(float(m.group(1)), 2)
                    if -100 <= val <= 200 and val != 1906.0:
                        print(f"NAAIM fetched from {url}: {val}")
                        return val, None
        except Exception as e:
            print(f"NAAIM source failed ({url}): {e}")
    return None, None
    
def fetch_cftc_spx():
    """直连 CFTC 官方免费 TFF API，获取标普 500 E-mini (13874A) 近 52 周机构持仓（仅作参考，不计分）"""
    url = "https://publicreporting.cftc.gov/resource/gpe5-46if.json"
    params = {
        "$where": "cftc_contract_market_code='13874A'",
        "$order": "report_date_as_yyyy_mm_dd DESC",
        "$limit": 52,
    }
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=15)
        r.raise_for_status()
        rows = r.json()
        if not isinstance(rows, list) or not rows:
            return None

        parsed = []
        for row in rows:
            try:
                oi = float(row.get("open_interest_all") or 0)
                if oi <= 0:
                    continue
                am_long = float(row.get("asset_mgr_positions_long") or row.get("asset_mgr_positions_long_all") or 0)
                am_short = float(row.get("asset_mgr_positions_short") or row.get("asset_mgr_positions_short_all") or 0)
                lev_long = float(row.get("lev_money_positions_long") or row.get("lev_money_positions_long_all") or 0)
                lev_short = float(row.get("lev_money_positions_short") or row.get("lev_money_positions_short_all") or 0)
                dt_raw = str(row.get("report_date_as_yyyy_mm_dd") or "")[:10]
                if len(dt_raw) < 10:
                    continue

                am_net = am_long - am_short
                lev_net = lev_long - lev_short
                parsed.append({
                    "date": dt_raw[5:].replace("-", "/"),
                    "am_net_k": round(am_net / 1000.0, 1),
                    "am_net_pct": round((am_net / oi) * 100.0, 2),
                    "lev_net_k": round(lev_net / 1000.0, 1),
                    "lev_net_pct": round((lev_net / oi) * 100.0, 2),
                })
            except (TypeError, ValueError):
                continue

        if not parsed:
            return None

        latest = parsed[0]
        prev = parsed[1] if len(parsed) > 1 else latest

        # 计算资管机构净多头占比的近 1 年（52周）历史分位数 (0-100%)
        series = [p["am_net_pct"] for p in parsed]
        below = sum(1 for x in series if x < latest["am_net_pct"])
        equal = sum(1 for x in series if x == latest["am_net_pct"])
        pct_rank_1y = round(((below + 0.5 * equal) / len(series)) * 100.0, 1)

        wow_change = round(latest["am_net_pct"] - prev["am_net_pct"], 2)
        result = {
            "report_date": latest["date"],
            "asset_mgr_net_pct": latest["am_net_pct"],
            "asset_mgr_net_k": latest["am_net_k"],
            "asset_mgr_pct_rank_1y": pct_rank_1y,
            "asset_mgr_wow_pct": wow_change,
            "lev_funds_net_pct": latest["lev_net_pct"],
            "lev_funds_net_k": latest["lev_net_k"],
            "weeks": len(parsed),
        }
        print(
            f"CFTC SPX ({latest['date']}): AssetMgr {latest['am_net_pct']}% "
            f"(1Y rank {pct_rank_1y}%), LevFunds {latest['lev_net_pct']}%"
        )
        return result
    except Exception as e:
        print(f"CFTC SPX fetch failed: {e}")
    return None


def fetch_put_call():
    """直接抓取 CBOE 官方每日 JSON 接口，回溯最近 10 个交易日计算真实 Equity Put/Call 10D 均值"""
    vals = []
    today = datetime.now(timezone.utc).date()

    for offset in range(0, 25):
        if len(vals) >= 10:
            break
        d = today - timedelta(days=offset)
        if d.weekday() >= 5:
            continue
        url = f"https://cdn.cboe.com/data/us/options/market_statistics/daily/{d.strftime('%Y-%m-%d')}_daily_options"
        try:
            r = requests.get(url, headers=HEADERS, timeout=10)
            if r.status_code != 200:
                continue
            payload = r.json()
            for item in payload.get("ratios", []):
                name = str(item.get("name", "")).upper()
                if "EQUITY" in name and "PUT/CALL" in name:
                    v = float(item["value"])
                    if 0.1 < v < 3.0:
                        vals.append(v)
                    break
        except Exception:
            continue

    if vals:
        avg_val = round(sum(vals) / len(vals), 2)
        print(f"CBOE Equity Put/Call ({len(vals)}D avg): {avg_val}")
        return avg_val

    try:
        r = requests.get("https://ycharts.com/indicators/cboe_equity_put_call_ratio", headers=HEADERS, timeout=15)
        r.raise_for_status()
        text = BeautifulSoup(r.text, "lxml").get_text(" ", strip=True)
        m = re.search(r"(\d\.\d{2})\s+for\s+[A-Z][a-z]{2}\s+\d{1,2},?\s+\d{4}", text)
        if m:
            val = round(float(m.group(1)), 2)
            print(f"YCharts Put/Call: {val}")
            return val
    except Exception as e:
        print(f"YCharts Put/Call failed: {e}")
    return None


def fetch_polymarket_fed():
    """从 Polymarket Gamma API 自动抓取下一次未结算 FOMC 会议的加/降息市场概率"""
    candidates = []
    try:
        r = requests.get(
            "https://gamma-api.polymarket.com/public-search",
            params={"q": "Fed decision in", "limit_per_type": 20},
            headers=HEADERS,
            timeout=15,
        )
        r.raise_for_status()
        payload = r.json()
        if isinstance(payload, dict) and isinstance(payload.get("events"), list):
            candidates.extend(payload["events"])
    except Exception as e:
        print(f"Polymarket public-search failed: {e}")

    if not candidates:
        try:
            r = requests.get(
                "https://gamma-api.polymarket.com/events",
                params={
                    "active": "true",
                    "closed": "false",
                    "limit": 100,
                    "order": "volume24hr",
                    "ascending": "false",
                },
                headers=HEADERS,
                timeout=15,
            )
            r.raise_for_status()
            if isinstance(r.json(), list):
                candidates.extend(r.json())
        except Exception as e:
            print(f"Polymarket events fallback failed: {e}")

    now_utc = datetime.now(timezone.utc)
    month_map = {
        "january": "1月", "february": "2月", "march": "3月", "april": "4月",
        "may": "5月", "june": "6月", "july": "7月", "august": "8月",
        "september": "9月", "october": "10月", "november": "11月", "december": "12月",
    }

    valid_events = []
    seen_slugs = set()
    for ev in candidates:
        slug = ev.get("slug") or ""
        if slug in seen_slugs:
            continue
        seen_slugs.add(slug)

        title = str(ev.get("title", ""))
        if not re.search(r"fed\s+decision\s+in\s+[a-z]+", title, re.IGNORECASE):
            continue
        if ev.get("closed") is True or ev.get("active") is False:
            continue

        end_raw = ev.get("endDate") or ev.get("end_date_iso") or ""
        try:
            end_dt = datetime.fromisoformat(str(end_raw).replace("Z", "+00:00"))
            if end_dt.tzinfo is None:
                end_dt = end_dt.replace(tzinfo=timezone.utc)
            if end_dt < now_utc:
                continue
        except Exception:
            end_dt = datetime.max.replace(tzinfo=timezone.utc)

        valid_events.append((end_dt, ev))

    if not valid_events:
        return None

    valid_events.sort(key=lambda x: x[0])
    _, target_ev = valid_events[0]

    markets = target_ev.get("markets") or []
    if (not markets or "outcomePrices" not in markets[0]) and target_ev.get("slug"):
        try:
            r = requests.get(
                "https://gamma-api.polymarket.com/events",
                params={"slug": target_ev["slug"]},
                headers=HEADERS,
                timeout=15,
            )
            r.raise_for_status()
            ev_list = r.json()
            if isinstance(ev_list, list) and ev_list:
                target_ev = ev_list[0]
                markets = target_ev.get("markets") or []
        except Exception as e:
            print(f"Polymarket slug detail fetch failed: {e}")

    title = str(target_ev.get("title", ""))
    m_month = re.search(r"in\s+([a-z]+)", title, re.IGNORECASE)
    meeting_cn = month_map.get(m_month.group(1).lower(), m_month.group(1)) if m_month else "下次"
    meeting_label = f"{meeting_cn}议息"

    parsed_outcomes = []
    for mkt in markets:
        if mkt.get("closed") is True:
            continue
        raw_label = str(mkt.get("groupItemTitle") or mkt.get("question") or "").strip()
        if not raw_label:
            continue

        prob = None
        try:
            bid = float(mkt.get("bestBid")) if mkt.get("bestBid") is not None else None
            ask = float(mkt.get("bestAsk")) if mkt.get("bestAsk") is not None else None
            if bid is not None and ask is not None and 0.0 <= bid <= ask <= 1.0 and (ask - bid) <= 0.12:
                prob = (bid + ask) / 2.0
        except (TypeError, ValueError):
            pass

        if prob is None:
            prices = mkt.get("outcomePrices")
            if isinstance(prices, str):
                try:
                    prices = json.loads(prices)
                except Exception:
                    prices = []
            if isinstance(prices, list) and len(prices) > 0:
                try:
                    prob = float(prices[0])
                except (TypeError, ValueError):
                    prob = None

        if prob is None:
            continue

        low = raw_label.lower()
        if "no change" in low or "hold" in low or "unchanged" in low:
            cn_action = "按兵不动"
            direction = "hold"
        elif "increase" in low or "hike" in low:
            bp = re.search(r"(\d+\+?)\s*bps?", low)
            cn_action = f"加息{bp.group(1)}bp" if bp else "加息"
            direction = "hike"
        elif "decrease" in low or "cut" in low:
            bp = re.search(r"(\d+\+?)\s*bps?", low)
            cn_action = f"降息{bp.group(1)}bp" if bp else "降息"
            direction = "cut"
        else:
            cn_action = raw_label[:14]
            direction = "other"

        parsed_outcomes.append({
            "action": cn_action,
            "direction": direction,
            "prob": round(prob * 100.0, 1),
        })

    if not parsed_outcomes:
        return None

    parsed_outcomes.sort(key=lambda x: x["prob"], reverse=True)
    top = parsed_outcomes[0]
    runner = parsed_outcomes[1] if len(parsed_outcomes) > 1 else None

    return {
        "meeting": meeting_label,
        "action": top["action"],
        "direction": top["direction"],
        "prob": top["prob"],
        "runner_up": f"次选 {runner['action']} {runner['prob']}%" if runner and runner["prob"] >= 0.5 else "",
        "event_title": title,
    }


def fetch_zq_implied_rate():
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
    data.pop("cds", None)
    data.pop("fedwatch", None)

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    data["updated_at"] = now_str

    # 1. 标普 500 及 50/200 日均线
    spx = fetch_yf_raw("^GSPC", period="1y")
    if not spx.empty:
        spx["MA50"] = spx["Close"].rolling(50).mean()
        spx["MA200"] = spx["Close"].rolling(200).mean()
        recent = spx.tail(15)
        latest = spx.iloc[-1]
        data["spx"] = {
            "dates": recent.index.strftime("%m/%d").tolist(),
            "close": [round(x, 2) for x in recent["Close"].tolist()],
            "ma50": round(float(latest["MA50"]), 2) if pd.notna(latest["MA50"]) else None,
            "ma200": round(float(latest["MA200"]), 2) if pd.notna(latest["MA200"]) else None,
            "latest": round(float(latest["Close"]), 2),
        }

    # 2. 费城半导体 SOX 及真实 SOX/SPX 50 日均值
    sox = fetch_yf_raw("^SOX", period="1y")
    if not sox.empty and not spx.empty:
        aligned = pd.concat(
            [sox["Close"].rename("sox"), spx["Close"].rename("spx")],
            axis=1,
        ).dropna()
        ratio_series = aligned["sox"] / aligned["spx"]
        ratio_50ma = ratio_series.rolling(50).mean().iloc[-1]

        recent_sox = sox.tail(15)
        sox_latest = float(sox["Close"].iloc[-1])
        spx_latest = float(spx["Close"].iloc[-1])
        data["sox"] = {
            "dates": recent_sox.index.strftime("%m/%d").tolist(),
            "close": [round(x, 2) for x in recent_sox["Close"].tolist()],
            "latest": round(sox_latest, 2),
            "sox_spx_ratio": round(sox_latest / spx_latest, 3),
            "sox_spx_50ma": round(float(ratio_50ma), 3) if pd.notna(ratio_50ma) else None,
        }

    # 3. CBOE VIX 期限结构
    vix_map = {"VIX1D": "^VIX1D", "VIX9D": "^VIX9D", "VIX": "^VIX", "VIX3M": "^VIX3M"}
    vix_vals = {}
    for label, sym in vix_map.items():
        df = fetch_yf_raw(sym, period="5d")
        if not df.empty:
            vix_vals[label] = round(float(df["Close"].iloc[-1]), 2)
    if "VIX" in vix_vals and "VIX3M" in vix_vals and vix_vals["VIX3M"] > 0:
        vix_vals["ratio"] = round(vix_vals["VIX"] / vix_vals["VIX3M"], 2)
    if vix_vals:
        data["vix_term"] = vix_vals

    # 4. FRED 高收益债利差（HY OAS）+ 分级信用利差（IG OAS / CCC OAS）
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

    # 5. FRED 美债利率 + 真实基准利率 (EFFR)
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

    # 6. Polymarket 美联储议息概率
    poly_fed = fetch_polymarket_fed()
    if poly_fed is not None:
        data["polymarket_fed"] = poly_fed
        print(f"Polymarket Fed: {poly_fed['meeting']} {poly_fed['action']} {poly_fed['prob']}%")

    # 7. 情绪与广度指标
    cnn = fetch_cnn_fear_greed()
    if cnn is not None:
        data["cnn"] = cnn

    s5fi = fetch_s5fi()
    if s5fi is not None:
        data["s5fi"] = s5fi

    aaii = fetch_aaii(limit=8)
    if aaii is not None:
        data["aaii"] = aaii

    naaim_val, naaim_date = fetch_naaim()
    if naaim_val is not None:
        data["naaim"] = naaim_val
        if naaim_date:
            data["naaim_date"] = naaim_date

    put_call = fetch_put_call()
    if put_call is not None:
        data["put_call"] = put_call

    # 8. CFTC 标普 500 期货机构持仓（仅参考展示，不参与打分）
    cftc_spx = fetch_cftc_spx()
    if cftc_spx is not None:
        data["cftc_spx"] = cftc_spx

    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Updated {DATA_FILE} at {now_str}")


if __name__ == "__main__":
    main()
