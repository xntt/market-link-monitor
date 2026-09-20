"""
币市 × 股市 联动监控（完整初始版）
- 标的：HOOD/MSTR/CRCL + BTC/ETH/SOL + Hood: PONS/CASHCAT/AI/TENDIES
- BTC/ETH/SOL：Binance 多域名 → yfinance → CoinGecko
- 15 分钟自动刷新 + 手动刷新
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

# =========================
# 配置
# =========================

STOCKS = [
    {"id": "HOOD", "symbol": "HOOD", "name": "Robinhood"},
    {"id": "MSTR", "symbol": "MSTR", "name": "MicroStrategy"},
    {"id": "CRCL", "symbol": "CRCL", "name": "Circle"},
]

MAJORS = [
    {"id": "BTC", "symbol": "BTCUSDT", "name": "Bitcoin"},
    {"id": "ETH", "symbol": "ETHUSDT", "name": "Ethereum"},
    {"id": "SOL", "symbol": "SOLUSDT", "name": "Solana"},
]

HOOD_TOKENS = [
    {"id": "PONS", "name": "PONS", "chain": "robinhood", "address": ""},
    {"id": "CASHCAT", "name": "CASHCAT", "chain": "robinhood", "address": ""},
    {"id": "AI", "name": "AI", "chain": "robinhood", "address": ""},
    {"id": "TENDIES", "name": "TENDIES", "chain": "robinhood", "address": ""},
]

ALERT_PCT_1H = 3.0
CORR_LOOKBACK = 30
REFRESH_SECONDS = 15 * 60

BINANCE_HOSTS = [
    "https://api.binance.com",
    "https://data-api.binance.vision",
    "https://api1.binance.com",
    "https://api2.binance.com",
]

YF_CRYPTO = {
    "BTCUSDT": "BTC-USD",
    "ETHUSDT": "ETH-USD",
    "SOLUSDT": "SOL-USD",
}

CG_IDS = {
    "BTCUSDT": "bitcoin",
    "ETHUSDT": "ethereum",
    "SOLUSDT": "solana",
}

DEXSCREENER_BASE = "https://api.dexscreener.com/latest/dex"

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "market-link-monitor/1.0"})


# =========================
# 工具请求
# =========================

def _binance_get(path: str, params: Optional[dict] = None, timeout: int = 12):
    last_err = None
    for host in BINANCE_HOSTS:
        try:
            r = SESSION.get(f"{host}{path}", params=params or {}, timeout=timeout)
            if r.status_code == 200:
                return r.json(), host
            last_err = f"{host} HTTP {r.status_code}"
        except Exception as e:
            last_err = f"{host} {e}"
    raise RuntimeError(last_err or "binance all hosts failed")


# =========================
# 美股
# =========================

def fetch_stock_snapshot(symbol: str) -> Dict[str, Any]:
    out = {
        "id": symbol,
        "type": "stock",
        "price": None,
        "chg_1h_pct": None,
        "chg_1d_pct": None,
        "volume": None,
        "hist_close": pd.Series(dtype=float),
        "source": "yfinance",
        "error": None,
    }
    try:
        t = yf.Ticker(symbol)
        try:
            fi = t.fast_info
            price = float(getattr(fi, "last_price", None) or getattr(fi, "lastPrice", None) or 0)
            prev = float(getattr(fi, "previous_close", None) or getattr(fi, "previousClose", None) or 0)
            volume = float(getattr(fi, "last_volume", None) or getattr(fi, "lastVolume", None) or 0)
        except Exception:
            price, prev, volume = 0.0, 0.0, 0.0

        hist = t.history(period="3mo", interval="1d")
        if (not price) and hist is not None and not hist.empty:
            price = float(hist["Close"].iloc[-1])
            if len(hist) > 1:
                prev = float(hist["Close"].iloc[-2])
            volume = float(hist["Volume"].iloc[-1]) if "Volume" in hist.columns else 0

        out["price"] = price if price else None
        out["volume"] = volume
        out["chg_1d_pct"] = ((price / prev) - 1) * 100 if price and prev else None
        if hist is not None and not hist.empty:
            out["hist_close"] = hist["Close"].astype(float)

        try:
            intraday = t.history(period="1d", interval="5m")
            if intraday is not None and len(intraday) >= 12:
                p0 = float(intraday["Close"].iloc[-12])
                p1 = float(intraday["Close"].iloc[-1])
                if p0:
                    out["chg_1h_pct"] = (p1 / p0 - 1) * 100
        except Exception:
            pass

        if not out["price"]:
            out["error"] = "no price"
    except Exception as e:
        out["error"] = str(e)
    return out


# =========================
# 主流币：Binance → yfinance → CoinGecko
# =========================

def fetch_major_from_yfinance(binance_symbol: str) -> Dict[str, Any]:
    yf_sym = YF_CRYPTO.get(binance_symbol)
    out = {
        "id": binance_symbol.replace("USDT", ""),
        "type": "major",
        "price": None,
        "chg_1h_pct": None,
        "chg_1d_pct": None,
        "volume": None,
        "quote_volume": None,
        "hist_close": pd.Series(dtype=float),
        "source": "yfinance",
        "error": None,
    }
    if not yf_sym:
        out["error"] = "no yfinance map"
        return out
    try:
        t = yf.Ticker(yf_sym)
        hist = t.history(period="3mo", interval="1d")
        if hist is None or hist.empty:
            out["error"] = "yfinance empty"
            return out
        price = float(hist["Close"].iloc[-1])
        prev = float(hist["Close"].iloc[-2]) if len(hist) > 1 else price
        out["price"] = price
        out["chg_1d_pct"] = (price / prev - 1) * 100 if prev else None
        out["volume"] = float(hist["Volume"].iloc[-1]) if "Volume" in hist.columns else None
        out["hist_close"] = hist["Close"].astype(float)
        try:
            intra = t.history(period="1d", interval="5m")
            if intra is not None and len(intra) >= 12:
                p0 = float(intra["Close"].iloc[-12])
                p1 = float(intra["Close"].iloc[-1])
                out["chg_1h_pct"] = (p1 / p0 - 1) * 100 if p0 else None
        except Exception:
            pass
    except Exception as e:
        out["error"] = str(e)
    return out


def fetch_major_from_coingecko(binance_symbol: str) -> Dict[str, Any]:
    cid = CG_IDS.get(binance_symbol)
    out = {
        "id": binance_symbol.replace("USDT", ""),
        "type": "major",
        "price": None,
        "chg_1h_pct": None,
        "chg_1d_pct": None,
        "volume": None,
        "quote_volume": None,
        "hist_close": pd.Series(dtype=float),
        "source": "coingecko",
        "error": None,
    }
    if not cid:
        out["error"] = "no cg id"
        return out
    try:
        r = SESSION.get(
            "https://api.coingecko.com/api/v3/coins/markets",
            params={"vs_currency": "usd", "ids": cid},
            timeout=15,
        )
        r.raise_for_status()
        arr = r.json()
        if not arr:
            out["error"] = "cg empty"
            return out
        j = arr[0]
        out["price"] = float(j.get("current_price") or 0) or None
        out["chg_1d_pct"] = float(j.get("price_change_percentage_24h") or 0)
        out["volume"] = float(j.get("total_volume") or 0)
        out["quote_volume"] = out["volume"]

        r2 = SESSION.get(
            f"https://api.coingecko.com/api/v3/coins/{cid}/market_chart",
            params={"vs_currency": "usd", "days": "90"},
            timeout=15,
        )
        if r2.status_code == 200:
            prices = r2.json().get("prices") or []
            if prices:
                out["hist_close"] = pd.Series([float(p[1]) for p in prices])
    except Exception as e:
        out["error"] = str(e)
    return out


def fetch_binance_snapshot(symbol: str) -> Dict[str, Any]:
    out = {
        "id": symbol.replace("USDT", ""),
        "type": "major",
        "price": None,
        "chg_1h_pct": None,
        "chg_1d_pct": None,
        "volume": None,
        "quote_volume": None,
        "hist_close": pd.Series(dtype=float),
        "source": None,
        "error": None,
    }

    # 1) Binance 多域名
    try:
        j, host = _binance_get("/api/v3/ticker/24hr", {"symbol": symbol})
        out["price"] = float(j["lastPrice"])
        out["chg_1d_pct"] = float(j["priceChangePercent"])
        out["volume"] = float(j["volume"])
        out["quote_volume"] = float(j["quoteVolume"])
        out["source"] = f"binance:{host}"

        try:
            kl, _ = _binance_get("/api/v3/klines", {"symbol": symbol, "interval": "1h", "limit": 2})
            if len(kl) >= 2:
                c0, c1 = float(kl[-2][4]), float(kl[-1][4])
                out["chg_1h_pct"] = (c1 / c0 - 1) * 100 if c0 else None
        except Exception:
            pass

        try:
            kl, _ = _binance_get("/api/v3/klines", {"symbol": symbol, "interval": "1d", "limit": 90})
            out["hist_close"] = pd.Series([float(x[4]) for x in kl])
        except Exception:
            pass

        if out["price"]:
            return out
    except Exception as e:
        out["error"] = f"binance: {e}"

    # 2) yfinance
    yf_out = fetch_major_from_yfinance(symbol)
    if yf_out.get("price"):
        if out.get("error"):
            yf_out["error"] = out["error"]
        return yf_out

    # 3) CoinGecko
    cg_out = fetch_major_from_coingecko(symbol)
    if cg_out.get("price"):
        cg_out["error"] = " | ".join(
            x for x in [out.get("error"), yf_out.get("error")] if x
        )
        return cg_out

    out["error"] = " | ".join(
        x
        for x in [
            out.get("error"),
            f"yf:{yf_out.get('error')}",
            f"cg:{cg_out.get('error')}",
        ]
        if x
    )
    return out


# =========================
# Hood / DEX
# =========================

def dexscreener_token(chain: str, address: str = "", name: str = "") -> Dict[str, Any]:
    out = {
        "id": name or (address[:8] if address else "?"),
        "type": "hood",
        "price": None,
        "chg_1h_pct": None,
        "chg_1d_pct": None,
        "volume": None,
        "buys_h1": None,
        "sells_h1": None,
        "txns_h1": None,
        "liquidity": None,
        "pair_url": None,
        "hist_close": pd.Series(dtype=float),
        "source": "dexscreener",
        "error": None,
    }

    pair = None
    try:
        if address:
            r = SESSION.get(f"{DEXSCREENER_BASE}/tokens/{address}", timeout=15)
            r.raise_for_status()
            pairs = r.json().get("pairs") or []
            cand = [
                p
                for p in pairs
                if (p.get("chainId") or "").lower() in (chain.lower(), "robinhood")
            ]
            pool = cand or pairs
            if pool:
                pair = sorted(
                    pool,
                    key=lambda x: float((x.get("liquidity") or {}).get("usd") or 0),
                    reverse=True,
                )[0]
        else:
            r = SESSION.get(f"{DEXSCREENER_BASE}/search", params={"q": name}, timeout=15)
            r.raise_for_status()
            pairs = r.json().get("pairs") or []
            pool = [
                p
                for p in pairs
                if (p.get("chainId") or "").lower() in (chain.lower(), "robinhood")
                and name.lower()
                in (((p.get("baseToken") or {}).get("symbol") or "").lower())
            ]
            if not pool:
                pool = [
                    p
                    for p in pairs
                    if (p.get("chainId") or "").lower() in (chain.lower(), "robinhood")
                ]
            if pool:
                pair = sorted(
                    pool,
                    key=lambda x: float((x.get("liquidity") or {}).get("usd") or 0),
                    reverse=True,
                )[0]
    except Exception as e:
        out["error"] = str(e)
        return out

    if not pair:
        out["error"] = "未找到交易对（建议填写合约地址）"
        return out

    try:
        out["price"] = float(pair.get("priceUsd") or 0) or None
        out["chg_1h_pct"] = float((pair.get("priceChange") or {}).get("h1") or 0)
        out["chg_1d_pct"] = float((pair.get("priceChange") or {}).get("h24") or 0)
        out["volume"] = float((pair.get("volume") or {}).get("h1") or 0)
        tx = (pair.get("txns") or {}).get("h1") or {}
        out["buys_h1"] = int(tx.get("buys") or 0)
        out["sells_h1"] = int(tx.get("sells") or 0)
        out["txns_h1"] = out["buys_h1"] + out["sells_h1"]
        out["liquidity"] = float((pair.get("liquidity") or {}).get("usd") or 0)
        out["pair_url"] = pair.get("url")
        sym = ((pair.get("baseToken") or {}).get("symbol")) or out["id"]
        out["id"] = sym
    except Exception as e:
        out["error"] = f"parse: {e}"
    return out


def fetch_exchange_volumes() -> pd.DataFrame:
    rows = []
    for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
        try:
            j, host = _binance_get("/api/v3/ticker/24hr", {"symbol": sym})
            rows.append(
                {
                    "exchange": host,
                    "symbol": sym,
                    "quote_volume_24h": float(j["quoteVolume"]),
                    "price_chg_pct": float(j["priceChangePercent"]),
                }
            )
        except Exception:
            y = fetch_major_from_yfinance(sym)
            if y.get("price") is not None:
                rows.append(
                    {
                        "exchange": "yfinance",
                        "symbol": sym,
                        "quote_volume_24h": y.get("volume"),
                        "price_chg_pct": y.get("chg_1d_pct"),
                    }
                )
    return pd.DataFrame(rows)


# =========================
# 分析
# =========================

def build_snapshot_table(snapshots: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for s in snapshots:
        rows.append(
            {
                "标的": s.get("id"),
                "类型": s.get("type"),
                "价格": s.get("price"),
                "1h%": s.get("chg_1h_pct"),
                "24h/日%": s.get("chg_1d_pct"),
                "量能": s.get("volume"),
                "买1h": s.get("buys_h1"),
                "卖1h": s.get("sells_h1"),
                "流动性$": s.get("liquidity"),
                "数据源": s.get("source"),
                "错误": s.get("error"),
            }
        )
    return pd.DataFrame(rows)


def correlation_matrix(snapshots: List[Dict[str, Any]]) -> Optional[pd.DataFrame]:
    series_map = {}
    for s in snapshots:
        hist = s.get("hist_close")
        if hist is None or len(hist) < 10:
            continue
        rets = pd.Series(hist).pct_change().dropna()
        if len(rets) < 10:
            continue
        series_map[s["id"]] = rets.reset_index(drop=True)

    if len(series_map) < 2:
        return None

    min_len = min(len(v) for v in series_map.values())
    min_len = min(min_len, CORR_LOOKBACK)
    data = {k: v.iloc[-min_len:].values for k, v in series_map.items()}
    return pd.DataFrame(data).corr()


def detect_alerts(snapshots: List[Dict[str, Any]], alert_pct: float) -> List[str]:
    alerts = []
    for s in snapshots:
        sid = s.get("id")
        c1 = s.get("chg_1h_pct")
        c24 = s.get("chg_1d_pct")
        if c1 is not None and abs(c1) >= alert_pct:
            alerts.append(f"⚡ {sid} 1h 异动 {c1:+.2f}%")
        if c24 is not None and abs(c24) >= alert_pct * 2:
            alerts.append(f"🔥 {sid} 日涨跌 {c24:+.2f}%")
        buys, sells = s.get("buys_h1"), s.get("sells_h1")
        if buys is not None and sells is not None and (buys + sells) >= 20:
            ratio = buys / max(sells, 1)
            if ratio >= 2.0:
                alerts.append(f"🟢 {sid} 1h 买盘占优 buys/sells={ratio:.2f}")
            elif ratio <= 0.5:
                alerts.append(f"🔴 {sid} 1h 卖盘占优 buys/sells={ratio:.2f}")
    return alerts


def lead_lag_hint(snapshots: List[Dict[str, Any]]) -> List[str]:
    items = [
        (s["id"], s.get("chg_1h_pct"))
        for s in snapshots
        if s.get("chg_1h_pct") is not None
    ]
    if len(items) < 2:
        return ["1h 数据不足，无法判断领先滞后"]

    items_sorted = sorted(items, key=lambda x: abs(x[1]), reverse=True)
    leader, lead_chg = items_sorted[0]
    hints = [f"1h 波动最大：{leader}（{lead_chg:+.2f}%）"]

    stocks = {s["id"]: s.get("chg_1h_pct") for s in snapshots if s.get("type") == "stock"}
    majors = {s["id"]: s.get("chg_1h_pct") for s in snapshots if s.get("type") == "major"}
    hoods = {s["id"]: s.get("chg_1h_pct") for s in snapshots if s.get("type") == "hood"}

    for h, hv in hoods.items():
        for sid, sv in {**stocks, **majors}.items():
            if hv is None or sv is None:
                continue
            if hv * sv > 0 and abs(hv) > 1 and abs(sv) > 0.3:
                hints.append(f"联动：{h}({hv:+.2f}%) 与 {sid}({sv:+.2f}%) 1h 同向")
    return hints[:12]


# =========================
# UI
# =========================

st.set_page_config(page_title="币股联动监控", page_icon="📊", layout="wide")
st.title("📊 币市 × 股市 联动监控")
st.caption("HOOD/MSTR/CRCL · BTC/ETH/SOL · PONS/CASHCAT/AI/TENDIES · 15 分钟自动刷新")

with st.sidebar:
    st.header("⚙️ 设置")
    st.write("Hood 代币合约（可选，填了更准）")
    hood_addresses = {}
    for tok in HOOD_TOKENS:
        hood_addresses[tok["id"]] = st.text_input(
            f"{tok['id']} address",
            value=tok.get("address") or "",
            key=f"addr_{tok['id']}",
        )
    alert_pct = st.number_input("1h 异动阈值 %", value=float(ALERT_PCT_1H), step=0.5)
    st.divider()
    manual = st.button("🔄 手动刷新", type="primary", use_container_width=True)
    st.caption(f"自动刷新：{REFRESH_SECONDS // 60} 分钟")

if "last_fetch_ts" not in st.session_state:
    st.session_state.last_fetch_ts = 0.0
if "cache_snap" not in st.session_state:
    st.session_state.cache_snap = None

now = time.time()
need_refresh = (
    manual
    or (now - st.session_state.last_fetch_ts >= REFRESH_SECONDS)
    or (st.session_state.cache_snap is None)
)

if need_refresh:
    with st.spinner("拉取行情中…"):
        snaps: List[Dict[str, Any]] = []

        for s in STOCKS:
            try:
                data = fetch_stock_snapshot(s["symbol"])
                data["id"] = s["id"]
                snaps.append(data)
            except Exception as e:
                snaps.append(
                    {"id": s["id"], "type": "stock", "error": str(e), "source": None}
                )

        for m in MAJORS:
            try:
                data = fetch_binance_snapshot(m["symbol"])
                data["id"] = m["id"]
                snaps.append(data)
            except Exception as e:
                snaps.append(
                    {"id": m["id"], "type": "major", "error": str(e), "source": None}
                )

        for tok in HOOD_TOKENS:
            addr = (hood_addresses.get(tok["id"]) or "").strip()
            data = dexscreener_token(tok["chain"], address=addr, name=tok["name"])
            data["id"] = tok["id"]
            data["type"] = "hood"
            snaps.append(data)

        try:
            ex_vol = fetch_exchange_volumes()
        except Exception:
            ex_vol = pd.DataFrame()

        st.session_state.cache_snap = {"snaps": snaps, "ex_vol": ex_vol}
        st.session_state.last_fetch_ts = time.time()

pack = st.session_state.cache_snap
snaps = pack["snaps"]
ex_vol = pack["ex_vol"]

last_dt = datetime.fromtimestamp(
    st.session_state.last_fetch_ts, tz=timezone.utc
).strftime("%Y-%m-%d %H:%M:%S UTC")
st.info(f"最后更新：{last_dt} · 保持页面打开约每 15 分钟自动刷新")

alerts = detect_alerts(snaps, float(alert_pct))
hints = lead_lag_hint(snaps)

c1, c2 = st.columns(2)
with c1:
    st.subheader("🚨 异动提示")
    if alerts:
        for a in alerts:
            st.write(a)
    else:
        st.write("暂无超过阈值的异动")
with c2:
    st.subheader("🔗 联动/领先提示（粗算）")
    for h in hints:
        st.write("· " + h)

st.divider()
st.subheader("📋 行情总表")
table = build_snapshot_table(snaps)

def _fmt_pct(x):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "—"
        return f"{float(x):+.2f}%"
    except Exception:
        return "—"

show = table.copy()
for col in ["1h%", "24h/日%"]:
    if col in show.columns:
        show[col] = show[col].apply(_fmt_pct)

st.dataframe(show, use_container_width=True, height=460)

st.subheader("📈 1h / 日涨跌对比")
plot_df = table.copy()
plot_df["1h%"] = pd.to_numeric(plot_df["1h%"], errors="coerce")
plot_df["24h/日%"] = pd.to_numeric(plot_df["24h/日%"], errors="coerce")

fig = go.Figure()
fig.add_bar(name="1h%", x=plot_df["标的"], y=plot_df["1h%"])
fig.add_bar(name="日%", x=plot_df["标的"], y=plot_df["24h/日%"])
fig.update_layout(barmode="group", height=400, margin=dict(l=20, r=20, t=30, b=20))
st.plotly_chart(fig, use_container_width=True)

st.subheader("🧮 日收益相关矩阵（有历史的标的）")
corr = correlation_matrix(snaps)
if corr is not None and not corr.empty:
    fig_c = px.imshow(
        corr, text_auto=".2f", color_continuous_scale="RdBu", zmin=-1, zmax=1
    )
    fig_c.update_layout(height=480, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_c, use_container_width=True)
else:
    st.write("历史序列不足（Hood 无日线时，相关主要在股票与 BTC/ETH/SOL）")

st.subheader("🏦 交易所代表量能（BTC/ETH/SOL）")
if ex_vol is not None and not ex_vol.empty:
    st.dataframe(ex_vol, use_container_width=True)
else:
    st.write("暂无数据")

st.subheader("🦊 Hood 代币 1h 买卖")
hood_rows = [s for s in snaps if s.get("type") == "hood"]
if hood_rows:
    hdf = pd.DataFrame(
        [
            {
                "代币": s.get("id"),
                "价格": s.get("price"),
                "1h%": s.get("chg_1h_pct"),
                "买笔数": s.get("buys_h1"),
                "卖笔数": s.get("sells_h1"),
                "1h成交额$": s.get("volume"),
                "流动性$": s.get("liquidity"),
                "链接": s.get("pair_url"),
                "错误": s.get("error"),
            }
            for s in hood_rows
        ]
    )
    st.dataframe(hdf, use_container_width=True)
else:
    st.write("无 Hood 数据")

st.divider()
st.markdown(
    """
### 说明
- **BTC/ETH/SOL** 顺序：Binance 多域名 → Yahoo(yfinance) → CoinGecko  
- 看表中 **数据源 / 错误** 列可判断当前走了哪一路  
- Hood 代币建议在左侧填写合约地址  
- Streamlit Cloud 休眠后重新打开会立刻刷新一次  
"""
)

elapsed = time.time() - st.session_state.last_fetch_ts
remain = max(0, REFRESH_SECONDS - int(elapsed))
st.caption(f"距下次自动刷新约 {remain // 60} 分 {remain % 60} 秒")

if remain <= 0:
    st.rerun()
else:
    time.sleep(min(30, remain))
    st.rerun()
