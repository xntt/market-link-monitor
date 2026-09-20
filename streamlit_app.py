"""
币市 × 股市 联动监控（初始版）
- 标的：HOOD/MSTR/CRCL + BTC/ETH/SOL + Hood链 PONS/CASHCAT/AI/TENDIES
- 15分钟自动刷新 + 手动刷新
- 涨跌对比、简单相关、量能变化、异动提示
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

# =========================
# 配置：按你之前的标的
# =========================

# 美股（yfinance ticker；CIRCLE 上市代码按 CRCL，不对请改）
STOCKS = [
    {"id": "HOOD", "symbol": "HOOD", "name": "Robinhood"},
    {"id": "MSTR", "symbol": "MSTR", "name": "MicroStrategy"},
    {"id": "CRCL", "symbol": "CRCL", "name": "Circle"},
]

# 主流币（Binance USDT 永续/现货 ticker）
MAJORS = [
    {"id": "BTC", "symbol": "BTCUSDT", "name": "Bitcoin"},
    {"id": "ETH", "symbol": "ETHUSDT", "name": "Ethereum"},
    {"id": "SOL", "symbol": "SOLUSDT", "name": "Solana"},
]

# Hood 链代币：请把 address 换成真实合约（0x...）
# 可先留空，用 DexScreener 搜索名；有地址更稳
HOOD_TOKENS = [
    {
        "id": "PONS",
        "name": "PONS",
        "chain": "robinhood",
        "address": "",  # TODO: 填合约地址
        "pair_url_hint": "https://dexscreener.com/robinhood",
    },
    {
        "id": "CASHCAT",
        "name": "CASHCAT",
        "chain": "robinhood",
        "address": "",
        "pair_url_hint": "https://dexscreener.com/robinhood",
    },
    {
        "id": "AI",
        "name": "AI",
        "chain": "robinhood",
        "address": "",
        "pair_url_hint": "https://dexscreener.com/robinhood",
    },
    {
        "id": "TENDIES",
        "name": "TENDIES",
        "chain": "robinhood",
        "address": "",
        "pair_url_hint": "https://dexscreener.com/robinhood",
    },
]

# 异动阈值
ALERT_PCT_1H = 3.0          # 1h 涨跌超过 ±3% 标红
ALERT_VOL_MULT = 2.0        # 量能相对前一段放大超过 2 倍
CORR_LOOKBACK = 30          # 相关用最近 N 根日线（股票+币对齐）
REFRESH_SECONDS = 15 * 60   # 15 分钟

BINANCE_BASE = "https://api.binance.com"
DEXSCREENER_BASE = "https://api.dexscreener.com/latest/dex"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "market-link-monitor/1.0"})


# =========================
# 数据拉取
# =========================

def fetch_stock_snapshot(symbol: str) -> Dict[str, Any]:
    """美股快照 + 近一段历史（用于相关）"""
    t = yf.Ticker(symbol)
    info = {}
    try:
        fi = t.fast_info
        price = float(getattr(fi, "last_price", None) or getattr(fi, "lastPrice", None) or 0)
        prev = float(getattr(fi, "previous_close", None) or getattr(fi, "previousClose", None) or 0)
        volume = float(getattr(fi, "last_volume", None) or getattr(fi, "lastVolume", None) or 0)
    except Exception:
        price, prev, volume = 0.0, 0.0, 0.0

    hist = t.history(period="3mo", interval="1d")
    chg_1d = ((price / prev) - 1) * 100 if prev else None

    # 近似 1h：用日内分钟线（若盘中）
    chg_1h = None
    try:
        intraday = t.history(period="1d", interval="5m")
        if len(intraday) >= 12:
            p0 = float(intraday["Close"].iloc[-12])
            p1 = float(intraday["Close"].iloc[-1])
            if p0:
                chg_1h = (p1 / p0 - 1) * 100
    except Exception:
        pass

    return {
        "id": symbol,
        "type": "stock",
        "price": price,
        "chg_1h_pct": chg_1h,
        "chg_1d_pct": chg_1d,
        "volume": volume,
        "hist_close": hist["Close"] if hist is not None and not hist.empty else pd.Series(dtype=float),
        "error": None if price else "no price",
    }


def fetch_binance_snapshot(symbol: str) -> Dict[str, Any]:
    """主流币：24h ticker + 1h K 线涨跌 + 近期日线"""
    out = {
        "id": symbol.replace("USDT", ""),
        "type": "major",
        "price": None,
        "chg_1h_pct": None,
        "chg_1d_pct": None,
        "volume": None,
        "quote_volume": None,
        "hist_close": pd.Series(dtype=float),
        "error": None,
    }
    try:
        r = SESSION.get(f"{BINANCE_BASE}/api/v3/ticker/24hr", params={"symbol": symbol}, timeout=12)
        r.raise_for_status()
        j = r.json()
        out["price"] = float(j["lastPrice"])
        out["chg_1d_pct"] = float(j["priceChangePercent"])
        out["volume"] = float(j["volume"])
        out["quote_volume"] = float(j["quoteVolume"])
    except Exception as e:
        out["error"] = f"ticker: {e}"
        return out

    try:
        r = SESSION.get(
            f"{BINANCE_BASE}/api/v3/klines",
            params={"symbol": symbol, "interval": "1h", "limit": 2},
            timeout=12,
        )
        r.raise_for_status()
        kl = r.json()
        if len(kl) >= 2:
            c0, c1 = float(kl[-2][4]), float(kl[-1][4])
            out["chg_1h_pct"] = (c1 / c0 - 1) * 100 if c0 else None
    except Exception as e:
        out["error"] = (out["error"] or "") + f" | 1h: {e}"

    try:
        r = SESSION.get(
            f"{BINANCE_BASE}/api/v3/klines",
            params={"symbol": symbol, "interval": "1d", "limit": 90},
            timeout=12,
        )
        r.raise_for_status()
        kl = r.json()
        closes = pd.Series([float(x[4]) for x in kl])
        out["hist_close"] = closes
    except Exception:
        pass

    return out


def dexscreener_token(chain: str, address: str = "", name: str = "") -> Dict[str, Any]:
    """
    Hood/DEX 代币。
    优先 address；否则用 search(name) 取流动性最高的 robinhood pair。
    """
    out = {
        "id": name or address[:8],
        "type": "hood",
        "price": None,
        "chg_1h_pct": None,
        "chg_1d_pct": None,
        "volume": None,
        "txns_h1": None,
        "buys_h1": None,
        "sells_h1": None,
        "liquidity": None,
        "pair_url": None,
        "hist_close": pd.Series(dtype=float),
        "error": None,
    }

    pair = None
    try:
        if address:
            url = f"{DEXSCREENER_BASE}/tokens/{address}"
            r = SESSION.get(url, timeout=15)
            r.raise_for_status()
            pairs = r.json().get("pairs") or []
            # 优先匹配 chain
            cand = [p for p in pairs if (p.get("chainId") or "").lower() in (chain.lower(), "robinhood")]
            pool = cand or pairs
            if pool:
                pair = sorted(pool, key=lambda x: float(x.get("liquidity", {}).get("usd") or 0), reverse=True)[0]
        else:
            r = SESSION.get(f"{DEXSCREENER_BASE}/search", params={"q": name}, timeout=15)
            r.raise_for_status()
            pairs = r.json().get("pairs") or []
            pool = [
                p
                for p in pairs
                if (p.get("chainId") or "").lower() in (chain.lower(), "robinhood")
                and name.lower() in ((p.get("baseToken") or {}).get("symbol") or "").lower()
            ]
            if not pool:
                pool = [p for p in pairs if (p.get("chainId") or "").lower() in (chain.lower(), "robinhood")]
            if pool:
                pair = sorted(pool, key=lambda x: float(x.get("liquidity", {}).get("usd") or 0), reverse=True)[0]
    except Exception as e:
        out["error"] = str(e)
        return out

    if not pair:
        out["error"] = "未找到交易对（请填合约地址）"
        return out

    try:
        out["price"] = float(pair.get("priceUsd") or 0)
        out["chg_1h_pct"] = float(pair.get("priceChange", {}).get("h1") or 0)
        out["chg_1d_pct"] = float(pair.get("priceChange", {}).get("h24") or 0)
        out["volume"] = float(pair.get("volume", {}).get("h1") or 0)
        vol24 = float(pair.get("volume", {}).get("h24") or 0)
        out["volume_24h"] = vol24
        tx = pair.get("txns", {}).get("h1") or {}
        out["buys_h1"] = int(tx.get("buys") or 0)
        out["sells_h1"] = int(tx.get("sells") or 0)
        out["txns_h1"] = out["buys_h1"] + out["sells_h1"]
        out["liquidity"] = float(pair.get("liquidity", {}).get("usd") or 0)
        out["pair_url"] = pair.get("url")
        out["id"] = (pair.get("baseToken") or {}).get("symbol") or out["id"]
    except Exception as e:
        out["error"] = f"parse: {e}"

    return out


def fetch_exchange_volumes() -> pd.DataFrame:
    """简易交易所现货 24h 报价量（Binance 代表 + 可扩展）"""
    rows = []
    # Binance 全市场 quoteVolume 汇总较重，这里用 BTC/ETH/SOL 代表量能变化
    for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
        try:
            r = SESSION.get(f"{BINANCE_BASE}/api/v3/ticker/24hr", params={"symbol": sym}, timeout=10)
            j = r.json()
            rows.append(
                {
                    "exchange": "binance",
                    "symbol": sym,
                    "quote_volume_24h": float(j["quoteVolume"]),
                    "price_chg_pct": float(j["priceChangePercent"]),
                }
            )
        except Exception:
            continue
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
                "量能(1h或日)": s.get("volume"),
                "买1h": s.get("buys_h1"),
                "卖1h": s.get("sells_h1"),
                "流动性$": s.get("liquidity"),
                "错误": s.get("error"),
            }
        )
    df = pd.DataFrame(rows)
    return df


def correlation_matrix(snapshots: List[Dict[str, Any]]) -> Optional[pd.DataFrame]:
    series_map = {}
    for s in snapshots:
        hist = s.get("hist_close")
        if hist is None or len(hist) < 10:
            continue
        # 用日收益
        rets = pd.Series(hist).pct_change().dropna()
        if len(rets) < 10:
            continue
        series_map[s["id"]] = rets.reset_index(drop=True)

    if len(series_map) < 2:
        return None

    # 对齐最短长度
    min_len = min(len(v) for v in series_map.values())
    min_len = min(min_len, CORR_LOOKBACK)
    data = {k: v.iloc[-min_len:].values for k, v in series_map.items()}
    df = pd.DataFrame(data)
    return df.corr()


def detect_alerts(snapshots: List[Dict[str, Any]]) -> List[str]:
    alerts = []
    for s in snapshots:
        sid = s.get("id")
        c1 = s.get("chg_1h_pct")
        c24 = s.get("chg_1d_pct")
        if c1 is not None and abs(c1) >= ALERT_PCT_1H:
            alerts.append(f"⚡ {sid} 1h 异动 {c1:+.2f}%")
        if c24 is not None and abs(c24) >= ALERT_PCT_1H * 2:
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
    """极简：比较 1h 涨跌绝对值，谁动得大 + 同向视为联动"""
    items = [(s["id"], s.get("chg_1h_pct")) for s in snapshots if s.get("chg_1h_pct") is not None]
    if len(items) < 2:
        return ["1h 数据不足，无法判断领先滞后"]

    items_sorted = sorted(items, key=lambda x: abs(x[1]), reverse=True)
    leader, lead_chg = items_sorted[0]
    hints = [f"1h 波动最大：{leader}（{lead_chg:+.2f}%）"]

    # 与股票/主流同向
    stocks = {s["id"]: s.get("chg_1h_pct") for s in snapshots if s.get("type") == "stock"}
    majors = {s["id"]: s.get("chg_1h_pct") for s in snapshots if s.get("type") == "major"}
    hoods = {s["id"]: s.get("chg_1h_pct") for s in snapshots if s.get("type") == "hood"}

    def same_sign(a, b):
        return a is not None and b is not None and a * b > 0

    for h, hv in hoods.items():
        for sid, sv in {**stocks, **majors}.items():
            if same_sign(hv, sv) and abs(hv) > 1 and abs(sv) > 0.3:
                hints.append(f"联动：{h}({hv:+.2f}%) 与 {sid}({sv:+.2f}%) 1h 同向")
    return hints[:12]


# =========================
# Streamlit UI
# =========================

st.set_page_config(page_title="币股联动监控", page_icon="📊", layout="wide")
st.title("📊 币市 × 股市 联动监控（初始版）")
st.caption("标的：HOOD/MSTR/CRCL · BTC/ETH/SOL · Hood: PONS/CASHCAT/AI/TENDIES · 每 15 分钟自动刷新")

# 侧边栏：可临时改地址
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
    st.caption(f"自动刷新间隔：{REFRESH_SECONDS // 60} 分钟")


# 刷新控制
if "last_fetch_ts" not in st.session_state:
    st.session_state.last_fetch_ts = 0.0
if "cache_snap" not in st.session_state:
    st.session_state.cache_snap = None

now = time.time()
need_refresh = manual or (now - st.session_state.last_fetch_ts >= REFRESH_SECONDS) or (
    st.session_state.cache_snap is None
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
                snaps.append({"id": s["id"], "type": "stock", "error": str(e)})

        for m in MAJORS:
            try:
                data = fetch_binance_snapshot(m["symbol"])
                data["id"] = m["id"]
                snaps.append(data)
            except Exception as e:
                snaps.append({"id": m["id"], "type": "major", "error": str(e)})

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

# 全局改阈值
ALERT_PCT_1H = float(alert_pct)

last_dt = datetime.fromtimestamp(st.session_state.last_fetch_ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
st.info(f"最后更新：{last_dt} · 下次自动刷新约在 15 分钟后（保持页面打开）")

# 异动
alerts = detect_alerts(snaps)
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

# 总表
st.subheader("📋 行情总表")
table = build_snapshot_table(snaps)

def _fmt_pct(x):
    return f"{x:+.2f}%" if pd.notna(x) else "—"

show = table.copy()
for col in ["1h%", "24h/日%"]:
    if col in show.columns:
        show[col] = show[col].apply(lambda v: _fmt_pct(v) if v is not None else "—")

st.dataframe(show, use_container_width=True, height=420)

# 涨跌对比图
st.subheader("📈 1h / 日涨跌对比")
plot_df = table.dropna(subset=["标的"]).copy()
plot_df["1h%"] = pd.to_numeric(plot_df["1h%"], errors="coerce")
plot_df["24h/日%"] = pd.to_numeric(plot_df["24h/日%"], errors="coerce")

fig = go.Figure()
fig.add_bar(name="1h%", x=plot_df["标的"], y=plot_df["1h%"])
fig.add_bar(name="日%", x=plot_df["标的"], y=plot_df["24h/日%"])
fig.update_layout(barmode="group", height=400, margin=dict(l=20, r=20, t=30, b=20))
st.plotly_chart(fig, use_container_width=True)

# 相关矩阵（有日线历史的标的）
st.subheader("🧮 日收益相关矩阵（有历史的标的）")
corr = correlation_matrix(snaps)
if corr is not None and not corr.empty:
    fig_c = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu", zmin=-1, zmax=1)
    fig_c.update_layout(height=480, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_c, use_container_width=True)
else:
    st.write("历史序列不足（Hood 代币初始版无日线，相关主要在股票与 BTC/ETH/SOL 之间）")

# 交易所量能
st.subheader("🏦 交易所代表量能（Binance BTC/ETH/SOL 24h Quote Volume）")
if ex_vol is not None and not ex_vol.empty:
    st.dataframe(ex_vol, use_container_width=True)
    st.caption("进阶可扩 DefiLlama / 多所 ticker，做环比突变预警（类似 perpdexlist）")
else:
    st.write("暂无数据")

# Hood 买卖对比
st.subheader("🦊 Hood 代币 1h 买卖笔数")
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
### 使用说明
1. 左侧可填 **PONS / CASHCAT / AI / TENDIES** 合约地址（推荐），不填则按名称在 DexScreener 搜索，可能不准。  
2. 保持页面打开会约每 **15 分钟** 自动刷新；也可点 **手动刷新**。  
3. Streamlit Cloud 休眠后，重新打开页面会立刻拉一次数。  
4. 下一步可加：Telegram 推送、多所量能突变、真实 lead-lag 相关。
"""
)

# 自动刷新：到点 rerun
elapsed = time.time() - st.session_state.last_fetch_ts
remain = max(0, REFRESH_SECONDS - int(elapsed))
st.caption(f"距下次自动刷新约 {remain // 60} 分 {remain % 60} 秒")
if remain <= 0:
    st.rerun()
else:
    # 轻量等待后检查（避免空转过狠）
    time.sleep(min(30, remain))
    st.rerun()
