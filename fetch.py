# fetch.py — GitHub Actions 定时生成信号摘要
# 设计要点：
# - 逐票抓历史 + 自动重试，避免批量下载在云端被限导致整批为空
# - 只用收盘价/成交量做指标（不依赖 fast_info 的盘中数据），凌晨跑更稳
# - watchlist 容错读取（公司名里可以有逗号）
# - 输出：docs/daily_summary.txt、docs/signals_us.csv、docs/signals_cn_hk.csv

import time
import re
from pathlib import Path
import pandas as pd
import numpy as np
import yfinance as yf

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "docs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

US_CSV = ROOT / "watchlist_us.csv"
CN_CSV = ROOT / "watchlist_cn.csv"

# ------------------ 市场代码 -> yfinance 符号映射 ------------------

def map_to_yahoo_symbol(market_symbol: str) -> str:
    """把 'SEHK:9992' / 'OTCPK:LKNC.Y' / 'SHSE:600519' 映射为 yfinance 符号"""
    market, symbol = market_symbol.split(":")
    market = market.strip().upper()
    symbol = symbol.strip().upper().replace(" ", "")
    if market.startswith("OTC"):                 # OTC ADR 如 LKNC.Y -> LKNCY
        return symbol.replace(".", "")
    if market in {"NYSE","NASDAQGS","NASDAQGM","NASDAQCM","NASDAQ","NASDQ"}:
        return symbol
    if market in {"SWX","SIX"}:                  # 瑞士
        return f"{symbol}.SW"
    if market in {"TSX"}:                        # 多伦多
        return f"{symbol}.TO"
    if market in {"XETRA","XTRA","FRA","FWB"}:   # 德股
        return f"{symbol}.DE"
    if market in {"SEHK","HKEX"}:                # 港股
        digits = re.sub(r"[^0-9]", "", symbol)
        return f"{digits.zfill(4)}.HK" if digits else f"{symbol}.HK"
    if market in {"SHSE","SSE","SHA"}:           # 上证
        return f"{symbol}.SS"
    if market in {"SZSE","SZE","SZN"}:           # 深证
        return f"{symbol}.SZ"
    return symbol

# 特例修正
MANUAL = {
    "Hermes": "HESAY",
    "LVMH": "LVMUY",
    "Celestica": "CLS.TO",
    "Roche": "ROG.SW",
    "Pony.ai": "PONY",
}

# ------------------ 读取 watchlist（容错） ------------------

def read_watchlist(path: Path) -> pd.DataFrame:
    """容错读取 CSV：最后两列视为 MarketSymbol、Owner，前面的全拼为 Name。"""
    rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.lower().startswith("name,"):  # 跳过表头/空行
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            owner = parts[-1]
            market = parts[-2]
            name = ",".join(parts[:-2]).strip()  # 名字允许包含逗号
            rows.append((name, market, owner))

    df = pd.DataFrame(rows, columns=["Name", "MarketSymbol", "Owner"])
    df["Ticker"] = [MANUAL.get(n, map_to_yahoo_symbol(ms)) for n, ms in zip(df["Name"], df["MarketSymbol"])]
    return df[["Name", "Ticker", "Owner", "MarketSymbol"]]

# ------------------ 逐票拉取 + 计算指标 ------------------

def compute_signals(df_watch: pd.DataFrame) -> pd.DataFrame:
    """逐票抓 400 交易日历史，算指标；抓不到也输出占位行。"""
    out, failed = [], []

    for _, row in df_watch.iterrows():
        name, tkr, owner = row["Name"], row["Ticker"], row["Owner"]

        # 最多重试 3 次
        hist = None
        for attempt in range(3):
            try:
                h = yf.Ticker(tkr).history(
                    period="400d", interval="1d", auto_adjust=False, raise_errors=False
                )
                if h is not None and not h.empty:
                    hist = h.dropna()
                    break
            except Exception:
                pass
            time.sleep(1.2 * (attempt + 1))

        if hist is None or hist.empty or "Close" not in hist.columns:
            failed.append(tkr)
            out.append({"Name": name, "Ticker": tkr, "Owner": owner})
            continue

        close = hist["Close"]
        vol = hist["Volume"] if "Volume" in hist.columns else pd.Series(dtype=float)

        last_close = float(close.iloc[-1])
        prev_close = float(close.iloc[-2]) if len(close) >= 2 else last_close

        # 昨收涨跌（凌晨跑最稳）
        pchg = (last_close / prev_close - 1.0) * 100.0

        # 52 周分位（0~1）
        win = close.tail(252)
        hi_52, lo_52 = float(win.max()), float(win.min())
        pidx = ((last_close - lo_52) / (hi_52 - lo_52)) if hi_52 != lo_52 else np.nan

        # 3D / 7D
        def nday(n: int):
            return (last_close / float(close.iloc[-(n + 1)]) - 1.0) * 100.0 if len(close) > n else np.nan

        chg3, chg7 = nday(3), nday(7)

        # 三连涨/跌
        rets = close.pct_change().dropna()
        up3 = bool((rets.tail(3) > 0).all())
        down3 = bool((rets.tail(3) < 0).all())

        # 量比（昨日成交量 / 近20日均量）
        vratio = np.nan
        if not vol.empty:
            last_vol = float(vol.iloc[-1])
            avg20 = float(vol.tail(20).mean()) if len(vol) >= 20 else np.nan
            vratio = (last_vol / avg20) if (avg20 and avg20 > 0) else np.nan

        out.append(
            {
                "Name": name,
                "Ticker": tkr,
                "Owner": owner,
                "Price Change %": pchg,
                "Price Index (0-1)": pidx,
                "3-D Change %": chg3,
                "7-D Change %": chg7,
                "Upward Trend (3d)": up3,
                "Downward Trend (3d)": down3,
                "Volume ÷ 20d Avg": vratio,
            }
        )

    if failed:
        print(f"⚠️ Failed tickers ({len(failed)}): {', '.join(sorted(set(failed)))}")

    return pd.DataFrame(out)

# ------------------ 文本汇总 ------------------

def fmt_pairs(df: pd.DataFrame, col: str) -> str:
    if df is None or df.empty or col not in df.columns:
        return ""
    d = df.dropna(subset=[col]).sort_values(col, ascending=False)
    if d.empty:
        return ""
    if "Volume" in col:
        return ", ".join(f"{r['Name']} ({r[col]:.2f}x)" for _, r in d.iterrows())
    else:
        return ", ".join(f"{r['Name']} ({r[col]:.2f})" for _, r in d.iterrows())

def build_text(by_region: dict[str, pd.DataFrame], owners: list[str]) -> str:
    lines: list[str] = []
    for owner in sorted(owners):
        lines.append(f"{owner}:")
        for region, df in by_region.items():
            sub = df[df["Owner"] == owner].copy()
            lines.append(f"{region}:")
            if sub.empty or "Price Change %" not in sub.columns:
                lines.append("No data.\n")
                continue

            sel_price = sub[sub["Price Change %"].abs() >= 3]
            sel_idx_hi = sub[sub["Price Index (0-1)"] >= 0.75]
            sel_idx_lo = sub[sub["Price Index (0-1)"] <= 0.25]
            sel_7d = sub[sub["7-D Change %"].abs() >= 10]
            sel_3d = sub[sub["3-D Change %"].abs() >= 5]
            up_list = sub[sub["Upward Trend (3d)"] == True]["Name"].tolist()
            down_list = sub[sub["Downward Trend (3d)"] == True]["Name"].tolist()
            sel_vol = sub[sub["Volume ÷ 20d Avg"] >= 2]

            lines.append("Price Change: " + (fmt_pairs(sel_price, "Price Change %") or "-"))
            idx_parts = []
            if not sel_idx_hi.empty:
                idx_parts.append(fmt_pairs(sel_idx_hi, "Price Index (0-1)"))
            if not sel_idx_lo.empty:
    # 低分位同样按从低到高或从高到低都行，这里保持降序即可
    idx_parts.append(fmt_pairs(sel_idx_lo, "Price Index (0-1)"))

