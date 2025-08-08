# fetch.py — 生成按负责人分组的信号摘要（GitHub Actions 用）
# - 不改 CSV 容错：用 pd.read_csv 直接读取
# - 逐票抓历史 + 重试，避免批量抓空
# - 用收盘价/成交量计算（昨收涨跌、52周分位、3D/7D、三连涨/跌、量比）
# - 输出：docs/daily_summary.txt、docs/signals_us.csv、docs/signals_cn_hk.csv

from pathlib import Path
import time
import re
import pandas as pd
import numpy as np
import yfinance as yf

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "docs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

US_CSV = ROOT / "watchlist_us.csv"
CN_CSV = ROOT / "watchlist_cn.csv"

# ---------- 市场代码 -> yfinance 符号 ----------
def map_to_yahoo_symbol(market_symbol: str) -> str:
    market, symbol = market_symbol.split(":")
    market = market.strip().upper()
    symbol = symbol.strip().upper().replace(" ", "")
    if market.startswith("OTC"):  # OTC ADR，如 LKNC.Y -> LKNCY
        return symbol.replace(".", "")
    if market in {"NYSE", "NASDAQGS", "NASDAQGM", "NASDAQCM", "NASDAQ", "NASDQ"}:
        return symbol
    if market in {"SWX", "SIX"}:
        return f"{symbol}.SW"
    if market in {"TSX"}:
        return f"{symbol}.TO"
    if market in {"XETRA", "XTRA", "FRA", "FWB"}:
        return f"{symbol}.DE"
    if market in {"SEHK", "HKEX"}:
        digits = re.sub(r"[^0-9]", "", symbol)
        return f"{digits.zfill(4)}.HK" if digits else f"{symbol}.HK"
    if market in {"SHSE", "SSE", "SHA"}:
        return f"{symbol}.SS"
    if market in {"SZSE", "SZE", "SZN"}:
        return f"{symbol}.SZ"
    return symbol

# 特例
MANUAL = {
    "Hermes": "HESAY",
    "LVMH": "LVMUY",
    "Celestica": "CLS.TO",
    "Roche": "ROG.SW",
    "Pony.ai": "PONY",
}

# ---------- 读取 watchlist（不做容错） ----------
def read_watchlist(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)  # 你已修好带逗号的行，这里直接读
    df["Ticker"] = [MANUAL.get(n, map_to_yahoo_symbol(ms)) for n, ms in zip(df["Name"], df["MarketSymbol"])]
    return df[["Name", "Ticker", "Owner", "MarketSymbol"]]

# ---------- 逐票抓历史 + 计算指标 ----------
def compute_signals(df_watch: pd.DataFrame) -> pd.DataFrame:
    out = []
    failed = []

    for _, row in df_watch.iterrows():
        name, tkr, owner = row["Name"], row["Ticker"], row["Owner"]

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
        # 昨收涨跌（百分比）
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

# ---------- 文本汇总 ----------
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

            sel_price  = sub[sub["Price Change %"].abs() >= 3]
            sel_idx_hi = sub[sub["Price Index (0-1)"] >= 0.75]
            sel_idx_lo = sub[sub["Price Index (0-1)"] <= 0.25]
            sel_7d     = sub[sub["7-D Change %"].abs() >= 10]
            sel_3d     = sub[sub["3-D Change %"].abs() >= 5]
            up_list    = sub[sub["Upward Trend (3d)"] == True]["Name"].tolist()
            down_list  = sub[sub["Downward Trend (3d)"] == True]["Name"].tolist()
            sel_vol    = sub[sub["Volume ÷ 20d Avg"] >= 2]

            lines.append("Price Change: " + (fmt_pairs(sel_price, "Price Change %") or "-"))
            idx_parts = []
            if not sel_idx_hi.empty:
                idx_parts.append(fmt_pairs(sel_idx_hi, "Price Index (0-1)"))
            if not sel_idx_lo.empty:
                idx_parts.append(fmt_pairs(sel_idx_lo, "Price Index (0-1)"))
            lines.append("Price Index: " + (", ".join([t for t in idx_parts if t]) or "-"))
            lines.append("7-D Change: " + (fmt_pairs(sel_7d, "7-D Change %") or "-"))
            lines.append("3-D Change: " + (fmt_pairs(sel_3d, "3-D Change %") or "-"))
            lines.append("Upward Trend: " + (", ".join(up_list) if up_list else "-"))
            lines.append("Downward Trend: " + (", ".join(down_list) if down_list else "-"))
            lines.append("Volume Percentage: " + (fmt_pairs(sel_vol, "Volume ÷ 20d Avg") or "-"))
            lines.append("")
    return "\n".join(lines).strip() + "\n"

# ---------- 主流程 ----------
def main():
    us = read_watchlist(US_CSV)
    cn = read_watchlist(CN_CSV)

    df_us = compute_signals(us)
    df_cn = compute_signals(cn)

    owners = sorted(set(df_us["Owner"]).union(set(df_cn["Owner"])))
    text = build_text({"US": df_us, "China/HK": df_cn}, owners)

    (OUT_DIR / "daily_summary.txt").write_text(text, encoding="utf-8")
    df_us.to_csv(OUT_DIR / "signals_us.csv", index=False)
    df_cn.to_csv(OUT_DIR / "signals_cn_hk.csv", index=False)

    print(text)

if __name__ == "__main__":
    main()
