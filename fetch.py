
import os, time, re
from pathlib import Path
import pandas as pd
import numpy as np
import yfinance as yf

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "docs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

US_CSV = ROOT / "watchlist_us.csv"
CN_CSV = ROOT / "watchlist_cn.csv"

def map_to_yahoo_symbol(market_symbol: str) -> str:
    market, symbol = market_symbol.split(":")
    market = market.strip().upper()
    symbol = symbol.strip().upper().replace(" ", "")
    if market.startswith("OTC"):  # OTC ADRs like LKNC.Y -> LKNCY
        return symbol.replace(".", "")
    if market in {"NYSE","NASDAQGS","NASDAQGM","NASDAQCM","NASDAQ","NASDQ"}:
        return symbol
    if market in {"SWX","SIX"}:
        return f"{symbol}.SW"
    if market in {"TSX"}:
        return f"{symbol}.TO"
    if market in {"XETRA","XTRA","FRA","FWB"}:
        return f"{symbol}.DE"
    if market in {"SEHK","HKEX"}:
        digits = re.sub(r"[^0-9]", "", symbol)
        return f"{digits.zfill(4)}.HK" if digits else f"{symbol}.HK"
    if market in {"SHSE","SSE","SHA"}:
        return f"{symbol}.SS"
    if market in {"SZSE","SZE","SZN"}:
        return f"{symbol}.SZ"
    return symbol

MANUAL = {
    "Hermes":"HESAY",
    "LVMH":"LVMUY",
    "Celestica":"CLS.TO",
    "Roche":"ROG.SW",
    "Pony.ai":"PONY",
}

def read_watchlist(path: Path):
    df = pd.read_csv(path)
    df["Ticker"] = [MANUAL.get(n, map_to_yahoo_symbol(ms)) for n, ms in zip(df["Name"], df["MarketSymbol"])]
    return df[["Name","Ticker","Owner","MarketSymbol"]]

def _download_batch(tickers, max_retries=3, pause=1.2):
    for i in range(max_retries):
        try:
            hist = yf.download(tickers=tickers, period="400d", interval="1d",
                               auto_adjust=False, progress=False, group_by="ticker", threads=True)
            return hist
        except Exception:
            time.sleep(pause * (i + 1))
    return None

def get_histories(tickers):
    CHUNK = 25
    parts = {}
    failed = set()
    for i in range(0, len(tickers), CHUNK):
        batch = tickers[i:i+CHUNK]
        hist = _download_batch(batch)
        if hist is None or hist.empty:
            for t in batch:
                try:
                    h = yf.Ticker(t).history(period="400d", interval="1d", auto_adjust=False)
                    if not h.empty:
                        parts[t] = h
                    else:
                        failed.add(t)
                except Exception:
                    failed.add(t)
        else:
            if isinstance(hist.columns, pd.MultiIndex):
                for t in batch:
                    try:
                        df = hist[t].dropna()
                        if not df.empty:
                            parts[t] = df
                        else:
                            failed.add(t)
                    except Exception:
                        failed.add(t)
            else:
                t = batch[0]
                df = hist.dropna()
                if not df.empty:
                    parts[t] = df
                else:
                    failed.add(t)
    return parts, failed

def compute_signals(df_watch):
    tickers = df_watch["Ticker"].dropna().unique().tolist()
    parts, failed = get_histories(tickers)

    rows = []
    for _, row in df_watch.iterrows():
        name, tkr, owner = row["Name"], row["Ticker"], row["Owner"]
        df = parts.get(tkr)
        if df is None or df.empty:
            rows.append({"Name":name,"Ticker":tkr,"Owner":owner})
            continue
        close = df["Close"]; vol = df["Volume"]
        last_close = float(close.iloc[-1])
        win = close.tail(252)
        hi_52, lo_52 = float(win.max()), float(win.min())
        rets = close.pct_change().dropna()
        up3 = bool((rets.tail(3) > 0).all())
        down3 = bool((rets.tail(3) < 0).all())

        def nday(n):
            return (last_close / float(close.iloc[-(n+1)]) - 1.0) * 100.0 if len(close) > n else np.nan
        chg3, chg7 = nday(3), nday(7)

        tk = yf.Ticker(tkr)
        fast = tk.fast_info
        price = getattr(fast, "last_price", np.nan) or last_close
        prev  = getattr(fast, "previous_close", np.nan) or (float(close.iloc[-2]) if len(close) >= 2 else last_close)
        pchg = (price / prev - 1.0) * 100.0
        pidx = ((price - lo_52) / (hi_52 - lo_52)) if hi_52 != lo_52 else np.nan  # 0~1
        vol_today = getattr(fast, "last_volume", np.nan)
        avg20 = float(vol.tail(20).mean()) if len(vol) >= 20 else np.nan
        vratio = (vol_today / avg20) if (avg20 and avg20 > 0) else np.nan

        rows.append({
            "Name":name,"Ticker":tkr,"Owner":owner,
            "Price Change %":pchg,"Price Index (0-1)":pidx,
            "3-D Change %":chg3,"7-D Change %":chg7,
            "Upward Trend (3d)":up3,"Downward Trend (3d)":down3,
            "Volume ÷ 20d Avg":vratio
        })
    return pd.DataFrame(rows)

def fmt_pairs(df, col):
    df = df.dropna(subset=[col]).copy()
    if not len(df):
        return ""
    if "Volume" in col:
        return ", ".join(f"{r['Name']} ({r[col]:.2f}x)" for _, r in df.sort_values(col, ascending=False).iterrows())
    if "Index" in col:
        return ", ".join(f"{r['Name']} ({r[col]:.2f})" for _, r in df.sort_values(col, ascending=False).iterrows())
    return ", ".join(f"{r['Name']} ({r[col]:.2f})" for _, r in df.sort_values(col, ascending=False).iterrows())

def build_text(df_all, owners):
    out = []
    order = sorted(owners)  # alphabetical by owner
    for owner in order:
        out.append(f"{owner}:")
        for region, df in df_all.items():
            sub = df[df["Owner"]==owner].copy()
            out.append(f"{region}:")
            if sub.empty or "Price Change %" not in sub.columns:
                out.append("No data.\n"); continue
            sel_price = sub[sub["Price Change %"].abs() >= 3]
            sel_index_high = sub[sub["Price Index (0-1)"] >= 0.75]
            sel_index_low  = sub[sub["Price Index (0-1)"] <= 0.25]
            sel_7d = sub[sub["7-D Change %"].abs() >= 10]
            sel_3d = sub[sub["3-D Change %"].abs() >= 5]
            up_trend = sub[sub["Upward Trend (3d)"]==True]["Name"].tolist()
            down_trend = sub[sub["Downward Trend (3d)"]==True]["Name"].tolist()
            sel_vol = sub[sub["Volume ÷ 20d Avg"] >= 2]

            out.append("Price Change: " + (fmt_pairs(sel_price, "Price Change %") or "-"))
            txt_idx = []
            if len(sel_index_high): txt_idx.append(fmt_pairs(sel_index_high, "Price Index (0-1)"))
            if len(sel_index_low):  txt_idx.append(fmt_pairs(sel_index_low, "Price Index (0-1)"))
            out.append("Price Index: " + (", ".join([t for t in txt_idx if t]) or "-"))
            out.append("7‑D Change: " + (fmt_pairs(sel_7d, "7-D Change %") or "-"))
            out.append("3‑D Change: " + (fmt_pairs(sel_3d, "3-D Change %") or "-"))
            out.append("Upward Trend: " + (", ".join(up_trend) if up_trend else "-"))
            out.append("Downward Trend: " + (", ".join(down_trend) if down_trend else "-"))
            out.append("Volume Percentage: " + (fmt_pairs(sel_vol, "Volume ÷ 20d Avg") or "-"))
            out.append("")
    return "\n".join(out).strip() + "\n"

def main():
    us = read_watchlist(US_CSV)
    cn = read_watchlist(CN_CSV)
    df_us = compute_signals(us)
    df_cn = compute_signals(cn)
    owners = sorted(set(df_us["Owner"]).union(set(df_cn["Owner"])))
    text = build_text({"US":df_us, "China/HK":df_cn}, owners)
    (OUT_DIR / "daily_summary.txt").write_text(text, encoding="utf-8")
    df_us.to_csv(OUT_DIR/"signals_us.csv", index=False)
    df_cn.to_csv(OUT_DIR/"signals_cn_hk.csv", index=False)
    print(text)

if __name__ == "__main__":
    main()
