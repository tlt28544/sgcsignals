import time
import yfinance as yf
import numpy as np
import pandas as pd

def compute_signals(df_watch):
    out = []
    failed = []

    for _, row in df_watch.iterrows():
        name, tkr, owner = row["Name"], row["Ticker"], row["Owner"]

        # 逐票抓，最多重试 3 次
        hist = None
        for attempt in range(3):
            try:
                h = yf.Ticker(tkr).history(period="400d", interval="1d", auto_adjust=False, raise_errors=False)
                if h is not None and not h.empty:
                    hist = h.dropna()
                    break
            except Exception:
                pass
            time.sleep(1.2 * (attempt + 1))

        if hist is None or hist.empty or "Close" not in hist.columns:
            failed.append(tkr)
            # 也写一行占位，避免整个人为空
            out.append({"Name": name, "Ticker": tkr, "Owner": owner})
            continue

        close = hist["Close"]
        vol   = hist["Volume"] if "Volume" in hist.columns else pd.Series(dtype=float)

        last_close = float(close.iloc[-1])
        prev_close = float(close.iloc[-2]) if len(close) >= 2 else last_close

        # 用两个收盘价算昨收涨跌（凌晨跑时更稳）
        pchg = (last_close / prev_close - 1.0) * 100.0

        # 52 周分位（0~1）
        win = close.tail(252)
        hi_52, lo_52 = float(win.max()), float(win.min())
        pidx = ((last_close - lo_52) / (hi_52 - lo_52)) if hi_52 != lo_52 else np.nan

        # 3D / 7D
        def nday(n):
            return (last_close / float(close.iloc[-(n+1)]) - 1.0) * 100.0 if len(close) > n else np.nan
        chg3, chg7 = nday(3), nday(7)

        # 三连涨 / 三连跌
        rets = close.pct_change().dropna()
        up3   = bool((rets.tail(3) > 0).all())
        down3 = bool((rets.tail(3) < 0).all())

        # 量比（昨日成交量 / 近20日均量）；若没 Volume 列则 NaN
        vratio = np.nan
        if not vol.empty:
            last_vol = float(vol.iloc[-1])
            avg20 = float(vol.tail(20).mean()) if len(vol) >= 20 else np.nan
            vratio = (last_vol / avg20) if (avg20 and avg20 > 0) else np.nan

        out.append({
            "Name": name, "Ticker": tkr, "Owner": owner,
            "Price Change %": pchg, "Price Index (0-1)": pidx,
            "3-D Change %": chg3, "7-D Change %": chg7,
            "Upward Trend (3d)": up3, "Downward Trend (3d)": down3,
            "Volume ÷ 20d Avg": vratio
        })

    # 打印一下方便排错
    if failed:
        print(f"⚠️ Failed tickers ({len(failed)}): {', '.join(sorted(set(failed)))}")

    return pd.DataFrame(out)
