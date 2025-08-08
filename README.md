# Signals (GitHub Actions)

- Edit `watchlist_us.csv` and `watchlist_cn.csv` (Name, MarketSymbol, Owner).
- Workflow runs on weekdays 01:00 UTC (see `.github/workflows/fetch.yml`) and on manual dispatch.
- Outputs:
  - `docs/daily_summary.txt` : human-readable summary grouped by Owner
  - `docs/signals_us.csv`, `docs/signals_cn_hk.csv` : raw metrics
- You can turn on GitHub Pages (Settings → Pages) with `docs/` as the root to browse the files online.
