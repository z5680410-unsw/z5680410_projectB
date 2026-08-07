"""Cek independen: crypto equal-weight growth of $1, Jan 2021 - Dec 2023.
Tidak memakai src/portfolios.py sama sekali - murni dari raw price data,
untuk memverifikasi pipeline oos_backtest tidak salah hitung.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from src import data_access

crypto = data_access.load_crypto_prices()
crypto = crypto[crypto["date"] <= "2023-12-31"].copy()
wide = crypto.pivot(index="date", columns="ticker", values="adjClose").sort_index()
returns = wide.pct_change()

oos_returns = returns.loc["2021-01-01":]
port_return = oos_returns.mean(axis=1)  # equal-weight = rata-rata baris, tanpa estimasi apapun
growth = (1 + port_return).cumprod()
n_days = port_return.dropna().shape[0]
cagr = growth.iloc[-1] ** (365 / n_days) - 1
vol = port_return.std() * (365 ** 0.5)

print(f"Periode: {oos_returns.index.min().date()} - {oos_returns.index.max().date()}")
print(f"n_days={n_days}  growth akhir=${growth.iloc[-1]:.3f}  CAGR={cagr:.3f}  "
      f"ann_vol={vol:.3f}  Sharpe={cagr/vol:.3f}")
