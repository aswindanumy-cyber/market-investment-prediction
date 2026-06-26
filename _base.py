"""
Shared constants, helpers, and formulas for all predictors.
Import from here — never duplicate across gold/silver/stock scripts.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from datetime import datetime

# ─────────────────────────────────────────────
# SENTIMENT CONSTANTS
# ─────────────────────────────────────────────
VERY_BULLISH = "VERY BULLISH"
BULLISH      = "BULLISH"
NEUTRAL      = "NEUTRAL"
BEARISH      = "BEARISH"
VERY_BEARISH = "VERY BEARISH"

# ─────────────────────────────────────────────
# DATA FETCH
# ─────────────────────────────────────────────
def fetch(ticker, period="5y"):
    """Download one ticker individually — avoids yfinance bulk column-order bug."""
    df = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False)["Close"]
    if isinstance(df, pd.DataFrame):
        df = df.squeeze()
    return df.dropna().rename(ticker)

# ─────────────────────────────────────────────
# TECHNICAL INDICATORS
# ─────────────────────────────────────────────
def sma(s, n): return s.rolling(n).mean()
def ema(s, n): return s.ewm(span=n, adjust=False).mean()

def rsi(s, n=14):
    d    = s.diff()
    gain = d.clip(lower=0).rolling(n).mean()
    loss = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))

def macd(s, fast=12, slow=26, sig=9):
    m = ema(s, fast) - ema(s, slow)
    return m, ema(m, sig)

def bollinger(s, n=20, k=2):
    mid = sma(s, n)
    std = s.rolling(n).std()
    return mid - k * std, mid, mid + k * std

# ─────────────────────────────────────────────
# PRICE TARGETS  (log-normal ±2σ√N)
# ─────────────────────────────────────────────
def price_targets(price_series, sigma=2):
    """
    Returns (t3, t12, t2030, monthly, fX, future_y, poly, mu, vol).
    Each target tuple = (bear, base, bull).
    Formula: p × exp(μN ± sigma×σ×√N)  — vol scales with √N, not N.
    """
    s        = price_series.dropna()
    log_ret  = np.log(s / s.shift(1)).dropna()
    lr_clean = log_ret[-126:].clip(-0.05, 0.05)
    mu       = lr_clean.mean()
    vol      = lr_clean.std()
    p        = s.iloc[-1]

    def _t(n):
        return (
            p * np.exp(mu * n - sigma * vol * np.sqrt(n)),
            p * np.exp(mu * n),
            p * np.exp(mu * n + sigma * vol * np.sqrt(n)),
        )

    t3  = _t(63)
    t12 = _t(252)

    monthly = s.resample("ME").last()
    X       = np.arange(len(monthly)).reshape(-1, 1)
    poly    = PolynomialFeatures(2)
    reg     = LinearRegression().fit(poly.fit_transform(X), monthly.values)
    m2030   = (datetime(2030, 12, 31) - monthly.index[-1]).days // 30
    fX      = np.arange(len(monthly), len(monthly) + m2030).reshape(-1, 1)
    fy      = reg.predict(poly.transform(fX))

    t2030_b = max(fy[-1], p)
    ann_v   = vol * np.sqrt(252)
    yrs     = (datetime(2030, 12, 31) - datetime.now()).days / 365
    t2030   = (
        t2030_b * np.exp(-ann_v * 0.3 * yrs),
        t2030_b,
        t2030_b * np.exp( ann_v * 0.6 * yrs),
    )
    return t3, t12, t2030, monthly, fX, fy, poly, mu, vol

def yearly_targets(price, mu, vol, macro_calendar):
    """
    Year-by-year prediction table with macro multipliers applied.
    macro_calendar: { year: (multiplier, sentiment_str, driver_text) }
    Returns list of (year, bear, base, bull, sentiment, driver).
    """
    rows = []
    for yr in range(2026, 2031):
        days   = (datetime(yr, 12, 31) - datetime.now()).days
        t_base = price * np.exp(mu * days)
        t_bull = price * np.exp(mu * days + 2 * vol * np.sqrt(days))
        t_bear = price * np.exp(mu * days - 2 * vol * np.sqrt(days))
        adj, sentiment, drivers = macro_calendar.get(yr, (1.0, NEUTRAL, "No specific event"))
        rows.append((yr, t_bear / adj, t_base * adj, t_bull * adj, sentiment, drivers))
    return rows

# ─────────────────────────────────────────────
# SIGNAL LABEL
# ─────────────────────────────────────────────
def signal_label(score):
    if score is None:  return "❓ N/A"
    if score >= 6.5:   return "🟢 BUY"
    elif score <= 4.0: return "🔴 SELL"
    else:              return "🟡 HOLD"

# ─────────────────────────────────────────────
# CHART HELPERS
# ─────────────────────────────────────────────
def dark_axes(axes):
    for ax in (axes if hasattr(axes, "__iter__") else [axes]):
        ax.set_facecolor("#1a1a1a")
        ax.tick_params(colors="#aaaaaa")
        ax.spines[:].set_color("#333333")

def fmt_date_axis(axes):
    for ax in (axes if hasattr(axes, "__iter__") else [axes]):
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

def print_yearly_table(yearly_rows, unit="$"):
    print(f"\n── Year-by-Year Prediction (macro-adjusted) ─────")
    print(f"  {'Year':<6} {'Sentiment':<13} {'Bear':>12}  {'Base':>12}  {'Bull':>12}   Key Driver")
    print("  " + "─" * 100)
    for yr, t_bear, t_base, t_bull, sentiment, drivers in yearly_rows:
        short = drivers[:58] + "..." if len(drivers) > 58 else drivers
        print(f"  {yr:<6} {sentiment:<13} {unit}{t_bear:>11,.2f}  {unit}{t_base:>11,.2f}  {unit}{t_bull:>11,.2f}   {short}")
