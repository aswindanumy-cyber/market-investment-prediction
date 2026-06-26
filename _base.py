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
# ─────────────────────────────────────────────
# LIVE MACRO FACTOR BUILDER  (Alpha Vantage News Sentiment)
#
# Each topic category has keyword triggers.  For each category we find
# the most relevant recent news article from Alpha Vantage and use its
# summary as the description — same (label, description) format as before,
# but fully live and forward-looking.
#
# Free API key: https://www.alphavantage.co/support/#api-key
# Set env var:  ALPHAVANTAGE_API_KEY=<your_key>
# Falls back to static descriptions if key is missing or API fails.
# ─────────────────────────────────────────────

import os
import time
import urllib.request
import json

# In-process cache: { cache_key: (timestamp, rows) }
# Shared across gold/silver within the same Docker process restart.
# Persisted to disk so it survives between loop runs (different python processes).
_MACRO_CACHE_FILE = "/app/output/.macro_cache.json"
_MACRO_CACHE_TTL  = 3600  # 1 hour — stays within free tier of 25 req/day

MACRO_TOPICS_SILVER = [
    ("Solar panel demand",    ["solar", "photovoltaic", "pv panel", "solar energy"]),
    ("EV & battery tech",     ["electric vehicle", "ev ", "battery", "charging"]),
    ("5G & electronics",      ["5g", "semiconductor", "electronics", "smartphone", "chip"]),
    ("Green energy mandate",  ["renewable", "net zero", "green energy", "wind farm", "climate"]),
    ("Gold/Silver Ratio",     ["gold silver ratio", "gsr", "silver undervalued"]),
    ("Central bank buying",   ["central bank", "fed reserve", "silver etf", "etf inflow"]),
    ("Supply constraints",    ["silver mine", "silver supply", "silver production", "silver deficit"]),
    ("Inflation hedge",       ["inflation", "cpi", "real rate", "dollar weakness", "hedge"]),
]

MACRO_TOPICS_GOLD = [
    ("De-dollarization",      ["de-dollarization", "dollar reserve", "brics", "dollar dump"]),
    ("US debt trajectory",    ["us debt", "national debt", "debt ceiling", "deficit"]),
    ("Fed rate cycle",        ["fed rate", "federal reserve", "rate cut", "rate hike", "fomc"]),
    ("Geopolitical risk",     ["geopolitical", "ukraine", "middle east", "taiwan", "war"]),
    ("ETF & retail demand",   ["gold etf", "etf inflow", "retail demand", "gold demand"]),
    ("Mining supply",         ["gold mine", "gold supply", "gold production", "mining"]),
    ("China & India demand",  ["china gold", "india gold", "central bank buying", "reserve"]),
    ("Inflation hedge",       ["inflation", "cpi", "real rate", "dollar weakness", "hedge"]),
]

# urllib.parse is stdlib — imported here so it is available to the functions below
import urllib.parse

import re

def _av_fetch_news(api_key, asset="GOLD", limit=50):
    """
    Two-pass fetch: first try asset-specific ETF tickers; if < 5 articles
    come back, fall through to broad commodities+economy topic search.
    Only valid stock/ETF symbols work in AV tickers param — no futures codes.
    """
    ticker_map = {"GOLD": "GLD,IAU", "SILVER": "SLV,SIVR"}
    tickers    = ticker_map.get(asset, "GLD")

    def _call(params):
        url = "https://www.alphavantage.co/query?function=NEWS_SENTIMENT" + params + f"&sort=RELEVANCE&limit={limit}&apikey={api_key}"
        try:
            with urllib.request.urlopen(url, timeout=10) as r:
                return json.loads(r.read().decode()).get("feed", [])
        except Exception:
            return []

    articles = _call(f"&tickers={urllib.parse.quote(tickers)}")
    if len(articles) < 5:
        # Broaden to commodity + macro topics when ticker returns too few hits
        articles = _call("&topics=commodities,economy_macro,finance")

    return articles

def _clean_title(title):
    """Strip boilerplate ticker prefixes then trim to 100 chars."""
    title = re.sub(r"^\([^)]+\)\s*", "", title)
    title = re.sub(r"^[A-Z]{1,5}:\s*", "", title)
    title = title.strip()
    return title[:100] + ("..." if len(title) > 100 else "")

def _best_match(articles, keywords):
    """
    Score each article against the category keywords using title+summary.
    Title keyword hit counts double to keep matching precise.
    Returns the cleaned title of the best match, or None.
    """
    scored = []
    for a in articles:
        title   = a.get("title", "").lower()
        summary = a.get("summary", "").lower()
        hits    = sum(2 for kw in keywords if kw in title) \
                + sum(1 for kw in keywords if kw in summary)
        if hits:
            scored.append((hits, a))
    if not scored:
        return None
    _, best = max(scored, key=lambda x: x[0])
    return _clean_title(best.get("title", ""))

def _cache_load(cache_key):
    try:
        with open(_MACRO_CACHE_FILE, "r") as f:
            store = json.load(f)
        entry = store.get(cache_key)
        if entry and (time.time() - entry["ts"]) < _MACRO_CACHE_TTL:
            return entry["rows"]
    except Exception:
        pass
    return None

def _cache_save(cache_key, rows):
    try:
        os.makedirs(os.path.dirname(_MACRO_CACHE_FILE), exist_ok=True)
        store = {}
        try:
            with open(_MACRO_CACHE_FILE, "r") as f:
                store = json.load(f)
        except Exception:
            pass
        store[cache_key] = {"ts": time.time(), "rows": rows}
        with open(_MACRO_CACHE_FILE, "w") as f:
            json.dump(store, f)
    except Exception:
        pass

def fetch_macro_factors(topic_list, asset="GOLD", fallbacks=None, api_key=None):
    """
    Returns list of (label, description) for each topic in topic_list.

    Priority per category:
      1. Matched article title from Alpha Vantage news (live)
      2. fallbacks dict entry for the label (live-data string from caller)
      3. Skipped if neither available

    Results cached 1 hour so free AV tier (25 req/day) is never exceeded.
    """
    key = api_key or os.environ.get("ALPHAVANTAGE_API_KEY", "")

    cached = _cache_load(asset)
    if cached is not None:
        news_map = {r[0]: r[1] for r in cached}
    elif key:
        articles = _av_fetch_news(key, asset=asset, limit=50)
        print(f"  [macro] fetched {len(articles)} articles from Alpha Vantage ({asset})")
        news_map = {}
        for label, keywords in topic_list:
            desc = _best_match(articles, keywords)
            if desc:
                news_map[label] = desc
        _cache_save(asset, list(news_map.items()))
    else:
        news_map = {}

    fb = fallbacks or {}
    rows = []
    for label, _ in topic_list:
        desc = news_map.get(label) or fb.get(label)
        if desc:
            rows.append((label, desc))

    return rows

def print_macro_factors(rows, header="Macro Tailwinds / Headwinds"):
    print(f"\n── {header} ──────────────────────────────")
    if not rows:
        print("  (set ALPHAVANTAGE_API_KEY env var for live macro news)")
        return
    for label, desc in rows:
        print(f"  ▸ {label:<24} {desc}")

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

# ─────────────────────────────────────────────
# LIVE NEWS  (Yahoo Finance, no API key needed)
# ─────────────────────────────────────────────
def fetch_news(ticker, max_items=8):
    """
    Returns list of (headline, url) for a ticker using yfinance.
    Falls back to empty list on any error so callers degrade gracefully.
    """
    try:
        items = yf.Ticker(ticker).news or []
        out = []
        for item in items[:max_items]:
            content = item.get("content", {})
            title   = content.get("title") or item.get("title", "")
            url     = (content.get("canonicalUrl", {}) or {}).get("url", "")
            if title:
                out.append((title.strip(), url))
        return out
    except Exception:
        return []

def print_news(ticker, label=None, max_items=8):
    """Print live news headlines for a ticker."""
    news = fetch_news(ticker, max_items)
    header = label or ticker
    print(f"\n── Live News: {header} ─────────────────────────────────")
    if not news:
        print("  (no news retrieved)")
        return
    for i, (title, url) in enumerate(news, 1):
        short_url = url.replace("https://", "").split("/")[0] if url else ""
        src = f"  [{short_url}]" if short_url else ""
        print(f"  {i}. {title}{src}")
