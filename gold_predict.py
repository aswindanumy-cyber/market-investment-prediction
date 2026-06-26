"""
Gold Price Predictor
- Short term signal: BUY / SELL / HOLD (based on technicals + macro)
- 3-month & 1-year price target
- 2030 bear / base / bull case

Dependencies:
  pip install yfinance pandas numpy scikit-learn matplotlib requests
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
from datetime import datetime, timedelta

# ─────────────────────────────────────────────
# 1. FETCH DATA
# ─────────────────────────────────────────────
def _fetch(ticker, period="5y"):
    df = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False)["Close"]
    if isinstance(df, pd.DataFrame):
        df = df.squeeze()
    return df.dropna().rename(ticker)

print("📡  Fetching market data...")
gold   = _fetch("GC=F")        # Gold futures USD/troy oz
silver = _fetch("SI=F")        # Silver futures USD/troy oz (co-mover)
usd    = _fetch("DX-Y.NYB")   # US Dollar Index
vix    = _fetch("^VIX")       # Fear Index
rates  = _fetch("^TNX")       # US 10Y Treasury Yield

print(f"✅  Gold data: {gold.index[0].date()} → {gold.index[-1].date()}  ({len(gold)} days)")
print(f"    Gold spot: ${gold.iloc[-1]:,.2f}/oz\n")

# Align all to gold index
raw = pd.DataFrame({
    "gold":   gold,
    "usd":    usd.reindex(gold.index, method="ffill"),
    "vix":    vix.reindex(gold.index, method="ffill"),
    "rates":  rates.reindex(gold.index, method="ffill"),
}).dropna(subset=["gold"])
raw.ffill(inplace=True)

# ─────────────────────────────────────────────
# 2. TECHNICAL INDICATORS
# ─────────────────────────────────────────────
def sma(s, n): return s.rolling(n).mean()
def ema(s, n): return s.ewm(span=n, adjust=False).mean()

def rsi(s, n=14):
    d = s.diff()
    gain = d.clip(lower=0).rolling(n).mean()
    loss = (-d.clip(upper=0)).rolling(n).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def macd(s, fast=12, slow=26, signal=9):
    m = ema(s, fast) - ema(s, slow)
    sig = ema(m, signal)
    return m, sig

def bollinger(s, n=20, k=2):
    mid = sma(s, n)
    std = s.rolling(n).std()
    return mid - k * std, mid, mid + k * std

ta = pd.DataFrame(index=raw.index)
ta["gold"]   = gold
ta["sma20"]  = sma(gold, 20)
ta["sma50"]  = sma(gold, 50)
ta["sma200"] = sma(gold, 200)
ta["ema12"]  = ema(gold, 12)
ta["rsi"]    = rsi(gold)
ta["macd"], ta["macd_sig"] = macd(gold)
ta["bb_lo"], ta["bb_mid"], ta["bb_hi"] = bollinger(gold)
ta["usd"]    = raw["usd"]
ta["vix"]    = raw["vix"]
ta["rates"]  = raw["rates"]
ta.dropna(inplace=True)

last = ta.iloc[-1]
price = last["gold"]

# ─────────────────────────────────────────────
# 3. SIGNAL SCORING (0–10 per factor, >5 = bullish)
# ─────────────────────────────────────────────
scores = {}

# Trend
scores["SMA20 > SMA50"]  = 8 if last["sma20"] > last["sma50"]  else 2
scores["SMA50 > SMA200"] = 8 if last["sma50"] > last["sma200"] else 2
scores["Price > SMA200"] = 8 if price > last["sma200"]          else 2

# Momentum
if last["rsi"] < 30:
    scores["RSI (oversold=buy)"] = 9   # oversold → buy signal
elif last["rsi"] > 70:
    scores["RSI (overbought)"]   = 2   # overbought → caution
else:
    scores["RSI (neutral)"]      = 5

scores["MACD cross"] = 7 if last["macd"] > last["macd_sig"] else 3

# Bollinger
if price < last["bb_lo"]:
    scores["Bollinger (below lo)"] = 9
elif price > last["bb_hi"]:
    scores["Bollinger (above hi)"] = 2
else:
    scores["Bollinger (mid)"]      = 5

# Macro: USD (inverse relationship with gold)
usd_1m = raw["usd"].iloc[-22] if len(raw) > 22 else raw["usd"].iloc[0]
usd_trend = last["usd"] - usd_1m
scores["USD trend (weak=gold up)"] = 3 if usd_trend > 0 else 8

# Macro: VIX (fear drives gold)
scores["VIX fear (>20=gold up)"]   = 8 if last["vix"] > 20 else 4

# Macro: Real rates (negative=gold up)
scores["10Y rates (<3=gold up)"]    = 7 if last["rates"] < 3 else 3

total_score = np.mean(list(scores.values()))

if total_score >= 6.5:
    signal = "🟢  BUY"
elif total_score <= 4.0:
    signal = "🔴  SELL"
else:
    signal = "🟡  HOLD"

# ─────────────────────────────────────────────
# 4. SHORT / MID TERM PRICE TARGETS
# ─────────────────────────────────────────────
# Use trailing 6-month returns, winsorized at ±5% to strip futures-roll outliers
returns   = gold.pct_change().dropna()
ret_clean = returns[-126:].clip(-0.05, 0.05)
mu_daily  = ret_clean.mean()
vol_daily = ret_clean.std()

days_3m  = 63
days_12m = 252

target_3m_base = price * (1 + mu_daily) ** days_3m
target_3m_bull = price * (1 + mu_daily + vol_daily) ** days_3m
target_3m_bear = price * (1 + mu_daily - vol_daily) ** days_3m

target_1y_base = price * (1 + mu_daily) ** days_12m
target_1y_bull = price * (1 + mu_daily + vol_daily) ** days_12m
target_1y_bear = price * (1 + mu_daily - vol_daily) ** days_12m

# Floor at zero to avoid negative prices
target_3m_bear = max(target_3m_bear, price * 0.5)
target_1y_bear = max(target_1y_bear, price * 0.3)

# ─────────────────────────────────────────────
# 5. 2030 LONG-TERM PREDICTION (Polynomial Regression)
# ─────────────────────────────────────────────
gold_monthly = gold.resample("ME").last()
X = np.arange(len(gold_monthly)).reshape(-1, 1)
y = gold_monthly.values

poly = PolynomialFeatures(degree=2)
Xp   = poly.fit_transform(X)
reg  = LinearRegression().fit(Xp, y)

# Project to 2030
months_to_2030 = (datetime(2030, 12, 31) - gold_monthly.index[-1]).days // 30
future_X = np.arange(len(gold_monthly), len(gold_monthly) + months_to_2030).reshape(-1, 1)
future_y = reg.predict(poly.transform(future_X))
target_2030_base = max(future_y[-1], price)  # floor at current

# Bear / bull based on historical annual volatility
annual_vol = returns.std() * np.sqrt(252)
years_to_2030 = (datetime(2030, 12, 31) - datetime.now()).days / 365
target_2030_bull = target_2030_base * (1 + annual_vol * 0.5) ** years_to_2030
target_2030_bear = target_2030_base * (1 - annual_vol * 0.3) ** years_to_2030

# ─────────────────────────────────────────────
# 6. MACRO CONTEXT (hardcoded analyst consensus as of 2025)
# ─────────────────────────────────────────────
macro_factors = [
    ("De-dollarization",        "Central banks globally dumping USD reserves → buying gold"),
    ("US debt trajectory",      "$36T+ national debt → inflation risk → gold hedge"),
    ("Fed rate cycle",          "Rate cuts weaken USD → bullish for gold"),
    ("Geopolitical risk",       "Middle East, Russia-Ukraine, Taiwan tension → safe haven demand"),
    ("ETF & retail demand",     "Gold ETF inflows trending up post-2024"),
    ("Mining supply constraints","Gold is hard to find; supply growth <2%/yr"),
    ("China & India demand",    "World's largest consumers increasing reserves"),
]

# ─────────────────────────────────────────────
# 7. PRINT REPORT
# ─────────────────────────────────────────────
print("=" * 60)
print("         GOLD PRICE PREDICTOR REPORT")
print(f"         Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("=" * 60)

print(f"\n💰  Current Gold Price:  ${price:,.2f} / oz")
print(f"📊  Signal Score:        {total_score:.1f} / 10")
print(f"🎯  Overall Signal:      {signal}\n")

print("── Technical Breakdown ─────────────────────────")
for k, v in scores.items():
    bar = "█" * int(v) + "░" * (10 - int(v))
    sentiment = "bullish" if v >= 6 else ("bearish" if v <= 4 else "neutral")
    print(f"  {k:<35} {bar}  {sentiment}")

print("\n── Price Targets ────────────────────────────────")
print(f"  3-Month  │ Bear: ${target_3m_bear:>8,.0f}  Base: ${target_3m_base:>8,.0f}  Bull: ${target_3m_bull:>8,.0f}")
print(f"  12-Month │ Bear: ${target_1y_bear:>8,.0f}  Base: ${target_1y_base:>8,.0f}  Bull: ${target_1y_bull:>8,.0f}")
print(f"  2030     │ Bear: ${target_2030_bear:>8,.0f}  Base: ${target_2030_base:>8,.0f}  Bull: ${target_2030_bull:>8,.0f}")

print("\n── Macro Tailwinds / Headwinds ──────────────────")
for factor, reason in macro_factors:
    print(f"  ▸ {factor:<28} {reason}")

print("\n── Key Levels ───────────────────────────────────")
print(f"  SMA 20:   ${last['sma20']:,.2f}    SMA 50:  ${last['sma50']:,.2f}    SMA 200: ${last['sma200']:,.2f}")
print(f"  BB Lower: ${last['bb_lo']:,.2f}    BB Mid:  ${last['bb_mid']:,.2f}    BB Upper: ${last['bb_hi']:,.2f}")
print(f"  RSI:      {last['rsi']:.1f}          MACD:    {last['macd']:.2f}          Signal: {last['macd_sig']:.2f}")
print(f"  USD Idx:  {last['usd']:.2f}         VIX:     {last['vix']:.2f}          10Y Yield: {last['rates']:.2f}%")

print("\n⚠️   DISCLAIMER: This is not financial advice.")
print("    Predictions are model-based estimates, not guarantees.")
print("=" * 60)

# ─────────────────────────────────────────────
# 8. CHART
# ─────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 12), facecolor="#0f0f0f")
fig.suptitle("Gold Price Predictor Dashboard", color="gold", fontsize=16, fontweight="bold")

ax1, ax2, ax3 = axes
for ax in axes:
    ax.set_facecolor("#1a1a1a")
    ax.tick_params(colors="#aaaaaa")
    ax.spines[:].set_color("#333333")

# Panel 1: Price + MAs + Bollinger + 2030 forecast
recent = ta[-500:]
ax1.plot(recent.index, recent["gold"],   color="gold",   lw=1.5, label="Gold")
ax1.plot(recent.index, recent["sma50"],  color="#4fc3f7", lw=1,  label="SMA 50",  alpha=0.8)
ax1.plot(recent.index, recent["sma200"], color="#ef5350", lw=1,  label="SMA 200", alpha=0.8)
ax1.fill_between(recent.index, recent["bb_lo"], recent["bb_hi"], alpha=0.1, color="gold")

# Future projection
future_dates = pd.date_range(gold_monthly.index[-1], periods=months_to_2030, freq="ME")
ax1.plot(future_dates, future_y, color="#69f0ae", lw=1.5, linestyle="--", label="2030 forecast (base)")
ax1.fill_between(
    [ta.index[-1], future_dates[-1]],
    [price, target_2030_bear], [price, target_2030_bull],
    alpha=0.15, color="#69f0ae", label="Bear–Bull range"
)
ax1.axhline(price, color="gold", lw=0.5, linestyle=":")
ax1.set_ylabel("Price (USD)", color="#aaaaaa")
ax1.legend(loc="upper left", facecolor="#1a1a1a", labelcolor="#cccccc", fontsize=8)
ax1.set_title("Price + Bollinger Bands + 2030 Projection", color="#cccccc", fontsize=10)

# Panel 2: RSI
ax2.plot(recent.index, recent["rsi"], color="#ce93d8", lw=1.2)
ax2.axhline(70, color="#ef5350", lw=0.8, linestyle="--", label="Overbought 70")
ax2.axhline(30, color="#69f0ae", lw=0.8, linestyle="--", label="Oversold 30")
ax2.axhline(50, color="#555555", lw=0.5)
ax2.fill_between(recent.index, recent["rsi"], 50,
    where=recent["rsi"] > 50, alpha=0.2, color="#ef5350")
ax2.fill_between(recent.index, recent["rsi"], 50,
    where=recent["rsi"] < 50, alpha=0.2, color="#69f0ae")
ax2.set_ylim(0, 100)
ax2.set_ylabel("RSI", color="#aaaaaa")
ax2.legend(loc="upper left", facecolor="#1a1a1a", labelcolor="#cccccc", fontsize=8)
ax2.set_title("RSI (14)", color="#cccccc", fontsize=10)

# Panel 3: MACD
ax3.plot(recent.index, recent["macd"],     color="#4fc3f7", lw=1.2, label="MACD")
ax3.plot(recent.index, recent["macd_sig"], color="#ef5350", lw=1.0, label="Signal")
hist = recent["macd"] - recent["macd_sig"]
ax3.bar(recent.index, hist,
    color=["#69f0ae" if v >= 0 else "#ef5350" for v in hist],
    alpha=0.5, width=1)
ax3.axhline(0, color="#555555", lw=0.5)
ax3.set_ylabel("MACD", color="#aaaaaa")
ax3.legend(loc="upper left", facecolor="#1a1a1a", labelcolor="#cccccc", fontsize=8)
ax3.set_title("MACD (12, 26, 9)", color="#cccccc", fontsize=10)

for ax in axes:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

plt.tight_layout()
plt.savefig("output/gold_prediction.png", dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
print("\n📈  Chart saved → gold_prediction.png")
plt.show()
