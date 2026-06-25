"""
Silver Price Predictor
- Short term signal: BUY / SELL / HOLD
- 3-month & 1-year price target
- 2030 bear / base / bull case

Silver differs from gold: 50%+ demand is industrial (solar, EVs, electronics).
Key extra drivers: Gold/Silver ratio, industrial output, green energy boom.

Dependencies:
  pip install yfinance pandas numpy scikit-learn matplotlib
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
# 1. FETCH DATA
# ─────────────────────────────────────────────
TICKERS = {
    "silver": "SI=F",     # Silver Futures
    "gold":   "GC=F",     # Gold (for GSR)
    "usd":    "DX-Y.NYB", # US Dollar Index
    "copper": "HG=F",     # Copper (industrial proxy)
    "sp500":  "^GSPC",    # Risk appetite
    "vix":    "^VIX",     # Fear
    "rates":  "^TNX",     # 10Y Treasury
    "slv":    "SLV",      # iShares Silver ETF (fund flow proxy)
}

print("📡  Fetching market data...")
raw = yf.download(
    list(TICKERS.values()),
    period="5y",
    interval="1d",
    auto_adjust=True,
    progress=False,
)["Close"]
raw.columns = list(TICKERS.keys())
raw.dropna(subset=["silver"], inplace=True)
raw.ffill(inplace=True)

silver = raw["silver"]
gold   = raw["gold"]
print(f"✅  Silver data: {raw.index[0].date()} → {raw.index[-1].date()}  ({len(raw)} days)\n")

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
    return m, ema(m, signal)

def bollinger(s, n=20, k=2):
    mid = sma(s, n)
    std = s.rolling(n).std()
    return mid - k * std, mid, mid + k * std

ta = pd.DataFrame(index=raw.index)
ta["silver"]  = silver
ta["sma20"]   = sma(silver, 20)
ta["sma50"]   = sma(silver, 50)
ta["sma200"]  = sma(silver, 200)
ta["rsi"]     = rsi(silver)
ta["macd"], ta["macd_sig"] = macd(silver)
ta["bb_lo"], ta["bb_mid"], ta["bb_hi"] = bollinger(silver)
ta["gsr"]     = gold / silver          # Gold/Silver Ratio
ta["copper"]  = raw["copper"]
ta["usd"]     = raw["usd"]
ta["vix"]     = raw["vix"]
ta["rates"]   = raw["rates"]
ta.dropna(inplace=True)

last  = ta.iloc[-1]
price = last["silver"]
gsr   = last["gsr"]

# ─────────────────────────────────────────────
# 3. SIGNAL SCORING
# ─────────────────────────────────────────────
scores = {}

# Trend
scores["SMA20 > SMA50"]         = 8 if last["sma20"] > last["sma50"]  else 2
scores["SMA50 > SMA200"]        = 8 if last["sma50"] > last["sma200"] else 2
scores["Price > SMA200"]        = 8 if price > last["sma200"]          else 2

# Momentum
rsi_val = last["rsi"]
if rsi_val < 30:   scores["RSI (oversold=buy)"]  = 9
elif rsi_val > 70: scores["RSI (overbought)"]    = 2
else:              scores["RSI (neutral)"]        = 5

scores["MACD cross"] = 7 if last["macd"] > last["macd_sig"] else 3

# Bollinger
if price < last["bb_lo"]:   scores["Bollinger (below lo)"] = 9
elif price > last["bb_hi"]: scores["Bollinger (above hi)"] = 2
else:                        scores["Bollinger (mid)"]      = 5

# Gold/Silver Ratio — historically mean ~65; high GSR = silver cheap vs gold
if gsr > 80:   scores["GSR >80 (silver undervalued)"] = 9
elif gsr > 65: scores["GSR 65-80 (slightly cheap)"]   = 6
else:          scores["GSR <65 (silver expensive)"]    = 3

# Copper trend (industrial proxy)
copper_1m = raw["copper"].iloc[-22] if len(raw) > 22 else raw["copper"].iloc[0]
scores["Copper rising (industrial)"] = 7 if last["copper"] > copper_1m else 3

# Macro
usd_1m = raw["usd"].iloc[-22] if len(raw) > 22 else raw["usd"].iloc[0]
scores["USD weak (silver up)"]        = 3 if last["usd"] > usd_1m else 8
scores["VIX fear (>20=safe haven)"]   = 7 if last["vix"] > 20    else 4
scores["10Y rates (<3=silver up)"]    = 7 if last["rates"] < 3    else 3

total_score = np.mean(list(scores.values()))

if total_score >= 6.5:   signal = "🟢  BUY"
elif total_score <= 4.0: signal = "🔴  SELL"
else:                    signal = "🟡  HOLD"

# ─────────────────────────────────────────────
# 4. PRICE TARGETS
# ─────────────────────────────────────────────
returns   = silver.pct_change().dropna()
mu_daily  = returns[-126:].mean()
vol_daily = returns[-126:].std()

t3b  = price * (1 + mu_daily - vol_daily) ** 63
t3   = price * (1 + mu_daily) ** 63
t3u  = price * (1 + mu_daily + vol_daily) ** 63
t12b = price * (1 + mu_daily - vol_daily) ** 252
t12  = price * (1 + mu_daily) ** 252
t12u = price * (1 + mu_daily + vol_daily) ** 252

# 2030 polynomial regression
silver_monthly = silver.resample("ME").last()
X  = np.arange(len(silver_monthly)).reshape(-1, 1)
y  = silver_monthly.values
Xp = PolynomialFeatures(2).fit_transform(X)
reg = LinearRegression().fit(Xp, y)

months_to_2030 = (datetime(2030, 12, 31) - silver_monthly.index[-1]).days // 30
future_X  = np.arange(len(silver_monthly), len(silver_monthly) + months_to_2030).reshape(-1, 1)
poly      = PolynomialFeatures(2)
poly.fit_transform(X)
future_y  = reg.predict(poly.transform(future_X))
t2030     = max(future_y[-1], price)

annual_vol    = returns.std() * np.sqrt(252)
years_to_2030 = (datetime(2030, 12, 31) - datetime.now()).days / 365
t2030_bull    = t2030 * (1 + annual_vol * 0.6) ** years_to_2030
t2030_bear    = t2030 * (1 - annual_vol * 0.3) ** years_to_2030

# ─────────────────────────────────────────────
# 5. MACRO / STRUCTURAL FACTORS
# ─────────────────────────────────────────────
macro_factors = [
    ("Solar panel demand",       "Each panel uses ~20g silver; solar capacity doubling every 3yrs"),
    ("EV & battery tech",        "Silver conductivity critical in EV charging & battery management"),
    ("5G & electronics",         "Every smartphone, chip, and PCB uses silver"),
    ("Green energy mandate",     "Global net-zero targets drive massive industrial silver demand"),
    ("Gold/Silver Ratio",        f"Current GSR {gsr:.1f}x — historical mean ~65x; reversion = upside"),
    ("Central bank buying",      "Less than gold, but ETF inflows picking up"),
    ("Supply constraints",       "Primary silver mines declining; mostly byproduct of copper/zinc mining"),
    ("Inflation hedge",          "Like gold but with industrial kicker = double tailwind"),
]

# ─────────────────────────────────────────────
# 6. REPORT
# ─────────────────────────────────────────────
print("=" * 60)
print("        SILVER PRICE PREDICTOR REPORT")
print(f"        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("=" * 60)

print(f"\n🥈  Current Silver Price: ${price:,.2f} / oz")
print(f"📊  Signal Score:         {total_score:.1f} / 10")
print(f"🎯  Overall Signal:       {signal}")
print(f"⚖️   Gold/Silver Ratio:    {gsr:.1f}x  (mean ~65x)\n")

print("── Technical Breakdown ─────────────────────────")
for k, v in scores.items():
    bar = "█" * int(v) + "░" * (10 - int(v))
    sentiment = "bullish" if v >= 6 else ("bearish" if v <= 4 else "neutral")
    print(f"  {k:<40} {bar}  {sentiment}")

print("\n── Price Targets ────────────────────────────────")
print(f"  3-Month  │ Bear: ${t3b:>7,.2f}  Base: ${t3:>7,.2f}  Bull: ${t3u:>7,.2f}")
print(f"  12-Month │ Bear: ${t12b:>7,.2f}  Base: ${t12:>7,.2f}  Bull: ${t12u:>7,.2f}")
print(f"  2030     │ Bear: ${t2030_bear:>7,.2f}  Base: ${t2030:>7,.2f}  Bull: ${t2030_bull:>7,.2f}")

print("\n── Macro & Industrial Tailwinds ─────────────────")
for factor, reason in macro_factors:
    print(f"  ▸ {factor:<28} {reason}")

print("\n── Key Levels ───────────────────────────────────")
print(f"  SMA 20:  ${last['sma20']:,.2f}    SMA 50: ${last['sma50']:,.2f}    SMA 200: ${last['sma200']:,.2f}")
print(f"  RSI: {last['rsi']:.1f}    MACD: {last['macd']:.3f}    Signal: {last['macd_sig']:.3f}")

print("\n⚠️   DISCLAIMER: Not financial advice. Model-based estimates only.")
print("=" * 60)

# ─────────────────────────────────────────────
# 7. CHART
# ─────────────────────────────────────────────
fig, axes = plt.subplots(4, 1, figsize=(14, 14), facecolor="#0f0f0f")
fig.suptitle("Silver Price Predictor Dashboard", color="silver", fontsize=16, fontweight="bold")

ax1, ax2, ax3, ax4 = axes
for ax in axes:
    ax.set_facecolor("#1a1a1a")
    ax.tick_params(colors="#aaaaaa")
    ax.spines[:].set_color("#333333")

recent = ta[-500:]
silver_color = "#C0C0C0"

# Panel 1: Price
ax1.plot(recent.index, recent["silver"],  color=silver_color, lw=1.5, label="Silver")
ax1.plot(recent.index, recent["sma50"],   color="#4fc3f7",    lw=1,   label="SMA 50",  alpha=0.8)
ax1.plot(recent.index, recent["sma200"],  color="#ef5350",    lw=1,   label="SMA 200", alpha=0.8)
ax1.fill_between(recent.index, recent["bb_lo"], recent["bb_hi"], alpha=0.1, color=silver_color)

future_dates = pd.date_range(silver_monthly.index[-1], periods=months_to_2030, freq="ME")
ax1.plot(future_dates, future_y, color="#69f0ae", lw=1.5, linestyle="--", label="2030 forecast")
ax1.fill_between(
    [ta.index[-1], future_dates[-1]],
    [price, t2030_bear], [price, t2030_bull],
    alpha=0.15, color="#69f0ae"
)
ax1.set_ylabel("Price USD/oz", color="#aaaaaa")
ax1.legend(loc="upper left", facecolor="#1a1a1a", labelcolor="#cccccc", fontsize=8)
ax1.set_title("Silver Price + Bollinger + 2030 Projection", color="#cccccc", fontsize=10)

# Panel 2: Gold/Silver Ratio
ax2.plot(recent.index, recent["gsr"], color="#ffd54f", lw=1.2, label="GSR")
ax2.axhline(65, color="#69f0ae", lw=0.8, linestyle="--", label="Mean ~65 (buy silver)")
ax2.axhline(80, color="#ef5350", lw=0.8, linestyle="--", label="Extreme 80 (very cheap)")
ax2.fill_between(recent.index, recent["gsr"], 65,
    where=recent["gsr"] > 65, alpha=0.2, color="#ef5350", label="Silver cheap zone")
ax2.set_ylabel("GSR", color="#aaaaaa")
ax2.legend(loc="upper left", facecolor="#1a1a1a", labelcolor="#cccccc", fontsize=8)
ax2.set_title("Gold/Silver Ratio (high = silver undervalued)", color="#cccccc", fontsize=10)

# Panel 3: RSI
ax3.plot(recent.index, recent["rsi"], color="#ce93d8", lw=1.2)
ax3.axhline(70, color="#ef5350", lw=0.8, linestyle="--")
ax3.axhline(30, color="#69f0ae", lw=0.8, linestyle="--")
ax3.axhline(50, color="#555555", lw=0.5)
ax3.fill_between(recent.index, recent["rsi"], 50, where=recent["rsi"] > 50, alpha=0.2, color="#ef5350")
ax3.fill_between(recent.index, recent["rsi"], 50, where=recent["rsi"] < 50, alpha=0.2, color="#69f0ae")
ax3.set_ylim(0, 100)
ax3.set_ylabel("RSI", color="#aaaaaa")
ax3.set_title("RSI (14)", color="#cccccc", fontsize=10)

# Panel 4: MACD
hist = recent["macd"] - recent["macd_sig"]
ax4.plot(recent.index, recent["macd"],     color="#4fc3f7", lw=1.2, label="MACD")
ax4.plot(recent.index, recent["macd_sig"], color="#ef5350", lw=1.0, label="Signal")
ax4.bar(recent.index, hist,
    color=["#69f0ae" if v >= 0 else "#ef5350" for v in hist], alpha=0.5, width=1)
ax4.axhline(0, color="#555555", lw=0.5)
ax4.set_ylabel("MACD", color="#aaaaaa")
ax4.legend(loc="upper left", facecolor="#1a1a1a", labelcolor="#cccccc", fontsize=8)
ax4.set_title("MACD (12, 26, 9)", color="#cccccc", fontsize=10)

for ax in axes:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

plt.tight_layout()
plt.savefig("output/silver_prediction.png", dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
print("\n📈  Chart saved → silver_prediction.png")
plt.show()
