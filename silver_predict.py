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

from _base import (
    fetch, sma, ema, rsi, macd, bollinger,
    price_targets, yearly_targets, signal_label,
    dark_axes, fmt_date_axis, print_yearly_table,
    VERY_BULLISH, BULLISH, NEUTRAL,
)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# ─────────────────────────────────────────────
# 1. FETCH DATA
# ─────────────────────────────────────────────
print("📡  Fetching market data...")
silver = fetch("SI=F")        # Silver futures USD/troy oz
gold   = fetch("GC=F")        # Gold futures USD/troy oz (for GSR)
usd    = fetch("DX-Y.NYB")   # US Dollar Index
copper = fetch("HG=F")       # Copper (industrial proxy)
vix    = fetch("^VIX")       # Fear index
rates  = fetch("^TNX")       # US 10Y Treasury yield

print(f"✅  Silver data: {silver.index[0].date()} → {silver.index[-1].date()}  ({len(silver)} days)")
print(f"    Silver spot: ${silver.iloc[-1]:.2f}/oz   Gold spot: ${gold.iloc[-1]:.2f}/oz\n")

ta = pd.DataFrame(index=silver.index)
ta["silver"]  = silver
ta["sma20"]   = sma(silver, 20)
ta["sma50"]   = sma(silver, 50)
ta["sma200"]  = sma(silver, 200)
ta["rsi"]     = rsi(silver)
ta["macd"], ta["macd_sig"] = macd(silver)
ta["bb_lo"], ta["bb_mid"], ta["bb_hi"] = bollinger(silver)
ta["gsr"]     = gold.reindex(silver.index, method="ffill") / silver
ta["copper"]  = copper.reindex(silver.index, method="ffill")
ta["usd"]     = usd.reindex(silver.index, method="ffill")
ta["vix"]     = vix.reindex(silver.index, method="ffill")
ta["rates"]   = rates.reindex(silver.index, method="ffill")
ta.dropna(inplace=True)

last  = ta.iloc[-1]
price = last["silver"]
gsr   = last["gsr"]

# ─────────────────────────────────────────────
# 3. SIGNAL SCORING
# ─────────────────────────────────────────────
scores = {}

scores["SMA20 > SMA50"]         = 8 if last["sma20"] > last["sma50"]  else 2
scores["SMA50 > SMA200"]        = 8 if last["sma50"] > last["sma200"] else 2
scores["Price > SMA200"]        = 8 if price > last["sma200"]          else 2

rsi_val = last["rsi"]
if rsi_val < 30:   scores["RSI (oversold=buy)"]  = 9
elif rsi_val > 70: scores["RSI (overbought)"]    = 2
else:              scores["RSI (neutral)"]        = 5

scores["MACD cross"] = 7 if last["macd"] > last["macd_sig"] else 3

if price < last["bb_lo"]:   scores["Bollinger (below lo)"] = 9
elif price > last["bb_hi"]: scores["Bollinger (above hi)"] = 2
else:                        scores["Bollinger (mid)"]      = 5

if gsr > 80:   scores["GSR >80 (silver undervalued)"] = 9
elif gsr > 65: scores["GSR 65-80 (slightly cheap)"]   = 6
else:          scores["GSR <65 (silver expensive)"]    = 3

copper_1m = ta["copper"].iloc[-22] if len(ta) > 22 else ta["copper"].iloc[0]
scores["Copper rising (industrial)"] = 7 if last["copper"] > copper_1m else 3

usd_1m = ta["usd"].iloc[-22] if len(ta) > 22 else ta["usd"].iloc[0]
scores["USD weak (silver up)"]      = 3 if last["usd"] > usd_1m else 8
scores["VIX fear (>20=safe haven)"] = 7 if last["vix"] > 20    else 4
scores["10Y rates (<3=silver up)"]  = 7 if last["rates"] < 3    else 3

total_score = np.mean(list(scores.values()))

if total_score >= 6.5:   signal = "🟢  BUY"
elif total_score <= 4.0: signal = "🔴  SELL"
else:                    signal = "🟡  HOLD"

# ─────────────────────────────────────────────
# 4+5. PRICE TARGETS + 2030 PROJECTION  (from _base)
# ─────────────────────────────────────────────
(t3b, t3, t3u), \
(t12b, t12, t12u), \
(t2030_bear, t2030, t2030_bull), \
silver_monthly, future_X, future_y, poly, mu_log, vol_log = price_targets(silver)

# ─────────────────────────────────────────────
# 5. YEAR-BY-YEAR MACRO CALENDAR
# ─────────────────────────────────────────────
SILVER_MACRO_CALENDAR = {
    2026: (1.10, BULLISH,
           "Solar installs record; EV scaling; Fed cuts weaken USD; GSR mean reversion ongoing"),
    2027: (1.15, VERY_BULLISH,
           "Green energy capex peak; 5G silver paste demand; supply deficit widens 3rd yr"),
    2028: (1.12, VERY_BULLISH,
           "AI data centers (silver cooling/PCBs); US election fiscal push; ETF inflows surge"),
    2029: (1.08, BULLISH,
           "EV penetration >35%; primary mine depletion; BRICS+ monetary silver demand"),
    2030: (1.15, VERY_BULLISH,
           "Net-zero peak demand; perovskite solar 10x silver; structural deficit 200M oz/yr"),
}

yearly_silver = yearly_targets(price, mu_log, vol_log, SILVER_MACRO_CALENDAR)

macro_factors = [
    ("Solar panel demand",   "Each panel uses ~20g silver; solar capacity doubling every 3yrs"),
    ("EV & battery tech",    "Silver conductivity critical in EV charging & battery management"),
    ("5G & electronics",     "Every smartphone, chip, and PCB uses silver"),
    ("Green energy mandate", "Global net-zero targets drive massive industrial silver demand"),
    ("Gold/Silver Ratio",    f"Current GSR {gsr:.1f}x — historical mean ~65x; reversion = upside"),
    ("Central bank buying",  "Less than gold, but ETF inflows picking up"),
    ("Supply constraints",   "Primary silver mines declining; mostly byproduct of copper/zinc mining"),
    ("Inflation hedge",      "Like gold but with industrial kicker = double tailwind"),
]

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
    bar       = "█" * int(v) + "░" * (10 - int(v))
    sentiment = "bullish" if v >= 6 else ("bearish" if v <= 4 else "neutral")
    print(f"  {k:<40} {bar}  {sentiment}")

print("\n── Short/Mid-Term Targets ───────────────────────")
print(f"  3-Month  │ Bear: ${t3b:>7,.2f}  Base: ${t3:>7,.2f}  Bull: ${t3u:>7,.2f}")
print(f"  12-Month │ Bear: ${t12b:>7,.2f}  Base: ${t12:>7,.2f}  Bull: ${t12u:>7,.2f}")

print_yearly_table(yearly_silver)

print("\n── Macro & Industrial Tailwinds ─────────────────")
for factor, reason in macro_factors:
    print(f"  ▸ {factor:<28} {reason}")

print("\n── Key Levels ───────────────────────────────────")
print(f"  SMA 20:  ${last['sma20']:,.2f}    SMA 50: ${last['sma50']:,.2f}    SMA 200: ${last['sma200']:,.2f}")
print(f"  RSI: {last['rsi']:.1f}    MACD: {last['macd']:.3f}    Signal: {last['macd_sig']:.3f}")

print("\n⚠️   DISCLAIMER: Not financial advice. Model-based estimates only.")
print("=" * 60)

# ─────────────────────────────────────────────
# 6. CHART
# ─────────────────────────────────────────────
fig, axes = plt.subplots(4, 1, figsize=(14, 14), facecolor="#0f0f0f")
fig.suptitle(
    f"Silver Price Predictor  —  ${price:.2f}/oz  |  GSR {gsr:.1f}x  |  {signal.strip()}",
    color="silver", fontsize=14, fontweight="bold"
)

ax1, ax2, ax3, ax4 = axes
dark_axes(axes)

recent       = ta[-500:]
silver_color = "#C0C0C0"
future_dates = pd.date_range(silver_monthly.index[-1], periods=months_to_2030, freq="ME")

# Panel 1: Price + BB + 2030
ax1.plot(recent.index, recent["silver"],  color=silver_color, lw=1.5, label="Silver spot")
ax1.plot(recent.index, recent["sma50"],   color="#4fc3f7",    lw=1,   label="SMA 50",  alpha=0.8)
ax1.plot(recent.index, recent["sma200"],  color="#ef5350",    lw=1,   label="SMA 200", alpha=0.8)
ax1.fill_between(recent.index, recent["bb_lo"], recent["bb_hi"], alpha=0.1, color=silver_color)
ax1.plot(future_dates, future_y, color="#69f0ae", lw=1.5, linestyle="--", label="2030 base")
ax1.fill_between(
    [ta.index[-1], future_dates[-1]],
    [price, t2030_bear], [price, t2030_bull],
    alpha=0.15, color="#69f0ae"
)
ax1.set_ylabel("Price USD/oz", color="#aaaaaa")
ax1.legend(loc="upper left", facecolor="#1a1a1a", labelcolor="#cccccc", fontsize=8)
ax1.set_title("Silver Spot Price + Bollinger + 2030 Projection", color="#cccccc", fontsize=10)

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
ax3.axhline(70, color="#ef5350", lw=0.8, linestyle="--", label="Overbought 70")
ax3.axhline(30, color="#69f0ae", lw=0.8, linestyle="--", label="Oversold 30")
ax3.axhline(50, color="#555555", lw=0.5)
ax3.fill_between(recent.index, recent["rsi"], 50, where=recent["rsi"] > 50, alpha=0.2, color="#ef5350")
ax3.fill_between(recent.index, recent["rsi"], 50, where=recent["rsi"] < 50, alpha=0.2, color="#69f0ae")
ax3.set_ylim(0, 100)
ax3.set_ylabel("RSI", color="#aaaaaa")
ax3.legend(loc="upper left", facecolor="#1a1a1a", labelcolor="#cccccc", fontsize=8)
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

fmt_date_axis(axes)

plt.tight_layout()
plt.savefig("output/silver_prediction.png", dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
print("\n📈  Chart saved → output/silver_prediction.png")
plt.show()
