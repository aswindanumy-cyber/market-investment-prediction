"""
Gold Price Predictor
- Short term signal: BUY / SELL / HOLD (based on technicals + macro)
- 3-month & 1-year price target
- 2030 bear / base / bull case

Dependencies:
  pip install yfinance pandas numpy scikit-learn matplotlib requests
"""

from _base import (
    fetch, sma, ema, rsi, macd, bollinger,
    price_targets, yearly_targets, signal_label,
    dark_axes, fmt_date_axis, print_yearly_table,
    fetch_macro_factors, print_macro_factors,
    MACRO_TOPICS_GOLD,
    VERY_BULLISH, BULLISH, NEUTRAL, BEARISH,
)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

print("📡  Fetching market data...")
gold   = fetch("GC=F")        # Gold futures USD/troy oz
silver = fetch("SI=F")        # Silver futures USD/troy oz (co-mover)
usd    = fetch("DX-Y.NYB")   # US Dollar Index
vix    = fetch("^VIX")       # Fear Index
rates  = fetch("^TNX")       # US 10Y Treasury Yield
crude  = fetch("CL=F")        # Crude oil (inflation proxy)

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
# 4+5. PRICE TARGETS + 2030 PROJECTION  (from _base)
# ─────────────────────────────────────────────
(target_3m_bear, target_3m_base, target_3m_bull), \
(target_1y_bear, target_1y_base, target_1y_bull), \
(target_2030_bear, target_2030_base, target_2030_bull), \
gold_monthly, future_X, future_y, poly, mu_log, vol_log = price_targets(gold)

# ─────────────────────────────────────────────
# 6. YEAR-BY-YEAR MACRO CALENDAR
# ─────────────────────────────────────────────
GOLD_MACRO_CALENDAR = {
    # multiplier applied on top of 2-sigma log-normal bull case.
    # Calibrated on real cycle data: 2018-2020 gold +75%, 2022-2025 gold +90%+.
    # BULLISH yr = ~1.15-1.20x; VERY_BULLISH = ~1.25-1.40x on bull leg.
    2026: (1.18, BULLISH,
           "Fed rate cuts → USD weakens; US debt ceiling crisis; CB buying 1000t+/yr; ETF inflows surge"),
    2027: (1.15, BULLISH,
           "US recession confirmed; flight-to-safety bid; de-dollarization: RU/CN/BRICS settle in gold"),
    2028: (1.30, VERY_BULLISH,
           "US election — massive fiscal stimulus pledged; real rates deeply negative; CB reserve shift peaks"),
    2029: (1.20, BULLISH,
           "Post-election debt explosion; US debt >$42T; stagflation risk; new mine supply at multi-decade low"),
    2030: (1.35, VERY_BULLISH,
           "BRICS+ commodity settlement system live; ESG blocks new mines; Asian middle class wealth surge"),
    # 2031-2035: same magnitude range as 2026-2030 above (1.10-1.25x/yr),
    # continuing the existing debt/de-dollarization/CB-buying thesis rather
    # than escalating it. Edit the text/numbers to your own view same as the
    # rows above; these are placeholders in the same style, not a claim.
    2031: (1.15, BULLISH,
           "CB gold buying continues at elevated pace; real yields stay compressed post-debt-crisis"),
    2032: (1.12, BULLISH,
           "US election cycle fiscal uncertainty; gold's monetary-hedge demand persists"),
    2033: (1.18, BULLISH,
           "Mine supply growth remains structurally low; reserve diversification away from USD continues"),
    2034: (1.10, NEUTRAL,
           "Demand growth moderates after a decade of CB buying; price consolidates prior gains"),
    2035: (1.20, BULLISH,
           "Decade-long de-dollarization trend matures; gold's reserve-asset share stabilizes higher"),
}

PREDICTION_SPECIAL_CALENDAR = {
    2027: (1.0, None),
    2028: (1.0, None),
    2029: (1.0, None),
    2030: (1.0, None),
    2031: (1.0, None),
    2032: (1.0, None),
    2033: (1.0, None),
    2034: (1.0, None),
    2035: (1.0, None),
}

yearly_gold = yearly_targets(price, mu_log, vol_log, GOLD_MACRO_CALENDAR, PREDICTION_SPECIAL_CALENDAR)


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

print("\n── Short/Mid-Term Targets ───────────────────────")
print(f"  3-Month  │ Bear: ${target_3m_bear:>8,.0f}  Base: ${target_3m_base:>8,.0f}  Bull: ${target_3m_bull:>8,.0f}")
print(f"  12-Month │ Bear: ${target_1y_bear:>8,.0f}  Base: ${target_1y_base:>8,.0f}  Bull: ${target_1y_bull:>8,.0f}")

print_yearly_table(yearly_gold, spot_price=price)

_dxy_v  = float(usd.iloc[-1])
_r_v    = float(rates.iloc[-1])
_vix_v  = float(vix.iloc[-1])
_cr_v   = float(crude.iloc[-1])
macro_rows = fetch_macro_factors(MACRO_TOPICS_GOLD, asset="GOLD", fallbacks={
    "De-dollarization":    f"DXY at {_dxy_v:.1f} — {'USD weak, central banks diversifying to gold' if _dxy_v < 100 else 'USD strong, de-doll. trend ongoing but suppressed'}",
    "US debt trajectory":  "US national debt exceeds $36T; fiscal deficit structurally bullish for gold",
    "Fed rate cycle":      f"US 10Y yield {_r_v:.2f}% — {'rate cuts expected, gold tailwind' if _r_v < 4 else 'high rates = headwind; watch for Fed pivot'}",
    "Geopolitical risk":   f"VIX {_vix_v:.1f} — {'elevated fear, safe-haven flows active' if _vix_v > 20 else 'low fear; geopolitical premium compressed'}",
    "ETF & retail demand": "Gold ETF inflows driven by inflation hedge demand and portfolio diversification",
    "Mining supply":       "Global gold mine output growth <2%/yr; new discoveries declining since 2015",
    "China & India demand":"World's two largest consumers increasing reserve allocations to gold",
    "Inflation hedge":     f"WTI crude ${_cr_v:.1f}/bbl — {'oil-driven inflation supports gold hedge demand' if _cr_v > 75 else 'low oil, mild inflation; gold hedge demand moderate'}",
})
print_macro_factors(macro_rows, "Macro Tailwinds / Headwinds (Gold)")

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
dark_axes(axes)

# Panel 1: Price + MAs + Bollinger + 2030 forecast
recent = ta[-500:]
ax1.plot(recent.index, recent["gold"],   color="gold",   lw=1.5, label="Gold")
ax1.plot(recent.index, recent["sma50"],  color="#4fc3f7", lw=1,  label="SMA 50",  alpha=0.8)
ax1.plot(recent.index, recent["sma200"], color="#ef5350", lw=1,  label="SMA 200", alpha=0.8)
ax1.fill_between(recent.index, recent["bb_lo"], recent["bb_hi"], alpha=0.1, color="gold")

# Future projection
future_dates = pd.date_range(gold_monthly.index[-1], periods=len(future_X), freq="ME")
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

fmt_date_axis(axes)

plt.tight_layout()
plt.savefig("output/gold_prediction.png", dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
print("\n📈  Chart saved → gold_prediction.png")
plt.show()
