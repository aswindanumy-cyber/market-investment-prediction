"""
Copper Price Predictor
- Short term signal: BUY / SELL / HOLD
- 3-month & 1-year price target
- 2030 bear / base / bull case

Copper is a pure industrial metal ("Dr. Copper") — no monetary safe-haven
demand like gold/silver. Key extra drivers: China manufacturing (>50% of
global demand), EV/grid electrification, data center buildout, mine supply.

Dependencies:
  pip install yfinance pandas numpy scikit-learn matplotlib
"""

from _base import (
    fetch, sma, ema, rsi, macd, bollinger,
    price_targets, yearly_targets, signal_label,
    dark_axes, fmt_date_axis, print_yearly_table,
    fetch_macro_factors, print_macro_factors,
    MACRO_TOPICS_COPPER,
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
copper = fetch("HG=F")       # Copper futures USD/lb
gold   = fetch("GC=F")        # Gold futures USD/troy oz (for Cu/Au ratio)
china  = fetch("FXI")        # iShares China Large-Cap ETF (China demand proxy)
usd    = fetch("DX-Y.NYB")   # US Dollar Index
vix    = fetch("^VIX")       # Fear index
rates  = fetch("^TNX")       # US 10Y Treasury yield

print(f"✅  Copper data: {copper.index[0].date()} → {copper.index[-1].date()}  ({len(copper)} days)")
print(f"    Copper spot: ${copper.iloc[-1]:.3f}/lb   Gold spot: ${gold.iloc[-1]:,.2f}/oz\n")

ta = pd.DataFrame(index=copper.index)
ta["copper"] = copper
ta["sma20"]  = sma(copper, 20)
ta["sma50"]  = sma(copper, 50)
ta["sma200"] = sma(copper, 200)
ta["rsi"]    = rsi(copper)
ta["macd"], ta["macd_sig"] = macd(copper)
ta["bb_lo"], ta["bb_mid"], ta["bb_hi"] = bollinger(copper)
ta["cgr"]    = copper / gold.reindex(copper.index, method="ffill") * 1000  # Cu/Au ratio ×1000 (readability)
ta["china"]  = china.reindex(copper.index, method="ffill")
ta["usd"]    = usd.reindex(copper.index, method="ffill")
ta["vix"]    = vix.reindex(copper.index, method="ffill")
ta["rates"]  = rates.reindex(copper.index, method="ffill")
ta.dropna(inplace=True)

last  = ta.iloc[-1]
price = last["copper"]
cgr   = last["cgr"]

# ─────────────────────────────────────────────
# 3. SIGNAL SCORING
# ─────────────────────────────────────────────
scores = {}

scores["SMA20 > SMA50"]  = 8 if last["sma20"] > last["sma50"]  else 2
scores["SMA50 > SMA200"] = 8 if last["sma50"] > last["sma200"] else 2
scores["Price > SMA200"] = 8 if price > last["sma200"]          else 2

rsi_val = last["rsi"]
if rsi_val < 30:   scores["RSI (oversold=buy)"] = 9
elif rsi_val > 70: scores["RSI (overbought)"]   = 2
else:              scores["RSI (neutral)"]       = 5

scores["MACD cross"] = 7 if last["macd"] > last["macd_sig"] else 3

if price < last["bb_lo"]:   scores["Bollinger (below lo)"] = 9
elif price > last["bb_hi"]: scores["Bollinger (above hi)"] = 2
else:                        scores["Bollinger (mid)"]      = 5

china_1m = ta["china"].iloc[-22] if len(ta) > 22 else ta["china"].iloc[0]
scores["China demand (FXI trend)"] = 8 if last["china"] > china_1m else 3

usd_1m = ta["usd"].iloc[-22] if len(ta) > 22 else ta["usd"].iloc[0]
scores["USD weak (copper up)"]      = 3 if last["usd"] > usd_1m else 8
scores["VIX calm (<20=risk-on)"]    = 7 if last["vix"] < 20    else 3   # opposite of gold/silver: copper wants risk-on
scores["10Y rates (<3=copper up)"]  = 7 if last["rates"] < 3    else 3

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
copper_monthly, future_X, future_y, poly, mu_log, vol_log = price_targets(copper)

# ─────────────────────────────────────────────
# 5. YEAR-BY-YEAR MACRO CALENDAR
# ─────────────────────────────────────────────
COPPER_MACRO_CALENDAR = {
    # Copper cycle driven by structural electrification demand vs. chronically
    # underinvested mine supply. 2021 cycle: $2.10→$4.90 (+130%) on EV/green boom.
    2026: (1.15, BULLISH,
           "China property stimulus stabilizes demand; grid modernization capex; Fed cuts weaken USD"),
    2027: (1.20, BULLISH,
           "EV penetration >30% globally; data center capex supercycle; ore grades decline further"),
    2028: (1.28, VERY_BULLISH,
           "Structural mine supply deficit widens (no major new projects since 2015); AI power buildout"),
    2029: (1.22, BULLISH,
           "Grid electrification capex peaks in US/EU; recycling can't offset primary demand gap"),
    2030: (1.32, VERY_BULLISH,
           "Net-zero transition requires 2x current copper output; supply response too slow; price discovery"),
}

yearly_copper = yearly_targets(price, mu_log, vol_log, COPPER_MACRO_CALENDAR)


print("=" * 60)
print("        COPPER PRICE PREDICTOR REPORT")
print(f"        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("=" * 60)

print(f"\n🟠  Current Copper Price: ${price:,.3f} / lb")
print(f"📊  Signal Score:         {total_score:.1f} / 10")
print(f"🎯  Overall Signal:       {signal}")
print(f"⚖️   Cu/Au Ratio (×1000):  {cgr:.2f}  (growth-optimism gauge)\n")

print("── Technical Breakdown ─────────────────────────")
for k, v in scores.items():
    bar       = "█" * int(v) + "░" * (10 - int(v))
    sentiment = "bullish" if v >= 6 else ("bearish" if v <= 4 else "neutral")
    print(f"  {k:<40} {bar}  {sentiment}")

print("\n── Short/Mid-Term Targets ───────────────────────")
print(f"  3-Month  │ Bear: ${t3b:>7,.3f}  Base: ${t3:>7,.3f}  Bull: ${t3u:>7,.3f}")
print(f"  12-Month │ Bear: ${t12b:>7,.3f}  Base: ${t12:>7,.3f}  Bull: ${t12u:>7,.3f}")

print_yearly_table(yearly_copper)

_cgr_v  = float(cgr)
_dxy_v  = float(usd.iloc[-1])
_r_v    = float(rates.iloc[-1])
_china_v = float(china.iloc[-1])
macro_rows = fetch_macro_factors(MACRO_TOPICS_COPPER, asset="COPPER", fallbacks={
    "China manufacturing PMI":    f"FXI at ${_china_v:.2f} — {'China equities firming, demand tailwind' if last['china'] > china_1m else 'China demand signal soft; watch stimulus'}",
    "EV & grid electrification":  "EV motors/wiring use ~80kg copper each — 3-4x an ICE vehicle; grid upgrades add structural demand",
    "Data center & AI buildout":  "Hyperscaler capex driving copper-intensive power/cooling infrastructure builds",
    "Green energy infrastructure":"Wind turbines use ~4t copper each; solar farms and transmission lines add multi-decade demand",
    "Global mine supply":         "No major new copper mine has come online since 2015; ore grades declining at existing mines",
    "Housing & construction":     "Residential/commercial wiring and plumbing remain a core structural demand pillar",
    "Inventory levels":           "LME/COMEX warehouse stocks near multi-year lows, tightening physical market",
    "US-China trade tensions":    f"DXY {_dxy_v:.1f}, 10Y {_r_v:.2f}% — {'trade friction risk to demand, offset by weak USD' if _dxy_v > 100 else 'macro backdrop supportive of industrial metals'}",
})
print_macro_factors(macro_rows, "Macro Tailwinds / Headwinds (Copper)")

print("\n── Key Levels ───────────────────────────────────")
print(f"  SMA 20:  ${last['sma20']:,.3f}    SMA 50: ${last['sma50']:,.3f}    SMA 200: ${last['sma200']:,.3f}")
print(f"  RSI: {last['rsi']:.1f}    MACD: {last['macd']:.4f}    Signal: {last['macd_sig']:.4f}")

print("\n⚠️   DISCLAIMER: Not financial advice. Model-based estimates only.")
print("=" * 60)

# ─────────────────────────────────────────────
# 6. CHART
# ─────────────────────────────────────────────
fig, axes = plt.subplots(4, 1, figsize=(14, 14), facecolor="#0f0f0f")
fig.suptitle(
    f"Copper Price Predictor  —  ${price:.3f}/lb  |  Cu/Au {cgr:.2f}  |  {signal.strip()}",
    color="#e07a3f", fontsize=14, fontweight="bold"
)

ax1, ax2, ax3, ax4 = axes
dark_axes(axes)

recent       = ta[-500:]
copper_color = "#e07a3f"
future_dates = pd.date_range(copper_monthly.index[-1], periods=len(future_X), freq="ME")

# Panel 1: Price + BB + 2030
ax1.plot(recent.index, recent["copper"], color=copper_color, lw=1.5, label="Copper spot")
ax1.plot(recent.index, recent["sma50"],  color="#4fc3f7",    lw=1,   label="SMA 50",  alpha=0.8)
ax1.plot(recent.index, recent["sma200"], color="#ef5350",    lw=1,   label="SMA 200", alpha=0.8)
ax1.fill_between(recent.index, recent["bb_lo"], recent["bb_hi"], alpha=0.1, color=copper_color)
ax1.plot(future_dates, future_y, color="#69f0ae", lw=1.5, linestyle="--", label="2030 base")
ax1.fill_between(
    [ta.index[-1], future_dates[-1]],
    [price, t2030_bear], [price, t2030_bull],
    alpha=0.15, color="#69f0ae"
)
ax1.set_ylabel("Price USD/lb", color="#aaaaaa")
ax1.legend(loc="upper left", facecolor="#1a1a1a", labelcolor="#cccccc", fontsize=8)
ax1.set_title("Copper Spot Price + Bollinger + 2030 Projection", color="#cccccc", fontsize=10)

# Panel 2: Copper/Gold Ratio
ax2.plot(recent.index, recent["cgr"], color="#ffd54f", lw=1.2, label="Cu/Au ×1000")
ax2.set_ylabel("Cu/Au Ratio", color="#aaaaaa")
ax2.legend(loc="upper left", facecolor="#1a1a1a", labelcolor="#cccccc", fontsize=8)
ax2.set_title("Copper/Gold Ratio (growth-optimism gauge)", color="#cccccc", fontsize=10)

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
plt.savefig("output/copper_prediction.png", dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
print("\n📈  Chart saved → output/copper_prediction.png")
plt.show()
