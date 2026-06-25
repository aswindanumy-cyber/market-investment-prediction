"""
Indonesian Stock Predictor (IDX)

Usage:
  python stock_ID_predict.py BBCA        → predict single stock (e.g. BBCA.JK)
  python stock_ID_predict.py TOP10       → scan & rank top 10 buy opportunities on IDX

Covers: signal scoring, 3-month / 1-year targets, 2030 projection, macro IDR factors.

Dependencies:
  pip install yfinance pandas numpy scikit-learn matplotlib
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from datetime import datetime

# ─────────────────────────────────────────────
# TOP 10 IDX UNIVERSE — Blue chips + high liquidity
# ─────────────────────────────────────────────
IDX_UNIVERSE = [
    "BBCA", "BBRI", "BMRI", "TLKM", "ASII",
    "GOTO", "UNVR", "ICBP", "KLBF", "INDF",
    "ANTM", "PTBA", "ADRO", "PGAS", "SMGR",
    "EXCL", "EMTK", "MIKA", "SIDO", "HEAL",
]

# ─────────────────────────────────────────────
# MACRO TICKERS (IDR context)
# ─────────────────────────────────────────────
MACRO = {
    "usd_idr": "IDR=X",    # USD/IDR exchange rate
    "ihsg":    "^JKSE",    # Jakarta Composite Index
    "vix":     "^VIX",     # Global fear index
    "crude":   "CL=F",     # Oil (Indonesia is commodity-linked)
    "rates":   "^TNX",     # US 10Y (influences BI rate)
}

# ─────────────────────────────────────────────
# HELPERS
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

def fetch_stock(ticker_code):
    """Download 5yr daily data for a single IDX stock."""
    t = ticker_code.upper()
    if not t.endswith(".JK"):
        t += ".JK"
    df = yf.download(t, period="5y", interval="1d", auto_adjust=True, progress=False)["Close"]
    df = df.dropna()
    if isinstance(df, pd.DataFrame):
        df = df.squeeze()
    return df, t

def fetch_macro():
    """Download macro context data."""
    raw = yf.download(
        list(MACRO.values()), period="5y", interval="1d",
        auto_adjust=True, progress=False
    )["Close"]
    raw.columns = list(MACRO.keys())
    raw.ffill(inplace=True)
    return raw

def score_stock(price_series, macro_df):
    """Return signal score (0–10), scores dict, last-row TA."""
    s = price_series.copy()
    ta = pd.DataFrame(index=s.index)
    ta["price"]   = s
    ta["sma20"]   = sma(s, 20)
    ta["sma50"]   = sma(s, 50)
    ta["sma200"]  = sma(s, 200)
    ta["rsi"]     = rsi(s)
    ta["macd"], ta["macd_sig"] = macd(s)
    ta["bb_lo"], ta["bb_mid"], ta["bb_hi"] = bollinger(s)

    # merge macro
    for col in ["usd_idr", "ihsg", "vix", "crude"]:
        if col in macro_df.columns:
            ta[col] = macro_df[col].reindex(ta.index, method="ffill")

    ta.dropna(inplace=True)
    if len(ta) < 10:
        return None, {}, None

    last  = ta.iloc[-1]
    price = last["price"]
    scores = {}

    # Trend
    scores["SMA20 > SMA50"]  = 8 if last["sma20"] > last["sma50"]  else 2
    scores["SMA50 > SMA200"] = 8 if last["sma50"] > last["sma200"] else 2
    scores["Price > SMA200"] = 8 if price > last["sma200"]          else 2

    # Momentum
    rv = last["rsi"]
    if rv < 30:   scores["RSI oversold"]   = 9
    elif rv > 70: scores["RSI overbought"] = 2
    else:         scores["RSI neutral"]    = 5

    scores["MACD cross"] = 7 if last["macd"] > last["macd_sig"] else 3

    # Bollinger
    if price < last["bb_lo"]:   scores["BB below low"] = 9
    elif price > last["bb_hi"]: scores["BB above high"] = 2
    else:                        scores["BB mid"]        = 5

    # Macro: IDR (weak IDR = pressure on imports / earnings)
    if "usd_idr" in ta.columns:
        idr_1m  = ta["usd_idr"].iloc[-22] if len(ta) > 22 else ta["usd_idr"].iloc[0]
        idr_now = last["usd_idr"]
        scores["IDR stable (<16500)"] = 7 if idr_now < 16500 else 3

    # Macro: IHSG trend
    if "ihsg" in ta.columns:
        ihsg_1m = ta["ihsg"].iloc[-22] if len(ta) > 22 else ta["ihsg"].iloc[0]
        scores["IHSG rising"] = 7 if last["ihsg"] > ihsg_1m else 3

    # Macro: VIX
    if "vix" in ta.columns:
        scores["VIX low (<20)"] = 7 if last["vix"] < 20 else 3

    total = np.mean(list(scores.values()))
    return total, scores, ta

def price_targets(price_series):
    """Return (3m bear/base/bull), (12m bear/base/bull), (2030 bear/base/bull)."""
    s = price_series.dropna()
    ret       = s.pct_change().dropna()
    mu        = ret[-126:].mean()
    vol       = ret[-126:].std()
    price     = s.iloc[-1]

    t3b  = price * (1 + mu - vol) ** 63
    t3   = price * (1 + mu)       ** 63
    t3u  = price * (1 + mu + vol) ** 63
    t12b = price * (1 + mu - vol) ** 252
    t12  = price * (1 + mu)       ** 252
    t12u = price * (1 + mu + vol) ** 252

    monthly = s.resample("ME").last()
    X  = np.arange(len(monthly)).reshape(-1, 1)
    y  = monthly.values
    poly = PolynomialFeatures(2)
    Xp   = poly.fit_transform(X)
    reg  = LinearRegression().fit(Xp, y)

    m2030   = (datetime(2030, 12, 31) - monthly.index[-1]).days // 30
    fX      = np.arange(len(monthly), len(monthly) + m2030).reshape(-1, 1)
    fy      = reg.predict(poly.transform(fX))
    t2030   = max(fy[-1], price)

    ann_vol       = ret.std() * np.sqrt(252)
    yrs_to_2030   = (datetime(2030, 12, 31) - datetime.now()).days / 365
    t2030_bull    = t2030 * (1 + ann_vol * 0.6) ** yrs_to_2030
    t2030_bear    = t2030 * (1 - ann_vol * 0.3) ** yrs_to_2030

    return (t3b, t3, t3u), (t12b, t12, t12u), (t2030_bear, t2030, t2030_bull), monthly, fX, fy, poly

def signal_label(score):
    if score >= 6.5:   return "🟢 BUY"
    elif score <= 4.0: return "🔴 SELL"
    else:              return "🟡 HOLD"

# ─────────────────────────────────────────────
# SINGLE STOCK REPORT
# ─────────────────────────────────────────────
def run_single(ticker_code):
    print(f"\n📡  Fetching {ticker_code.upper()}.JK + macro data...")
    macro_df = fetch_macro()
    price_series, full_ticker = fetch_stock(ticker_code)

    if price_series.empty:
        print(f"❌  No data found for {full_ticker}. Check the ticker code.")
        return

    price = price_series.iloc[-1]
    print(f"✅  {full_ticker}: {len(price_series)} days  |  Latest: Rp {price:,.0f}\n")

    total_score, scores, ta = score_stock(price_series, macro_df)
    if ta is None:
        print("❌  Not enough data to score."); return

    t3, t12, t2030, monthly, fX, fy, poly = price_targets(price_series)
    last = ta.iloc[-1]

    print("=" * 62)
    print(f"   IDX STOCK REPORT — {full_ticker}")
    print(f"   Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 62)
    print(f"\n💰  Price:   Rp {price:>12,.0f}")
    print(f"📊  Score:   {total_score:.1f} / 10")
    print(f"🎯  Signal:  {signal_label(total_score)}\n")

    print("── Technical Breakdown ─────────────────────────────")
    for k, v in scores.items():
        bar  = "█" * int(v) + "░" * (10 - int(v))
        mood = "bullish" if v >= 6 else ("bearish" if v <= 4 else "neutral")
        print(f"  {k:<35} {bar}  {mood}")

    print("\n── Price Targets (IDR) ─────────────────────────────")
    print(f"  3-Month  │ Bear: {t3[0]:>10,.0f}  Base: {t3[1]:>10,.0f}  Bull: {t3[2]:>10,.0f}")
    print(f"  12-Month │ Bear: {t12[0]:>10,.0f}  Base: {t12[1]:>10,.0f}  Bull: {t12[2]:>10,.0f}")
    print(f"  2030     │ Bear: {t2030[0]:>10,.0f}  Base: {t2030[1]:>10,.0f}  Bull: {t2030[2]:>10,.0f}")

    print("\n── IDX Macro Context ────────────────────────────────")
    macro_factors = [
        ("Bank Indonesia rate",  "BI rate affects cost of capital for all IDX companies"),
        ("USD/IDR",             f"Current {last.get('usd_idr', 0):,.0f} — above 16500 = earnings pressure"),
        ("Commodity exports",    "Indonesia exports coal, CPO, nickel — commodity cycle matters"),
        ("Consumer growth",      "270M population with rising middle class = long-term demand"),
        ("2030 IKN / infra",     "New capital city & infra spending = construction/bank boost"),
        ("Nickel & EV supply",   "Indonesia has 40%+ global nickel reserves — EV boom tailwind"),
    ]
    for f, r in macro_factors:
        print(f"  ▸ {f:<28} {r}")

    print("\n── Key Levels ───────────────────────────────────────")
    print(f"  SMA20: {last['sma20']:>10,.0f}  SMA50: {last['sma50']:>10,.0f}  SMA200: {last['sma200']:>10,.0f}")
    print(f"  RSI:   {last['rsi']:>10.1f}  MACD:  {last['macd']:>10.2f}  Signal: {last['macd_sig']:>10.2f}")
    print(f"  BB Lo: {last['bb_lo']:>10,.0f}  BB Mid: {last['bb_mid']:>9,.0f}  BB Hi:  {last['bb_hi']:>10,.0f}")

    print("\n⚠️   DISCLAIMER: Not financial advice.")
    print("=" * 62)

    # Chart
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), facecolor="#0f0f0f")
    fig.suptitle(f"IDX Stock Predictor — {full_ticker}", color="#4fc3f7", fontsize=15, fontweight="bold")
    ax1, ax2, ax3 = axes
    for ax in axes:
        ax.set_facecolor("#1a1a1a"); ax.tick_params(colors="#aaaaaa"); ax.spines[:].set_color("#333333")

    recent = ta[-500:]
    clr    = "#4fc3f7"

    ax1.plot(recent.index, recent["price"],   color=clr,       lw=1.5, label=full_ticker)
    ax1.plot(recent.index, recent["sma50"],   color="#ffd54f", lw=1,   label="SMA50",  alpha=0.8)
    ax1.plot(recent.index, recent["sma200"],  color="#ef5350", lw=1,   label="SMA200", alpha=0.8)
    ax1.fill_between(recent.index, recent["bb_lo"], recent["bb_hi"], alpha=0.08, color=clr)
    future_dates = pd.date_range(monthly.index[-1], periods=len(fX), freq="ME")
    ax1.plot(future_dates, fy, color="#69f0ae", lw=1.5, linestyle="--", label="2030 forecast")
    ax1.fill_between([ta.index[-1], future_dates[-1]],
        [price, t2030[0]], [price, t2030[2]], alpha=0.12, color="#69f0ae")
    ax1.set_ylabel("Price (IDR)", color="#aaaaaa")
    ax1.legend(loc="upper left", facecolor="#1a1a1a", labelcolor="#cccccc", fontsize=8)
    ax1.set_title("Price + Bollinger + 2030 Projection", color="#cccccc", fontsize=10)

    ax2.plot(recent.index, recent["rsi"], color="#ce93d8", lw=1.2)
    ax2.axhline(70, color="#ef5350", lw=0.8, linestyle="--", label="Overbought")
    ax2.axhline(30, color="#69f0ae", lw=0.8, linestyle="--", label="Oversold")
    ax2.axhline(50, color="#555555", lw=0.5)
    ax2.fill_between(recent.index, recent["rsi"], 50, where=recent["rsi"] > 50, alpha=0.2, color="#ef5350")
    ax2.fill_between(recent.index, recent["rsi"], 50, where=recent["rsi"] < 50, alpha=0.2, color="#69f0ae")
    ax2.set_ylim(0, 100); ax2.set_ylabel("RSI", color="#aaaaaa")
    ax2.legend(loc="upper left", facecolor="#1a1a1a", labelcolor="#cccccc", fontsize=8)
    ax2.set_title("RSI (14)", color="#cccccc", fontsize=10)

    hist = recent["macd"] - recent["macd_sig"]
    ax3.plot(recent.index, recent["macd"],     color="#4fc3f7", lw=1.2, label="MACD")
    ax3.plot(recent.index, recent["macd_sig"], color="#ef5350", lw=1.0, label="Signal")
    ax3.bar(recent.index, hist, color=["#69f0ae" if v >= 0 else "#ef5350" for v in hist], alpha=0.5, width=1)
    ax3.axhline(0, color="#555555", lw=0.5)
    ax3.set_ylabel("MACD", color="#aaaaaa")
    ax3.legend(loc="upper left", facecolor="#1a1a1a", labelcolor="#cccccc", fontsize=8)
    ax3.set_title("MACD (12, 26, 9)", color="#cccccc", fontsize=10)

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    plt.tight_layout()
    fname = f"{ticker_code.upper()}_prediction.png"
    plt.savefig(f"output/{fname}", dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
    print(f"\n📈  Chart saved → {fname}")
    plt.show()

# ─────────────────────────────────────────────
# TOP 10 SCANNER
# ─────────────────────────────────────────────
def run_top10():
    print("\n📡  Scanning IDX universe for top opportunities...")
    print(f"    Checking {len(IDX_UNIVERSE)} stocks...\n")

    macro_df = fetch_macro()
    results  = []

    for code in IDX_UNIVERSE:
        try:
            s, ticker = fetch_stock(code)
            if s.empty or len(s) < 60:
                continue
            score, scores, ta = score_stock(s, macro_df)
            if score is None:
                continue
            price = s.iloc[-1]
            t3, t12, t2030, *_ = price_targets(s)
            upside_3m  = (t3[1]  / price - 1) * 100
            upside_1y  = (t12[1] / price - 1) * 100
            results.append({
                "ticker":     ticker,
                "price":      price,
                "score":      score,
                "signal":     signal_label(score),
                "upside_3m":  upside_3m,
                "upside_1y":  upside_1y,
                "t3_base":    t3[1],
                "t12_base":   t12[1],
                "t2030_base": t2030[1],
            })
            mood = "🟢" if score >= 6.5 else ("🔴" if score <= 4 else "🟡")
            print(f"  {mood} {ticker:<12} Score: {score:.1f}  Price: Rp {price:>10,.0f}  3m upside: {upside_3m:>+.1f}%")
        except Exception as e:
            print(f"  ⚠️  {code}: {e}")

    if not results:
        print("No results."); return

    df = pd.DataFrame(results).sort_values("score", ascending=False).head(10)

    print("\n" + "=" * 72)
    print("   TOP 10 IDX SWEET SPOTS — RANKED BY SIGNAL SCORE")
    print(f"   Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 72)
    print(f"\n{'#':<3} {'Ticker':<12} {'Signal':<10} {'Score':>5} {'Price':>12} {'3m%':>7} {'1y%':>7} {'2030 Base':>14}")
    print("-" * 72)
    for i, row in enumerate(df.itertuples(), 1):
        print(
            f"{i:<3} {row.ticker:<12} {row.signal:<10} {row.score:>5.1f} "
            f"Rp {row.price:>10,.0f} {row.upside_3m:>+6.1f}% {row.upside_1y:>+6.1f}% "
            f"Rp {row.t2030_base:>12,.0f}"
        )
    print("=" * 72)
    print("\n⚠️   DISCLAIMER: Not financial advice. Scan based on technical signals only.")

    # Summary chart
    fig, ax = plt.subplots(figsize=(12, 6), facecolor="#0f0f0f")
    ax.set_facecolor("#1a1a1a"); ax.tick_params(colors="#aaaaaa"); ax.spines[:].set_color("#333333")
    colors = ["#69f0ae" if s >= 6.5 else ("#ef5350" if s <= 4.0 else "#ffd54f") for s in df["score"]]
    bars = ax.barh(df["ticker"], df["score"], color=colors, edgecolor="#333333")
    ax.axvline(6.5, color="#69f0ae", lw=1, linestyle="--", label="Buy threshold 6.5")
    ax.axvline(4.0, color="#ef5350", lw=1, linestyle="--", label="Sell threshold 4.0")
    ax.set_xlim(0, 10)
    ax.set_xlabel("Signal Score (0–10)", color="#aaaaaa")
    ax.set_title("Top IDX Opportunities — Signal Score Ranking", color="#4fc3f7", fontsize=13, fontweight="bold")
    ax.legend(facecolor="#1a1a1a", labelcolor="#cccccc")
    for bar, score in zip(bars, df["score"]):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                f"{score:.1f}", va="center", color="#cccccc", fontsize=9)
    plt.tight_layout()
    plt.savefig("output/top10_IDX_prediction.png", dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
    print("\n📈  Chart saved → top10_IDX_prediction.png")
    plt.show()

# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    arg = sys.argv[1].strip().upper() if len(sys.argv) > 1 else None

    if arg is None:
        print("Usage:")
        print("  python stock_ID_predict.py BBCA        # single stock")
        print("  python stock_ID_predict.py TOP10       # scan top 10 buys")
        sys.exit(0)

    if arg == "TOP10":
        run_top10()
    else:
        run_single(arg)
