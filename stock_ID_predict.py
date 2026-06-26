"""
Indonesian Stock Predictor (IDX)

Usage:
  python stock_ID_predict.py TOP10       → 2 tables: best buy now + best upcoming dividend
  python stock_ID_predict.py HALAL       → same but halal-screened only
  python stock_ID_predict.py BBCA        → single stock full report

Halal criteria (real, not govt trust):
  1. Listed on IDX Sharia index (ISSI) — objective IDX screening
  2. Debt ratio < 30%  (total_liabilities / total_assets from balance sheet)

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
from datetime import datetime, timedelta

# ─────────────────────────────────────────────
# IDX UNIVERSE
# ─────────────────────────────────────────────
IDX_UNIVERSE = [
    "BBCA", "BBRI", "BMRI", "TLKM", "ASII",
    "GOTO", "UNVR", "ICBP", "KLBF", "INDF",
    "ANTM", "PTBA", "ADRO", "PGAS", "SMGR",
    "EXCL", "EMTK", "MIKA", "SIDO", "HEAL",
    "CPIN", "JPFA", "ITMG", "INCO", "MDKA",
    "BBNI", "BRIS", "BTPS", "ESSA", "DNET",
    "HMSP", "GGRM", "MAPI", "ACES", "RALS",
    "BYAN", "MEDC", "HRUM", "CTRA", "PWON",
    "SMRA", "BSDE",
]

# ─────────────────────────────────────────────
# ISSI (IDX Sharia Stock Index) — objective list
# Source: IDX/OJK, updated every 6 months
# ─────────────────────────────────────────────
ISSI_LIST = {
    "TLKM", "ASII", "KLBF", "SIDO", "HEAL", "MIKA",
    "ANTM", "PTBA", "ADRO", "ITMG", "INCO", "MDKA",
    "PGAS", "SMGR", "EXCL", "EMTK", "DNET", "ESSA",
    "ICBP", "INDF", "CPIN", "JPFA", "BRIS", "BTPS",
    "ACES", "MAPI", "RALS", "MEDC", "HRUM", "CTRA",
    "PWON", "SMRA", "BSDE", "BYAN",
}

HALAL_DEBT_THRESHOLD = 0.30

# ─────────────────────────────────────────────
# MACRO TICKERS
# ─────────────────────────────────────────────
MACRO = {
    "usd_idr": "IDR=X",
    "ihsg":    "^JKSE",
    "vix":     "^VIX",
    "crude":   "CL=F",
    "rates":   "^TNX",
}

# ─────────────────────────────────────────────
# INDICATORS
# ─────────────────────────────────────────────
def sma(s, n): return s.rolling(n).mean()
def ema(s, n): return s.ewm(span=n, adjust=False).mean()

def rsi(s, n=14):
    d = s.diff()
    g = d.clip(lower=0).rolling(n).mean()
    l = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))

def macd(s, fast=12, slow=26, sig=9):
    m = ema(s, fast) - ema(s, slow)
    return m, ema(m, sig)

def bollinger(s, n=20, k=2):
    mid = sma(s, n)
    std = s.rolling(n).std()
    return mid - k * std, mid, mid + k * std

# ─────────────────────────────────────────────
# HALAL SCREENING (real: ISSI + debt ratio)
# ─────────────────────────────────────────────
def screen_halal(code):
    """
    Returns (is_halal, debt_ratio, reason).
    Criteria:
      1. Must be in ISSI (IDX Sharia index)
      2. total_liabilities / total_assets < 30%
    """
    if code.upper() not in ISSI_LIST:
        return False, None, "Not in ISSI"

    try:
        tk = yf.Ticker(f"{code}.JK")
        bs = tk.balance_sheet
        if bs is None or bs.empty:
            return False, None, "ISSI ✓ | No balance sheet data"

        col          = bs.columns[0]
        total_assets = None
        total_liab   = None

        for row in bs.index:
            rl = str(row).lower()
            if "total assets" in rl:
                total_assets = bs.loc[row, col]
            if "total liab" in rl:
                total_liab = bs.loc[row, col]

        if total_assets is None or total_liab is None or total_assets == 0:
            return False, None, "ISSI ✓ | Missing balance sheet rows"

        debt_ratio = float(total_liab) / float(total_assets)
        if debt_ratio < HALAL_DEBT_THRESHOLD:
            return True, debt_ratio, f"ISSI ✓ | Debt {debt_ratio:.1%} < 30%"
        else:
            return False, debt_ratio, f"ISSI ✓ | Debt {debt_ratio:.1%} ≥ 30%"

    except Exception as e:
        return False, None, f"ISSI ✓ | Error: {e}"

# ─────────────────────────────────────────────
# FETCH HELPERS
# ─────────────────────────────────────────────
def fetch_price(code):
    t  = f"{code.upper()}.JK"
    df = yf.download(t, period="5y", interval="1d", auto_adjust=True, progress=False)["Close"]
    df = df.dropna()
    if isinstance(df, pd.DataFrame):
        df = df.squeeze()
    return df, t

def fetch_macro():
    raw = yf.download(
        list(MACRO.values()), period="5y", interval="1d",
        auto_adjust=True, progress=False
    )["Close"]
    raw.columns = list(MACRO.keys())
    raw.ffill(inplace=True)
    return raw

def get_dividend_info(code):
    t = f"{code.upper()}.JK"
    try:
        tk   = yf.Ticker(t)
        divs = tk.dividends
        info = tk.fast_info

        if divs.empty:
            return _empty_div()

        now_utc = pd.Timestamp.now(tz="UTC")
        one_yr  = now_utc - pd.DateOffset(years=1)
        recent  = divs[divs.index >= one_yr]
        annual  = float(recent.sum())

        price = getattr(info, "last_price", None) or 0
        yld   = (annual / price * 100) if price > 0 else 0

        last_ex = divs.index[-1]
        freq    = len(recent)

        # Estimate next ex-date from average interval
        avg_interval = 365 / freq if freq > 0 else 365
        next_ex      = last_ex + pd.DateOffset(days=avg_interval)
        days_to_next = int((next_ex - now_utc).days)

        # Override with yfinance calendar if available
        try:
            cal = tk.calendar
            if cal is not None and "Ex-Dividend Date" in cal:
                ex_raw = cal["Ex-Dividend Date"]
                if ex_raw:
                    ex_cal = pd.Timestamp(ex_raw, tz="UTC")
                    if ex_cal > now_utc:
                        next_ex      = ex_cal
                        days_to_next = int((ex_cal - now_utc).days)
        except Exception:
            pass

        return {
            "annual_div":         annual,
            "div_yield":          yld,
            "last_ex":            last_ex.date(),
            "next_ex":            next_ex.date() if next_ex is not None else None,
            "days_to_next":       days_to_next,
            "freq":               freq,
            "divs":               divs,
            "last_div_per_share": float(divs.iloc[-1]),
        }
    except Exception:
        return _empty_div()

def _empty_div():
    return {
        "annual_div": 0, "div_yield": 0, "last_ex": None,
        "next_ex": None, "days_to_next": None, "freq": 0,
        "divs": pd.Series(dtype=float), "last_div_per_share": 0,
    }

# ─────────────────────────────────────────────
# SIGNAL SCORING
# ─────────────────────────────────────────────
def score_stock(price_series, macro_df):
    s  = price_series.copy()
    ta = pd.DataFrame(index=s.index)
    ta["price"]  = s
    ta["sma20"]  = sma(s, 20)
    ta["sma50"]  = sma(s, 50)
    ta["sma200"] = sma(s, 200)
    ta["rsi"]    = rsi(s)
    ta["macd"], ta["macd_sig"] = macd(s)
    ta["bb_lo"], ta["bb_mid"], ta["bb_hi"] = bollinger(s)
    for col in ["usd_idr", "ihsg", "vix"]:
        if col in macro_df.columns:
            ta[col] = macro_df[col].reindex(ta.index, method="ffill")
    ta.dropna(inplace=True)
    if len(ta) < 10:
        return None, {}, ta

    last  = ta.iloc[-1]
    price = last["price"]
    sc    = {}

    sc["SMA20 > SMA50"]  = 8 if last["sma20"] > last["sma50"]  else 2
    sc["SMA50 > SMA200"] = 8 if last["sma50"] > last["sma200"] else 2
    sc["Price > SMA200"] = 8 if price > last["sma200"]          else 2

    rv = last["rsi"]
    if rv < 30:   sc["RSI oversold"]   = 9
    elif rv > 70: sc["RSI overbought"] = 2
    else:         sc["RSI neutral"]    = 5

    sc["MACD cross"] = 7 if last["macd"] > last["macd_sig"] else 3

    if price < last["bb_lo"]:   sc["BB below low"]  = 9
    elif price > last["bb_hi"]: sc["BB above high"] = 2
    else:                        sc["BB mid"]        = 5

    if "usd_idr" in ta.columns:
        sc["IDR stable"] = 7 if last["usd_idr"] < 16500 else 3
    if "ihsg" in ta.columns:
        ihsg_1m = ta["ihsg"].iloc[-22] if len(ta) > 22 else ta["ihsg"].iloc[0]
        sc["IHSG rising"] = 7 if last["ihsg"] > ihsg_1m else 3
    if "vix" in ta.columns:
        sc["VIX low"] = 7 if last["vix"] < 20 else 3

    return np.mean(list(sc.values())), sc, ta

def signal_label(score):
    if score is None:  return "❓ N/A"
    if score >= 6.5:   return "🟢 BUY"
    elif score <= 4.0: return "🔴 SELL"
    else:              return "🟡 HOLD"

# ─────────────────────────────────────────────
# PRICE TARGETS
# ─────────────────────────────────────────────
def price_targets(price_series):
    s         = price_series.dropna()
    ret       = s.pct_change().dropna()
    ret_clean = ret[-126:].clip(-0.05, 0.05)   # winsorize: strip gap/halt outliers
    mu        = ret_clean.mean()
    vol       = ret_clean.std()
    p         = s.iloc[-1]

    t3  = (max(p*(1+mu-vol)**63,  p*0.5), p*(1+mu)**63,  p*(1+mu+vol)**63)
    t12 = (max(p*(1+mu-vol)**252, p*0.3), p*(1+mu)**252, p*(1+mu+vol)**252)

    monthly = s.resample("ME").last()
    X       = np.arange(len(monthly)).reshape(-1, 1)
    poly    = PolynomialFeatures(2)
    reg     = LinearRegression().fit(poly.fit_transform(X), monthly.values)
    m2030   = (datetime(2030, 12, 31) - monthly.index[-1]).days // 30
    fX      = np.arange(len(monthly), len(monthly) + m2030).reshape(-1, 1)
    fy      = reg.predict(poly.transform(fX))
    t2030_b = max(fy[-1], p)
    ann_v   = ret_clean.std() * np.sqrt(252)
    yrs     = (datetime(2030, 12, 31) - datetime.now()).days / 365
    t2030   = (t2030_b*(1-ann_v*0.3)**yrs, t2030_b, t2030_b*(1+ann_v*0.6)**yrs)
    return t3, t12, t2030, monthly, fX, fy, poly

# ─────────────────────────────────────────────
# CHART — 4 panels (like gold/silver)
# ─────────────────────────────────────────────
def plot_stock(code, ta, price_series, div_info, t3, t12, t2030, monthly, fX, fy, poly, score, halal_ok=None, debt_ratio=None):
    price        = ta.iloc[-1]["price"]
    future_dates = pd.date_range(monthly.index[-1], periods=len(fX), freq="ME")

    if halal_ok is True:
        halal_label = f"✅ HALAL  Debt {debt_ratio:.0%}"
        title_color = "#69f0ae"
    elif halal_ok is False:
        halal_label = "❌ NOT HALAL"
        title_color = "#ef5350"
    else:
        halal_label = ""
        title_color = "#4fc3f7"

    div_label = f"  |  Div: {div_info['div_yield']:.1f}%/yr" if div_info["div_yield"] > 0 else ""

    fig, axes = plt.subplots(4, 1, figsize=(14, 16), facecolor="#0f0f0f")
    fig.suptitle(
        f"IDX — {code}.JK  {halal_label}{div_label}  |  Signal: {signal_label(score)}",
        color=title_color, fontsize=13, fontweight="bold"
    )
    ax1, ax2, ax3, ax4 = axes
    for ax in axes:
        ax.set_facecolor("#1a1a1a")
        ax.tick_params(colors="#aaaaaa")
        ax.spines[:].set_color("#333333")

    recent = ta[-500:]

    # Panel 1: Price + BB + SMA + 2030 + div markers
    ax1.plot(recent.index, recent["price"],  color="#4fc3f7", lw=1.5, label=code)
    ax1.plot(recent.index, recent["sma50"],  color="#ffd54f", lw=1,   label="SMA50",  alpha=0.8)
    ax1.plot(recent.index, recent["sma200"], color="#ef5350", lw=1,   label="SMA200", alpha=0.8)
    ax1.fill_between(recent.index, recent["bb_lo"], recent["bb_hi"], alpha=0.08, color="#4fc3f7")
    ax1.plot(future_dates, fy, color="#69f0ae", lw=1.5, linestyle="--", label="2030 base")
    ax1.fill_between(
        [ta.index[-1], future_dates[-1]],
        [price, t2030[0]], [price, t2030[2]],
        alpha=0.12, color="#69f0ae"
    )
    divs = div_info.get("divs", pd.Series(dtype=float))
    if not divs.empty:
        for dd in [d for d in divs.index if d >= recent.index[0]]:
            ax1.axvline(dd, color="#ffd54f", lw=0.6, alpha=0.6)
    if div_info.get("next_ex"):
        nex = pd.Timestamp(str(div_info["next_ex"]), tz="UTC")
        ax1.axvline(nex, color="#ff9800", lw=1.4, linestyle="--",
                    label=f"Next ex-div: {div_info['next_ex']} ({div_info['days_to_next']}d)")

    ax1.set_ylabel("Price (IDR)", color="#aaaaaa")
    ax1.legend(loc="upper left", facecolor="#1a1a1a", labelcolor="#cccccc", fontsize=8)
    ax1.set_title("Price + Bollinger + Div ex-dates (yellow▏orange=next) + 2030", color="#cccccc", fontsize=9)

    # Panel 2: RSI
    ax2.plot(recent.index, recent["rsi"], color="#ce93d8", lw=1.2)
    ax2.axhline(70, color="#ef5350", lw=0.8, linestyle="--", label="Overbought 70")
    ax2.axhline(30, color="#69f0ae", lw=0.8, linestyle="--", label="Oversold 30")
    ax2.axhline(50, color="#555555", lw=0.5)
    ax2.fill_between(recent.index, recent["rsi"], 50, where=recent["rsi"] > 50, alpha=0.2, color="#ef5350")
    ax2.fill_between(recent.index, recent["rsi"], 50, where=recent["rsi"] < 50, alpha=0.2, color="#69f0ae")
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("RSI", color="#aaaaaa")
    ax2.legend(loc="upper left", facecolor="#1a1a1a", labelcolor="#cccccc", fontsize=8)
    ax2.set_title("RSI (14)", color="#cccccc", fontsize=9)

    # Panel 3: MACD
    hist = recent["macd"] - recent["macd_sig"]
    ax3.plot(recent.index, recent["macd"],     color="#4fc3f7", lw=1.2, label="MACD")
    ax3.plot(recent.index, recent["macd_sig"], color="#ef5350", lw=1.0, label="Signal")
    ax3.bar(recent.index, hist,
            color=["#69f0ae" if v >= 0 else "#ef5350" for v in hist], alpha=0.5, width=1)
    ax3.axhline(0, color="#555555", lw=0.5)
    ax3.set_ylabel("MACD", color="#aaaaaa")
    ax3.legend(loc="upper left", facecolor="#1a1a1a", labelcolor="#cccccc", fontsize=8)
    ax3.set_title("MACD (12, 26, 9)", color="#cccccc", fontsize=9)

    # Panel 4: Dividend history
    if not divs.empty:
        recent_divs = divs[-20:]
        bar_colors  = ["#ff9800" if i == len(recent_divs) - 1 else "#ffd54f"
                       for i in range(len(recent_divs))]
        ax4.bar(range(len(recent_divs)), recent_divs.values, color=bar_colors, alpha=0.8)
        ax4.set_xticks(range(len(recent_divs)))
        ax4.set_xticklabels(
            [str(d.date()) for d in recent_divs.index],
            rotation=45, ha="right", fontsize=7, color="#aaaaaa"
        )
        ax4.set_ylabel("Div/Share (IDR)", color="#aaaaaa")
        yld_str = f"{div_info['div_yield']:.1f}% yield/yr  |  {div_info['freq']}x/yr"
        if div_info.get("next_ex"):
            yld_str += f"  |  Next ex: {div_info['next_ex']} ({div_info['days_to_next']} days)"
        ax4.set_title(f"Dividend History  —  {yld_str}", color="#cccccc", fontsize=9)
    else:
        ax4.text(0.5, 0.5, "No dividend data", transform=ax4.transAxes,
                 ha="center", va="center", color="#888888", fontsize=12)
        ax4.set_title("Dividend History", color="#cccccc", fontsize=9)

    for ax in (ax1, ax2, ax3):
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    plt.tight_layout()
    fname = f"output/{code}_prediction.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
    print(f"📈  Chart saved → {fname}")
    plt.show()

# ─────────────────────────────────────────────
# SINGLE STOCK
# ─────────────────────────────────────────────
def run_single(ticker_code):
    code = ticker_code.upper().replace(".JK", "")
    print(f"\n📡  Fetching {code}.JK + macro + dividend + balance sheet...")
    macro_df  = fetch_macro()
    price_s, full = fetch_price(code)
    if price_s.empty:
        print(f"❌  No data for {full}"); return

    halal_ok, debt_ratio, halal_desc = screen_halal(code)
    div_info  = get_dividend_info(code)
    score, scores, ta = score_stock(price_s, macro_df)
    t3, t12, t2030, monthly, fX, fy, poly = price_targets(price_s)
    price = price_s.iloc[-1]

    print("=" * 68)
    print(f"   IDX STOCK REPORT — {full}")
    print(f"   Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 68)
    print(f"\n💰  Price:        Rp {price:>12,.0f}")
    print(f"📊  Signal Score: {score:.1f} / 10")
    print(f"🎯  Signal:       {signal_label(score)}")
    print(f"{'✅' if halal_ok else '❌'}  Halal:        {'HALAL' if halal_ok else 'NOT HALAL'} — {halal_desc}")

    if div_info["div_yield"] > 0:
        print(f"\n💵  Dividend:")
        print(f"    Annual:       Rp {div_info['annual_div']:,.2f}/share  ({div_info['div_yield']:.2f}%)")
        print(f"    Frequency:    {div_info['freq']}x per year")
        print(f"    Last ex-date: {div_info['last_ex']}")
        if div_info.get("next_ex"):
            print(f"    Next ex-date: {div_info['next_ex']}  ({div_info['days_to_next']} days)")
        print(f"    Last payout:  Rp {div_info['last_div_per_share']:,.2f}/share")
        lots_example = 10
        thp = div_info["last_div_per_share"] * lots_example * 100 * 0.90
        print(f"    THP example:  Rp {div_info['last_div_per_share']:,.0f} × {lots_example} lots × 100 − 10% PPh = Rp {thp:,.0f}")
    else:
        print(f"\n💵  Dividend:     None / no data")

    print("\n── Technical Breakdown ──────────────────────────────────")
    for k, v in scores.items():
        bar  = "█" * int(v) + "░" * (10 - int(v))
        mood = "bullish" if v >= 6 else ("bearish" if v <= 4 else "neutral")
        print(f"  {k:<35} {bar}  {mood}")

    print("\n── Price Targets (IDR) ──────────────────────────────────")
    print(f"  3-Month  │ Bear: {t3[0]:>12,.0f}  Base: {t3[1]:>12,.0f}  Bull: {t3[2]:>12,.0f}")
    print(f"  12-Month │ Bear: {t12[0]:>12,.0f}  Base: {t12[1]:>12,.0f}  Bull: {t12[2]:>12,.0f}")
    print(f"  2030     │ Bear: {t2030[0]:>12,.0f}  Base: {t2030[1]:>12,.0f}  Bull: {t2030[2]:>12,.0f}")
    print("\n⚠️   DISCLAIMER: Not financial advice.")
    print("=" * 68)

    plot_stock(code, ta, price_s, div_info, t3, t12, t2030, monthly, fX, fy, poly, score, halal_ok, debt_ratio)

# ─────────────────────────────────────────────
# SCANNER CORE
# ─────────────────────────────────────────────
def scan_universe(halal_only=False):
    universe = list(dict.fromkeys(IDX_UNIVERSE))
    print(f"\n📡  Scanning {len(universe)} IDX stocks{'  (halal filter ON)' if halal_only else ''}...")
    macro_df = fetch_macro()
    rows     = []

    for code in universe:
        try:
            price_s, ticker = fetch_price(code)
            if price_s.empty or len(price_s) < 60:
                continue
            price = price_s.iloc[-1]

            halal_ok, debt_ratio, halal_desc = screen_halal(code)
            if halal_only and not halal_ok:
                continue

            score, _, _ = score_stock(price_s, macro_df)
            t3, t12, t2030, *_ = price_targets(price_s)
            div_info = get_dividend_info(code)

            rows.append({
                "code":         code,
                "ticker":       ticker,
                "price":        price,
                "score":        score or 0,
                "signal":       signal_label(score),
                "halal_ok":     halal_ok,
                "debt_ratio":   debt_ratio,
                "halal_desc":   halal_desc,
                "div_yield":    div_info["div_yield"],
                "annual_div":   div_info["annual_div"],
                "last_div":     div_info["last_div_per_share"],
                "last_ex":      div_info["last_ex"],
                "next_ex":      div_info["next_ex"],
                "days_to_next": div_info["days_to_next"],
                "freq":         div_info["freq"],
                "upside_3m":    (t3[1] / price - 1) * 100,
                "upside_1y":    (t12[1] / price - 1) * 100,
            })

            h = "✅" if halal_ok else "❌"
            d = f"  💵{div_info['div_yield']:.1f}%" if div_info["div_yield"] > 0 else ""
            n = (f"  ⏰ex:{div_info['days_to_next']}d"
                 if div_info["days_to_next"] is not None and 0 < div_info["days_to_next"] <= 30
                 else "")
            print(f"  {signal_label(score)} {h} {ticker:<12} {(score or 0):.1f}  Rp {price:>10,.0f}{d}{n}")
        except Exception as e:
            print(f"  ⚠️  {code}: {e}")

    return pd.DataFrame(rows)

# ─────────────────────────────────────────────
# TABLE 1 — Best buy now
# ─────────────────────────────────────────────
def print_best_buy(df, label):
    top = df.sort_values("score", ascending=False).head(10).reset_index(drop=True)
    print("\n" + "═" * 88)
    print(f"   📈  {label} — BEST BUY NOW  (by signal score)")
    print(f"   Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("═" * 88)
    print(f"{'#':<3} {'Ticker':<11} {'H':<2} {'Signal':<10} {'Score':>5} {'Price':>12} {'Div%':>6} {'3m%':>6} {'1y%':>6}")
    print("─" * 88)
    for i, r in enumerate(top.itertuples(), 1):
        h = "✅" if r.halal_ok else "❌"
        print(
            f"{i:<3} {r.ticker:<11} {h} {r.signal:<10} "
            f"{r.score:>5.1f} Rp {r.price:>9,.0f} {r.div_yield:>5.1f}% {r.upside_3m:>+5.1f}% {r.upside_1y:>+5.1f}%"
        )
    print("═" * 88)
    return top

# ─────────────────────────────────────────────
# TABLE 2 — Best dividend ≤30 days, payout yield ≥10%
# ─────────────────────────────────────────────
MIN_PAYOUT_YIELD = 0.10   # div_per_share / price_per_share >= 10% per payout

def print_best_dividend(df, label):
    """
    Filter: ex-date within 30 days AND (last_div / price) >= 10%.
    Ranked by payout yield descending. Top 20.
    """
    # per-payout yield = last dividend / current price (not annualised)
    df = df.copy()
    df["payout_yield"] = df.apply(
        lambda r: (r["last_div"] / r["price"]) if r["price"] > 0 else 0, axis=1
    )
    df["div_per_lot"]   = df["last_div"] * 100
    df["price_per_lot"] = df["price"]   * 100

    mask = (
        (df["last_div"] > 0) &
        (df["days_to_next"].notna()) &
        (df["days_to_next"] > 0) &
        (df["days_to_next"] <= 30) &
        (df["payout_yield"] >= MIN_PAYOUT_YIELD)
    )
    div_df = df[mask].sort_values("payout_yield", ascending=False).head(20).reset_index(drop=True)

    print("\n" + "═" * 115)
    print(f"   💵  {label} — UPCOMING EX-DIVIDEND ≤30 DAYS  &  PAYOUT ≥10% per lot  (top 20, biggest THP first)")
    print(f"   Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("═" * 115)

    if div_df.empty:
        print("   ⚠️  No stocks passed the filter (ex≤30d AND payout≥10%).")
        print("        Relaxing to: ex≤30d only, any yield — top 20:\n")
        fallback = df[
            (df["last_div"] > 0) &
            (df["days_to_next"].notna()) &
            (df["days_to_next"] > 0) &
            (df["days_to_next"] <= 30)
        ].sort_values("payout_yield", ascending=False).head(20).reset_index(drop=True)
        div_df = fallback

    print(f"{'#':<3} {'Ticker':<11} {'H':<2} {'Signal':<10} "
          f"{'Payout%':>8} {'Div/lot':>12} {'Price/lot':>12} {'THP@10lot':>12} {'Next Ex':>12} {'Days':>5}")
    print("─" * 115)
    for i, r in enumerate(div_df.itertuples(), 1):
        h       = "✅" if r.halal_ok else "❌"
        next_ex = str(r.next_ex) if r.next_ex else "est."
        days    = str(int(r.days_to_next)) if r.days_to_next is not None else "?"
        thp     = r.div_per_lot * 10 * 0.90   # 10 lots, after 10% PPh
        print(
            f"{i:<3} {r.ticker:<11} {h} {r.signal:<10} "
            f"{r.payout_yield:>7.1%} Rp {r.div_per_lot:>9,.0f} Rp {r.price_per_lot:>9,.0f} "
            f"Rp {thp:>9,.0f} {next_ex:>12} {days:>5}d"
        )

    print("─" * 115)
    print("  THP = Div/lot × lots × (1 − 10% PPh)   |   example above = 10 lots")
    print("  Payout% = dividend per share ÷ price per share  (single payout, not annualised)")
    print("═" * 115)
    print("\n⚠️   DISCLAIMER: Not financial advice.")
    return div_df

# ─────────────────────────────────────────────
# SCANNER CHART
# ─────────────────────────────────────────────
def plot_scanner(buy_df, div_df, title):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), facecolor="#0f0f0f")
    fig.suptitle(title, color="#4fc3f7", fontsize=14, fontweight="bold")
    for ax in (ax1, ax2):
        ax.set_facecolor("#1a1a1a"); ax.tick_params(colors="#aaaaaa"); ax.spines[:].set_color("#333333")

    # Left: best buy score
    bd = buy_df.sort_values("score", ascending=True)
    bar_c = ["#69f0ae" if s >= 6.5 else ("#ef5350" if s <= 4.0 else "#ffd54f") for s in bd["score"]]
    bars  = ax1.barh(bd["ticker"], bd["score"], color=bar_c, edgecolor="#333")
    ax1.axvline(6.5, color="#69f0ae", lw=1, linestyle="--", label="BUY ≥6.5")
    ax1.axvline(4.0, color="#ef5350", lw=1, linestyle="--", label="SELL ≤4.0")
    ax1.set_xlim(0, 10); ax1.set_xlabel("Signal Score", color="#aaaaaa")
    ax1.legend(facecolor="#1a1a1a", labelcolor="#cccccc", fontsize=8)
    ax1.set_title("📈 Best Buy Now — Signal Score", color="#cccccc", fontsize=11)
    for bar, row in zip(bars, bd.itertuples()):
        ax1.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                 f"{row.score:.1f}  {'✅' if row.halal_ok else '❌'}",
                 va="center", color="#ccc", fontsize=8)

    # Right: payout yield % per lot (color = days urgency)
    if not div_df.empty:
        _dd = div_df.copy()
        if "payout_yield" not in _dd.columns:
            _dd["payout_yield"] = _dd.apply(
                lambda r: (r["last_div"] / r["price"]) if r["price"] > 0 else 0, axis=1
            )
        dd      = _dd.sort_values("payout_yield", ascending=True)
        day_c   = ["#ff5722" if (d or 999) <= 7 else ("#ff9800" if (d or 999) <= 14 else "#ffd54f")
                   for d in dd["days_to_next"]]
        bars2   = ax2.barh(dd["ticker"], dd["payout_yield"] * 100, color=day_c, edgecolor="#333")
        ax2.axvline(10, color="#69f0ae", lw=1, linestyle="--", label="≥10% threshold")
        ax2.set_xlabel("Payout Yield % per lot", color="#aaaaaa")
        ax2.legend(facecolor="#1a1a1a", labelcolor="#cccccc", fontsize=8)
        ax2.set_title("💵 Best Dividend ≤30 days  (🔴≤7d 🟠≤14d 🟡≤30d)", color="#cccccc", fontsize=11)
        for bar, row in zip(bars2, dd.itertuples()):
            days = f"{int(row.days_to_next)}d" if row.days_to_next is not None else "?"
            ax2.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                     f"{row.payout_yield:.1%}  ex:{days}  {'✅' if row.halal_ok else '❌'}",
                     va="center", color="#ccc", fontsize=8)
    else:
        ax2.text(0.5, 0.5, "No upcoming\ndividend ≤30 days", transform=ax2.transAxes,
                 ha="center", va="center", color="#888888", fontsize=13)
        ax2.set_title("💵 Upcoming Dividend", color="#cccccc", fontsize=11)

    plt.tight_layout()
    safe  = title.replace(" ", "_")[:25]
    fname = f"output/{safe}_scanner.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
    print(f"\n📈  Chart saved → {fname}")
    plt.show()

# ─────────────────────────────────────────────
# MODES
# ─────────────────────────────────────────────
def run_top10():
    df     = scan_universe(halal_only=False)
    buy_df = print_best_buy(df, "IDX TOP10")
    div_df = print_best_dividend(df, "IDX TOP10")
    plot_scanner(buy_df, div_df, "IDX TOP10 — Buy Now vs Dividend")

def run_halal():
    df     = scan_universe(halal_only=True)
    buy_df = print_best_buy(df, "IDX HALAL")
    div_df = print_best_dividend(df, "IDX HALAL")
    plot_scanner(buy_df, div_df, "IDX HALAL — Buy Now vs Dividend")

def run_dividen():
    """Top 20 halal stocks — ex-date ≤30 days AND payout per lot ≥10% of price per lot."""
    df     = scan_universe(halal_only=True)
    div_df = print_best_dividend(df, "IDX DIVIDEN")
    if div_df.empty:
        return

    # payout_yield col may not be in div_df if it came from print_best_dividend's fallback
    if "payout_yield" not in div_df.columns:
        div_df["payout_yield"] = div_df.apply(
            lambda r: (r["last_div"] / r["price"]) if r["price"] > 0 else 0, axis=1
        )
    if "div_per_lot" not in div_df.columns:
        div_df["div_per_lot"] = div_df["last_div"] * 100

    dd    = div_df.sort_values("payout_yield", ascending=True)
    h_fig = max(6, len(dd) * 0.45)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, h_fig), facecolor="#0f0f0f")
    fig.suptitle(
        "💵 IDX DIVIDEN — Ex-Dividend ≤30 days  &  Payout ≥10%/lot  (top 20)",
        color="#ffd54f", fontsize=13, fontweight="bold"
    )
    for ax in (ax1, ax2):
        ax.set_facecolor("#1a1a1a"); ax.tick_params(colors="#aaaaaa"); ax.spines[:].set_color("#333333")

    # Left: payout yield %
    day_c = ["#ff5722" if (d or 999) <= 7 else ("#ff9800" if (d or 999) <= 14 else "#ffd54f")
              for d in dd["days_to_next"]]
    bars1 = ax1.barh(dd["ticker"], dd["payout_yield"] * 100, color=day_c, edgecolor="#333")
    ax1.axvline(10, color="#69f0ae", lw=1, linestyle="--", label="10% threshold")
    ax1.set_xlabel("Payout Yield % (single payout)", color="#aaaaaa")
    ax1.legend(facecolor="#1a1a1a", labelcolor="#cccccc", fontsize=8)
    ax1.set_title("Payout %  (🔴≤7d 🟠≤14d 🟡≤30d)", color="#cccccc", fontsize=10)
    for bar, row in zip(bars1, dd.itertuples()):
        days = f"{int(row.days_to_next)}d" if row.days_to_next is not None else "?"
        ax1.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                 f"{row.payout_yield:.1%}  ex:{days}  {'✅' if row.halal_ok else '❌'}",
                 va="center", color="#ccc", fontsize=8)

    # Right: THP for 10 lots after PPh
    dd["thp_10lots"] = dd["div_per_lot"] * 10 * 0.90
    bars2 = ax2.barh(dd["ticker"], dd["thp_10lots"], color=day_c, edgecolor="#333")
    ax2.set_xlabel("THP @ 10 lots, after 10% PPh  (IDR)", color="#aaaaaa")
    ax2.set_title("THP Estimate — 10 Lots", color="#cccccc", fontsize=10)
    for bar, row in zip(bars2, dd.itertuples()):
        ax2.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height()/2,
                 f"Rp {row.thp_10lots:,.0f}",
                 va="center", color="#ccc", fontsize=8)

    plt.tight_layout()
    fname = "output/DIVIDEN_scanner.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
    print(f"📈  Chart saved → {fname}")
    plt.show()

# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    arg = sys.argv[1].strip().upper() if len(sys.argv) > 1 else None
    if arg is None:
        print("Usage: python stock_ID_predict.py [TOP10 | HALAL | TICKER]")
        sys.exit(0)
    if arg == "TOP10":   run_top10()
    elif arg == "HALAL": run_halal()
    else:                run_single(arg)
