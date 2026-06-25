"""
Single entry point for all predictors.

Usage:
  python run.py gold
  python run.py silver
  python run.py TOP10    → best buy now + best upcoming dividend (all stocks)
  python run.py HALAL    → same, halal-screened only
  python run.py DIVIDEN  → top dividend ≤30 days across all stocks
  python run.py BBCA     → single IDX stock
"""

import sys

if len(sys.argv) < 2:
    print("Usage: python run.py [gold | silver | TOP10 | HALAL | DIVIDEN | TICKER]")
    sys.exit(1)

mode = sys.argv[1].strip().upper()

if mode == "GOLD":
    import gold_predict

elif mode == "SILVER":
    import silver_predict

elif mode == "TOP10":
    import stock_ID_predict
    stock_ID_predict.run_top10()

elif mode == "HALAL":
    import stock_ID_predict
    stock_ID_predict.run_halal()

elif mode == "DIVIDEN":
    import stock_ID_predict
    stock_ID_predict.run_dividen()

else:
    import stock_ID_predict
    stock_ID_predict.run_single(mode)
