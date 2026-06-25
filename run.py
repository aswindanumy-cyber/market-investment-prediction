"""
Single entry point for all predictors.

Usage:
  python run.py gold
  python run.py silver
  python run.py BBCA
  python run.py TOP10
"""

import sys

if len(sys.argv) < 2:
    print("Usage: python run.py [gold | silver | STOCK_CODE | TOP10]")
    sys.exit(1)

mode = sys.argv[1].strip().upper()

if mode == "GOLD":
    import gold_predict
elif mode == "SILVER":
    import silver_predict
else:
    sys.argv[1] = mode
    import stock_ID_predict
    if mode == "TOP10":
        stock_ID_predict.run_top10()
    else:
        stock_ID_predict.run_single(mode)
