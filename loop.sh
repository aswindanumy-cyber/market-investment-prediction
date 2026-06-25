#!/bin/sh
while true; do
  for MODE in gold silver TOP10 HALAL DIVIDEN; do
    echo ""
    echo "════════════════════════════════════════"
    echo "  MODE: $MODE"
    echo "════════════════════════════════════════"
    python run.py "$MODE"
    echo ""
    echo "⏱  Switching to next mode in 10s..."
    sleep 10
  done
done
