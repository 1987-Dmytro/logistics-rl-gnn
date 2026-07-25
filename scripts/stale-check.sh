#!/usr/bin/env bash
# brain-init generic (module M1): warn if the curated block of hot.md is older than the last
# commit (day-level) — a sign of "day closed without /close". Silent when in sync. Always exits 0.
cd "$(dirname "$0")/.." || exit 0
HOT=knowledge/hot.md
[ -f "$HOT" ] || exit 0
LU=$(grep -m1 -oE '\*\*Last update:\*\* *[0-9]{4}-[0-9]{2}-[0-9]{2}' "$HOT" | grep -oE '[0-9]{4}-[0-9]{2}-[0-9]{2}')
GD=$(git log -1 --format=%cs 2>/dev/null)
if [ -n "$LU" ] && [ -n "$GD" ] && [ "$LU" \< "$GD" ]; then
  echo "⚠️ stale-check: hot.md curated ($LU) is older than the last commit ($GD) — run /close"
fi
exit 0
