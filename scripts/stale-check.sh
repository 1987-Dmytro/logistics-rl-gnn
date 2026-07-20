#!/usr/bin/env bash
# brain-init generic (модуль M1): предупредить, если curated-блок hot.md старше последнего
# коммита (day-level) — признак «день закрыт без /close». Тихий при синхроне. Exit 0 всегда.
cd "$(dirname "$0")/.." || exit 0
HOT=knowledge/hot.md
[ -f "$HOT" ] || exit 0
LU=$(grep -m1 -oE '\*\*Last update:\*\* *[0-9]{4}-[0-9]{2}-[0-9]{2}' "$HOT" | grep -oE '[0-9]{4}-[0-9]{2}-[0-9]{2}')
GD=$(git log -1 --format=%cs 2>/dev/null)
if [ -n "$LU" ] && [ -n "$GD" ] && [ "$LU" \< "$GD" ]; then
  echo "⚠️ stale-check: hot.md curated ($LU) старше последнего коммита ($GD) — обнови curated или сделай /close"
fi
exit 0
