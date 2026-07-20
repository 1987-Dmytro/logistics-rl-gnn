#!/usr/bin/env python3
"""Stop-хук (brain-init generic): daily-log stub + регенерация knowledge/index.md + vault-state.

Никакого yaml — plain JSON. Идемпотентен, exit 0 всегда.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KN = ROOT / "knowledge"


def ensure_daily_log():
    logs = KN / "daily_logs"
    logs.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    p = logs / f"{today}.md"
    if not p.exists():
        tmpl = KN / "templates" / "daily-log.md"
        body = (
            tmpl.read_text(encoding="utf-8").replace("{{DATE}}", today)
            if tmpl.is_file()
            else f"---\ntype: daily_log\ndate: {today}\n---\n\n# {today}\n\n## Sessions\n"
        )
        p.write_text(body, encoding="utf-8")
    with p.open("a", encoding="utf-8") as f:
        f.write(f"- {datetime.now().strftime('%H:%M')}: session ended (auto)\n")
    return p


def regen_index():
    lines = [
        "# knowledge/ index",
        "",
        f"*Авто-генерируется Stop-хуком "
        f"({datetime.now().strftime('%Y-%m-%d %H:%M')}); руками не править.*",
        "",
    ]
    counts = {}
    for d in sorted(p for p in KN.iterdir() if p.is_dir() and not p.name.startswith(".")):
        files = sorted(d.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
        counts[d.name] = len(files)
        if not files:
            continue
        lines.append(f"## {d.name}/ ({len(files)})")
        for p in files[:15]:
            lines.append(f"- [[{p.stem}]]")
        if len(files) > 15:
            lines.append(f"- … ещё {len(files) - 15}")
        lines.append("")
    root_files = sorted(KN.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
    counts["_root"] = len(root_files)
    (KN / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return counts


def main():
    if not KN.is_dir():
        print("brain-session-end: knowledge/ отсутствует — пропуск")
        return
    p = ensure_daily_log()
    counts = regen_index()
    (KN / ".vault-state.json").write_text(
        json.dumps(
            {"last_update": datetime.now().isoformat(timespec="seconds"), "counts": counts},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"brain-session-end: OK (log={p.name}, index regen, {sum(counts.values())} files)")


if __name__ == "__main__":
    sys.exit(main())
