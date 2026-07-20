#!/usr/bin/env python3
"""Валидатор [[wikilinks]] (brain-init generic, модуль M4).

Скан: knowledge/**/*.md + native memory-dir проекта. Резолв: basename ноты (без .md) ИЛИ
frontmatter `name:`-slug (memory). Битые — в stdout. Exit 0 всегда (non-blocking).
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WIKILINK = re.compile(r"\[\[([^\]]+)\]\]")


def memory_dir():
    slug = re.sub(r"[^a-zA-Z0-9]", "-", str(ROOT))
    return Path.home() / ".claude" / "projects" / slug / "memory"


def targets():
    names = set()
    for base in (ROOT / "knowledge", memory_dir()):
        if not base.is_dir():
            continue
        for p in base.rglob("*.md"):
            names.add(p.stem)
            try:
                head = p.read_text(encoding="utf-8", errors="replace")[:400]
            except OSError:
                continue
            m = re.search(r"^name:\s*(.+)$", head, re.M)
            if m:
                names.add(m.group(1).strip())
    return names


def main():
    known = targets()
    broken = []
    for base in (ROOT / "knowledge", memory_dir()):
        if not base.is_dir():
            continue
        for p in base.rglob("*.md"):
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            # `[[пример]]` в код-спанах и <!-- комментах --> — упоминания, не линки
            text = re.sub(r"`[^`\n]*`|<!--.*?-->", "", text, flags=re.DOTALL)
            for m in WIKILINK.finditer(text):
                name = m.group(1).split("|")[0].split("#")[0].strip().removesuffix(".md")
                if name and name not in known:
                    broken.append(
                        (
                            str(
                                p.relative_to(ROOT.parent if base != memory_dir() else base.parent)
                            ),
                            name,
                        )
                    )
    if broken:
        print(f"check-wikilinks: {len(broken)} битых линков")
        for f, n in broken[:30]:
            print(f"  - [[{n}]] в {f}")
    else:
        print("check-wikilinks: OK, битых нет")


if __name__ == "__main__":
    sys.exit(main())
