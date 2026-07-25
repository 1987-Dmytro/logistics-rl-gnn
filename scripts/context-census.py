#!/usr/bin/env python3
"""Boot-budget census (brain-init generic, module M3): a rough estimate of the Tier-0 tax.

Sum of bytes//4 over: ~/.claude/CLAUDE.md · ~/CLAUDE.md · ./CLAUDE.md (+@import, 1 level) ·
MEMORY.md (cap 25KB — beyond that nothing loads) · knowledge/hot.md · .claude/rules/*.md WITHOUT
paths: (those load every session). One line of output; warns above 9K. Exits 0.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HOME = Path.home()
TARGET_KTOK = 9.0
MEMORY_CAP = 25 * 1024


def nbytes(p, cap=None):
    try:
        n = p.stat().st_size
        return min(n, cap) if cap else n
    except OSError:
        return 0


def main():
    total = 0
    srcs = 0
    claude_mds = [HOME / ".claude" / "CLAUDE.md", HOME / "CLAUDE.md", ROOT / "CLAUDE.md"]
    for p in claude_mds:
        b = nbytes(p)
        if b:
            total += b
            srcs += 1
            # @import, 1 level (inline expansion counts toward the budget)
            try:
                for m in re.finditer(
                    r"(?:^|\s)@([\w./~-]+\.md)", p.read_text(encoding="utf-8", errors="replace")
                ):
                    total += nbytes((p.parent / m.group(1)).resolve())
            except OSError:
                pass
    slug = re.sub(r"[^a-zA-Z0-9]", "-", str(ROOT))
    total += nbytes(HOME / ".claude" / "projects" / slug / "memory" / "MEMORY.md", cap=MEMORY_CAP)
    total += nbytes(ROOT / "knowledge" / "hot.md")
    for p in (ROOT / ".claude" / "rules").glob("*.md"):
        try:
            head = p.read_text(encoding="utf-8", errors="replace")[:300]
        except OSError:
            continue
        if "paths:" not in head:  # without paths: — always loaded
            total += nbytes(p)
    ktok = total / 4 / 1000
    warn = (
        f"  ⚠️ > {TARGET_KTOK}K target — de-bloat: "
        f"rules+paths: / shrink the MEMORY index / curate hot.md"
        if ktok > TARGET_KTOK
        else ""
    )
    print(f"brain-census: {ktok:.1f}Ktok boot tax{warn}")


if __name__ == "__main__":
    sys.exit(main())
