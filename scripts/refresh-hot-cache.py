#!/usr/bin/env python3
"""Regenerates the AUTO section of knowledge/hot.md (brain-init generic; a copy in scripts/).

Marker contract (verbatim, MUST NOT change — the curated block below END survives every refresh):
  <!-- AUTO-GEN START (refreshed by scripts/refresh-hot-cache.py) -->
  <!-- AUTO-GEN END (everything below preserved across refreshes) -->
Fail-safe: markers broken / not exactly one each → warn and leave the file untouched.
"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HOT = ROOT / "knowledge" / "hot.md"
START = "<!-- AUTO-GEN START (refreshed by scripts/refresh-hot-cache.py) -->"
END = "<!-- AUTO-GEN END (everything below preserved across refreshes) -->"


def sh(*args):
    try:
        return subprocess.run(args, capture_output=True, text=True, cwd=ROOT).stdout.strip()
    except Exception:
        return ""


def build_auto():
    out = [
        START,
        "",
        "# Hot Cache",
        "",
        f"**Auto-refreshed:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (every SessionStart)",
        f"**Branch:** `{sh('git', 'branch', '--show-current') or '—'}`",
        "",
    ]
    log = sh("git", "log", "--oneline", "-5")
    if log:
        out += ["## 🔀 Recent commits (top 5)", "", "```", log, "```", ""]
    goals = ROOT / "knowledge" / "goals" / "INDEX.md"  # feature-detect (module M2)
    if goals.is_file():
        rows = [
            ln
            for ln in goals.read_text(encoding="utf-8").splitlines()
            if ln.startswith("| ") and "---" not in ln and not ln.startswith("| Goal")
        ]
        if rows:
            out += ["## 🎯 Active goals", ""] + rows[:5] + [""]
    dec_dir = ROOT / "knowledge" / "decisions"
    decs = (
        sorted(dec_dir.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)[:3]
        if dec_dir.is_dir()
        else []
    )
    if decs:
        out += ["## 📋 Recent decisions", ""]
        for p in decs:
            head = next(
                (
                    ln.lstrip("# ").strip()
                    for ln in p.read_text(encoding="utf-8").splitlines()
                    if ln.startswith("# ")
                ),
                p.stem,
            )
            out.append(f"- `{p.name}` — {head}")
        out.append("")
    logs_dir = ROOT / "knowledge" / "daily_logs"
    logs = sorted(logs_dir.glob("2*.md"), reverse=True)[:3] if logs_dir.is_dir() else []
    if logs:
        out += ["## 📅 Recent daily logs", ""] + [f"- `{p.name}`" for p in logs] + [""]
    out.append(END)
    return "\n".join(out)


def main():
    if not HOT.is_file():
        print(f"refresh-hot-cache: {HOT} is missing — skipped (scaffold has not created it yet?)")
        return
    text = HOT.read_text(encoding="utf-8")
    if text.count(START) != 1 or text.count(END) != 1:
        print("⚠️ refresh-hot-cache: AUTO-GEN markers are broken — file NOT touched (fix them)")
        return
    curated = text.split(END, 1)[1]
    HOT.write_text(build_auto() + curated, encoding="utf-8")
    print(f"refresh-hot-cache: OK ({HOT.stat().st_size} bytes)")


if __name__ == "__main__":
    sys.exit(main())
