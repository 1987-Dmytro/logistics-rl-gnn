"""Phase 8 — сводные финальные метрики «Было/Стало» (LinkedIn-кейс).

Числа ТОЛЬКО из durable-артефактов (provenance цепочкой), ничего не перегоняется:
  • greedy / OR-Tools — полный вектор из `baselines.json` ([[0002-baselines]]);
  • система (полир. портфель) — полный вектор из `system_metrics.json` (eval_system.py, cost-парити
    к [[0009]] 631.6€); латентность re-plan из `polish_summary.json` (dynamic-агрегаты 0009).
Пишет results/final_metrics.json + MD-таблицу (docs/final_metrics.md, в git). ПАРИТИ-СТРАЖ: cost
greedy/OR/системы обязаны сойтись с decision 0002/0009 (иначе артефакты разъехались — assert).

Запуск: python scripts/final_metrics.py   (после eval_system.py + при наличии durable json)
"""

from __future__ import annotations

import json
from pathlib import Path

_RES = Path("results")
_OUT_JSON = _RES / "final_metrics.json"
_OUT_MD = Path("docs/final_metrics.md")

# якоря парити (decision-числа) — артефакты обязаны сойтись
_ANCHOR = {"greedy": 825.38, "ortools": 611.14, "system": 631.62, "step3_portfolio": 766.14}
_TOL = 0.5


def _load(name: str) -> dict:
    p = _RES / name
    if not p.exists():
        raise FileNotFoundError(f"нет {p} — durable-артефакт отсутствует (см. decision)")
    return json.loads(p.read_text())


def _agg(bl: dict, method: str) -> dict:
    a = bl[method]["agg"]
    return {
        "cost_eur": -a["reward"]["mean"],
        "distance_km": a["distance_km"]["mean"],
        "time_min": a["time_min"]["mean"],
        "vehicles": a["vehicles_used"]["mean"],
        "on_time_pct": a["on_time_pct"]["mean"],
        "unserved": a["unserved"]["mean"],
    }


def build() -> dict:
    bl = _load("baselines.json")
    sysm = _load("system_metrics.json")
    polish = _load("polish_summary.json")

    greedy = _agg(bl, "greedy")
    ortools = _agg(bl, "ortools")
    sm = sysm["means"]
    system = {
        "cost_eur": sm["cost_eur"], "distance_km": sm["distance_km"], "time_min": sm["time_min"],
        "vehicles": sm["vehicles_used"], "on_time_pct": sm["on_time_pct"],
        "unserved": sm["unserved"],
    }
    # латентность re-plan (dynamic 0009): rl=система(портфель+polish), greedy, ortools
    lat = polish["dynamic"]["aggregates"]
    latency = {
        "system_ms": lat["rl"]["latency_ms_median"],
        "greedy_ms": lat["greedy"]["latency_ms_median"],
        "ortools_ms": lat["ortools"]["latency_ms_median"],
    }

    # ПАРИТИ-СТРАЖ: durable-числа сходятся с decision-якорями
    for name, val in (("greedy", greedy["cost_eur"]), ("ortools", ortools["cost_eur"]),
                      ("system", system["cost_eur"])):
        assert abs(val - _ANCHOR[name]) < _TOL, (
            f"ПАРИТИ FAIL: {name} cost {val:.2f} != anchor {_ANCHOR[name]} (артефакты разъехались)"
        )

    def pct(a, b):  # (a−b)/b — насколько a ниже/выше b
        return a / b - 1.0

    deltas = {
        "cost_vs_greedy": pct(system["cost_eur"], greedy["cost_eur"]),
        "cost_vs_ortools": pct(system["cost_eur"], ortools["cost_eur"]),
        "distance_vs_greedy": pct(system["distance_km"], greedy["distance_km"]),
        "time_vs_greedy": pct(system["time_min"], greedy["time_min"]),
        "vehicles_vs_greedy": pct(system["vehicles"], greedy["vehicles"]),
        "reaction_speedup_vs_ortools": latency["ortools_ms"] / latency["system_ms"],
    }
    return {
        "note": "Было/Стало: greedy vs полир. портфель на ИДЕНТИЧНЫХ инстансах (seeds 0–9, "
        "full-62, free-flow), единый scorer (№3). cost/пробег/время/машины — статика; "
        "латентность — re-plan-реакция (dynamic 0009). Числа из durable json (парити decision).",
        "rows": {"greedy": greedy, "ortools": ortools, "system": system},
        "latency_replan": latency,
        "deltas": deltas,
        "provenance": {
            "greedy_ortools": {"src": "baselines.json", "decision": "0002",
                               "seeds": bl["run"]["seeds"],
                               "ortools_version": bl["run"]["ortools_version"]},
            "system_full_vector": {"src": "system_metrics.json", "decision": "0009 (парити)",
                                   "ckpt_sha16": sysm["provenance"]["checkpoint"]["sha256_16"],
                                   "cost_anchor_0009": sysm["durable_cost_anchor_0009"]},
            "latency": {"src": "polish_summary.json", "decision": "0009 (dynamic 5×6 событий)"},
            "step3_portfolio_eur": _ANCHOR["step3_portfolio"],  # 0008 цепочка (до polish)
        },
    }


def _fmt_md(m: dict) -> str:
    d, lat = m["deltas"], m["latency_replan"]
    g, o, s = m["rows"]["greedy"], m["rows"]["ortools"], m["rows"]["system"]

    def mh(x):  # маш-ч из минут
        return x / 60.0

    lines = [
        "# Финальные метрики — «Было/Стало» (Аугсбург, seeds 0–9, full-62)",
        "",
        "> greedy (Было) vs полир. портфель (Стало) на ИДЕНТИЧНЫХ инстансах, единый scorer; "
        "OR-Tools — верхняя планка. Числа из durable-артефактов (parity к decision 0002/0009).",
        "",
        "| Метрика | greedy (Было) | OR-Tools | Система (Стало) | Δ vs greedy | Δ vs OR |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Издержки, € | {g['cost_eur']:.1f} | {o['cost_eur']:.1f} | "
        f"**{s['cost_eur']:.1f}** | **{d['cost_vs_greedy']:+.1%}** | {d['cost_vs_ortools']:+.1%} |",
        f"| Пробег, км (топл. прокси) | {g['distance_km']:.1f} | {o['distance_km']:.1f} | "
        f"**{s['distance_km']:.1f}** | **{d['distance_vs_greedy']:+.1%}** | — |",
        f"| Машино-часы в наряде* | {mh(g['time_min']):.1f} | {mh(o['time_min']):.1f} | "
        f"**{mh(s['time_min']):.1f}** | **{d['time_vs_greedy']:+.1%}** | — |",
        f"| Машин задействовано | {g['vehicles']:.1f} | {o['vehicles']:.1f} | "
        f"**{s['vehicles']:.1f}** | {d['vehicles_vs_greedy']:+.1%} | — |",
        f"| On-time, % | {g['on_time_pct']:.0f} | {o['on_time_pct']:.0f} | "
        f"{s['on_time_pct']:.0f} | — | — |",
        f"| Не обслужено | {g['unserved']:.1f} | {o['unserved']:.1f} | "
        f"{s['unserved']:.1f} | — | — |",
        f"| Латентность re-plan | {lat['greedy_ms']:.0f} мс | {lat['ortools_ms']:.0f} мс | "
        f"{lat['system_ms']:.0f} мс | — | **×{d['reaction_speedup_vs_ortools']:.1f} быстрее OR** |",
        "",
        f"\\*Машино-часы = travel + простой-ожидание окон + сервис. Выигрыш "
        f"**{d['time_vs_greedy']:+.1%}** — почти весь от **сокращения простоя** (окна): пробег "
        f"~flat ({d['distance_vs_greedy']:+.1%}), сервис идентичен (те же 62 аптеки). Это экономия "
        "ЧАСОВ НАРЯДА (труд), не километража.",
        "",
        f"**Итог:** издержки **{d['cost_vs_greedy']:+.1%}** и часы наряда "
        f"**{d['time_vs_greedy']:+.1%}** к greedy (планирование окон, не пробег — он ~flat), "
        f"в **{d['cost_vs_ortools']:+.1%}** к OR-Tools. Реакция на событие: нейро-старт ~15мс "
        f"(потолок скорости), деплой-система (portfolio+polish) {lat['system_ms']:.0f}мс = "
        f"**×{d['reaction_speedup_vs_ortools']:.1f}** к OR-Tools ПРИ том же качестве (+3.4%). "
        "Гарантия ≥ greedy по построению (0008).",
        "",
        "<sub>Провенанс: baselines.json (0002) · system_metrics.json (парити 0009 631.6€) · "
        "polish_summary.json (0009, латентность re-plan 5×6 событий; нейро-floor 14–19мс — "
        "ablation 0010, качество-инфериор). Статика — seeds 0–9 full-62. Вне git (№1).</sub>",
    ]
    return "\n".join(lines)


def main() -> None:
    m = build()
    _OUT_JSON.write_text(json.dumps(m, ensure_ascii=False, indent=2))
    md = _fmt_md(m)
    _OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    _OUT_MD.write_text(md + "\n")
    print(md)
    print(f"\n→ {_OUT_JSON}  +  {_OUT_MD}")


if __name__ == "__main__":
    main()
