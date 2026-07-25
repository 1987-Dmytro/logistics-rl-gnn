"""Phase 9 — кастомные сценарии: загрузчик, валидация, регрессия дефолта.

Загрузчик проверяем БЕЗ солвера (секунды): резолв имён, день недели → реальные окна, K/Q,
события, стражи (Σdemand ≤ K·Q, достижимость). Отдельно — регрессия: дефолтный вторник обязан
остаться байт-в-байт прежним (сценарные параметры добавлены как None-дефолты, а не как новый
путь исполнения). Тяжёлый end-to-end демо на friday_south — под skipif (веса/снапшот вне git).
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import pytest

yaml = pytest.importorskip("yaml")  # pyyaml — опц. депа [data]

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "scripts"))
_SNAP = _ROOT / "data" / "snapshots" / "augsburg_20260720"
_CKPT = _ROOT / "results" / "policy_pomo_congestion.pt"
_SM = _ROOT / "results" / "system_metrics.json"
_HAS_SNAP = (_SNAP / "meta.json").exists()
_SCEN = _ROOT / "scenarios"

from logistics_rl_gnn.config import instance as im  # noqa: E402
from logistics_rl_gnn.config import scenario as sc  # noqa: E402

_NAMES = {1: "Vita-Apotheke", 2: "Linden-Apotheke", 7: "Ludwigs-Apotheke", 11: "Bergius-Apotheke"}


def _write(tmp_path: Path, body: dict) -> Path:
    p = tmp_path / "scen.yaml"
    p.write_text(yaml.safe_dump(body, allow_unicode=True), encoding="utf-8")
    return p


# ---------- регрессия дефолта (сценарные параметры не должны его тронуть) ----------


@pytest.mark.skipif(not _HAS_SNAP, reason="нет снапшота (запрет №1)")
def test_default_instance_unchanged():
    """Дефолтный вторник байт-в-байт как до сценариев: n, спрос, окна (hash), K/Q/старт в meta."""
    inst = im.generate_instance(snapshot_dir=_SNAP, seed=0)
    assert len(inst.demand) == 63 and int(inst.demand.sum()) == 476
    assert hashlib.sha256(np.ascontiguousarray(inst.windows)).hexdigest()[:16] == "058f906f3db84b35"
    assert im.fleet_of(inst) == (im.FLEET_SIZE, float(im.VEHICLE_CAP))
    assert im.dispatch_offset_min(inst) == 0.0
    assert inst.start_datetime.hour == im.DISPATCH_OPEN_H and inst.horizon_s == 36000


# ---------- резолв имён (без снапшота) ----------


def test_resolve_stop_exact_substring_fuzzy_id():
    assert sc.resolve_stop("Vita-Apotheke", _NAMES) == 1  # точное
    assert sc.resolve_stop("vita-apotheke", _NAMES) == 1  # регистр
    assert sc.resolve_stop("Ludwigs", _NAMES) == 7  # уникальная подстрока
    assert sc.resolve_stop("Vita Apotheke", _NAMES) == 1  # fuzzy (дефис потерян)
    assert sc.resolve_stop(11, _NAMES) == 11 and sc.resolve_stop("#11", _NAMES) == 11


def test_event_clock_is_exact():
    """«08:55» — ровно 55.0 мин от старта: float-остаток печатал бы 08:54 (событие «раньше»)."""
    assert sc._at_min("08:55", open_h=8, horizon_min=600) == 55.0
    assert sc._at_min("09:20", open_h=8, horizon_min=600) == 80.0
    assert sc._at_min("09:20", open_h=8.5, horizon_min=570) == 50.0  # смена с 08:30


def test_resolve_stop_errors_are_actionable():
    with pytest.raises(ValueError, match="не найдена") as e:
        sc.resolve_stop("Аптека №1", _NAMES)
    assert "Похожие" in str(e.value)  # ошибка подсказывает кандидатов
    with pytest.raises(ValueError, match="неоднозначно"):
        sc.resolve_stop("Apotheke", _NAMES)  # подстрока у всех
    with pytest.raises(ValueError, match="нет в снапшоте"):
        sc.resolve_stop(999, _NAMES, known={1, 2, 7, 11})


# ---------- загрузка сценариев из репозитория ----------


@pytest.mark.skipif(not _HAS_SNAP, reason="нет снапшота (запрет №1)")
def test_friday_south_loads():
    """Подмножество + 2 машины + пятничные окна + один traffic-инцидент в координатах аптеки."""
    s = sc.load_scenario(_SCEN / "friday_south.yaml", snapshot_dir=_SNAP, seed=0)
    inst = s.instance
    assert s.weekday == 4 and s.fleet == (2, 80.0)
    assert len(inst.demand) - 1 == len(s.requested_stops) - len(s.dropped_stops)
    assert float(inst.demand.sum()) <= s.fleet[0] * s.fleet[1]  # страж Σdemand ≤ K·Q
    assert inst.meta["delivery_weekday"] == 4
    (ev,) = s.events
    assert ev.kind == "traffic" and ev.at_min == pytest.approx(50.0)  # 08:50 от 08:00
    vita = inst.snapshot_stops.index(sc.resolve_stop("Vita-Apotheke", s.names))
    assert tuple(ev.incident.center) == tuple(inst.coords[vita])
    # спрос-оверрайд применён поверх розыгрыша
    assert inst.demand[vita] == 14


@pytest.mark.skipif(not _HAS_SNAP, reason="нет снапшота (запрет №1)")
def test_monday_rush_loads_two_incidents():
    """`all` + два traffic-события в утренний пик; второе — закрытие (δ=∞), зоны разные."""
    s = sc.load_scenario(_SCEN / "monday_rush.yaml", snapshot_dir=_SNAP, seed=0)
    assert s.weekday == 0 and s.fleet == (im.FLEET_SIZE, float(im.VEHICLE_CAP))
    assert len(s.instance.demand) - 1 == 62 and s.requested_stops == []
    assert [e.kind for e in s.events] == ["traffic", "traffic"]
    assert [e.at_min for e in s.events] == [25.0, 50.0]
    assert not np.isinf(s.events[0].incident.magnitude)
    assert np.isinf(s.events[1].incident.magnitude)  # closure: true → δ=∞
    assert tuple(s.events[0].incident.center) != tuple(s.events[1].incident.center)


@pytest.mark.skipif(not _HAS_SNAP, reason="нет снапшота (запрет №1)")
def test_events_are_sorted_by_time(tmp_path):
    """Порядок в YAML произвольный — загрузчик обязан выстроить события по времени (триггер =
    последнее): без сортировки демо взяло бы триггером не то событие."""
    p = _write(tmp_path, {"name": "order", "pharmacies": ["Vita-Apotheke", "Linden-Apotheke"],
                          "fleet": {"K": 2}, "events": [
                              {"at": "10:00", "type": "breakdown"},
                              {"at": "08:30", "type": "traffic", "where": "Vita-Apotheke"}]})
    s = sc.load_scenario(p, snapshot_dir=_SNAP, seed=0)
    assert [e.kind for e in s.events] == ["traffic", "breakdown"]
    assert [e.at_min for e in s.events] == [30.0, 120.0]


# ---------- валидация ----------


@pytest.mark.skipif(not _HAS_SNAP, reason="нет снапшота (запрет №1)")
def test_infeasible_fleet_rejected(tmp_path):
    """Σdemand > K·Q — существующий страж инстанса, ошибка указывает на файл сценария."""
    p = _write(tmp_path, {"name": "tiny", "pharmacies": "all", "fleet": {"K": 1, "Q": 10}})
    with pytest.raises(ValueError, match="инфеасибл"):
        sc.load_scenario(p, snapshot_dir=_SNAP, seed=0)


@pytest.mark.skipif(not _HAS_SNAP, reason="нет снапшота (запрет №1)")
def test_dispatch_start_shifts_congestion_clock(tmp_path):
    """dispatch_start сдвигает и окна, И профиль c(dow,h): в 09:00 берётся c(9), не c(8)."""
    from logistics_rl_gnn.config import congestion as cg
    from logistics_rl_gnn.env.events import congestion_for

    p = _write(tmp_path, {"name": "late", "dispatch_start": "09:00", "weekday": 1,
                          "pharmacies": ["Vita-Apotheke", "Linden-Apotheke"],
                          "events": [{"at": "09:30", "type": "traffic", "where": "Vita-Apotheke"}]})
    s = sc.load_scenario(p, snapshot_dir=_SNAP, seed=0)
    inst = s.instance
    assert inst.start_datetime.hour == 9 and inst.horizon_s == 9 * 3600
    assert im.dispatch_offset_min(inst) == 60.0
    tr = congestion_for(inst, dow=1, offset_min=im.dispatch_offset_min(inst))
    t0 = inst.time_matrix[0, 1] / 60.0
    assert tr.time(0, 1, 0.0) == pytest.approx(t0 * cg.congestion(1, 9))
    assert tr.time(0, 1, 0.0) != pytest.approx(t0 * cg.congestion(1, 8))
    # событие: 30 мин от старта смены, но в АБСОЛЮТНОМ клоке инцидента — 90 (иначе зона «оживёт»
    # на час раньше, чем сказано в YAML)
    (ev,) = s.events
    assert ev.at_min == 30.0 and ev.incident.t_start_min == 90.0
    assert ev.incident.at_node(ev.incident.center, 90.0) != 0.0  # активен в свой момент
    assert ev.incident.at_node(ev.incident.center, 30.0) == 0.0  # и НЕ активен часом раньше


@pytest.mark.skipif(not _HAS_SNAP, reason="нет снапшота (запрет №1)")
def test_fleet_from_meta_reaches_env(tmp_path):
    """K/Q/день сценария доезжают до среды, а не теряются в def-time дефолтах VRPEnv (8×80)."""
    from logistics_rl_gnn.env.events import make_dynamic_env

    p = _write(tmp_path, {"name": "smallq", "weekday": "friday", "fleet": {"K": 3, "Q": 40},
                          "pharmacies": ["Vita-Apotheke", "Linden-Apotheke"]})
    inst = sc.load_scenario(p, snapshot_dir=_SNAP, seed=0).instance
    env = make_dynamic_env(inst, travel=None)
    assert (env.K, env.Q, env.dow) == (3, 40.0, 4)


@pytest.mark.skipif(not _HAS_SNAP, reason="нет снапшота (запрет №1)")
def test_event_validation(tmp_path):
    """Событие вне смены / у аптеки не из списка / неизвестного типа → внятная ошибка."""
    base = {"name": "x", "pharmacies": ["Vita-Apotheke", "Linden-Apotheke"], "fleet": {"K": 2}}
    with pytest.raises(ValueError, match="вне диспетч-смены"):
        sc.load_scenario(_write(tmp_path, {**base, "events": [
            {"at": "19:30", "type": "breakdown"}]}), snapshot_dir=_SNAP)
    with pytest.raises(ValueError, match="не входит в инстанс"):
        sc.load_scenario(_write(tmp_path, {**base, "events": [
            {"at": "09:00", "type": "traffic", "where": "Ludwigs-Apotheke"}]}), snapshot_dir=_SNAP)
    with pytest.raises(ValueError, match="неизвестен"):
        sc.load_scenario(_write(tmp_path, {**base, "events": [
            {"at": "09:00", "type": "meteorite"}]}), snapshot_dir=_SNAP)
    # `closure` не в params → тихо давал бы обычную пробку вместо закрытия
    with pytest.raises(ValueError, match="неизвестные поля"):
        sc.load_scenario(_write(tmp_path, {**base, "events": [
            {"at": "09:00", "type": "traffic", "where": "Vita-Apotheke", "closure": True}]}),
            snapshot_dir=_SNAP)
    with pytest.raises(ValueError, match="неизвестные params"):
        sc.load_scenario(_write(tmp_path, {**base, "events": [
            {"at": "09:00", "type": "urgent", "where": "Vita-Apotheke",
             "params": {"boxes": 5}}]}), snapshot_dir=_SNAP)


@pytest.mark.skipif(not _HAS_SNAP, reason="нет снапшота (запрет №1)")
def test_schema_typos_rejected(tmp_path):
    """Опечатка в поле/дне недели не проглатывается молча (тихий дефолт = ложный сценарий)."""
    with pytest.raises(ValueError, match="неизвестные поля"):
        sc.load_scenario(_write(tmp_path, {"name": "x", "pharmacys": "all"}), snapshot_dir=_SNAP)
    with pytest.raises(ValueError, match="не распознан"):
        sc.load_scenario(_write(tmp_path, {"name": "x", "weekday": "freitag"}), snapshot_dir=_SNAP)
    with pytest.raises(ValueError, match="вне списка pharmacies"):
        sc.load_scenario(_write(tmp_path, {"name": "x", "pharmacies": ["Vita-Apotheke"],
                                           "demand": {"Linden-Apotheke": 5}}), snapshot_dir=_SNAP)
    with pytest.raises(ValueError, match="только K и Q"):  # опечатка → тихий дефолтный флот
        sc.load_scenario(_write(tmp_path, {"name": "x", "fleet": {"K": 2, "Qty": 40}}),
                         snapshot_dir=_SNAP)
    with pytest.raises(ValueError, match=r"> Q="):  # стоп тяжелее машины — не увезти никогда
        sc.load_scenario(_write(tmp_path, {"name": "x", "pharmacies": ["Vita-Apotheke"],
                                           "fleet": {"K": 2, "Q": 40},
                                           "demand": {"Vita-Apotheke": 50}}), snapshot_dir=_SNAP)


# ---------- end-to-end демо на сценарии (тяжёлый) ----------


@pytest.mark.skipif(not (_CKPT.exists() and _SM.exists() and _HAS_SNAP),
                    reason="ckpt/snapshot/system_metrics вне git (запрет №1)")
def test_demo_on_friday_south_end_to_end(tmp_path):
    """Тот же пайплайн и рендер на сценарии: 5 артефактов, флот сценария соблюдён, всё обслужено,
    в листе — честное «durable-якоря нет» (запрет №4)."""
    demo = pytest.importorskip("demo")
    s = demo.run_demo(seed=0, event_kind="traffic", out_dir=str(tmp_path), open_maps=False,
                      scenario_path=str(_SCEN / "friday_south.yaml"))

    assert s["scenario"] == "friday_south" and s["used_model"] is True
    by_name = {Path(f).name: f for f in s["files"]}
    assert set(by_name) == {"1_morning_plan.html", "route_sheet.md", "2_incident_no_replan.html",
                            "3_incident_replan.html", "compare.html"}
    for f in s["files"]:
        assert Path(f).exists() and Path(f).stat().st_size > 0
    assert s["unserved"] == 0 and s["n_served"] + s["n_pending"] == 14
    # цена в шапке карты == числу в выводе демо (тот же страж, что у дефолтного прогона)
    banner = re.search(r'data-demo-price="([0-9.]+)"',
                       Path(by_name["1_morning_plan.html"]).read_text(encoding="utf-8"))
    assert banner and abs(float(banner.group(1)) - s["morning_cost"]) < 1e-3
    sheet = Path(by_name["route_sheet.md"]).read_text(encoding="utf-8")
    assert "K = 2 машин" in sheet and "urable-якоря нет" in sheet  # флот сценария + честная шапка
    assert "per_seed" not in sheet  # чужой якорь в кастомный сценарий не подмешан
    # вклад модели измерен обеими сторонами; модель может лишь добавить кандидатов в пул плана
    c = s["contribution"]
    assert c["plan_without"] is not None and c["plan_with"] is not None
    assert c["plan_with"] <= c["plan_without"] + 1e-9
    assert c["replan_without"] is not None  # отдельный портфель без RL, а не строка из таблицы
