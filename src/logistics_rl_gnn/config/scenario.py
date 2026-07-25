"""Кастомный сценарий дня из YAML → Instance + поток событий (Phase 9, приёмка).

Сценарий описывает ДЕНЬ ДИСПЕТЧЕРА на тех же реальных данных снапшота: какой день недели
(окна аптек берутся из настоящих `opening_hours` этого дня + профиль congestion c(dow,h)),
какие аптеки (по именам / stop-id / `all`), какой флот K/Q, какие события и когда.

Ничего не «моделируем заново»: инстанс строит `generate_instance` (те же реальные окна и тот
же страж Σdemand ≤ K·Q + достижимость в T_max), события — те же классы `env.events`, что
крутит харнесс 0004. Сценарий лишь ПАРАМЕТРИЗУЕТ их вместо seeded-потока.

Схема (все поля кроме `name` опциональны):

    name: friday_south           # имя сценария (шапки/логи)
    weekday: friday              # 0..6 или monday..sunday; дефолт — DELIVERY_WEEKDAY (вторник)
    dispatch_start: "08:00"      # час открытия диспетч-окна; дефолт DISPATCH_OPEN_H
    pharmacies: all              # или список имён/stop-id: ["Vita-Apotheke", 12, ...]
    demand: {"Vita-Apotheke": 20}  # оверрайды спроса (боксы) поверх сид-розыгрыша
    fleet: {K: 2, Q: 80}         # флот сценария
    events:
      - {at: "09:15", type: traffic,   where: "Linden-Apotheke",
         params: {magnitude: 1.5, radius_km: 1.5, duration_min: 45}}   # closure: true → δ=∞
      - {at: "09:40", type: breakdown}
      - {at: "10:05", type: urgent, where: 12, params: {demand: 10, window_min: 45}}

Имена резолвятся по `names.parquet` снапшота: точное совпадение → уникальная подстрока →
fuzzy (difflib); иначе ValueError со списком похожих. Время событий — «мин от старта смены»
для состояния эпизода и АБСОЛЮТНЫЕ минуты от DISPATCH_OPEN_H для инцидента (так их читает
CongestionTravel) — сдвиг учитывается при dispatch_start ≠ DISPATCH_OPEN_H.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

from logistics_rl_gnn.config import instance as im
from logistics_rl_gnn.config.congestion import (
    INCIDENT_DUR_RANGE_MIN,
    INCIDENT_MAG_RANGE,
    INCIDENT_RADIUS_KM,
)
from logistics_rl_gnn.env.events import BreakdownEvent, TrafficEvent, UrgentEvent
from logistics_rl_gnn.env.travel import Incident

_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
_KINDS = ("traffic", "breakdown", "urgent")
_FUZZY_CUTOFF = 0.75  # ниже — не угадываем, а просим уточнить (список похожих в ошибке)


@dataclass
class Scenario:
    """Загруженный сценарий: готовый инстанс + события + что именно попросили и что выпало."""

    name: str
    weekday: int
    instance: object
    events: list
    path: Path
    requested_stops: list[int] = field(default_factory=list)  # snapshot-стопы из YAML
    dropped_stops: list[int] = field(default_factory=list)  # закрыты в этот день → выпали
    names: dict = field(default_factory=dict)  # {snapshot_stop: name}

    @property
    def fleet(self) -> tuple[int, float]:
        return im.fleet_of(self.instance)

    def label(self, stop: int) -> str:
        return self.names.get(int(stop), f"stop #{int(stop)}")


# ---------- имена аптек ----------


def load_names(snapshot_dir) -> dict[int, str]:
    """{snapshot_stop: name} из names.parquet снапшота; нет файла → {} (резолв только по id)."""
    p = Path(snapshot_dir) / "names.parquet"
    if not p.exists():
        return {}
    df = pd.read_parquet(p)
    return {
        int(s): str(n).strip()
        for s, n in zip(df["stop"], df["name"], strict=False)
        if pd.notna(n) and str(n).strip()
    }


def resolve_stop(token, names: dict[int, str], *, known: set[int] | None = None) -> int:
    """Имя/stop-id из YAML → snapshot-стоп. Точное → подстрока → fuzzy; иначе внятная ошибка."""
    if isinstance(token, bool):
        raise ValueError(f"аптека задана булевым значением {token!r} — ожидалось имя или stop-id")
    if isinstance(token, int) or (isinstance(token, str) and token.strip().lstrip("#").isdigit()):
        stop = int(str(token).strip().lstrip("#"))
        if known is not None and stop not in known:
            raise ValueError(f"stop-id {stop} нет в снапшоте (есть 0..{max(known)})")
        return stop
    q = str(token).strip()
    if not names:
        raise ValueError(f"аптека «{q}»: нет names.parquet в снапшоте — задавай stop-id числом")
    ql = q.casefold()
    exact = [s for s, n in names.items() if n.casefold() == ql]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        raise ValueError(f"имя «{q}» неоднозначно: стопы {sorted(exact)} — уточни stop-id")
    sub = [s for s, n in names.items() if ql in n.casefold()]
    if len(sub) == 1:
        return sub[0]
    if len(sub) > 1:
        raise ValueError(
            f"имя «{q}» неоднозначно — подходят: "
            + ", ".join(f"«{names[s]}» (stop {s})" for s in sorted(sub)[:5])
        )
    by_name: dict[str, list[int]] = {}  # одноимённые аптеки → СПИСОК стопов, иначе fuzzy молча
    for s, n in names.items():          # выбрал бы одну из них (точное совпадение честно падает)
        by_name.setdefault(n.casefold(), []).append(s)
    close = difflib.get_close_matches(ql, list(by_name), n=3, cutoff=_FUZZY_CUTOFF)
    if close:  # уверенное fuzzy-совпадение
        hit = by_name[close[0]]
        if len(hit) > 1:
            raise ValueError(f"имя «{q}» ≈ «{names[hit[0]]}» неоднозначно: стопы {sorted(hit)} "
                             f"— задай stop-id")
        return hit[0]
    hint = difflib.get_close_matches(ql, list(by_name), n=3, cutoff=0.3) or list(by_name)[:3]
    raise ValueError(
        f"аптека «{q}» не найдена в names.parquet. Похожие: "
        + ", ".join(f"«{names[by_name[c][0]]}» (stop {by_name[c][0]})" for c in hint)
    )


# ---------- парсинг полей ----------


def _weekday(v) -> int:
    if v is None:
        return im.DELIVERY_WEEKDAY
    if isinstance(v, int) and not isinstance(v, bool):
        if not 0 <= v <= 6:
            raise ValueError(f"weekday {v} вне 0..6 (0 = понедельник)")
        return v
    s = str(v).strip().casefold()
    for i, name in enumerate(_WEEKDAYS):
        if s in (name, name[:3]):
            return i
    raise ValueError(f"weekday «{v}» не распознан (0..6 или {', '.join(_WEEKDAYS)})")


def _hour(v, field_name: str) -> float:
    """«08:30» / 8 / 8.5 → час суток (float)."""
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        h = float(v)
    else:
        parts = str(v).strip().split(":")
        try:
            h = float(parts[0]) + (float(parts[1]) / 60.0 if len(parts) > 1 else 0.0)
        except ValueError as e:
            raise ValueError(f"{field_name} «{v}»: ожидался формат ЧЧ:ММ") from e
    if not 0.0 <= h < 24.0:
        raise ValueError(f"{field_name} «{v}» вне суток")
    return h


def _at_min(v, *, open_h: float, horizon_min: float) -> float:
    """Время события «09:15» → минуты от старта смены. Вне смены → внятная ошибка.

    round(…, 6) — иначе 08:55 даёт 54.999999… мин, и часы демо печатают «08:54» (событие как бы
    на минуту раньше, чем в YAML).
    """
    at = round((_hour(v, "at") - open_h) * 60.0, 6)
    if not 0.0 <= at < horizon_min:
        end_h = open_h + horizon_min / 60.0
        raise ValueError(
            f"событие в «{v}» вне диспетч-смены {open_h:02.0f}:00–{end_h:02.0f}:00 "
            f"({at:.0f} мин от старта)"
        )
    return at


_EVENT_KEYS = {"at", "type", "where", "params"}
_PARAM_KEYS = {  # опечатка в params молчала бы: `closure` не там → инцидент вместо закрытия
    "traffic": {"magnitude", "radius_km", "duration_min", "closure"},
    "urgent": {"demand", "window_min"},
    "breakdown": set(),
}


def _build_event(spec: dict, inst, *, open_h: float, names: dict, snap_stops: list):
    """Одно описание события из YAML → объект env.events (тот же класс, что у харнесса 0004)."""
    kind = str(spec.get("type", "")).strip().casefold()
    if kind not in _KINDS:
        raise ValueError(f"тип события «{spec.get('type')}» неизвестен (ожидалось {_KINDS})")
    bad = set(spec) - _EVENT_KEYS
    if bad:
        raise ValueError(f"событие «{kind}» в «{spec.get('at')}»: неизвестные поля {sorted(bad)} "
                         f"(параметры кладутся в params: {sorted(_PARAM_KEYS[kind]) or '—'})")
    horizon_min = inst.horizon_s / 60.0
    at = _at_min(spec.get("at"), open_h=open_h, horizon_min=horizon_min)
    p = dict(spec.get("params") or {})
    bad_p = set(p) - _PARAM_KEYS[kind]
    if bad_p:
        raise ValueError(f"событие «{kind}» в «{spec.get('at')}»: неизвестные params "
                         f"{sorted(bad_p)} (ожидалось {sorted(_PARAM_KEYS[kind]) or '—'})")
    if kind == "breakdown":
        return BreakdownEvent(at)

    if spec.get("where") is None:
        raise ValueError(f"событие «{kind}» в «{spec.get('at')}» требует поле where (аптека)")
    stop = resolve_stop(spec["where"], names, known=set(snap_stops))
    if stop not in snap_stops:
        raise ValueError(
            f"аптека «{spec['where']}» (stop {stop}) не входит в инстанс сценария "
            "(не в списке pharmacies или закрыта в этот день)"
        )
    idx = snap_stops.index(stop)  # instance-local индекс (как idx у seeded-потока)
    if kind == "urgent":
        return UrgentEvent(at, {
            "idx": idx,
            "demand": int(p.get("demand", im.DEMAND_RANGE[1])),
            "delta_s": float(p.get("window_min", 45.0)) * 60.0,
        })
    mag = float("inf") if bool(p.get("closure")) else float(
        p.get("magnitude", sum(INCIDENT_MAG_RANGE) / 2)
    )
    inc = Incident(
        center=tuple(inst.coords[idx]),
        radius_km=float(p.get("radius_km", INCIDENT_RADIUS_KM)),
        magnitude=mag,
        # инцидент живёт в АБСОЛЮТНОМ клоке congestion (от DISPATCH_OPEN_H), событие — от старта
        t_start_min=at + im.dispatch_offset_min(inst),
        duration_min=float(p.get("duration_min", sum(INCIDENT_DUR_RANGE_MIN) / 2)),
    )
    return TrafficEvent(at, inc)


# ---------- загрузка ----------


def load_scenario(path, *, snapshot_dir=None, seed: int = im.DEFAULT_SEED) -> Scenario:
    """YAML → Scenario. Валидация — существующими стражами (Σdemand ≤ K·Q, достижимость)."""
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"нет файла сценария: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: ожидался YAML-словарь, получено {type(raw).__name__}")
    unknown = set(raw) - {"name", "weekday", "dispatch_start", "pharmacies", "demand", "fleet",
                          "events"}
    if unknown:
        raise ValueError(f"{path}: неизвестные поля {sorted(unknown)}")

    snap = Path(snapshot_dir) if snapshot_dir else im._latest_snapshot_dir()
    if snap is None:
        raise FileNotFoundError("нет снапшота — сначала `python scripts/build_snapshot.py`")
    names = load_names(snap)
    all_stops = set(pd.read_parquet(Path(snap) / "nodes.parquet")["stop"].astype(int))

    weekday = _weekday(raw.get("weekday"))
    start = raw.get("dispatch_start")
    open_h = _hour(start, "dispatch_start") if start is not None else None
    fleet = raw.get("fleet") or {}
    if not isinstance(fleet, dict):
        raise ValueError(f"{path}: fleet — словарь {{K: …, Q: …}}, получено {fleet!r}")
    if set(fleet) - {"K", "k", "Q", "q"}:  # опечатка тихо вернула бы дефолтный флот 8×80
        raise ValueError(f"{path}: fleet — только K и Q, получено {sorted(fleet)}")
    k = fleet.get("K", fleet.get("k"))
    q = fleet.get("Q", fleet.get("q"))

    ph = raw.get("pharmacies", "all")
    if isinstance(ph, str) and ph.strip().casefold() == "all":
        requested, include = [], None
    elif isinstance(ph, list):
        requested = [resolve_stop(t, names, known=all_stops) for t in ph]
        include = set(requested)
        if not include:
            raise ValueError(f"{path}: pharmacies — пустой список")
    else:
        raise ValueError(f"{path}: pharmacies — «all» или список имён/stop-id, получено {ph!r}")

    overrides = {
        resolve_stop(t, names, known=all_stops): int(v)
        for t, v in (raw.get("demand") or {}).items()
    }
    bad = sorted(set(overrides) - include) if include is not None else []
    if bad:
        raise ValueError(f"{path}: demand-оверрайд для аптек вне списка pharmacies: {bad}")

    try:
        inst = im.generate_instance(
            snap, seed=seed, delivery_weekday=weekday, include_stops=include,
            demand_overrides=overrides, fleet_size=k, vehicle_cap=q, dispatch_open_h=open_h,
        )
    except (ValueError, AssertionError) as e:  # страж инстанса → адресуем к файлу сценария
        raise ValueError(f"{path}: сценарий невалиден — {e}") from e

    open_eff = float(inst.meta["dispatch_open_h"])
    spec = raw.get("events") or []
    if not isinstance(spec, list) or any(not isinstance(s, dict) for s in spec):
        raise ValueError(f"{path}: events — список словарей [{{at, type, …}}], получено {spec!r}")
    events = [_build_event(dict(s), inst, open_h=open_eff, names=names,
                           snap_stops=list(inst.snapshot_stops)) for s in spec]
    events.sort(key=lambda e: e.at_min)
    # выпавшие стопы: из запрошенных (подмножество) либо все закрытые в этот день (pharmacies: all)
    known = set(inst.snapshot_stops)
    dropped = ([s for s in requested if s not in known] if requested
               else list(inst.excluded_stops))
    return Scenario(
        name=str(raw.get("name") or path.stem), weekday=weekday, instance=inst, events=events,
        path=path, requested_stops=requested, dropped_stops=dropped, names=names,
    )
