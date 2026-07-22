---
type: runbook
id: run-train-on-server
date: 2026-07-21
tags: [runbook, training, server, pomo, phase6b]
---

# Обучение на Ryzen-сервере (NJ)

**Назначение:** воспроизводимый полный прогон обучения (POMO / dynamic) на сервере NJ
(Ryzen 9 9900X, 12C/24T, 96 ГБ, CPU-only). Код едет с мака через git, данные — rsync'ом,
запуск в tmux, веса и метрики забираем обратно. Мак — только для smoke; полные прогоны здесь.

> **Переменные окружения (задать под себя):**
> `SERVER` — SSH-хост сервера · `REPO=~/logistics-rl-gnn` — путь репо на сервере ·
> `SNAP=augsburg_20260720` — активный снапшот данных · `TRAIN=scripts/train_pomo.py` —
> актуальный train-скрипт фазы (для Шага 2 заменить на dynamic-раннер).

## Предусловия

- `ssh $SERVER` работает.
- Актуальный коммит **запушен** с мака (сервер тянет версию из истории, не рабочее дерево).
- На маке собран снапшот `data/snapshots/$SNAP/` (иначе — Phase 2 `scripts/build_snapshot.py`).

## Шаги

### 1. Синхронизировать код (git)

```bash
# на маке
git push
# на сервере
ssh $SERVER
cd $REPO && git pull
```

### 2. Синхронизировать данные (rsync — снапшот вне git)

Снапшот в gitignore (запрет №1), переносим явно. Так гарантируем **идентичные** данные:
ре-билд заново дёрнул бы OSM и мог разойтись → сломал бы сравнение с бейзлайнами.

```bash
# на маке
rsync -av data/snapshots/$SNAP/  "$SERVER:$REPO/data/snapshots/$SNAP/"
```

### 3. Окружение (один раз)

```bash
# на сервере, в $REPO
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-колёса (AMD)
pip install -e ".[data,env,baselines,model]"
```

### 4. Паритет среды

```bash
pytest -q     # тот же зелёный, что на маке — подтверждает, что версия и окружение сходятся
```

### 5. Запуск в tmux

```bash
tmux new -s train
# потоки = физические ядра (12); сравнить 12 vs 24 (SMT) на своей нагрузке
OMP_NUM_THREADS=12 python -u $TRAIN 2>&1 | tee results/pomo_$(date +%Y%m%d).log
# Ctrl-b d — отцепиться (прогон живёт дальше); tmux attach -t train — вернуться
```

Сервер не засыпает — `caffeinate` не нужен (в отличие от мака-Air).

### 6. Мониторинг

```bash
tail -f results/pomo_*.log  # ep N|train(g±%)|val(g±%[OR±%])|mem±%|H||g||std|es N/patience|fr HASH
```

Здоровые признаки: `train g↓` · `|g|>0` · энтропия не 0 и не максимум · `std>0` (shared baseline не
вырожден) · `fr` (хеш инстансов) меняется каждую эпоху (RNG свеж, не повтор драйвов) · `mem` НЕ растёт
монотонно (растёт = memorization: train↓ при застывшем val). **Гейт
первых ~15 эпох:** если `val` плоский (в пределах шума) с ~эпохи 3 и `es` только копится, не улучшаясь
→ сигнал отбора слаб (val насытился у эвристики, как в прошлом прогоне) → поднять `val_n_range=(62,62)`
(лычаг в config/pomo.py — val/test на deployment-размере) и перезапустить. `es N/patience` → early-stop.

### 7. Забрать результат

```bash
# на маке (refit пишет policy_pomo_refit.pt — glob заберёт и его, и старый best 770.4€)
rsync -av "$SERVER:$REPO/results/policy_pomo_*.pt"  results/
rsync -av "$SERVER:$REPO/results/pomo_*.log"        results/
```

Веса вне git (запрет №1) → в git коммитим только метрики/сводку + decision с числами. Refit пишет в
`policy_pomo_refit.pt` — старый `policy_pomo_best.pt` (770.4€) НЕ затирается (нужен для сравнения).

## Воспроизводимость (обязательно)

Рядом с весами залогировать: `seed`, config (N_starts / epochs / lr / clip), версии
(`torch`, `torch-geometric`, `ortools`) — по образцу `results/baselines.json`.
Без этого прогон не повторить, и «Стало» нельзя защитить.

## Troubleshooting

- **torch тянет CUDA/тяжёлую сборку** → ставить строго с `--index-url …/whl/cpu` **до** `pip install -e`.
- **torch-geometric не собирается** → сначала torch (CPU), затем PyG; сверить совместимость версий.
- **Медленно / троттлит** → тюнинг `OMP_NUM_THREADS` (12 физических обычно лучше 24 c SMT); `htop` — следить, что не свопит.
- **`pytest` красный на сервере, зелёный на маке** → разошлись версии: сверить `pip freeze` ключевых пакетов, пересобрать `.venv`.

## Связи

- Обучение: [[0006-pomo-static]] · [[0003-phase6-training]]
- Данные/снапшот: [[0001-mdp-spec]] (Phase 2)
