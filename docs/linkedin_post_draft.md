# LinkedIn post drafts — Dynamic Pharmacy Delivery Routing (GNN + RL)

> Drafts only — final review by the author. Every number is from a durable seeded artifact.
> Honesty rules enforced: **no static/dynamic conflation**, **no "we beat OR-Tools"** (static quality
> is *within 3.4 % of* OR-Tools; the edge vs OR-Tools is *reaction latency*, a different setting).
> Replace `<repo-url>` before posting.

---

## Variant A — business / "before → after" (EN)

**Before → after on real-world delivery routing.**

I built a dynamic route planner for same-day medication delivery to 62 pharmacies in Augsburg — on
the real OpenStreetMap road network, with real opening-hours time windows, not a toy grid.

On the static daily plan, vs the status-quo greedy dispatcher:
• **−23.5 % operating cost**
• **−39.6 % vehicle-hours on duty** (mostly cutting idle waiting at closed time windows — same
  kilometres, far less waiting)
• Lands **within 3.4 %** of a strong static OR-Tools reference

And in the dynamic setting — a mid-day disruption (traffic, breakdown, urgent order), replanning the
residual problem — it **re-plans in under 0.7 s**, about **2.9× faster** than an OR-Tools re-solve,
so a dispatcher gets an answer in the time it takes to read the alert.

Stack: Graph Neural Network + Reinforcement Learning (POMO) for construction, a candidate portfolio,
and classical local-search polish — every number seeded and reproducible.

**Honest caveat:** the static quality here comes from the classical local-search polish, not the
neural net — given the same wall-clock, OR-Tools actually edges out the system on static cost. Where
GNN+RL genuinely earns its place is **reaction speed**: an instant re-plan when the day changes.

Write-up, code, and the honest trade-offs: <repo-url>

#OperationsResearch #ReinforcementLearning #Logistics #GraphNeuralNetworks #VehicleRouting

---

## Variant A — business / "before → after" (RU)

**«Было → Стало» на реальной задаче маршрутизации.**

Собрал динамический планировщик маршрутов для доставки медикаментов в 62 аптеки Аугсбурга — на
реальной дорожной сети OpenStreetMap, с реальными окнами по часам работы аптек, а не на игрушечной
сетке.

На статическом дневном плане, против статус-кво (жадный диспетчер):
• **−23.5 % операционных издержек**
• **−39.6 % машино-часов в наряде** (в основном за счёт простоя у закрытых окон — те же километры,
  куда меньше ожидания)
• В пределах **+3.4 %** от сильного статического ориентира OR-Tools

А в динамике — событие среди дня (пробка, поломка, срочный заказ), перестроение остаточной задачи —
система **перестраивает маршрут за <0.7 с**, примерно **в 2.9× быстрее** пересчёта OR-Tools:
диспетчер получает ответ за время прочтения уведомления.

Стек: Graph Neural Network + Reinforcement Learning (POMO) для построения, портфель кандидатов и
классическая local-search полировка — каждое число с зафиксированным seed и воспроизводимо.

**Честная оговорка:** статическое качество здесь даёт классическая local-search полировка, а не
нейросеть — при равном wall-clock OR-Tools статически даже чуть обходит систему. Реальный вклад
GNN+RL — **скорость реакции**: мгновенное перестроение, когда день меняется.

Разбор, код и честные компромиссы: <repo-url>

#OperationsResearch #ReinforcementLearning #Logistics #МашинноеОбучение #VehicleRouting

---

## Variant B — technical / research honesty (EN)

**An honest negative result is still a result.**

I spent this project asking one question on real Augsburg delivery data (62 pharmacies, real OSM
network): **does a GNN + RL policy actually beat classical routing on quality?**

Short answer: **no — and I made that hard to fake.**
• The final "RL-by-quality" test was a **pre-registered gate committed to git before the training
  code** (one run, no retry). It failed.
• Comparisons were **paired** on identical seeds; a byte-identical greedy candidate made "never worse
  than greedy" true *by construction*, not by trust.
• A **time-matched** benchmark removed the usual "you under-budgeted OR-Tools" objection: given equal
  wall-clock, OR-Tools matches the system's static quality in under a second.

So where does the neural machinery genuinely help? **Reaction latency and candidate diversity.** In
the dynamic setting it re-plans in ~0.7 s vs ~2 s for an OR-Tools re-solve — an edge in *speed of
reaction*, not in solution cost.

Meanwhile the deployed system still delivers **−23.5 % cost vs the status-quo heuristic**, within
3.4 % of the static OR-Tools optimum. The quality lever turned out to be classical local search, not
the policy — and saying so plainly is the point.

Full decision log, methodology, and reproducible numbers: <repo-url>

#ReinforcementLearning #OperationsResearch #MachineLearning #ResearchIntegrity #VehicleRouting

---

## Variant B — technical / research honesty (RU)

**Честный отрицательный результат — тоже результат.**

Весь проект я проверял один вопрос на реальных данных Аугсбурга (62 аптеки, реальная OSM-сеть):
**действительно ли GNN + RL-политика бьёт классику по качеству маршрутов?**

Короткий ответ: **нет — и я постарался сделать это трудно-подделываемым.**
• Финальный тест «RL по качеству» был **предрегистрированным гейтом, закоммиченным в git ДО кода
  обучения** (один заход, без ретрая). Не пройден.
• Сравнения — **парные** на одинаковых seed; байт-идентичный greedy-кандидат делал «не хуже greedy»
  истиной *по построению*, а не по доверию.
• **Time-matched** бенчмарк снял возражение «вы недодали бюджет OR-Tools»: при равном wall-clock
  OR-Tools достигает статик-качества системы меньше чем за секунду.

Где же нейросеть реально помогает? **Латентность реакции и разнообразие кандидатов.** В динамике
перестроение ~0.7 с против ~2 с у OR-Tools-пересчёта — преимущество в *скорости реакции*, не в
стоимости решения.

При этом деплой-система даёт **−23.5 % издержек против статус-кво**, в пределах 3.4 % от статического
оптимума OR-Tools. Рычагом качества оказался классический local search, а не политика — и сказать это
прямо и есть суть.

Полный журнал решений, методология и воспроизводимые числа: <repo-url>

#ReinforcementLearning #OperationsResearch #МашинноеОбучение #ResearchIntegrity #VehicleRouting
