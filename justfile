# logistics-rl-gnn — dev-таргеты (just <target>)

# Установить проект с dev-зависимостями
install:
    pip install -e ".[dev]"

# Прогнать тесты
test:
    pytest -q

# Линт
lint:
    ruff check .

# Формат
fmt:
    ruff format .

# Линт + тесты (CI-гейт)
check: lint test
