# logistics-rl-gnn — dev targets (just <target>)

# Install the project with dev dependencies
install:
    pip install -e ".[dev]"

# Run the tests
test:
    pytest -q

# Lint
lint:
    ruff check .

# Format
fmt:
    ruff format .

# Lint + tests (the CI gate)
check: lint test
