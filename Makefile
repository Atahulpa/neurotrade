.PHONY: check lint fmt types test install

# Lance tous les contrôles qualité d'un coup
check: lint types test

# Lint ruff
lint:
	uv run ruff check src/ tests/

# Formatage automatique (black + ruff --fix)
fmt:
	uv run black src/ tests/
	uv run ruff check --fix src/ tests/

# Vérification de types mypy
types:
	uv run mypy src/

# Tests pytest avec couverture
test:
	uv run pytest --cov=neurotrade --cov-report=term-missing

# Installation de l'environnement (première fois)
install:
	uv sync
