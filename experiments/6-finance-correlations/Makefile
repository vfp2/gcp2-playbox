VENV=.venv
PY=$(VENV)/bin/python
PIP=$(VENV)/bin/pip
RUFF=$(VENV)/bin/ruff
MYPY=$(VENV)/bin/mypy
PYTEST=$(VENV)/bin/pytest

.PHONY: dev run serve test lint typecheck build

dev:
	python3 -m venv $(VENV)
	$(PY) -m pip install -U pip
	$(PY) -m pip install -e .[dev]

dev-market: dev
	$(PIP) install --only-binary=:all: '.[market]'

run:
	$(PY) main.py run --log-level INFO

serve:
	$(PY) main.py serve --host 0.0.0.0 --port 8050

test:
	$(PYTEST) -q

lint:
	$(RUFF) check .

typecheck:
	$(MYPY) app

build:
	$(PIP) freeze > requirements.txt




