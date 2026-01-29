# Makefile for smart-energy-ea
# Usage:
#   make help

.PHONY: help setup install run-api run-dashboard al-online al-offline-grid upload-minio format lint clean

PYTHON ?= python
PIP ?= pip

# -------- Help --------
help:
	@echo "Targets:"
	@echo "  setup            Create venv and install deps"
	@echo "  install          Install deps (no venv creation)"
	@echo "  run-api          Run FastAPI retraining service (dev)"
	@echo "  run-dashboard    Run Streamlit AL dashboard"
	@echo "  al-online        Run online AL experiment (with simulator)"
	@echo "  al-offline-grid  Run offline AL grid experiment"
	@echo "  upload-minio     Upload dataset+model to MinIO (integration script)"
	@echo "  format           Format code (ruff/black/isort if installed)"
	@echo "  lint             Lint code (ruff if installed)"
	@echo "  clean            Remove caches and temp artifacts"

# -------- Environment --------
setup:
	$(PYTHON) -m venv venv-smart-energy
	@echo "Activate venv:"
	@echo "  Windows (PowerShell): .\\venv-smart-energy\\Scripts\\Activate.ps1"
	@echo "  Linux/Mac: source venv-smart-energy/bin/activate"
	@echo "Then run: make install"

install:
	$(PIP) install -r requirements.txt

# -------- Run services --------
run-api:
	$(PYTHON) -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

run-dashboard:
	$(PYTHON) -m streamlit run src/apps/streamlit_active_learning_dashboard.py

# -------- Experiments --------
al-online:
	$(PYTHON) src/experiments/run_online_active_learning_with_simulator.py --strategy entropy --init 100 --batch 50 --iters 40

al-offline-grid:
	$(PYTHON) src/experiments/run_offline_active_learning_grid.py

# -------- Integration --------
upload-minio:
	$(PYTHON) src/integration/upload_dataset_and_model_to_minio.py

# -------- Quality (optional, only if tools installed) --------
format:
	@echo "Formatting (skipping if tools not installed)..."
	-ruff format .
	-black .
	-isort .

lint:
	@echo "Linting (skipping if ruff not installed)..."
	-ruff check .

# -------- Cleanup --------
clean:
	@echo "Cleaning caches and temp files..."
	-rm -rf __pycache__ .pytest_cache
	-find . -type d -name "__pycache__" -prune -exec rm -rf {} \; 2>/dev/null || true
	-find . -type d -name ".ipynb_checkpoints" -prune -exec rm -rf {} \; 2>/dev/null || true
