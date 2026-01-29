# Smart Energy Active Learning

This repository contains experimental code, analysis, and prototype services
developed as part of **Active Learning research for Smart Energy security assessment**.
The work is carried out in the context of the **HumAIne project** at the
Jožef Stefan Institute (JSI).

--- 

## Scope of this repository

This repository focuses on:
- Active Learning strategies for power grid security classification
- Simulation-based labeling (digital twin / oracle)
- Offline and online experimental pipelines
- Prototype retraining API and lightweight dashboards

⚠️ **Note**: This is a research and experimentation repository.
It is **not** an official production deployment.

---

## Quickstart

The fastest way to run the project locally:

```bash
make run-api
make run-dashboard
make al-online
```

This will:
- start the FastAPI retraining service
- launch the Streamlit Active Learning dashboard
- run an online Active Learning experiment with simulator-based labels

---

## Environment setup

1. Create and activate a virtual environment:
```bash
python -m venv venv-smart-energy
```

2. Activate the environment:
- Windows (PowerShell):
```bash
.\venv-smart-energy\Scripts\Activate.ps1
```
- Linux / macOS:
```bash
source venv-smart-energy/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
```bash
cp .env.example .env
```
Edit `.env` with the required credentials and endpoints.

---

## Repository structure

```text
smart-energy-ea/
├── app/            # FastAPI retraining service
├── src/            # Core logic, experiments, dashboards
├── notebooks/      # Exploratory and analysis notebooks
├── data/           # Input datasets & digital twin definitions
├── tables/         # Experiment outputs (CSV/XLSX)
├── figures/        # Generated plots
├── reports/        # Interim reports and summaries
├── docs/           # Project documentation
├── README.md                              
```

---

## Where to look next

- **Retraining API details**  
  → `app/README.md`

- **Source code & experiment overview**  
  → `src/README.md`

- **Detailed Active Learning pipeline explanation**  
  → `docs/ACTIVE_LEARNING_GUIDE.md` (if present)

---

## Related links

- HumAIne Project: https://humaine-horizon.eu/
- Jožef Stefan Institute (JSI): https://www.ijs.si/

---

## Author

**Gašper Leskovec**  
Jožef Stefan Institute  
Active Learning & Smart Energy research








