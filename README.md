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
├── retraining-api/ # FastAPI retraining service (deployed on Atena)
│   └── app/        # API source: main.py, schemas, train, MinIO IO
├── src/            # Core logic, experiments, dashboards
├── notebooks/      # Exploratory and analysis notebooks
├── data/           # Input datasets & digital twin definitions
├── tables/         # Experiment outputs (CSV/XLSX)
├── figures/        # Generated plots
├── reports/        # Historical interim reports and architecture notes. Not actively maintained.
├── docs/           # Project documentation (see docs/INDEX.md)
├── powershell/     # Historical Windows experiment runner scripts. Paths reference old layout and may not work.
└── README.md
```

---

## Where to look next

- **Retraining API details**  
  → `retraining-api/app/README.md`

- **Source code & experiment overview**  
  → `src/README.md`

- **Detailed Active Learning pipeline explanation**  
  → `docs/active-learning-project-guide.md`

- **System architecture & workflow** — component map and message flow  
  → `docs/architecture.md` + `docs/sequence-current.md`

- **Dataset description** — features, labels, data generation pipeline  
  → `docs/smart-energy-data-description.md`

- **API integration contract** (for UBITECH)  
  → `retraining-api/INTEGRATION_CONTRACT.md`

- **Documentation index (all docs explained)**  
  → `docs/INDEX.md`

---

## AI agent workflow

This repository uses three files for AI-assisted development (Claude Code):

- `project_tracker.md` — the single source of truth for current goals, active tasks, decisions, and open questions. **Read this before any significant code or paper edit.**
- `AGENTS.md` — instructions for AI agents (role, coding rules, communication style).
- `CLAUDE.md` — Claude Code-specific overrides and hooks.

These files are gitignored by design so they remain local and editable without cluttering the shared history. A collaborator cloning the repo will not see them; ask the project owner for a copy if you need them.

---

## Related links

- HumAIne Project: https://humaine-horizon.eu/
- Jožef Stefan Institute (JSI): https://www.ijs.si/

---

## Author

**Gašper Leskovec**  
Jožef Stefan Institute  
Active Learning & Smart Energy research








