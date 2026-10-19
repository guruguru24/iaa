# IAA AI Hackathon — Actuarial Pricing Across Markets with AI Assistance

Solo entry by Suguru Fujita. Predicts 2025 expected ultimate claim cost (risk premium) for private motor,
home buildings and home contents, then adapts to a new market (Week 2), with an AI-assisted, governed workflow.

## Repository map
| Path | Content |
|---|---|
| `CLAUDE.md` | Instructions for AI assistants working in this repo (rules, workflow, review gates) |
| `docs/00–06` | Brief, EDA checklist, modelling standards, validation & guardrails, governance, Week 2 playbook, report/video outline |
| `logs/` | AI usage log, decision log, data quality log, assumptions register |
| `src/` | Pipeline code (data, target, features, models, validation, projection, submission) |
| `tests/` | pytest suite |
| `notebooks/` | EDA and reporting notebooks (call `src/`) |
| `configs/config.yaml` | Seeds, paths, assumptions, hyper-parameters |
| `data/` | Raw/processed data (git-ignored) |
| `outputs/` | Submissions, models, reports |

## Reproduce
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q
python -m src.pipeline --market w1   # (to be implemented)
```
