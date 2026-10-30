# IAA AI Hackathon — Actuarial Pricing Across Markets with AI Assistance

Solo entry by Suguru Fujita. Predicts 2025 expected ultimate claim cost (risk premium) for private motor,
home buildings and home contents, then adapts to a new market (Week 2), with an AI-assisted, governed workflow.

## Repository map
| Path | Content |
|---|---|
| `CLAUDE.md` | Instructions for AI assistants working in this repo (rules, workflow, review gates) |
| `docs/00–07` | Brief, EDA checklist, modelling standards, validation & guardrails, governance, Week 2 playbook, report/video outline, office premium |
| `logs/` | AI usage log, decision log, data quality log, assumptions register |
| `src/` | Pipeline code (data, target, development, trend, features, models, validation, Week 2 pipeline, office premium, submission) |
| `tests/` | pytest suite |
| `scripts/` | Owner-run analysis scripts (validation, diagnostics, market shift, candidates) printing compact evidence |
| `app/` | Streamlit pricing tool |
| `docs/report/` | Write-up and video script |
| `configs/config.yaml` | Seeds, paths, assumptions, hyper-parameters |
| `data/` | Raw/processed data (git-ignored) |
| `outputs/` | Submissions, models, reports |

## Results
| | Validation D² (temporal) | Leaderboard |
|---|---|---|
| Week 1 (developed) | 0.139 pooled (train 2021–23 → 2024) | private 0.13363 (rank 30), public 0.12480 |
| Week 2 (emerging) | motor 0.082, buildings 0.102, contents 0.102 (train 2023 → 2024) | public 0.15691 (private published after the close) |

Final files: Week 1 `outputs/submissions/w1_blend_20261024_0007_d4aa68f.csv` + `candidates/w1_final_b3_20261024_0024.csv`
(tag `w1-final`); Week 2 `outputs/submissions/w2_20261028_2009_82f4d73.csv` + `candidates/w2_nb_high_20261028_1800.csv`
(tag `w2-final`). Write-up: `docs/report/report.pdf` (source `docs/report/report_draft.md`); video script: `docs/report/video_script.md`;
office premium build-up: `outputs/reports/w2_office_premium*.{csv,md}`.

## Where to look (evidence)
- Every decision, with alternatives and evidence: `logs/decision_log.md` (submissions table at the end)
- Every AI contribution, how it was verified and whether it was accepted: `logs/ai_usage_log.md`
- Data issues and their resolution: `logs/data_quality_log.md`; assumptions: `logs/assumptions_register.md`
- Commit hashes: the history was rewritten once before publishing (author e-mail only; every file tree is
  unchanged). Hashes quoted in file names and logs (e.g. `82f4d73`) are the original ones; `docs/commit_map.csv`
  maps each to its published commit.

## Reproduce
Data are not in the repo: put the official `developed` folder in `data/raw/w1/` and `emerging` in `data/raw/w2/`.
```bash
python -m venv .venv && .venv\Scripts\activate   # Windows (source .venv/bin/activate elsewhere)
pip install -r requirements.txt
pytest -q                                              # 65 tests
python -m src.pipeline --market w1 --model blend       # Week 1 final model (b3 config) -> outputs/submissions/
python -m src.pipeline_w2                              # Week 2 build + new-business sensitivity files
python scripts/make_office_premium.py                  # office premium for the 20 reference policies
streamlit run app/pricing_tool.py                      # pricing tool
```
Assumptions, seeds and model choices are in `configs/config.yaml`. Scripts in `scripts/` reproduce the validation
evidence cited in the report (each writes to `outputs/reports/`).
