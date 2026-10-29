# Orchestrator Soft Handoff — Generation 0 to Generation 1

**Author**: Generation 0 Orchestrator (`teamwork_orchestrator`)  
**Working Directory**: `d:\ML-Project\.agents\teamwork\orchestrator\`  
**Target Successor**: Generation 1 Orchestrator (`self`)  
**Date**: 2026-10-29T15:52:00Z  
**Parent Conversation ID**: `77860de7-28c2-46fe-9ffd-5d40f671bc90`

---

## 1. Milestone State

| # | Milestone | Status | Key Artifacts / Outputs | Next Steps for Successor |
|---|-----------|:------:|-------------------------|--------------------------|
| M0 | Survey Phase | **DONE** | `PROJECT.md`, `TEST_INFRA.md`, survey reports | Complete. |
| M1 | Environment & Version Control | **PASSED** (Gate 100%) | Git branch `feat/ltr-and-hurdle-architecture`, `lightgbm==4.7.0`, `tests/conftest.py` | Complete. 23/23 baseline tests pass. |
| M2 | Race-Aligned Ranking Metrics | **PASSED** (Gate 100%) | `src/evaluation/ranking_metrics.py`, `tests/test_ranking_metrics.py` (19 tests) | Complete. 42/42 tests pass. |
| M3 | Learning-to-Rank Architecture | **WORKER DONE** (Awaiting Gate) | `src/models/ltr_ranker.py`, `tests/test_ltr_ranker.py` (19 tests), `EXPERIMENT_FREEZE.yaml` | **Run Milestone 3 Gate**: Spawn 2 Reviewers, 2 Challengers, and 1 Forensic Auditor (`teamwork_auditor`) to verify Worker M3. |
| M4 | Two-Stage Hurdle Architecture | **NOT STARTED** | `src/models/hurdle_model.py`, `tests/test_hurdle_model.py` | Dispatch Worker M4 once M3 Gate passes. |
| M5 | Experiment Runner & Table 8 Benchmark | **NOT STARTED** | `src/run_v6_experiments.py`, `reports/tables/table8_ranking_hurdle_benchmark.csv` | Dispatch Worker M5 once M4 Gate passes. |
| M6 | Diagnostic Visualizations | **NOT STARTED** | `reports/figures/fig12_ranking_and_hurdle_eval.png` (300 DPI, 4 panels) | Dispatch Worker M6 once M5 Gate passes. |
| M7 | Final Milestone: E2E Validation, Coverage Hardening, Docs, Git Commit | **NOT STARTED** | Pass 100% tests, Tier 5 adversarial coverage, update walkthrough, commit on `feat/ltr-and-hurdle-architecture` | Final acceptance & user report. |

---

## 2. Active Subagents
- All 16 subagents spawned in Generation 0 have completed their tasks and delivered handoff reports.
- **Pending Subagents**: None currently running.
- **Worker M3** has completed `src/models/ltr_ranker.py` and reported `linear_rank` as the winning mapping with 61/61 tests passing. The immediate next action for Generation 1 is to evaluate the Milestone 3 Gate (Reviewers, Challengers, Auditor).

---

## 3. Key Artifacts & Context
- `ORIGINAL_REQUEST.md`: `d:\ML-Project\.agents\teamwork\ORIGINAL_REQUEST.md` (authoritative user requirements).
- `PROJECT.md`: `d:\ML-Project\PROJECT.md` (global architecture, milestones, interface contracts, feature inventory).
- `TEST_INFRA.md`: `d:\ML-Project\TEST_INFRA.md` (test framework & tier specifications).
- `GATE_STATUS.md`: `d:\ML-Project\.agents\teamwork\orchestrator\GATE_STATUS.md` (verdicts for M1 and M2 gates — both PASSED).
- `EXPERIMENT_FREEZE.yaml`: Updated with `v6_ranking_and_hurdle` containing the winning `linear_rank` configuration.
- Total passing tests currently: **61 passed, 0 failed in 7.12s**.

---

## 4. Pending Decisions & Critical Context for Successor
1. **Milestone 3 Gate**:
   - Worker M3 delivered `src/models/ltr_ranker.py` and `tests/test_ltr_ranker.py`.
   - Successor must immediately dispatch the M3 Gate team:
     - 2 Reviewers (`teamwork_reviewer`) to review LTR models, early stopping, and typing.
     - 2 Challengers (`teamwork_challenger`) to stress-test `predict_podium_top3` non-duplicate guarantee, tie-breaking on `grid_effective`, and PyTorch ListMLE stability.
     - 1 Forensic Auditor (`teamwork_auditor`) to audit for zero hardcoding, zero cheating, and genuine mathematical optimization.
2. **Milestone 4 (Two-Stage Hurdle)**:
   - Once M3 Gate passes, dispatch Worker M4 for `src/models/hurdle_model.py`:
     - Stage 1: Calibrated DNF hazard classifier predicting $P(\text{is\_dnf} = 1 \mid X)$.
     - Stage 2: Classified pace model trained strictly on `is_classified == 1`.
     - Law of Total Probability combination: $\mathbb{E}[\text{Pos}] = (1 - p_{\text{dnf}})\hat{y}_{\text{finish}} + p_{\text{dnf}}\bar{y}_{\text{dnf}}$ and $P(\text{Podium}) = (1 - p_{\text{dnf}}) P(\text{Podium} \mid \text{Finish})$.
     - Top-3 podium selection strictly guaranteeing 3 unique drivers per race.
     - Tests in `tests/test_hurdle_model.py`.
3. **Milestone 5 & 6**:
   - `src/run_v6_experiments.py` generating `table8_ranking_hurdle_benchmark.csv` and Markdown table.
   - `plot_fig12_ranking_and_hurdle_eval()` generating `reports/figures/fig12_ranking_and_hurdle_eval.png` (4 panels, 300 DPI).
4. **Milestone 7 (Final Acceptance)**:
   - Run full test suite (must pass 100%).
   - Verify all 5 acceptance criteria checkboxes in `ORIGINAL_REQUEST.md`.
   - Update project walkthrough documentation artifact.
   - Commit all changes on branch `feat/ltr-and-hurdle-architecture`.
   - Report final results back to parent (`77860de7-28c2-46fe-9ffd-5d40f671bc90`).

---

## 5. Hard Constraints Reminder for Successor
- NEVER write, modify, or create source code files directly.
- NEVER run build/test commands yourself — require workers to do so.
- NEVER investigate or explore the problem at the code level — dispatch Explorers / Reviewers / Challengers.
- You MAY use file-editing tools ONLY for metadata/state files (.md) in your `.agents/teamwork/` folder.
- Hard Audit Enforcement: Forensic Auditor verdict is a BINARY VETO.
- Never reuse a subagent after it has delivered its handoff.
- Pass parent ID `77860de7-28c2-46fe-9ffd-5d40f671bc90` for status reporting and final completion.
