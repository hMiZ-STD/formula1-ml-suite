## 2026-10-29T15:05:31Z
You are Worker M1 (teamwork_worker).
Your working directory is: d:\ML-Project\.agents\teamwork\worker_m1\

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

You must read and strictly adhere to:
- d:\ML-Project\.agents\teamwork\ORIGINAL_REQUEST.md
- d:\ML-Project\PROJECT.md
- Explorer survey reports:
  - d:\ML-Project\.agents\teamwork\explorer_survey_1\handoff.md
  - d:\ML-Project\.agents\teamwork\explorer_survey_3\handoff.md

Your exclusive write ownership:
- `pyproject.toml`
- `tests/conftest.py`

Your task — Milestone 1 (Environment and Version Control):
1. Version Control:
   - Run git command to create and switch to branch `feat/ltr-and-hurdle-architecture`:
     `git checkout -b feat/ltr-and-hurdle-architecture`
   - Verify `git status` and `git branch` confirm you are on `feat/ltr-and-hurdle-architecture`.
2. Add Dependency:
   - In `pyproject.toml`, add `"lightgbm>=4.0.0"` under `project.dependencies`.
   - Install LightGBM in the Python environment: `pip install "lightgbm>=4.0.0"` (or `python -m pip install "lightgbm>=4.0.0"`).
   - Verify installation: `python -c "import lightgbm; print('LightGBM version:', lightgbm.__version__)"`.
3. Suppress Loky Core Warning:
   - Create `tests/conftest.py` containing:
     ```python
     """Pytest shared configuration and global environment hooks."""
     import os
     # Suppress joblib loky physical core detection warning on Windows
     os.environ["LOKY_MAX_CPU_COUNT"] = "4"
     ```
4. Regression & Baseline Verification:
   - Run the complete existing test suite:
     `pytest tests/ -v --tb=short`
   - All 23 existing tests must pass with 0 failures, 0 errors, and 0 warnings.
5. Report:
   - Write a complete handoff report to:
     `d:\ML-Project\.agents\teamwork\worker_m1\handoff.md`
   - Include commands run, terminal outputs, verification results, git branch confirmation.
   - Keep your `progress.md` updated in your working directory.
   - Send a message to parent when done.
