## 2026-10-29T16:01:48Z
You are Worker M4 (teamwork_worker).
Your working directory is: d:\ML-Project\.agents\teamwork\worker_m4\

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

You must read and strictly adhere to:
- d:\ML-Project\.agents\teamwork\ORIGINAL_REQUEST.md
- d:\ML-Project\PROJECT.md
- Reference plan: C:/Users/hamiz/.gemini/antigravity/brain/20fe5515-5671-4c16-be43-e50f0230ca78/plan_ltr_and_hurdle_architecture.md
- Explorer survey reports:
  - d:\ML-Project\.agents\teamwork\explorer_survey_1\handoff.md
  - d:\ML-Project\.agents\teamwork\explorer_survey_2\handoff.md

Your exclusive write ownership:
- `src/models/hurdle_model.py`
- `tests/test_hurdle_model.py`

Your task — Milestone 4 (Two-Stage Hurdle Architecture):
1. Implement `src/models/hurdle_model.py`:
   - Educational child-friendly ELI5 analogy at top of module:
     "Imagine running a race with an unpredictable slippery banana peel at Turn 1. Hurdle uses two brains: Brain 1 predicts if your engine will break down or slip on a peel; Brain 2 predicts how fast you drive assuming your car stays in one piece. We then combine them with the Law of Total Probability so a random mechanical breakdown doesn't ruin the model's estimate of a fast driver's true speed!"
   - Strict Python type hints and mathematical docstrings.
   - `F1HurdleModel`:
     - Stage 1 (DNF Hazard Classifier):
       - Predicts $P(\text{is\_dnf} = 1 \mid X)$.
       - Estimator: `HistGradientBoostingClassifier(loss='log_loss', max_depth=4, l2_regularization=1.0, random_state=42)` (or LightGBM classifier).
       - Calibrated on validation data (`val.parquet`) via `CalibratedClassifierCV(..., method='sigmoid', cv='prefit')` for calibrated probabilities $p_{\text{dnf}} \in [0, 1]$.
     - Stage 2 (Classified Pace Model):
       - Trained strictly on classified starters (`df["is_classified"] == 1`). Must assert or guarantee zero DNF rows leak into Stage 2!
       - Dual capability: predicts on-track finish position $\hat{y}_{\text{finish}}$ (e.g. Ridge / GBDT / LightGBM regressor on `positionOrder`) AND on-track podium probability $P(\text{Podium} \mid \text{Finish})$ (e.g. classifier or pace transformation).
     - Law of Total Probability Recombination:
       - Expected finishing position:
         $$\mathbb{E}[\text{Position}_i \mid X_i] = (1 - p_{\text{dnf}, i}) \cdot \hat{y}_{\text{finish}, i} + p_{\text{dnf}, i} \cdot \bar{y}_{\text{dnf}}$$
         where $\bar{y}_{\text{dnf}}$ is computed dynamically from training data ($\approx 18.04$).
       - Hurdle podium probability:
         $$P(\text{Podium}_i \mid X_i) = (1 - p_{\text{dnf}, i}) \cdot P(\text{Podium}_i \mid \text{Finish}_i, X_i)$$
     - `predict_podium_top3(df)`:
       - Strictly guarantees exactly 3 unique podium finishers per race ($3 \times N_{\text{races}}$).
       - Secondary tie-breaking on `grid_effective` ascending (closer to pole wins ties).
       - Handles single-race and multi-race DataFrames.
2. Implement comprehensive unit tests in `tests/test_hurdle_model.py`:
   - Test Stage 1 hazard probabilities are bounded in $[0, 1]$ and calibrated.
   - Test Stage 2 training purity: assert zero DNF rows used in Stage 2 training (`assert (y_train_stage2_dnf == 0).all()`).
   - Test Total Probability recombination bounds ($p_{\text{dnf}} \in [0, 1] \implies \mathbb{E}[\text{Pos}]$ between $\hat{y}_{\text{finish}}$ and $\bar{y}_{\text{dnf}}$).
   - Test `predict_podium_top3`: assert strictly 3 unique drivers per race across all 41 validation races and 128 test races.
   - Test edge cases: extreme $p_{\text{dnf}} = 1.0$, extreme $p_{\text{dnf}} = 0.0$, tied scores.
3. Run verification:
   - Run `pytest tests/test_hurdle_model.py -v --tb=short`
   - Run full regression suite: `pytest tests/ -v --tb=short` (must pass 100% with zero regressions).
4. Write detailed handoff report to `d:\ML-Project\.agents\teamwork\worker_m4\handoff.md`.
5. Send a message to parent when complete.
