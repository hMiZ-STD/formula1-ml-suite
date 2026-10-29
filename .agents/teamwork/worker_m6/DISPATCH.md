# Dispatch Assignment: Milestone 6 — Diagnostic Visualizations (Figure 12)

**Task**: Implement `plot_fig12_ranking_and_hurdle_eval()` in `src/visualization/plots.py` and generate `reports/figures/fig12_ranking_and_hurdle_eval.png` (300 DPI, 4 panels) adhering to project standards and educational ELI5 analogies.

**Scope & Deliverables**:
1. Add `plot_fig12_ranking_and_hurdle_eval()` to `src/visualization/plots.py` with full docstring, type hints, child-friendly analogy, and 4 diagnostic panels on a 2x2 grid.
2. Ensure the function can be invoked standalone or integrated cleanly into `src/run_v6_experiments.py` (e.g. `if __name__ == "__main__":` or called by experiment runner).
3. Execute the function to generate `reports/figures/fig12_ranking_and_hurdle_eval.png` at strictly 300 DPI.
4. Verify that the image exists, is valid (PNG format, 300 DPI, non-empty, ~400KB-1MB), and that all 4 panels render clearly with titles, axes, legends, and annotations.
5. Ensure 100% backward compatibility: existing plotting functions (`plot_fig1` through `plot_fig11`) must remain completely intact.
6. Verify full test suite passes with zero warnings: `pytest tests/ -v`.


## 2026-10-29T16:45:40Z
You are the Diagnostic Visualizations Worker for Milestone 6 of the Formula 1 Machine Learning Engineering Suite v6.0.

Your working directory is: d:\ML-Project\.agents\teamwork\worker_m6
You MUST read the following authoritative files before starting work:
- d:\ML-Project\.agents\teamwork\ORIGINAL_REQUEST.md
- d:\ML-Project\PROJECT.md
- d:\ML-Project\.agents\teamwork\worker_m6\DISPATCH.md
- d:\ML-Project\src\visualization\plots.py
- d:\ML-Project\reports\tables\table8_ranking_hurdle_benchmark.csv
- d:\ML-Project\EXPERIMENT_FREEZE.yaml
- d:\ML-Project\src\run_v6_experiments.py

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

YOUR ASSIGNMENT:
1. Implement `plot_fig12_ranking_and_hurdle_eval()` in `src/visualization/plots.py`:
   - Follow project standards, PEP 8, strict type hints, and educational ELI5 child-friendly analogies.
   - 2x2 grid layout saved at 300 DPI to `reports/figures/fig12_ranking_and_hurdle_eval.png`.
   - Panel A (Top-Left): NDCG@K comparison (K=1, 3, 10) across key benchmark models (Grid Heuristic baseline, Pointwise v5.0 models, LambdaMART linear_rank, Plackett-Luce neural ranker, Two-Stage Hurdle model). Data should be drawn from `reports/tables/table8_ranking_hurdle_benchmark.csv` (or passed DataFrame).
   - Panel B (Top-Right): Hurdle Stage 1 DNF Brier calibration curve (`sklearn.calibration.calibration_curve`). Fit or use Stage 1 hazard classifier on `train_core` / `val` and evaluate on `test.parquet`. Plot mean predicted DNF probability vs true fraction of DNFs with diagonal reference line and annotated Brier score.
   - Panel C (Bottom-Left): Test Podium F1 comparison: Pointwise static threshold tau vs. Post-hoc rank() <= 3 vs. LTR (linear_rank) vs. Hurdle model, visually highlighting the ranking penalty suffered by pointwise models and the zero-sum performance of LTR.
   - Panel D (Bottom-Right): Regression residual distribution (y - \hat{y}) comparing standard unmodeled OLS regression (or Ridge/LinearRegression baseline) against the Two-Stage Hurdle model expected position E[Pos] on test data, illustrating how decoupling mechanical DNF hazard eliminates catastrophic positive tail error (>10 positions) and elevates classified pace R2.
   - Make the function self-contained with reasonable defaults so calling `plot_fig12_ranking_and_hurdle_eval()` with no arguments loads datasets and Table 8 automatically and generates the figure.
2. In `src/run_v6_experiments.py`, update `run_v6_benchmarks()` to call `plot_fig12_ranking_and_hurdle_eval()` after generating Table 8 and updating freeze, so that running the experiment runner automatically updates Figure 12 as well.
3. Also provide a standalone CLI entrypoint in `src/visualization/plots.py`:
   `if __name__ == "__main__": plot_fig12_ranking_and_hurdle_eval()`
4. Execute the plotting code and verify:
   - Command: `python src/visualization/plots.py` (or `python src/run_v6_experiments.py`)
   - Verify `reports/figures/fig12_ranking_and_hurdle_eval.png` exists, is valid PNG, non-empty, and rendered at 300 DPI.
   - Verify all 73 existing tests pass: `pytest tests/ -v --tb=short`.
   - Ensure zero regressions to existing functions in `plots.py` (`plot_fig1` through `plot_fig11`).
5. Write a comprehensive handoff report to `d:\ML-Project\.agents\teamwork\worker_m6\handoff.md` with Observation, Logic Chain, Caveats, Conclusion, and Verification Method.
6. Send a message to orchestrator upon completion.
