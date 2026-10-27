# Formula 1 Machine Learning Engineering Standards & Coding Guidelines
### Version 5.0 — Authoritative Project Standards

This document establishes the software engineering, machine learning rigor, and pedagogical documentation standards for the entire Formula 1 Machine Learning Suite. It guarantees that our codebase is production-grade, mathematically sound, reproducible, and deeply educational.

---

## 1. Pedagogical Commenting & "Explain Like I'm 5" (ELI5) Child-Friendly Code Standards

A core requirement of this project is that **anyone reading the code—from a beginner student to an interviewer—can immediately understand what every line does and WHY it exists**. 

Every file must follow these documentation rules:

### A. The "Why This Matters" Child-Friendly Analogy
Above every complex block, algorithm, or data transformation, provide a plain-English, intuitive explanation (an "ELI5" explanation):
```python
# -------------------------------------------------------------------------
# ELI5 Explanation (Child-Friendly Analogy):
# Imagine you are predicting whether a student passes today's exam.
# You CANNOT use today's exam grade to predict today's exam grade!
# That would be cheating (data leakage). You can only look at tests
# they took yesterday, last week, or last month.
#
# That is why we use .shift(1): it pushes today's race results into the future,
# ensuring the driver's rolling average only uses races that happened BEFORE today.
# -------------------------------------------------------------------------
df["driver_norm_finish_p5"] = (
    df.groupby("driverId")["norm_finish"]
      .transform(lambda s: s.shift(1).rolling(5, min_periods=3).mean())
)
```

### B. First-Principles Formula Breakdown
Do not write complex formulas on a single line without explaining each piece:
- **What is the numerator?** (e.g., "how many races the driver won")
- **What is the denominator?** (e.g., "how many races the driver started")
- **What happens on edge cases like division by zero?** (e.g., "if zero starts, output 0.0 instead of crashing")

### C. Step-by-Step Numbered Sections
Break complex functions into clearly labeled steps:
```python
# Step 1: Find all teammates in the same car for this race
# Step 2: Calculate the average finish of those teammates (excluding this driver!)
# Step 3: Subtract teammate average from driver finish (isolates driver skill from car speed)
```

### D. Mathematical Intuition in Docstrings
Every scratch algorithm (`linear_gd.py`, `kmeans_scratch.py`, `pca_scratch.py`) must include an `Intuition` section in its docstring:
- **Gradient Descent**: "Like rolling a marble down a bowl until it settles at the lowest point."
- **K-Means**: "Like dropping pins on a map and having people walk to their closest pin, then moving each pin to the center of its crowd, repeating until nobody moves."
- **PCA**: "Like shining a flashlight at a 3D object to cast the clearest, most spread-out 2D shadow possible."

---

## 2. Python Code Quality & Style

- **PEP 8 Compliance**: 4-space indentation, descriptive `snake_case` variables/functions, and `PascalCase` classes.
- **Strict Type Annotations**: Every function signature must specify parameter and return types using `typing` (`np.ndarray`, `pd.DataFrame`, `tuple[np.ndarray, np.ndarray]`, `Optional[float]`, etc.).
- **NumPy/Google Docstrings**: Class and function docstrings must document:
  - Intuition & Summary.
  - Parameters (type, shape, description).
  - Returns (type, shape, description).
  - Raises (exceptions and when they trigger).
- **No Hardcoded Magic Numbers**: All constants (e.g., `RANDOM_SEED = 42`, `MIN_CAREER_STARTS = 20`, `SUPERVISED_START_YEAR = 1994`) must be declared at the module top with an explanation of why that specific value was chosen.

---

## 3. Machine Learning & Anti-Leakage Standards

1. **Strict Temporal Partitioning**:
   - Supervised data uses **1994–2024** only.
   - `train_core` (1994–2016), `val` (2017–2018), `test` (2019–2024).
   - Season-aligned split (no race split mid-season).
   - The test set is touched **exactly once** after writing `EXPERIMENT_FREEZE.yaml`.

2. **Pipeline Containment**:
   - `SimpleImputer(strategy='median', add_indicator=True)` and `StandardScaler()` must live **inside** the Scikit-Learn `Pipeline`.
   - Never fit scalers or imputers on the full dataset before splitting or cross-validation.

3. **Feature Whitelist Enforcement**:
   - All models must select inputs from `ALLOWED_FEATURES` only.
   - An assertion in tests must verify that `FORBIDDEN_FEATURES` (race points, laps, milliseconds, status, etc.) are strictly absent.

4. **Race-Aligned Cross-Validation**:
   - Always use `race_time_folds()` which returns positional indices (`np.flatnonzero`).
   - All drivers from the same race must stay in the same fold.
   - Validation folds must strictly succeed training folds in time.

5. **Grid and Qualifying Safety**:
   - `grid == 0` is encoded as `grid_zero_flag = 1` and `grid_effective = field_max + 1` (placed last in field). Never treat 0 as pole!
   - Qualifying gaps are Q1-relative only (`q_gap_pct`), with lap time outliers (>150% median) flagged as NaN.
   - Sprint weekends (2021+) have `grid_vs_quali_delta = NaN`.

---

## 4. Mathematical Rigor & Scratch Algorithms

1. **Vectorization**: Use pure vectorized NumPy matrix operations (`X @ theta`, `np.linalg.eigh`, `np.linalg.lstsq`). Avoid slow Python `for` loops over rows.
2. **Analytical Proof of Stability**:
   - In Gradient Descent, derive the maximum safe learning rate analytically:
     $$\alpha_{\text{safe}} = 0.8 \times \frac{2}{\lambda_{\text{max}}(X^T X / m)}$$
   - Demonstrate divergence empirically with $\alpha_{\text{unsafe}} = 2.2 \times \frac{2}{\lambda_{\text{max}}}$.
3. **Parity Testing**:
   - Scratch Gradient Descent must achieve cost parity with closed-form `lstsq`:
     $$\frac{J_{\text{GD}} - J_{\text{lstsq}}}{\max(1, |J_{\text{lstsq}}|)} < 10^{-6}$$
   - Scratch K-Means must match Scikit-Learn WCSS within $10^{-4}$ given identical initial centroids.
   - Scratch PCA eigenvalues and EVR must match Scikit-Learn within $10^{-6}$ (sign-corrected).

---

## 5. Visualization & Reporting Standards

- **Resolution**: All 11 figures saved at **300 DPI** to `reports/figures/`.
- **Theme**: Professional, publication-ready Seaborn/Matplotlib styling (`sns.set_theme(style="whitegrid")`).
- **Completeness**: Every plot must have clear labels, units, informative titles, legends, and annotated points (e.g., Elbow point, optimal $\alpha$, early-stop epoch).
- **Validation-Only Complexity Curves**: Figure 4 complexity curves must use training and validation curves, **never** test curves.

---

## 6. Audit-First Gate

- No feature code is frozen until the **Table 1 Data Audit** outputs are saved to `data/processed/audit/`:
  - `non_starter_status_ids.json`
  - `sprint_race_ids.json`
  - `q_coverage_by_season.csv`
  - `constructor_canonical_map.json`
  - `table1_audit.csv`
- All downstream code must read these frozen JSON artifacts rather than hardcoding arbitrary guesses.
