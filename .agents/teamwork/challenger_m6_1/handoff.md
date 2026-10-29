# Milestone 6 Challenger 1 Handoff Report: Image Metadata, Pixel Properties, & Data Fidelity Audit

**Author**: Milestone 6 Empirical Challenger 1 (`challenger_m6_1`)  
**Parent Orchestrator**: `baa442d4-0226-4bdc-8a4b-922c0f4e6855`  
**Date**: 2026-10-29  
**Branch**: `feat/ltr-and-hurdle-architecture`  
**Audit Target**: `reports/figures/fig12_ranking_and_hurdle_eval.png` & `reports/tables/table8_ranking_hurdle_benchmark.csv`  
**Gate Evaluation Verdict**: **APPROVE**  

---

## 1. Observation

### 1.1 Authoritative Requirements & Target Specifications
From dispatch instructions and `PROJECT.md` line 152:
- Image file: `reports/figures/fig12_ranking_and_hurdle_eval.png`.
- Format: PNG.
- Dimensions: Exactly 4800x3600 pixels (16x12 inches at 300 DPI).
- DPI metadata: Strictly 300 DPI.
- File size: Non-zero, non-corrupted, reasonable size (~1 MB).
- Content integrity: 4 quadrants corresponding to Panels A, B, C, D must have active, varied, non-solid pixels (axes, titles, labels, curves, bars, legends; no empty or broken axes).
- Numerical consistency: Metrics plotted in Panel A (NDCG@K) and Panel C (Podium F1 and deltas) must match `reports/tables/table8_ranking_hurdle_benchmark.csv`.

### 1.2 Empirical Metadata & File Properties Audit
Executed inspection via Pillow (`PIL.Image`) on `reports/figures/fig12_ranking_and_hurdle_eval.png`:
- **Format**: `PNG`
- **Color Mode**: `RGBA` (4 channels, 8 bits/channel)
- **Pixel Dimensions**: `(4800, 3600)` (width: 4800 px, height: 3600 px)
- **File Size**: `1,063,071 bytes` (`1038.16 KB` / ~1.01 MB)
- **Embedded DPI Metadata**: `im.info['dpi'] == (299.9994, 299.9994)` (`int(round(dpi[0])) == 300`), matching Matplotlib's 300 DPI standard pHYs physical meter chunk ($11811\text{ px/m} \times 0.0254\text{ m/in} = 299.9994\text{ DPI}$).
- **Software Metadata**: `Matplotlib version3.10.3, https://matplotlib.org/`

### 1.3 Empirical Quadrant Pixel Integrity Audit
Split image array `arr` (`shape: (3600, 4800, 4)`) into four quadrants:
1. **Panel A (Top-Left, [0:1800, 0:2400])**:
   - Mean RGB: `[216.66, 223.62, 207.34]`, Std RGB: `[76.28, 51.97, 81.34]`, Overall Std: `71.34`.
   - Non-white pixels: `31.18%` (`1,347,005` pixels).
   - Unique RGB colors: `296` (vector-rendered bar chart with anti-aliasing blends).
   - Status: Active bar chart, non-blank, fully rendered.
2. **Panel B (Top-Right, [0:1800, 2400:4800])**:
   - Mean RGB: `[248.13, 247.54, 247.55]`, Std RGB: `[33.54, 35.53, 35.52]`, Overall Std: `34.88`.
   - Non-white pixels: `6.75%` (`291,685` pixels).
   - Unique RGB colors: `555` (calibration curve, diagonal reference line, square markers, text box, grid lines).
   - Status: Active calibration curve, non-blank, fully rendered.
3. **Panel C (Bottom-Left, [1800:3600, 0:2400])**:
   - Mean RGB: `[217.77, 232.87, 230.45]`, Std RGB: `[73.99, 46.28, 53.97]`, Overall Std: `59.61`.
   - Non-white pixels: `24.87%` (`1,074,387` pixels).
   - Unique RGB colors: `2,034` (hatched pattern bars, solid bars, numerical text labels, axes).
   - Status: Active comparison chart, non-blank, fully rendered.
4. **Panel D (Bottom-Right, [1800:3600, 2400:4800])**:
   - Mean RGB: `[245.69, 243.92, 243.30]`, Std RGB: `[38.30, 36.40, 38.93]`, Overall Std: `37.90`.
   - Non-white pixels: `22.32%` (`964,012` pixels).
   - Unique RGB colors: `3,974` (shaded red error span, multi-colored KDE curves, text box, grid lines).
   - Status: Active distribution plot, non-blank, fully rendered.

### 1.4 Empirical Numerical Consistency with Table 8
Intercepted matplotlib artists directly from `plot_fig12_ranking_and_hurdle_eval` execution and verified against `reports/tables/table8_ranking_hurdle_benchmark.csv`:

#### Panel A (Race-Aligned NDCG@K Comparison):
- Total bar patches: Exactly 24 (8 models $\times$ 3 bars: K=1, 3, 10).
- Comparison of rendered bar height vs Table 8 values:
  1. Grid Heuristic: NDCG@1 = 0.6271 (exp: 0.6271), NDCG@3 = 0.7315 (exp: 0.7315), NDCG@10 = 0.8156 (exp: 0.8156) [PASS, $\Delta = 0.0000$]
  2. KNN: NDCG@1 = 0.6149 (exp: 0.6149), NDCG@3 = 0.7263 (exp: 0.7263), NDCG@10 = 0.8205 (exp: 0.8205) [PASS, $\Delta = 0.0000$]
  3. Decision Tree: NDCG@1 = 0.8608 (exp: 0.8608), NDCG@3 = 0.8086 (exp: 0.8086), NDCG@10 = 0.8958 (exp: 0.8958) [PASS, $\Delta = 0.0000$]
  4. SVM (RBF): NDCG@1 = 0.5672 (exp: 0.5672), NDCG@3 = 0.6973 (exp: 0.6973), NDCG@10 = 0.8035 (exp: 0.8035) [PASS, $\Delta = 0.0000$]
  5. Dual-Branch 1D-CNN: NDCG@1 = 0.6706 (exp: 0.6706), NDCG@3 = 0.7347 (exp: 0.7347), NDCG@10 = 0.8374 (exp: 0.8374) [PASS, $\Delta = 0.0000$]
  6. LambdaMART (linear_rank): NDCG@1 = 0.8063 (exp: 0.8063), NDCG@3 = 0.7731 (exp: 0.7731), NDCG@10 = 0.8618 (exp: 0.8618) [PASS, $\Delta = 0.0000$]
  7. Plackett-Luce Neural Ranker: NDCG@1 = 0.6955 (exp: 0.6955), NDCG@3 = 0.7600 (exp: 0.7600), NDCG@10 = 0.8446 (exp: 0.8446) [PASS, $\Delta = 0.0000$]
  8. Two-Stage Hurdle: NDCG@1 = 0.6486 (exp: 0.6486), NDCG@3 = 0.7417 (exp: 0.7417), NDCG@10 = 0.8305 (exp: 0.8305) [PASS, $\Delta = 0.0000$]

#### Panel C (Podium F1 Comparison & Ranking Penalty Deltas):
- Total bar patches: Exactly 12 (4 static + 4 top-3 + 4 tournament).
- Pointwise static vs top-3:
  1. KNN: Static F1 = 0.6840 (exp: 0.6840), Top-3 F1 = 0.6641 (exp: 0.6641), Rendered text delta: `-0.020` [PASS]
  2. Decision Tree: Static F1 = 0.6803 (exp: 0.6803), Top-3 F1 = 0.6849 (exp: 0.6849), Rendered text delta: `+0.005` [PASS]
  3. SVM (RBF): Static F1 = 0.6709 (exp: 0.6709), Top-3 F1 = 0.6562 (exp: 0.6562), Rendered text delta: `-0.015` [PASS]
  4. 1D-CNN: Static F1 = 0.6791 (exp: 0.6791), Top-3 F1 = 0.6615 (exp: 0.6615), Rendered text delta: `-0.018` [PASS]
- Tournament / Hurdle models:
  1. LambdaMART (linear_rank): F1 = 0.6771 (exp: 0.6771), Rendered text score: `0.6771` [PASS]
  2. Plackett-Luce Neural Ranker: F1 = 0.6823 (exp: 0.6823), Rendered text score: `0.6823` [PASS]
  3. Two-Stage Hurdle: F1 = 0.6693 (exp: 0.6693), Rendered text score: `0.6693` [PASS]
  4. Two-Stage Hurdle (Exp Pos): F1 = 0.6745 (exp: 0.6745), Rendered text score: `0.6745` [PASS]

#### Panel B & Panel D Computed Values:
- Panel B: Brier score loss = `0.1281` (computed on test set `is_dnf` vs `predict_hazard`), matching info box and Table 8 Hurdle setup. Observed DNF rate = `14.5%`, Mean predicted hazard = `20.2%`.
- Panel D:
  - Standard OLS $R^2$ (All Data) = `0.412`, OLS $R^2$ (Classified Starters) = `0.5002`.
  - Two-Stage Hurdle Stage 2 Pace $R^2$ (Classified Starters) = `0.6587` (exactly matches Table 8 `Test_R2_Classified` column = `0.6587`).
  - Two-Stage Hurdle $\mathbb{E}[\text{Pos}] R^2$ (All Data) = `0.433` (exactly matches Table 8 `Test_R2_All` column = `0.4326`).

### 1.5 Automated Regression Test Suite
Executed `pytest tests/ -v --tb=short`:
- Result: **76 passed in 35.47s** (100% pass rate, 0 failures, 0 warnings).

---

## 2. Logic Chain

1. **Premise 1 (Image Spec Adherence)**: Observation 1.2 confirms through PIL inspection that `reports/figures/fig12_ranking_and_hurdle_eval.png` is PNG, exactly 4800x3600 pixels, exactly 300 DPI (299.9994 DPI unrounded), and 1038.16 KB.
2. **Premise 2 (Panel Rendering Integrity)**: Observation 1.3 proves that every quadrant contains substantial graphical content (between 6.75% and 31.18% non-background pixels, color std dev between 34.88 and 71.34, hundreds to thousands of unique colors). No quadrant is solid, blank, missing axes, or corrupted.
3. **Premise 3 (Numerical Grounding in Table 8)**: Observation 1.4 confirms through direct matplotlib artist inspection that all 24 bars in Panel A and all 12 bars in Panel C match the values in `reports/tables/table8_ranking_hurdle_benchmark.csv` within $1 \times 10^{-5}$. Furthermore, the delta annotations for pointwise ranking penalties and tournament scores in Panel C match Table 8 calculations exactly.
4. **Premise 4 (System Stability & Compatibility)**: Observation 1.5 demonstrates that the entire test suite (76 tests covering ranking metrics, LTR, Hurdle, and all 12 figures) passes with zero errors or regressions.

Therefore, the deliverable satisfies all acceptance criteria without defect.

---

## 3. Adversarial Challenge Report

### Challenge Summary
**Overall risk assessment**: **LOW**

### Challenges Evaluated

#### Challenge 1: DPI Metadata Precision (Matplotlib floating-point representation)
- **Assumption challenged**: Embedded DPI metadata must strictly be 300 DPI without truncation or corruption.
- **Attack scenario**: Matplotlib stores DPI via PNG pHYs chunk as pixels per meter (`11811 px/m`). When loaded back into Pillow, `im.info['dpi']` yields `(299.9994, 299.9994)`. Could downstream ingestion pipelines requiring exact integer 300 fail?
- **Blast radius**: Low. Standard image readers round `dpi` to the nearest integer (`int(round(dpi[0])) == 300`).
- **Mitigation verified**: `test_plots.py` tests both `int(round(dpi[0])) == 300` and validates resolution $4800 \times 3600$ at $16 \times 12$ inches, perfectly preserving physical print dimensions.

#### Challenge 2: Panel A Bar Ordering vs Legend Correspondence
- **Assumption challenged**: Grouped bars in Panel A correctly map left-to-right to NDCG@1, NDCG@3, and NDCG@10 respectively.
- **Attack scenario**: If bar offsets `(x - w, x, x + w)` were misaligned with the legend colors (`#1f77b4`, `#ff7f0e`, `#2ca02c`), viewers would misinterpret ranking quality across K.
- **Blast radius**: Medium (metric inversion).
- **Stress test result**: Verified `ax_a.patches` order: first 8 patches correspond to `x - w` with height equal to NDCG@1, middle 8 patches to `x` with height equal to NDCG@3, and final 8 patches to `x + w` with height equal to NDCG@10. All heights match Table 8.

#### Challenge 3: Pointwise Ranking Penalty Delta Signs in Panel C
- **Assumption challenged**: Ranking penalty $\Delta = \text{F1}_{\text{Top3}} - \text{F1}_{\text{Static}}$ correctly reflects sign and magnitude.
- **Attack scenario**: KNN drops from 0.6840 to 0.6641 ($\Delta = -0.0199 \to -0.020$). Decision Tree rises from 0.6803 to 0.6849 ($\Delta = +0.0046 \to +0.005$). An inverted subtraction would show a false penalty for Decision Tree or false gain for KNN.
- **Blast radius**: Medium.
- **Stress test result**: Inspected rendered text strings in `ax_c.texts`. The rendered strings are `'-0.020'`, `'+0.005'`, `'-0.015'`, and `'-0.018'`, matching mathematical deltas and color-coded appropriately (green for positive, red for negative).

### Stress Test Results Table
| Scenario / Test | Expected Behavior | Actual Behavior | Result |
|---|---|---|---|
| PNG format verification | `im.format == 'PNG'` | `'PNG'` | **PASS** |
| Resolution verification | `im.size == (4800, 3600)` | `(4800, 3600)` | **PASS** |
| DPI metadata verification | `int(round(dpi[0])) == 300` | `300` (`299.9994`) | **PASS** |
| File size check | Non-zero, ~1 MB | `1038.16 KB` (1,063,071 B) | **PASS** |
| Quadrant A (NDCG@K) content | Non-white > 2%, Std > 10 | Non-white: 31.18%, Std: 71.34 | **PASS** |
| Quadrant B (Calibration) content | Non-white > 2%, Std > 10 | Non-white: 6.75%, Std: 34.88 | **PASS** |
| Quadrant C (Podium F1) content | Non-white > 2%, Std > 10 | Non-white: 24.87%, Std: 59.61 | **PASS** |
| Quadrant D (Residual KDE) content | Non-white > 2%, Std > 10 | Non-white: 22.32%, Std: 37.90 | **PASS** |
| Panel A data vs Table 8 | Max absolute diff < 1e-5 across 24 bars | Max diff: 0.0000 | **PASS** |
| Panel C data vs Table 8 | Max absolute diff < 1e-5 across 12 bars | Max diff: 0.0000 | **PASS** |
| Full pytest test suite | 76 passed, 0 failures | 76 passed in 35.47s | **PASS** |

### Unchallenged Areas
- Subjective font styling and color palette aesthetics (conforms to seaborn/matplotlib publication standards; within design discretion).

---

## 4. Caveats
- No caveats. The audit was conducted empirically with multiple automated scripts and direct inspection of memory artists and physical image assets.

---

## 5. Conclusion

**Verdict: APPROVE**

Milestone 6 (Figure 12 Diagnostic Visualizations) satisfies all image format, dimensional, resolution, pixel integrity, and numerical fidelity requirements:
1. `reports/figures/fig12_ranking_and_hurdle_eval.png` exists, is a valid PNG, measures exactly 4800x3600 pixels (16x12 inches at 300 DPI), carries 300 DPI metadata, and is 1038.16 KB.
2. All 4 panels are active, contain dense non-white pixel graphics, and show zero empty or corrupted subplots.
3. All plotted values in Panel A (NDCG@K) and Panel C (Podium F1 and ranking penalties) match `reports/tables/table8_ranking_hurdle_benchmark.csv` to 4 decimal places with zero discrepancies.
4. All 76 tests in the project test suite pass with 100% success rate.

---

## 6. Verification Method

To independently verify this evaluation, run the following commands in powershell:

1. **Verify Image Metadata & DPI**:
   ```powershell
   python -c "
   from PIL import Image
   with Image.open('reports/figures/fig12_ranking_and_hurdle_eval.png') as im:
       assert im.format == 'PNG'
       assert im.size == (4800, 3600)
       assert int(round(im.info['dpi'][0])) == 300
       print('Image metadata audit: PASSED')
   "
   ```

2. **Verify Quadrant Pixel Variances**:
   ```powershell
   python -c "
   from PIL import Image
   import numpy as np
   img = Image.open('reports/figures/fig12_ranking_and_hurdle_eval.png')
   arr = np.array(img)
   quads = [arr[0:1800, 0:2400], arr[0:1800, 2400:4800], arr[1800:3600, 0:2400], arr[1800:3600, 2400:4800]]
   for idx, q in enumerate(quads):
       rgb = q[:, :, :3]
       white_mask = (rgb > 250).all(axis=2)
       non_white_pct = (1.0 - white_mask.mean()) * 100.0
       assert non_white_pct > 5.0 and rgb.std() > 20.0
   print('Quadrant pixel integrity audit: PASSED')
   "
   ```

3. **Verify Full Automated Test Suite**:
   ```powershell
   pytest tests/ -v --tb=short
   ```
   *Expected output*: `76 passed` in ~35s.

*Invalidation conditions*:
- Any discrepancy between Table 8 values and Figure 12 bars exceeding $10^{-4}$.
- Any panel having empty axes or $< 5\%$ non-background pixels.
- Any failure in `pytest tests/`.
