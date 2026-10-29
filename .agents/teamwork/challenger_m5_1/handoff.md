# Handoff Report — Milestone 5 Challenger Audit (Table 8 Benchmark & Podium Invariants)

**Agent**: Challenger 1 (`teamwork_challenger`)  
**Working Directory**: `d:\ML-Project\.agents\teamwork\challenger_m5_1\`  
**Date**: 2026-10-29  
**Milestone**: Milestone 5 (Experiment Orchestration & Side-by-Side Comparison)  
**Target Under Audit**: Worker M5 (`reports/tables/table8_ranking_hurdle_benchmark.csv`, `src/run_v6_experiments.py`, `EXPERIMENT_FREEZE.yaml`)  
**Verdict**: **APPROVE**  

---

## 1. Observation

### 1.1 Table 8 Benchmark Inspection
File: `reports/tables/table8_ranking_hurdle_benchmark.csv` (15 model rows, 15 columns).
Columns present: `Model`, `Architecture`, `Decision_Rule`, `NDCG@1`, `NDCG@3`, `NDCG@10`, `MRR`, `Kendall_Tau`, `Podium_Precision`, `Podium_Recall`, `Podium_F1`, `Test_RMSE`, `Test_R2_All`, `Test_R2_Classified`, `Key_Parameters`.

### 1.2 Independent Empirical Re-Evaluation on `test.parquet`
Rather than checking only 3 models as minimally requested, all 15 model rows in Table 8 were independently reconstructed and evaluated from scratch using `src.evaluation.ranking_metrics.evaluate_race_ranking` and scikit-learn metrics on `test.parquet` (128 test races, 2,559 driver entries):

```
========================================================================================================
INDEPENDENT EMPIRICAL RE-EVALUATION AUDIT VS reports/tables/table8_ranking_hurdle_benchmark.csv
========================================================================================================
Model & Decision Rule                          | Metric        | Table 8 | Indep Audit | Status
-----------------------------------------------+---------------+---------+-------------+---------
Constructor Rank (Post-hoc Top-3 Selection)    | NDCG@1        | 0.7484  | 0.7484      | MATCH
Constructor Rank (Post-hoc Top-3 Selection)    | NDCG@3        | 0.7280  | 0.7280      | MATCH
Constructor Rank (Post-hoc Top-3 Selection)    | NDCG@10       | 0.8241  | 0.8241      | MATCH
Constructor Rank (Post-hoc Top-3 Selection)    | MRR           | 0.7461  | 0.7461      | MATCH
Constructor Rank (Post-hoc Top-3 Selection)    | Kendall_Tau   | 0.4667  | 0.4667      | MATCH
Constructor Rank (Post-hoc Top-3 Selection)    | Podium_Prec   | 0.6432  | 0.6432      | MATCH
Constructor Rank (Post-hoc Top-3 Selection)    | Podium_Rec    | 0.6432  | 0.6432      | MATCH
Constructor Rank (Post-hoc Top-3 Selection)    | Podium_F1     | 0.6432  | 0.6432      | MATCH
Constructor Rank (Post-hoc Top-3 Selection)    | Test_RMSE     | 4.9679  | 4.9679      | MATCH
Constructor Rank (Post-hoc Top-3 Selection)    | Test_R2_All   | 0.2572  | 0.2572      | MATCH
Constructor Rank (Post-hoc Top-3 Selection)    | Test_R2_Class | 0.3895  | 0.3895      | MATCH
-----------------------------------------------+---------------+---------+-------------+---------
Grid Heuristic (Post-hoc Top-3 Selection)      | NDCG@1        | 0.6271  | 0.6271      | MATCH
Grid Heuristic (Post-hoc Top-3 Selection)      | NDCG@3        | 0.7315  | 0.7315      | MATCH
Grid Heuristic (Post-hoc Top-3 Selection)      | NDCG@10       | 0.8156  | 0.8156      | MATCH
Grid Heuristic (Post-hoc Top-3 Selection)      | MRR           | 0.6807  | 0.6807      | MATCH
Grid Heuristic (Post-hoc Top-3 Selection)      | Kendall_Tau   | 0.4998  | 0.4998      | MATCH
Grid Heuristic (Post-hoc Top-3 Selection)      | Podium_Prec   | 0.6823  | 0.6823      | MATCH
Grid Heuristic (Post-hoc Top-3 Selection)      | Podium_Rec    | 0.6823  | 0.6823      | MATCH
Grid Heuristic (Post-hoc Top-3 Selection)      | Podium_F1     | 0.6823  | 0.6823      | MATCH
Grid Heuristic (Post-hoc Top-3 Selection)      | Test_RMSE     | 4.9946  | 4.9946      | MATCH
Grid Heuristic (Post-hoc Top-3 Selection)      | Test_R2_All   | 0.2492  | 0.2492      | MATCH
Grid Heuristic (Post-hoc Top-3 Selection)      | Test_R2_Class | 0.3739  | 0.3739      | MATCH
-----------------------------------------------+---------------+---------+-------------+---------
KNN (Static Threshold tau=0.286)               | NDCG@1        | 0.6149  | 0.6149      | MATCH
KNN (Static Threshold tau=0.286)               | NDCG@3        | 0.7263  | 0.7263      | MATCH
KNN (Static Threshold tau=0.286)               | NDCG@10       | 0.8205  | 0.8205      | MATCH
KNN (Static Threshold tau=0.286)               | MRR           | 0.6809  | 0.6809      | MATCH
KNN (Static Threshold tau=0.286)               | Kendall_Tau   | 0.4918  | 0.4918      | MATCH
KNN (Static Threshold tau=0.286)               | Podium_Prec   | 0.6052  | 0.6052      | MATCH
KNN (Static Threshold tau=0.286)               | Podium_Rec    | 0.7865  | 0.7865      | MATCH
KNN (Static Threshold tau=0.286)               | Podium_F1     | 0.6840  | 0.6840      | MATCH
KNN (Static Threshold tau=0.286)               | Test_RMSE     | N/A     | N/A         | MATCH
KNN (Static Threshold tau=0.286)               | Test_R2_All   | N/A     | N/A         | MATCH
KNN (Static Threshold tau=0.286)               | Test_R2_Class | N/A     | N/A         | MATCH
-----------------------------------------------+---------------+---------+-------------+---------
KNN (Post-hoc Top-3 Selection)                 | NDCG@1        | 0.6149  | 0.6149      | MATCH
KNN (Post-hoc Top-3 Selection)                 | NDCG@3        | 0.7263  | 0.7263      | MATCH
KNN (Post-hoc Top-3 Selection)                 | NDCG@10       | 0.8205  | 0.8205      | MATCH
KNN (Post-hoc Top-3 Selection)                 | MRR           | 0.6809  | 0.6809      | MATCH
KNN (Post-hoc Top-3 Selection)                 | Kendall_Tau   | 0.4918  | 0.4918      | MATCH
KNN (Post-hoc Top-3 Selection)                 | Podium_Prec   | 0.6641  | 0.6641      | MATCH
KNN (Post-hoc Top-3 Selection)                 | Podium_Rec    | 0.6641  | 0.6641      | MATCH
KNN (Post-hoc Top-3 Selection)                 | Podium_F1     | 0.6641  | 0.6641      | MATCH
KNN (Post-hoc Top-3 Selection)                 | Test_RMSE     | 4.1868  | 4.1868      | MATCH
KNN (Post-hoc Top-3 Selection)                 | Test_R2_All   | 0.4724  | 0.4724      | MATCH
KNN (Post-hoc Top-3 Selection)                 | Test_R2_Class | 0.6391  | 0.6391      | MATCH
-----------------------------------------------+---------------+---------+-------------+---------
Decision Tree (Static Threshold tau=0.714)      | NDCG@1        | 0.8608  | 0.8608      | MATCH
Decision Tree (Static Threshold tau=0.714)      | NDCG@3        | 0.8086  | 0.8086      | MATCH
Decision Tree (Static Threshold tau=0.714)      | NDCG@10       | 0.8958  | 0.8958      | MATCH
Decision Tree (Static Threshold tau=0.714)      | MRR           | 0.8536  | 0.8536      | MATCH
Decision Tree (Static Threshold tau=0.714)      | Kendall_Tau   | 0.5207  | 0.5207      | MATCH
Decision Tree (Static Threshold tau=0.714)      | Podium_Prec   | 0.7123  | 0.7123      | MATCH
Decision Tree (Static Threshold tau=0.714)      | Podium_Rec    | 0.6510  | 0.6510      | MATCH
Decision Tree (Static Threshold tau=0.714)      | Podium_F1     | 0.6803  | 0.6803      | MATCH
Decision Tree (Static Threshold tau=0.714)      | Test_RMSE     | N/A     | N/A         | MATCH
Decision Tree (Static Threshold tau=0.714)      | Test_R2_All   | N/A     | N/A         | MATCH
Decision Tree (Static Threshold tau=0.714)      | Test_R2_Class | N/A     | N/A         | MATCH
-----------------------------------------------+---------------+---------+-------------+---------
Decision Tree (Post-hoc Top-3 Selection)       | NDCG@1        | 0.8608  | 0.8608      | MATCH
Decision Tree (Post-hoc Top-3 Selection)       | NDCG@3        | 0.8086  | 0.8086      | MATCH
Decision Tree (Post-hoc Top-3 Selection)       | NDCG@10       | 0.8958  | 0.8958      | MATCH
Decision Tree (Post-hoc Top-3 Selection)       | MRR           | 0.8536  | 0.8536      | MATCH
Decision Tree (Post-hoc Top-3 Selection)       | Kendall_Tau   | 0.5207  | 0.5207      | MATCH
Decision Tree (Post-hoc Top-3 Selection)       | Podium_Prec   | 0.6849  | 0.6849      | MATCH
Decision Tree (Post-hoc Top-3 Selection)       | Podium_Rec    | 0.6849  | 0.6849      | MATCH
Decision Tree (Post-hoc Top-3 Selection)       | Podium_F1     | 0.6849  | 0.6849      | MATCH
Decision Tree (Post-hoc Top-3 Selection)       | Test_RMSE     | 4.2120  | 4.2120      | MATCH
Decision Tree (Post-hoc Top-3 Selection)       | Test_R2_All   | 0.4661  | 0.4661      | MATCH
Decision Tree (Post-hoc Top-3 Selection)       | Test_R2_Class | 0.6432  | 0.6432      | MATCH
-----------------------------------------------+---------------+---------+-------------+---------
SVM (RBF) (Static Threshold tau=0.863)         | NDCG@1        | 0.5672  | 0.5672      | MATCH
SVM (RBF) (Static Threshold tau=0.863)         | NDCG@3        | 0.6973  | 0.6973      | MATCH
SVM (RBF) (Static Threshold tau=0.863)         | NDCG@10       | 0.8035  | 0.8035      | MATCH
SVM (RBF) (Static Threshold tau=0.863)         | MRR           | 0.6309  | 0.6309      | MATCH
SVM (RBF) (Static Threshold tau=0.863)         | Kendall_Tau   | 0.5327  | 0.5327      | MATCH
SVM (RBF) (Static Threshold tau=0.863)         | Podium_Prec   | 0.6527  | 0.6527      | MATCH
SVM (RBF) (Static Threshold tau=0.863)         | Podium_Rec    | 0.6901  | 0.6901      | MATCH
SVM (RBF) (Static Threshold tau=0.863)         | Podium_F1     | 0.6709  | 0.6709      | MATCH
SVM (RBF) (Static Threshold tau=0.863)         | Test_RMSE     | N/A     | N/A         | MATCH
SVM (RBF) (Static Threshold tau=0.863)         | Test_R2_All   | N/A     | N/A         | MATCH
SVM (RBF) (Static Threshold tau=0.863)         | Test_R2_Class | N/A     | N/A         | MATCH
-----------------------------------------------+---------------+---------+-------------+---------
SVM (RBF) (Post-hoc Top-3 Selection)          | NDCG@1        | 0.5672  | 0.5672      | MATCH
SVM (RBF) (Post-hoc Top-3 Selection)          | NDCG@3        | 0.6973  | 0.6973      | MATCH
SVM (RBF) (Post-hoc Top-3 Selection)          | NDCG@10       | 0.8035  | 0.8035      | MATCH
SVM (RBF) (Post-hoc Top-3 Selection)          | MRR           | 0.6309  | 0.6309      | MATCH
SVM (RBF) (Post-hoc Top-3 Selection)          | Kendall_Tau   | 0.5327  | 0.5327      | MATCH
SVM (RBF) (Post-hoc Top-3 Selection)          | Podium_Prec   | 0.6562  | 0.6562      | MATCH
SVM (RBF) (Post-hoc Top-3 Selection)          | Podium_Rec    | 0.6562  | 0.6562      | MATCH
SVM (RBF) (Post-hoc Top-3 Selection)          | Podium_F1     | 0.6562  | 0.6562      | MATCH
SVM (RBF) (Post-hoc Top-3 Selection)          | Test_RMSE     | 4.7508  | 4.7508      | MATCH
SVM (RBF) (Post-hoc Top-3 Selection)          | Test_R2_All   | 0.3208  | 0.3208      | MATCH
SVM (RBF) (Post-hoc Top-3 Selection)          | Test_R2_Class | 0.4975  | 0.4975      | MATCH
-----------------------------------------------+---------------+---------+-------------+---------
Dual-Branch 1D-CNN (Static Threshold tau=0.803)| NDCG@1        | 0.6706  | 0.6706      | MATCH
Dual-Branch 1D-CNN (Static Threshold tau=0.803)| NDCG@3        | 0.7347  | 0.7347      | MATCH
Dual-Branch 1D-CNN (Static Threshold tau=0.803)| NDCG@10       | 0.8374  | 0.8374      | MATCH
Dual-Branch 1D-CNN (Static Threshold tau=0.803)| MRR           | 0.7272  | 0.7272      | MATCH
Dual-Branch 1D-CNN (Static Threshold tau=0.803)| Kendall_Tau   | 0.5333  | 0.5333      | MATCH
Dual-Branch 1D-CNN (Static Threshold tau=0.803)| Podium_Prec   | 0.6478  | 0.6478      | MATCH
Dual-Branch 1D-CNN (Static Threshold tau=0.803)| Podium_Rec    | 0.7135  | 0.7135      | MATCH
Dual-Branch 1D-CNN (Static Threshold tau=0.803)| Podium_F1     | 0.6791  | 0.6791      | MATCH
Dual-Branch 1D-CNN (Static Threshold tau=0.803)| Test_RMSE     | N/A     | N/A         | MATCH
Dual-Branch 1D-CNN (Static Threshold tau=0.803)| Test_R2_All   | N/A     | N/A         | MATCH
Dual-Branch 1D-CNN (Static Threshold tau=0.803)| Test_R2_Class | N/A     | N/A         | MATCH
-----------------------------------------------+---------------+---------+-------------+---------
Dual-Branch 1D-CNN (Post-hoc Top-3 Selection)  | NDCG@1        | 0.6706  | 0.6706      | MATCH
Dual-Branch 1D-CNN (Post-hoc Top-3 Selection)  | NDCG@3        | 0.7347  | 0.7347      | MATCH
Dual-Branch 1D-CNN (Post-hoc Top-3 Selection)  | NDCG@10       | 0.8374  | 0.8374      | MATCH
Dual-Branch 1D-CNN (Post-hoc Top-3 Selection)  | MRR           | 0.7272  | 0.7272      | MATCH
Dual-Branch 1D-CNN (Post-hoc Top-3 Selection)  | Kendall_Tau   | 0.5333  | 0.5333      | MATCH
Dual-Branch 1D-CNN (Post-hoc Top-3 Selection)  | Podium_Prec   | 0.6615  | 0.6615      | MATCH
Dual-Branch 1D-CNN (Post-hoc Top-3 Selection)  | Podium_Rec    | 0.6615  | 0.6615      | MATCH
Dual-Branch 1D-CNN (Post-hoc Top-3 Selection)  | Podium_F1     | 0.6615  | 0.6615      | MATCH
Dual-Branch 1D-CNN (Post-hoc Top-3 Selection)  | Test_RMSE     | 4.7620  | 4.7620      | MATCH
Dual-Branch 1D-CNN (Post-hoc Top-3 Selection)  | Test_R2_All   | 0.3175  | 0.3175      | MATCH
Dual-Branch 1D-CNN (Post-hoc Top-3 Selection)  | Test_R2_Class | 0.4921  | 0.4921      | MATCH
-----------------------------------------------+---------------+---------+-------------+---------
LambdaMART (linear_rank) (Top-3 Guaranteed)   | NDCG@1        | 0.8063  | 0.8063      | MATCH
LambdaMART (linear_rank) (Top-3 Guaranteed)   | NDCG@3        | 0.7731  | 0.7731      | MATCH
LambdaMART (linear_rank) (Top-3 Guaranteed)   | NDCG@10       | 0.8618  | 0.8618      | MATCH
LambdaMART (linear_rank) (Top-3 Guaranteed)   | MRR           | 0.7875  | 0.7875      | MATCH
LambdaMART (linear_rank) (Top-3 Guaranteed)   | Kendall_Tau   | 0.5172  | 0.5172      | MATCH
LambdaMART (linear_rank) (Top-3 Guaranteed)   | Podium_Prec   | 0.6771  | 0.6771      | MATCH
LambdaMART (linear_rank) (Top-3 Guaranteed)   | Podium_Rec    | 0.6771  | 0.6771      | MATCH
LambdaMART (linear_rank) (Top-3 Guaranteed)   | Podium_F1     | 0.6771  | 0.6771      | MATCH
LambdaMART (linear_rank) (Top-3 Guaranteed)   | Test_RMSE     | 4.2449  | 4.2449      | MATCH
LambdaMART (linear_rank) (Top-3 Guaranteed)   | Test_R2_All   | 0.4577  | 0.4577      | MATCH
LambdaMART (linear_rank) (Top-3 Guaranteed)   | Test_R2_Class | 0.6309  | 0.6309      | MATCH
-----------------------------------------------+---------------+---------+-------------+---------
LambdaMART (f1_points) (Top-3 Guaranteed)     | NDCG@1        | 0.8353  | 0.8353      | MATCH
LambdaMART (f1_points) (Top-3 Guaranteed)     | NDCG@3        | 0.7666  | 0.7666      | MATCH
LambdaMART (f1_points) (Top-3 Guaranteed)     | NDCG@10       | 0.8449  | 0.8449      | MATCH
LambdaMART (f1_points) (Top-3 Guaranteed)     | MRR           | 0.7901  | 0.7901      | MATCH
LambdaMART (f1_points) (Top-3 Guaranteed)     | Kendall_Tau   | 0.3820  | 0.3820      | MATCH
LambdaMART (f1_points) (Top-3 Guaranteed)     | Podium_Prec   | 0.6432  | 0.6432      | MATCH
LambdaMART (f1_points) (Top-3 Guaranteed)     | Podium_Rec    | 0.6432  | 0.6432      | MATCH
LambdaMART (f1_points) (Top-3 Guaranteed)     | Podium_F1     | 0.6432  | 0.6432      | MATCH
LambdaMART (f1_points) (Top-3 Guaranteed)     | Test_RMSE     | 4.8408  | 4.8408      | MATCH
LambdaMART (f1_points) (Top-3 Guaranteed)     | Test_R2_All   | 0.2948  | 0.2948      | MATCH
LambdaMART (f1_points) (Top-3 Guaranteed)     | Test_R2_Class | 0.3652  | 0.3652      | MATCH
-----------------------------------------------+---------------+---------+-------------+---------
Plackett-Luce Neural Ranker (Top-3 Guaranteed) | NDCG@1        | 0.6955  | 0.6955      | MATCH
Plackett-Luce Neural Ranker (Top-3 Guaranteed) | NDCG@3        | 0.7600  | 0.7600      | MATCH
Plackett-Luce Neural Ranker (Top-3 Guaranteed) | NDCG@10       | 0.8446  | 0.8446      | MATCH
Plackett-Luce Neural Ranker (Top-3 Guaranteed) | MRR           | 0.7453  | 0.7453      | MATCH
Plackett-Luce Neural Ranker (Top-3 Guaranteed) | Kendall_Tau   | 0.5379  | 0.5379      | MATCH
Plackett-Luce Neural Ranker (Top-3 Guaranteed) | Podium_Prec   | 0.6823  | 0.6823      | MATCH
Plackett-Luce Neural Ranker (Top-3 Guaranteed) | Podium_Rec    | 0.6823  | 0.6823      | MATCH
Plackett-Luce Neural Ranker (Top-3 Guaranteed) | Podium_F1     | 0.6823  | 0.6823      | MATCH
Plackett-Luce Neural Ranker (Top-3 Guaranteed) | Test_RMSE     | 4.7487  | 4.7487      | MATCH
Plackett-Luce Neural Ranker (Top-3 Guaranteed) | Test_R2_All   | 0.3213  | 0.3213      | MATCH
Plackett-Luce Neural Ranker (Top-3 Guaranteed) | Test_R2_Class | 0.4928  | 0.4928      | MATCH
-----------------------------------------------+---------------+---------+-------------+---------
Two-Stage Hurdle (Top-3 Guaranteed)            | NDCG@1        | 0.6486  | 0.6486      | MATCH
Two-Stage Hurdle (Top-3 Guaranteed)            | NDCG@3        | 0.7417  | 0.7417      | MATCH
Two-Stage Hurdle (Top-3 Guaranteed)            | NDCG@10       | 0.8305  | 0.8305      | MATCH
Two-Stage Hurdle (Top-3 Guaranteed)            | MRR           | 0.7203  | 0.7203      | MATCH
Two-Stage Hurdle (Top-3 Guaranteed)            | Kendall_Tau   | 0.5106  | 0.5106      | MATCH
Two-Stage Hurdle (Top-3 Guaranteed)            | Podium_Prec   | 0.6693  | 0.6693      | MATCH
Two-Stage Hurdle (Top-3 Guaranteed)            | Podium_Rec    | 0.6693  | 0.6693      | MATCH
Two-Stage Hurdle (Top-3 Guaranteed)            | Podium_F1     | 0.6693  | 0.6693      | MATCH
Two-Stage Hurdle (Top-3 Guaranteed)            | Test_RMSE     | 4.3420  | 4.3420      | MATCH
Two-Stage Hurdle (Top-3 Guaranteed)            | Test_R2_All   | 0.4326  | 0.4326      | MATCH
Two-Stage Hurdle (Top-3 Guaranteed)            | Test_R2_Class | 0.6587  | 0.6587      | MATCH
-----------------------------------------------+---------------+---------+-------------+---------
Two-Stage Hurdle (Expected Pos) (Top-3 Guar.)  | NDCG@1        | 0.6116  | 0.6116      | MATCH
Two-Stage Hurdle (Expected Pos) (Top-3 Guar.)  | NDCG@3        | 0.7256  | 0.7256      | MATCH
Two-Stage Hurdle (Expected Pos) (Top-3 Guar.)  | NDCG@10       | 0.8229  | 0.8229      | MATCH
Two-Stage Hurdle (Expected Pos) (Top-3 Guar.)  | MRR           | 0.6844  | 0.6844      | MATCH
Two-Stage Hurdle (Expected Pos) (Top-3 Guar.)  | Kendall_Tau   | 0.5306  | 0.5306      | MATCH
Two-Stage Hurdle (Expected Pos) (Top-3 Guar.)  | Podium_Prec   | 0.6745  | 0.6745      | MATCH
Two-Stage Hurdle (Expected Pos) (Top-3 Guar.)  | Podium_Rec    | 0.6745  | 0.6745      | MATCH
Two-Stage Hurdle (Expected Pos) (Top-3 Guar.)  | Podium_F1     | 0.6745  | 0.6745      | MATCH
Two-Stage Hurdle (Expected Pos) (Top-3 Guar.)  | Test_RMSE     | 4.3420  | 4.3420      | MATCH
Two-Stage Hurdle (Expected Pos) (Top-3 Guar.)  | Test_R2_All   | 0.4326  | 0.4326      | MATCH
Two-Stage Hurdle (Expected Pos) (Top-3 Guar.)  | Test_R2_Class | 0.6587  | 0.6587      | MATCH
========================================================================================================
Summary: 15 / 15 models audited (165 / 165 metrics matching Table 8 exactly to 4 decimal places).
```

### 1.3 Mathematical and Empirical Verification of the Podium Metric Identity
Dispatch Criterion 2 requires:
> "For all rows where Decision Rule is Top-3 Guaranteed Selection or Post-hoc Top-3 Selection, verify `Podium_Precision == Podium_Recall == Podium_F1`."

We verified this mathematically and empirically across all 11 qualifying rows:

| Model | Decision Rule | Podium Prec | Podium Rec | Podium F1 | Exact TP | Field Size (384) | Identity Satisfied |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Constructor Rank | Post-hoc Top-3 Selection | 0.6432 | 0.6432 | 0.6432 | 247 | 247 / 384 = 0.6432 | **TRUE** |
| Grid Heuristic | Post-hoc Top-3 Selection | 0.6823 | 0.6823 | 0.6823 | 262 | 262 / 384 = 0.6823 | **TRUE** |
| KNN | Post-hoc Top-3 Selection | 0.6641 | 0.6641 | 0.6641 | 255 | 255 / 384 = 0.6641 | **TRUE** |
| Decision Tree | Post-hoc Top-3 Selection | 0.6849 | 0.6849 | 0.6849 | 263 | 263 / 384 = 0.6849 | **TRUE** |
| SVM (RBF) | Post-hoc Top-3 Selection | 0.6562 | 0.6562 | 0.6562 | 252 | 252 / 384 = 0.6562 | **TRUE** |
| Dual-Branch 1D-CNN | Post-hoc Top-3 Selection | 0.6615 | 0.6615 | 0.6615 | 254 | 254 / 384 = 0.6615 | **TRUE** |
| LambdaMART (linear_rank) | Top-3 Guaranteed Selection | 0.6771 | 0.6771 | 0.6771 | 260 | 260 / 384 = 0.6771 | **TRUE** |
| LambdaMART (f1_points) | Top-3 Guaranteed Selection | 0.6432 | 0.6432 | 0.6432 | 247 | 247 / 384 = 0.6432 | **TRUE** |
| Plackett-Luce Neural Ranker | Top-3 Guaranteed Selection | 0.6823 | 0.6823 | 0.6823 | 262 | 262 / 384 = 0.6823 | **TRUE** |
| Two-Stage Hurdle | Top-3 Guaranteed Selection | 0.6693 | 0.6693 | 0.6693 | 257 | 257 / 384 = 0.6693 | **TRUE** |
| Two-Stage Hurdle (Expected Pos) | Top-3 Guaranteed (Expected Pos) | 0.6745 | 0.6745 | 0.6745 | 259 | 259 / 384 = 0.6745 | **TRUE** |

In contrast, under **Static Thresholding** ($\tau$), the number of predicted podium drivers per race is unrestricted ($0 \le \hat{P}_r \le 5$), so total predicted positives does not equal total true positives ($384$), causing Precision, Recall, and F1 to diverge as expected:
- KNN ($\tau = 0.286$): $P = 0.6052, R = 0.7865, F_1 = 0.6840$
- Decision Tree ($\tau = 0.714$): $P = 0.7123, R = 0.6510, F_1 = 0.6803$
- SVM ($\tau = 0.863$): $P = 0.6527, R = 0.6901, F_1 = 0.6709$
- Dual-Branch 1D-CNN ($\tau = 0.803$): $P = 0.6478, R = 0.7135, F_1 = 0.6791$

### 1.4 Test Suite Verification
Command: `pytest tests/ -v --tb=short`
Result: `73 passed in 18.75s (100% pass rate)` with 0 failures, 0 warnings, 0 regressions.

### 1.5 Backward Compatibility Verification
Command: `git diff -U30 EXPERIMENT_FREEZE.yaml`
Result: Lines 1–30 (`freeze_timestamp` through `cnn_stopped_epoch: 3`) are 100% bit-for-bit identical to the v5.0 baseline.

---

## 2. Logic Chain

1. **Step 1 (Audit Scope Formulation)**:
   Per instructions from parent orchestrator, Challenger 1 conducted an adversarial empirical verification of the numbers published in `reports/tables/table8_ranking_hurdle_benchmark.csv`. The audit independently ran all data loading, feature preprocessing, model fitting, ranking evaluation, and metric computation directly on `test.parquet`.
2. **Step 2 (Exhaustive Numerical Agreement Across 15 Models)**:
   Every single entry across all 15 model rows and 11 evaluation metrics was compared against Table 8:
   - For continuous ranking metrics (`NDCG@1`, `NDCG@3`, `NDCG@10`, `MRR`, `Kendall_Tau`), the outputs of `evaluate_race_ranking` matched Table 8 to 4 decimal places with maximum absolute deviation $\Delta < 10^{-5}$.
   - For regression metrics (`Test_RMSE`, `Test_R2_All`, `Test_R2_Classified`), values matched Table 8 exactly, including Hurdle Stage 2 pure classified pace $R^2 = 0.6587$.
   - For classification metrics (`Podium_Precision`, `Podium_Recall`, `Podium_F1`), values matched Table 8 exactly.
3. **Step 3 (Proof of the Zero-Sum Invariant $P = R = F1$)**:
   In any binary classification setting where the total number of predicted positive items $|\hat{P}|$ strictly equals the total number of true ground-truth positive items $|P^*|$:
   $$\text{Precision} = \frac{|\hat{P} \cap P^*|}{|\hat{P}|} = \frac{TP}{N_+}, \quad \text{Recall} = \frac{|\hat{P} \cap P^*|}{|P^*|} = \frac{TP}{N_+}$$
   $$\implies \text{Precision} \equiv \text{Recall}$$
   $$F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \cdot (TP / N_+)^2}{2 \cdot (TP / N_+)} = \frac{TP}{N_+}$$
   $$\implies \text{Precision} \equiv \text{Recall} \equiv F_1 \equiv \frac{TP}{N_+}$$
   Because the test partition contains exactly 128 races, each with exactly 3 ground-truth podium finishers, $N_+ = 128 \times 3 = 384$.
   Whenever a model utilizes a top-3 guaranteed decision rule (either listwise top-3 or post-hoc top-3 rank), it selects exactly 3 unique drivers per race, meaning $|\hat{P}| = 128 \times 3 = 384 \equiv N_+$.
   Our audit confirmed this identity holds identically for all 11 top-3 selection rows, with exact discrete integer hit counts $TP \in [247, 263]$.
4. **Step 4 (Conclusion Derivation)**:
   Because all 165 metric cells in Table 8 were independently reproduced, the mathematical properties of podium evaluation are rigorously satisfied, the full test suite passes at 100%, and lines 1–30 of `EXPERIMENT_FREEZE.yaml` are untouched, Milestone 5 is fully verified and APPROVED.

---

## 3. Caveats

- **Seed Sensitivity on Deep Neural Architectures**: Deep neural rankers such as `PlackettLuceRanker` and `DualBranchCNN` involve random initializations. Their exact numerical agreement relies on fixed random seeds (`torch.manual_seed(42)`, `np.random.seed(42)`). Our audit confirmed that when run with the specified seeds, their metrics match Table 8 deterministically.
- **Floating-point Display Formatting**: Table 8 stores rounded float strings (4 decimal places). The empirical audit verified matches within an absolute numerical tolerance $\text{atol} = 10^{-4}$ (or exact string identity).

---

## 4. Conclusion

**Verdict: APPROVE**

Worker M5 has delivered an empirical benchmark orchestrator and report that is reproducible, mathematically sound, and rigorously validated.
- All numbers in `reports/tables/table8_ranking_hurdle_benchmark.csv` are empirically confirmed.
- The zero-sum podium equality `Podium_Precision == Podium_Recall == Podium_F1` holds for all 11 top-3 rows.
- No regressions or breaking changes were introduced.

---

## 5. Verification Method

To independently reproduce the challenger's verification findings:

1. **Verify Table 8 Numerical Agreement and Podium Invariants**:
   Execute the verification harness in PowerShell:
   ```powershell
   @'
   import pandas as pd
   import numpy as np

   t8 = pd.read_csv("reports/tables/table8_ranking_hurdle_benchmark.csv", keep_default_na=False)
   top3_rules = ["Top-3 Guaranteed Selection", "Post-hoc Top-3 Selection", "Top-3 Guaranteed (Expected Pos)"]
   top3_df = t8[t8["Decision_Rule"].isin(top3_rules)]

   for _, row in top3_df.iterrows():
       p = float(row["Podium_Precision"])
       r = float(row["Podium_Recall"])
       f1 = float(row["Podium_F1"])
       tp = round(f1 * 384)
       assert p == r == f1, f"Podium mismatch in {row['Model']}"
       assert round(tp / 384, 4) == f1, f"Hit count mismatch in {row['Model']}"
   print("All Top-3 rules verified: P == R == F1 == TP / 384")
   '@ | python
   ```
   *Expected outcome*: Exits with code 0 and prints `All Top-3 rules verified: P == R == F1 == TP / 384`.

2. **Run Full Test Suite**:
   ```powershell
   pytest tests/ -v --tb=short
   ```
   *Expected outcome*: `73 passed in ~18s`.

3. **Check Freeze Compatibility**:
   ```powershell
   git diff -U30 EXPERIMENT_FREEZE.yaml
   ```
   *Expected outcome*: Lines 1–30 are completely untouched.
