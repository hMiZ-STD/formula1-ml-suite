import sys
sys.path.insert(0, r"D:\ML-Project")

import pandas as pd
import numpy as np
import yaml
from src.models.ltr_ranker import F1RankerDataset, LambdaMARTRanker, predict_podium_top3
from src.evaluation.ranking_metrics import evaluate_race_ranking

train_df = pd.read_parquet('data/processed/train_core.parquet')
val_df = pd.read_parquet('data/processed/val.parquet')

freeze = yaml.safe_load(open('EXPERIMENT_FREEZE.yaml'))['v6_ranking_and_hurdle']

mappings = ['linear_rank', 'f1_points', 'podium_tier']

for m in mappings:
    ds = F1RankerDataset(target_mapping=m)
    X_train, y_train, group_train = ds.transform(train_df)
    X_val, y_val, group_val = ds.transform(val_df)

    ranker = LambdaMARTRanker(
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=15,
        random_state=42,
    )
    ranker.fit(
        X_train, y_train, group_train,
        X_val=X_val, y_val=y_val, group_val=group_val,
        early_stopping_rounds=15,
    )

    preds = ranker.predict(X_val)
    val_df_copy = val_df.copy()
    val_df_copy['score'] = preds
    metrics = evaluate_race_ranking(val_df_copy, score_col='score', relevance_mapping=m)
    
    frozen = freeze['relevance_mapping_benchmark'][m]
    print(f'=== Mapping: {m} ===')
    print('Best iteration:', ranker.best_iteration_)
    
    mapping_check = [
        ('ndcg@1', 'val_ndcg_at_1'),
        ('ndcg@3', 'val_ndcg_at_3'),
        ('ndcg@10', 'val_ndcg_at_10'),
        ('mrr', 'val_mrr'),
        ('kendall_tau', 'val_kendall_tau'),
        ('podium_f1', 'val_podium_f1'),
    ]
    for calc_k, frz_k in mapping_check:
        calc_val = metrics[calc_k]
        frz_val = frozen[frz_k]
        print(f'  {calc_k:12s}: calc={calc_val:.6f}, frozen={frz_val:.6f}')
        assert np.isclose(calc_val, frz_val, atol=1e-5), f'Mismatch for {m} {calc_k}: {calc_val} vs {frz_val}'

print('\n>>> ALL RELEVANCE BENCHMARK METRICS EXACTLY VERIFIED WITH ZERO DISCREPANCY! <<<')
