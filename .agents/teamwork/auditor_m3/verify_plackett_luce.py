import sys
sys.path.insert(0, r"D:\ML-Project")

import pandas as pd
import numpy as np
import torch
from src.models.ltr_ranker import F1RankerDataset, PlackettLuceRanker

train_df = pd.read_parquet('data/processed/train_core.parquet')
val_df = pd.read_parquet('data/processed/val.parquet')

ds = F1RankerDataset(target_mapping="linear_rank")
X_train, y_train, group_train = ds.transform(train_df)
X_val, y_val, group_val = ds.transform(val_df)

ranker = PlackettLuceRanker(
    input_dim=17,
    hidden_dim=32,
    lr=1e-3,
    epochs=3,
    batch_size_races=16,
    random_state=42,
)

initial_weights = {k: v.clone() for k, v in ranker.net.state_dict().items()}

ranker.fit(
    X_train, y_train, group_train,
    X_val=X_val, y_val=y_val, group_val=group_val
)

final_weights = ranker.net.state_dict()

# Check that weights actually updated
weights_updated = False
for k in initial_weights:
    diff = (final_weights[k] - initial_weights[k]).abs().sum().item()
    print(f"Weight diff for {k}: {diff:.6f}")
    if diff > 1e-4:
        weights_updated = True

assert weights_updated, "CRITICAL INTEGRITY FAILURE: Neural network weights did not update during fit!"
print("PLACKETT-LUCE NEURAL WEIGHT UPDATE EMPIRICALLY CONFIRMED!")

# Verify prediction
preds = ranker.predict(X_val)
assert preds.shape == (820,)
assert not np.isnan(preds).any()
assert len(np.unique(preds)) > 800
print(f"Predictions unique count: {len(np.unique(preds))} / 820")
print("PLACKETT-LUCE NEURAL RANKER AUDIT CLEAN!")
