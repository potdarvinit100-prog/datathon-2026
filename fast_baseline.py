"""FAST baseline: single GroupShuffleSplit holdout (30 batteries), LightGBM MAE.
Then retrain on full data for submission.
"""
import pandas as pd, numpy as np, lightgbm as lgb, sys
from sklearn.model_selection import GroupShuffleSplit

TR = 'data/train (1).csv'; TE = 'data/test (1).csv'
TARGETS = ['RUL_cycles_to_80pct', 'SOH_pct']
ID_COLS = ['RecordID', 'BatteryID', 'VehicleID']

print("loading...", flush=True)
tr = pd.read_csv(TR, low_memory=False)
te = pd.read_csv(TE, low_memory=False)
groups = tr['BatteryID'].values
y = {t: tr[t].values for t in TARGETS}

feat_cols = [c for c in tr.columns if c not in ID_COLS + TARGETS]
X = tr[feat_cols].copy(); Xte = te[feat_cols].copy()
cat_cols = X.select_dtypes(include='object').columns.tolist()
for c in cat_cols:
    X[c] = X[c].astype('category'); Xte[c] = Xte[c].astype('category')

def score_fn(yr, pr, ys, ps):
    return 0.5*(np.mean(np.abs(yr-pr))/5000) + 0.5*(np.mean(np.abs(ys-ps))/30.04)

gss = GroupShuffleSplit(n_splits=1, test_size=30, random_state=42)
tri, vai = next(gss.split(X, groups=groups))
print(f"train rows: {len(tri)}, valid rows: {len(vai)}, valid batteries: {len(np.unique(groups[vai]))}", flush=True)

params = dict(objective='mae', learning_rate=0.08, num_leaves=63, min_child_samples=40,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=8)

models = {}
for t in TARGETS:
    dtr = lgb.Dataset(X.iloc[tri], label=y[t][tri], categorical_feature=cat_cols)
    dva = lgb.Dataset(X.iloc[vai], label=y[t][vai], categorical_feature=cat_cols)
    m = lgb.train(params, dtr, num_boost_round=1500, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    models[t] = m
    print(f"{t}: best_iter={m.best_iteration}", flush=True)

pr = models[TARGETS[0]].predict(X.iloc[vai])
ps = models[TARGETS[1]].predict(X.iloc[vai])
s = score_fn(y[TARGETS[0]][vai], pr, y[TARGETS[1]][vai], ps)
print(f"VALID score={s:.5f} | RUL_MAE={np.mean(np.abs(y[TARGETS[0]][vai]-pr)):.1f} | SOH_MAE={np.mean(np.abs(y[TARGETS[1]][vai]-ps)):.3f}", flush=True)

# Retrain on FULL data with best iters, predict test
preds = {}
for t in TARGETS:
    dall = lgb.Dataset(X, label=y[t], categorical_feature=cat_cols)
    m = lgb.train(params, dall, num_boost_round=models[t].best_iteration)
    preds[t] = m.predict(Xte)
    print(f"final {t} trained", flush=True)

sub = pd.DataFrame({'RecordID': te['RecordID'], 'RUL_cycles_to_80pct': preds[TARGETS[0]], 'SOH_pct': preds[TARGETS[1]]})
sub.to_csv('output/sub_fast_lgb.csv', index=False)
print("saved output/sub_fast_lgb.csv", sub.shape, flush=True)
