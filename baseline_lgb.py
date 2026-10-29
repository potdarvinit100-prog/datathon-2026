"""Baseline: LightGBM with GroupKFold by BatteryID (test has unseen batteries).
Metric: 0.5*(MAE_RUL/5000) + 0.5*(MAE_SOH/30.04), lower is better.
"""
import pandas as pd, numpy as np, lightgbm as lgb
from sklearn.model_selection import GroupKFold

TR = 'data/train (1).csv'; TE = 'data/test (1).csv'
TARGETS = ['RUL_cycles_to_80pct', 'SOH_pct']
ID_COLS = ['RecordID', 'BatteryID', 'VehicleID']

tr = pd.read_csv(TR, low_memory=False)
te = pd.read_csv(TE, low_memory=False)
groups = tr['BatteryID'].values

y_rul = tr['RUL_cycles_to_80pct'].values
y_soh = tr['SOH_pct'].values

feat_cols = [c for c in tr.columns if c not in ID_COLS + TARGETS]
X = tr[feat_cols].copy(); Xte = te[feat_cols].copy()

cat_cols = X.select_dtypes(include='object').columns.tolist()
for c in cat_cols:
    X[c] = X[c].astype('category'); Xte[c] = Xte[c].astype('category')

def score_fn(y_true_r, y_pred_r, y_true_s, y_pred_s):
    return 0.5*(np.mean(np.abs(y_true_r-y_pred_r))/5000) + 0.5*(np.mean(np.abs(y_true_s-y_pred_s))/30.04)

params = dict(objective='mae', learning_rate=0.05, num_leaves=63, min_child_samples=40,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
              lambda_l1=0.1, verbose=-1, num_threads=8)

gkf = GroupKFold(n_splits=5)
oof_r = np.zeros(len(X)); oof_s = np.zeros(len(X))
pred_r = np.zeros(len(Xte)); pred_s = np.zeros(len(Xte))

for fold, (tri, vai) in enumerate(gkf.split(X, groups=groups)):
    dtr = lgb.Dataset(X.iloc[tri], label=y_rul[tri], categorical_feature=cat_cols)
    dva = lgb.Dataset(X.iloc[vai], label=y_rul[vai], categorical_feature=cat_cols)
    m_r = lgb.train(params, dtr, num_boost_round=3000, valid_sets=[dva],
                    callbacks=[lgb.early_stopping(100, verbose=False)])
    dtr = lgb.Dataset(X.iloc[tri], label=y_soh[tri], categorical_feature=cat_cols)
    dva = lgb.Dataset(X.iloc[vai], label=y_soh[vai], categorical_feature=cat_cols)
    m_s = lgb.train(params, dtr, num_boost_round=3000, valid_sets=[dva],
                    callbacks=[lgb.early_stopping(100, verbose=False)])
    oof_r[vai] = m_r.predict(X.iloc[vai]); oof_s[vai] = m_s.predict(X.iloc[vai])
    pred_r += m_r.predict(Xte)/5; pred_s += m_s.predict(Xte)/5
    fs = score_fn(y_rul[vai], oof_r[vai], y_soh[vai], oof_s[vai])
    print(f"fold {fold}: score={fs:.5f} (RUL_MAE={np.mean(np.abs(y_rul[vai]-oof_r[vai])):.1f}, SOH_MAE={np.mean(np.abs(y_soh[vai]-oof_s[vai])):.3f})", flush=True)

print(f"\nOOF combined score: {score_fn(y_rul, oof_r, y_soh, oof_s):.5f}")
print(f"OOF RUL MAE: {np.mean(np.abs(y_rul-oof_r)):.1f} | SOH MAE: {np.mean(np.abs(y_soh-oof_s)):.3f}")

sub = pd.DataFrame({'RecordID': te['RecordID'], 'RUL_cycles_to_80pct': pred_r, 'SOH_pct': pred_s})
sub.to_csv('output/sub_baseline_lgb.csv', index=False)
print("saved output/sub_baseline_lgb.csv", sub.shape)
