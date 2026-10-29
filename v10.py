"""V10: PHYSICS CORRECTION. RUL = base * C, learn C via ML."""
import pandas as pd, numpy as np, lightgbm as lgb, xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit

TR = 'data/train (1).csv'; TE = 'data/test (1).csv'
ID_COLS = ['RecordID', 'BatteryID', 'VehicleID']

def add_features(df):
    d = df.copy()
    d['charge_efficiency'] = d['EnergyCharged_kWh'] / (d['ChargingDuration_min'] + 1)
    d['temp_range'] = d['MaxBatteryTemperature_C'] - d['MinBatteryTemperature_C']
    d['stress_total'] = (d['ChargingStressScore'] + d['DrivingStressScore'] + d['ThermalStressScore']
                         + d['RoadStressScore'] + d['SOCStressScore'] + d['BatteryStressScore'])
    d['soc_swing'] = d['ChargingEndSOC_pct'] - d['ChargingStartSOC_pct']
    d['temp_x_crate'] = d['BatteryTemperature_C'] * d['C_Rate']
    d['cycle_x_resistance'] = d['Cycle'] * d['InternalResistance_Ohm']
    d['soh_decay_x_age'] = d['SOHDecayRate_pctPerCycle'] * d['BatteryAgeMonths']
    return d

print("loading...", flush=True)
tr_raw = pd.read_csv(TR, low_memory=False); te_raw = pd.read_csv(TE, low_memory=False)
y_rul = tr_raw['RUL_cycles_to_80pct'].values; y_soh = tr_raw['SOH_pct'].values
groups = tr_raw['BatteryID'].values

# Physics base: RUL_base = (SOH-80)/decay
decay = tr_raw['SOHDecayRate_pctPerCycle'].clip(lower=1e-6).values
soh = tr_raw['SOH_pct'].values
rul_base = np.clip((soh - 80) / decay, 0, 20000)
# Correction factor C = RUL_true / RUL_base (avoid div by zero)
mask = rul_base > 1
C = np.ones(len(y_rul))
C[mask] = (y_rul[mask] / rul_base[mask]).clip(0.1, 10.0)
print(f"C stats: mean={C[mask].mean():.3f}, std={C[mask].std():.3f}", flush=True)

tr = add_features(tr_raw)
feat = [c for c in tr.columns if c not in ID_COLS + ['RUL_cycles_to_80pct', 'SOH_pct']]
# Add rul_base as feature too
tr['rul_base'] = rul_base
feat.append('rul_base')
X = tr[feat].copy()
cat_cols = X.select_dtypes(include='object').columns.tolist()
for c in cat_cols: X[c] = X[c].astype('category')
Xx = X.copy()
for c in cat_cols: Xx[c] = Xx[c].cat.codes
Xn = X.reset_index(drop=True); Xxn = Xx.reset_index(drop=True)

gss = GroupShuffleSplit(n_splits=1, test_size=30, random_state=42)
tri, vai = next(gss.split(Xn, groups=groups))
lp = dict(objective='mae', learning_rate=0.08, num_leaves=63, min_child_samples=40,
          feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=8)
xp = dict(objective='reg:absoluteerror', learning_rate=0.08, max_depth=8, subsample=0.8,
          colsample_bytree=0.8, reg_lambda=1.0, nthread=8, tree_method='hist')

def train_ens(Xtr, ytr, Xva, yva, Xtrx, Xvax, nr=1500):
    dtr = lgb.Dataset(Xtr, label=ytr, categorical_feature=cat_cols)
    dva = lgb.Dataset(Xva, label=yva, categorical_feature=cat_cols)
    ml = lgb.train(lp, dtr, num_boost_round=nr, valid_sets=[dva],
                   callbacks=[lgb.early_stopping(50, verbose=False)])
    mx = xgb.train(xp, xgb.DMatrix(Xtrx, label=ytr), num_boost_round=nr,
                   evals=[(xgb.DMatrix(Xvax, label=yva), 'v')],
                   early_stopping_rounds=50, verbose_eval=False)
    pv = (ml.predict(Xva) + mx.predict(xgb.DMatrix(Xvax))) / 2
    return pv, ml, mx

# Learn C (correction factor)
pv_C, ml_C, mx_C = train_ens(Xn.iloc[tri], C[tri], Xn.iloc[vai], C[vai], Xxn.iloc[tri], Xxn.iloc[vai])
# Reconstruct RUL
pv_rul = rul_base[vai] * pv_C
mae_r = np.mean(np.abs(y_rul[vai] - pv_rul))
print(f"RUL via correction MAE={mae_r:.1f}", flush=True)

# SOH model (standard)
feat_soh = [c for c in feat if c != 'rul_base']
cat_soh = [c for c in cat_cols if c in feat_soh]
dtr = lgb.Dataset(Xn[feat_soh].iloc[tri], label=y_soh[tri], categorical_feature=cat_soh)
dva = lgb.Dataset(Xn[feat_soh].iloc[vai], label=y_soh[vai], categorical_feature=cat_soh)
ml_soh = lgb.train(lp, dtr, num_boost_round=1500, valid_sets=[dva],
                   callbacks=[lgb.early_stopping(50, verbose=False)])
mx_soh = xgb.train(xp, xgb.DMatrix(Xxn[feat_soh].iloc[tri], label=y_soh[tri]), num_boost_round=1500,
                   evals=[(xgb.DMatrix(Xxn[feat_soh].iloc[vai], label=y_soh[vai]), 'v')],
                   early_stopping_rounds=50, verbose_eval=False)
pv_soh = (ml_soh.predict(Xn[feat_soh].iloc[vai]) + mx_soh.predict(xgb.DMatrix(Xxn[feat_soh].iloc[vai]))) / 2
mae_s = np.mean(np.abs(y_soh[vai] - pv_soh))
score = 0.5*(mae_r/5000) + 0.5*(mae_s/30.04)
print(f"SOH MAE={mae_s:.4f} | V10 VALID score={score:.5f}", flush=True)

# Final: retrain on full, predict test
ml_Cf = lgb.train(lp, lgb.Dataset(Xn, label=C, categorical_feature=cat_cols), num_boost_round=ml_C.best_iteration)
mx_Cf = xgb.train(xp, xgb.DMatrix(Xxn, label=C), num_boost_round=mx_C.best_iteration)
ml_sf = lgb.train(lp, lgb.Dataset(Xn[feat_soh], label=y_soh, categorical_feature=cat_soh), num_boost_round=ml_soh.best_iteration)
mx_sf = xgb.train(xp, xgb.DMatrix(Xxn[feat_soh], label=y_soh), num_boost_round=mx_soh.best_iteration)

te = add_features(te_raw)
Xte = te[feat].copy()
for c in cat_cols: Xte[c] = Xte[c].astype('category')
Xtex = Xte.copy()
for c in cat_cols: Xtex[c] = Xtex[c].cat.codes
Xten = Xte.reset_index(drop=True); Xtexn = Xtex.reset_index(drop=True)
# SOH first
Xte_soh = te[feat_soh].copy()
for c in cat_soh: Xte_soh[c] = Xte_soh[c].astype('category')
Xtex_soh = Xte_soh.copy()
for c in cat_soh: Xtex_soh[c] = Xtex_soh[c].cat.codes
test_soh = (ml_sf.predict(Xte_soh.reset_index(drop=True)) + mx_sf.predict(xgb.DMatrix(Xtex_soh.reset_index(drop=True))))/2
# RUL base from predicted SOH
decay_te = te_raw['SOHDecayRate_pctPerCycle'].clip(lower=1e-6).values
rul_base_te = np.clip((test_soh - 80) / decay_te, 0, 20000)
# Need rul_base feature for C prediction - use test_soh-based
Xte['rul_base'] = rul_base_te
Xtex['rul_base'] = rul_base_te
# Re-encode after adding
for c in cat_cols:
    if Xte[c].dtype == 'object': Xte[c] = Xte[c].astype('category')
Xten2 = Xte.reset_index(drop=True); Xtexn2 = Xtex.reset_index(drop=True)
for c in cat_cols:
    if c in Xtexn2.columns: Xtexn2[c] = Xtexn2[c].cat.codes
test_C = (ml_Cf.predict(Xten2) + mx_Cf.predict(xgb.DMatrix(Xtexn2)))/2
test_rul = rul_base_te * test_C
sub = pd.DataFrame({'RecordID': te_raw['RecordID'], 'RUL_cycles_to_80pct': test_rul, 'SOH_pct': test_soh})
sub.to_csv('output/sub_v10.csv', index=False)
print("saved output/sub_v10.csv", sub.shape, flush=True)
