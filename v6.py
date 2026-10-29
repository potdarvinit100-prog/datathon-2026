"""V6: ANTI-OVERFIT. Strong regularization to close train(83)->valid(144) gap."""
import pandas as pd, numpy as np, lightgbm as lgb, xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit

TR = 'data/train (1).csv'; TE = 'data/test (1).csv'
ID_COLS = ['RecordID', 'BatteryID', 'VehicleID']

def add_features(df, soh=None):
    d = df.copy()
    if soh is not None: s = soh
    elif 'SOH_pct' in d.columns: s = d['SOH_pct']
    else: s = None
    if s is not None:
        d['rul_ratio'] = ((s - 80) / d['SOHDecayRate_pctPerCycle'].clip(lower=1e-6)).clip(upper=20000)
    d['charge_efficiency'] = d['EnergyCharged_kWh'] / (d['ChargingDuration_min'] + 1)
    d['temp_range'] = d['MaxBatteryTemperature_C'] - d['MinBatteryTemperature_C']
    d['stress_total'] = (d['ChargingStressScore'] + d['DrivingStressScore'] + d['ThermalStressScore']
                         + d['RoadStressScore'] + d['SOCStressScore'] + d['BatteryStressScore'])
    d['soc_swing'] = d['ChargingEndSOC_pct'] - d['ChargingStartSOC_pct']
    d['temp_x_crate'] = d['BatteryTemperature_C'] * d['C_Rate']
    d['soh_decay_x_age'] = d['SOHDecayRate_pctPerCycle'] * d['BatteryAgeMonths']
    d['cycle_x_resistance'] = d['Cycle'] * d['InternalResistance_Ohm']
    return d

print("loading...", flush=True)
tr_raw = pd.read_csv(TR, low_memory=False); te_raw = pd.read_csv(TE, low_memory=False)
y_rul = tr_raw['RUL_cycles_to_80pct'].values; y_soh = tr_raw['SOH_pct'].values
groups = tr_raw['BatteryID'].values
tr = add_features(tr_raw)
feat = [c for c in tr.columns if c not in ID_COLS + ['RUL_cycles_to_80pct', 'SOH_pct']]
X = tr[feat].copy()
cat_cols = X.select_dtypes(include='object').columns.tolist()
for c in cat_cols: X[c] = X[c].astype('category')
Xx = X.copy()
for c in cat_cols: Xx[c] = Xx[c].cat.codes
Xn = X.reset_index(drop=True); Xxn = Xx.reset_index(drop=True)

gss = GroupShuffleSplit(n_splits=1, test_size=30, random_state=42)
tri, vai = next(gss.split(Xn, groups=groups))

# STRONG regularization
lp = dict(objective='mae', learning_rate=0.05, num_leaves=31, min_child_samples=200,
          feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1,
          reg_lambda=10.0, reg_alpha=1.0, verbose=-1, num_threads=8)
xp = dict(objective='reg:absoluteerror', learning_rate=0.05, max_depth=6, subsample=0.7,
          colsample_bytree=0.7, reg_lambda=10.0, reg_alpha=1.0,
          nthread=8, tree_method='hist')

def ens2(Xtr, ytr, Xva, yva, Xtrx, Xvax, nr=2000):
    dtr = lgb.Dataset(Xtr, label=ytr, categorical_feature=cat_cols)
    dva = lgb.Dataset(Xva, label=yva, categorical_feature=cat_cols)
    ml = lgb.train(lp, dtr, num_boost_round=nr, valid_sets=[dva],
                   callbacks=[lgb.early_stopping(100, verbose=False)])
    mx = xgb.train(xp, xgb.DMatrix(Xtrx, label=ytr), num_boost_round=nr,
                   evals=[(xgb.DMatrix(Xvax, label=yva), 'v')],
                   early_stopping_rounds=100, verbose_eval=False)
    pv = (ml.predict(Xva) + mx.predict(xgb.DMatrix(Xvax))) / 2
    return pv, (ml.best_iteration, mx.best_iteration)

def score_fn(yr, pr, ys, ps):
    return 0.5*(np.mean(np.abs(yr-pr))/5000) + 0.5*(np.mean(np.abs(ys-ps))/30.04)

pv_soh, it_soh = ens2(Xn.iloc[tri], y_soh[tri], Xn.iloc[vai], y_soh[vai], Xxn.iloc[tri], Xxn.iloc[vai])
print(f"SOH MAE={np.mean(np.abs(y_soh[vai]-pv_soh)):.4f}", flush=True)
pv_rul, it_rul = ens2(Xn.iloc[tri], y_rul[tri], Xn.iloc[vai], y_rul[vai], Xxn.iloc[tri], Xxn.iloc[vai])
mae_r = np.mean(np.abs(y_rul[vai]-pv_rul))
s = score_fn(y_rul[vai], pv_rul, y_soh[vai], pv_soh)
print(f"RUL MAE={mae_r:.1f} | V6 VALID score={s:.5f} iters={it_rul}", flush=True)

# Final retrain on full
feat_soh = [c for c in feat if c != 'rul_ratio']
cat_soh = [c for c in cat_cols if c in feat_soh]
ml = lgb.train(lp, lgb.Dataset(Xn[feat_soh], label=y_soh, categorical_feature=cat_soh), num_boost_round=it_soh[0])
mx = xgb.train(xp, xgb.DMatrix(Xxn[feat_soh], label=y_soh), num_boost_round=it_soh[1])
te_base = add_features(te_raw)
Xte0 = te_base[feat_soh].copy()
for c in cat_soh: Xte0[c] = Xte0[c].astype('category')
Xtex0 = Xte0.copy()
for c in cat_soh: Xtex0[c] = Xtex0[c].cat.codes
test_soh = (ml.predict(Xte0.reset_index(drop=True)) + mx.predict(xgb.DMatrix(Xtex0.reset_index(drop=True))))/2

te = add_features(te_raw, soh=test_soh)
Xte = te[feat].copy()
for c in cat_cols: Xte[c] = Xte[c].astype('category')
Xtex = Xte.copy()
for c in cat_cols: Xtex[c] = Xtex[c].cat.codes
Xten = Xte.reset_index(drop=True); Xtexn = Xtex.reset_index(drop=True)
ml2 = lgb.train(lp, lgb.Dataset(Xn, label=y_rul, categorical_feature=cat_cols), num_boost_round=it_rul[0])
mx2 = xgb.train(xp, xgb.DMatrix(Xxn, label=y_rul), num_boost_round=it_rul[1])
test_rul = (ml2.predict(Xten) + mx2.predict(xgb.DMatrix(Xtexn)))/2
sub = pd.DataFrame({'RecordID': te_raw['RecordID'], 'RUL_cycles_to_80pct': test_rul, 'SOH_pct': test_soh})
sub.to_csv('output/sub_v6.csv', index=False)
print("saved output/sub_v6.csv", sub.shape, flush=True)
