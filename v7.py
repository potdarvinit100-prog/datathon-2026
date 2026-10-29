"""V7: NUMERICAL-ONLY + physics. Drop categoricals to force universal patterns."""
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
        d['soh_margin'] = s - 80
    # numerical only
    d['charge_efficiency'] = d['EnergyCharged_kWh'] / (d['ChargingDuration_min'] + 1)
    d['temp_range'] = d['MaxBatteryTemperature_C'] - d['MinBatteryTemperature_C']
    d['stress_total'] = (d['ChargingStressScore'] + d['DrivingStressScore'] + d['ThermalStressScore']
                         + d['RoadStressScore'] + d['SOCStressScore'] + d['BatteryStressScore'])
    d['soc_swing'] = d['ChargingEndSOC_pct'] - d['ChargingStartSOC_pct']
    d['temp_x_crate'] = d['BatteryTemperature_C'] * d['C_Rate']
    d['soh_decay_x_age'] = d['SOHDecayRate_pctPerCycle'] * d['BatteryAgeMonths']
    d['cycle_x_resistance'] = d['Cycle'] * d['InternalResistance_Ohm']
    d['cycle_sq'] = d['Cycle'] ** 2
    d['discharge_rate'] = d['DepthOfDischarge_pct'] / (d['DischargeDuration_min'] + 1)
    d['regen_ratio'] = d['EnergyRegenerated_kWh'] / (d['EnergyConsumed_kWh'] + 1)
    return d

print("loading...", flush=True)
tr_raw = pd.read_csv(TR, low_memory=False); te_raw = pd.read_csv(TE, low_memory=False)
y_rul = tr_raw['RUL_cycles_to_80pct'].values; y_soh = tr_raw['SOH_pct'].values
groups = tr_raw['BatteryID'].values
tr = add_features(tr_raw)
# NUMERICAL ONLY - drop all object columns
num_cols = [c for c in tr.columns if c not in ID_COLS + ['RUL_cycles_to_80pct', 'SOH_pct']
            and tr[c].dtype != 'object']
print(f"using {len(num_cols)} numerical features", flush=True)
X = tr[num_cols].copy()
Xn = X.reset_index(drop=True)

gss = GroupShuffleSplit(n_splits=1, test_size=30, random_state=42)
tri, vai = next(gss.split(Xn, groups=groups))
lp = dict(objective='mae', learning_rate=0.08, num_leaves=63, min_child_samples=40,
          feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=8)
xp = dict(objective='reg:absoluteerror', learning_rate=0.08, max_depth=8, subsample=0.8,
          colsample_bytree=0.8, reg_lambda=1.0, nthread=8, tree_method='hist')

def ens2(Xtr, ytr, Xva, yva, nr=1500):
    ml = lgb.train(lp, lgb.Dataset(Xtr, label=ytr), num_boost_round=nr, valid_sets=[lgb.Dataset(Xva, label=yva)],
                   callbacks=[lgb.early_stopping(50, verbose=False)])
    mx = xgb.train(xp, xgb.DMatrix(Xtr, label=ytr), num_boost_round=nr,
                   evals=[(xgb.DMatrix(Xva, label=yva), 'v')],
                   early_stopping_rounds=50, verbose_eval=False)
    pv = (ml.predict(Xva) + mx.predict(xgb.DMatrix(Xva))) / 2
    return pv, (ml.best_iteration, mx.best_iteration)

def score_fn(yr, pr, ys, ps):
    return 0.5*(np.mean(np.abs(yr-pr))/5000) + 0.5*(np.mean(np.abs(ys-ps))/30.04)

pv_soh, it_soh = ens2(Xn.iloc[tri], y_soh[tri], Xn.iloc[vai], y_soh[vai])
print(f"SOH MAE={np.mean(np.abs(y_soh[vai]-pv_soh)):.4f}", flush=True)
pv_rul, it_rul = ens2(Xn.iloc[tri], y_rul[tri], Xn.iloc[vai], y_rul[vai])
mae_r = np.mean(np.abs(y_rul[vai]-pv_rul))
s = score_fn(y_rul[vai], pv_rul, y_soh[vai], pv_soh)
print(f"RUL MAE={mae_r:.1f} | V7 VALID score={s:.5f} iters={it_rul}", flush=True)

# Final
feat_soh = [c for c in num_cols if c not in ('rul_ratio', 'soh_margin')]
ml = lgb.train(lp, lgb.Dataset(Xn[feat_soh], label=y_soh), num_boost_round=it_soh[0])
mx = xgb.train(xp, xgb.DMatrix(Xn[feat_soh], label=y_soh), num_boost_round=it_soh[1])
te_base = add_features(te_raw)
Xte0 = te_base[feat_soh]
test_soh = (ml.predict(Xte0.reset_index(drop=True)) + mx.predict(xgb.DMatrix(Xte0.reset_index(drop=True))))/2
te = add_features(te_raw, soh=test_soh)
Xte = te[num_cols]
ml2 = lgb.train(lp, lgb.Dataset(Xn, label=y_rul), num_boost_round=it_rul[0])
mx2 = xgb.train(xp, xgb.DMatrix(Xn, label=y_rul), num_boost_round=it_rul[1])
Xten = Xte.reset_index(drop=True)
test_rul = (ml2.predict(Xten) + mx2.predict(xgb.DMatrix(Xten)))/2
sub = pd.DataFrame({'RecordID': te_raw['RecordID'], 'RUL_cycles_to_80pct': test_rul, 'SOH_pct': test_soh})
sub.to_csv('output/sub_v7.csv', index=False)
print("saved output/sub_v7.csv", sub.shape, flush=True)
