"""V11: CatBoost for RUL (different algorithm, handles categoricals well)."""
import pandas as pd, numpy as np, lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import GroupShuffleSplit

TR = 'data/train (1).csv'; TE = 'data/test (1).csv'
ID_COLS = ['RecordID', 'BatteryID', 'VehicleID']

def add_features(df, soh=None):
    d = df.copy()
    if soh is not None: s = soh
    elif 'SOH_pct' in d.columns: s = d['SOH_pct']
    else: s = None
    if s is not None:
        d['rul_ratio'] = ((np.asarray(s) - 80) / d['SOHDecayRate_pctPerCycle'].clip(lower=1e-6)).clip(upper=20000)
    d['charge_efficiency'] = d['EnergyCharged_kWh'] / (d['ChargingDuration_min'] + 1)
    d['temp_range'] = d['MaxBatteryTemperature_C'] - d['MinBatteryTemperature_C']
    d['stress_total'] = (d['ChargingStressScore'] + d['DrivingStressScore'] + d['ThermalStressScore']
                         + d['RoadStressScore'] + d['SOCStressScore'] + d['BatteryStressScore'])
    d['soc_swing'] = d['ChargingEndSOC_pct'] - d['ChargingStartSOC_pct']
    d['temp_x_crate'] = d['BatteryTemperature_C'] * d['C_Rate']
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
cat_idx = [X.columns.get_loc(c) for c in cat_cols]
Xn = X.reset_index(drop=True)

gss = GroupShuffleSplit(n_splits=1, test_size=30, random_state=42)
tri, vai = next(gss.split(Xn, groups=groups))

print("training CatBoost RUL...", flush=True)
cb = CatBoostRegressor(iterations=800, learning_rate=0.08, depth=8, l2_leaf_reg=3.0,
                       loss_function='MAE', random_seed=42, verbose=False, thread_count=8)
cb.fit(Xn.iloc[tri], y_rul[tri], cat_features=cat_idx, eval_set=(Xn.iloc[vai], y_rul[vai]),
       early_stopping_rounds=50, verbose=False)
pv_rul = cb.predict(Xn.iloc[vai])
mae_r = np.mean(np.abs(y_rul[vai] - pv_rul))
print(f"CatBoost RUL MAE={mae_r:.1f}", flush=True)

# SOH via LightGBM
Xa = tr[feat].copy()
for c in cat_cols: Xa[c] = Xa[c].astype('category')
Xan = Xa.reset_index(drop=True)
lp = dict(objective='mae', learning_rate=0.08, num_leaves=63, min_child_samples=40, verbose=-1, num_threads=8)
msoh = lgb.train(lp, lgb.Dataset(Xan.iloc[tri], label=y_soh[tri], categorical_feature=cat_cols),
                 num_boost_round=800,
                 valid_sets=[lgb.Dataset(Xan.iloc[vai], label=y_soh[vai], categorical_feature=cat_cols)],
                 callbacks=[lgb.early_stopping(50, verbose=False)])
pv_soh = msoh.predict(Xan.iloc[vai])
mae_s = np.mean(np.abs(y_soh[vai] - pv_soh))
score = 0.5*(mae_r/5000) + 0.5*(mae_s/30.04)
print(f"SOH MAE={mae_s:.4f} | V11 VALID score={score:.5f}", flush=True)

# Final
cb_f = CatBoostRegressor(iterations=cb.best_iteration_, learning_rate=0.08, depth=8, l2_leaf_reg=3.0,
                         loss_function='MAE', random_seed=42, verbose=False, thread_count=8)
cb_f.fit(Xn, y_rul, cat_features=cat_idx, verbose=False)
msoh_f = lgb.train(lp, lgb.Dataset(Xan, label=y_soh, categorical_feature=cat_cols),
                   num_boost_round=msoh.best_iteration)
te = add_features(te_raw)
Xte = te[feat].copy()
Xten = Xte.reset_index(drop=True)
# SOH first for ratio
feat_soh = [c for c in feat if c != 'rul_ratio']
Xte_soh = te[feat_soh].copy()
for c in cat_cols:
    if c in Xte_soh.columns: Xte_soh[c] = Xte_soh[c].astype('category')
test_soh = msoh_f.predict(Xte_soh.reset_index(drop=True))
# RUL with ratio
te2 = add_features(te_raw, soh=test_soh)
Xte2 = te2[feat].copy()
test_rul = cb_f.predict(Xte2.reset_index(drop=True))
sub = pd.DataFrame({'RecordID': te_raw['RecordID'], 'RUL_cycles_to_80pct': test_rul, 'SOH_pct': test_soh})
sub.to_csv('output/sub_v11.csv', index=False)
print("saved output/sub_v11.csv", sub.shape, flush=True)
