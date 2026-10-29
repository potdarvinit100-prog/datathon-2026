"""V9: MLP neural net for RUL (different algorithm). sklearn MLPRegressor."""
import pandas as pd, numpy as np, lightgbm as lgb
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
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
    d['cycle_x_resistance'] = d['Cycle'] * d['InternalResistance_Ohm']
    return d

print("loading...", flush=True)
tr_raw = pd.read_csv(TR, low_memory=False); te_raw = pd.read_csv(TE, low_memory=False)
y_rul = tr_raw['RUL_cycles_to_80pct'].values; y_soh = tr_raw['SOH_pct'].values
groups = tr_raw['BatteryID'].values
tr = add_features(tr_raw)
# numerical only for MLP
num_cols = [c for c in tr.columns if c not in ID_COLS + ['RUL_cycles_to_80pct', 'SOH_pct']
            and tr[c].dtype != 'object']
X = tr[num_cols].copy().fillna(0)
Xn = X.reset_index(drop=True)

gss = GroupShuffleSplit(n_splits=1, test_size=30, random_state=42)
tri, vai = next(gss.split(Xn, groups=groups))

scaler = StandardScaler()
Xtr_s = scaler.fit_transform(Xn.iloc[tri])
Xva_s = scaler.transform(Xn.iloc[vai])

print("training MLP...", flush=True)
mlp = MLPRegressor(hidden_layer_sizes=(128, 64), activation='relu', solver='adam',
                   alpha=0.001, max_iter=200, early_stopping=True, validation_fraction=0.1,
                   n_iter_no_change=10, verbose=False, random_state=42)
mlp.fit(Xtr_s, y_rul[tri])
pv_mlp = mlp.predict(Xva_s)
mae_mlp = np.mean(np.abs(y_rul[vai] - pv_mlp))
print(f"MLP RUL MAE={mae_mlp:.1f}", flush=True)

# SOH via LightGBM (fast, good)
feat_all = [c for c in tr.columns if c not in ID_COLS + ['RUL_cycles_to_80pct', 'SOH_pct']]
Xa = tr[feat_all].copy()
for c in Xa.select_dtypes(include='object').columns: Xa[c] = Xa[c].astype('category')
cat_cols = Xa.select_dtypes(include='category').columns.tolist()
Xan = Xa.reset_index(drop=True)
lp = dict(objective='mae', learning_rate=0.08, num_leaves=63, min_child_samples=40, verbose=-1, num_threads=8)
msoh = lgb.train(lp, lgb.Dataset(Xan.iloc[tri], label=y_soh[tri], categorical_feature=cat_cols),
                 num_boost_round=500,
                 valid_sets=[lgb.Dataset(Xan.iloc[vai], label=y_soh[vai], categorical_feature=cat_cols)],
                 callbacks=[lgb.early_stopping(50, verbose=False)])
pv_soh = msoh.predict(Xan.iloc[vai])
print(f"SOH MAE={np.mean(np.abs(y_soh[vai]-pv_soh)):.4f}", flush=True)
score = 0.5*(mae_mlp/5000) + 0.5*(np.mean(np.abs(y_soh[vai]-pv_soh))/30.04)
print(f"V9 VALID score={score:.5f}", flush=True)

# Final: retrain MLP on full + predict test
Xfull_s = scaler.fit_transform(Xn)
mlp_final = MLPRegressor(hidden_layer_sizes=(128, 64), activation='relu', solver='adam',
                         alpha=0.001, max_iter=200, early_stopping=True, validation_fraction=0.1,
                         n_iter_no_change=10, verbose=False, random_state=42)
mlp_final.fit(Xfull_s, y_rul)
# SOH final
msoh_f = lgb.train(lp, lgb.Dataset(Xan, label=y_soh, categorical_feature=cat_cols),
                   num_boost_round=msoh.best_iteration)
te_base = add_features(te_raw)
Xte_soh = te_base[feat_all].copy()
for c in cat_cols: Xte_soh[c] = Xte_soh[c].astype('category')
test_soh = msoh_f.predict(Xte_soh.reset_index(drop=True))
# RUL: need ratio with predicted SOH
te2 = add_features(te_raw, soh=test_soh)
Xte = te2[num_cols].copy().fillna(0)
Xte_s = scaler.transform(Xte.reset_index(drop=True))
test_rul = mlp_final.predict(Xte_s)
sub = pd.DataFrame({'RecordID': te_raw['RecordID'], 'RUL_cycles_to_80pct': test_rul, 'SOH_pct': test_soh})
sub.to_csv('output/sub_v9.csv', index=False)
print("saved output/sub_v9.csv", sub.shape, flush=True)
