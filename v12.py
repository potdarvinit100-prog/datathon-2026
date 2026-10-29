"""V12: 3-fold robust ensemble of V5 (variance reduction)."""
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
        d['rul_ratio'] = ((np.asarray(s) - 80) / d['SOHDecayRate_pctPerCycle'].clip(lower=1e-6)).clip(upper=20000)
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
tr = add_features(tr_raw)
feat = [c for c in tr.columns if c not in ID_COLS + ['RUL_cycles_to_80pct', 'SOH_pct']]
X = tr[feat].copy()
cat_cols = X.select_dtypes(include='object').columns.tolist()
for c in cat_cols: X[c] = X[c].astype('category')
Xx = X.copy()
for c in cat_cols: Xx[c] = Xx[c].cat.codes
Xn = X.reset_index(drop=True); Xxn = Xx.reset_index(drop=True)

lp = dict(objective='mae', learning_rate=0.08, num_leaves=63, min_child_samples=40,
          feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=8)
xp = dict(objective='reg:absoluteerror', learning_rate=0.08, max_depth=8, subsample=0.8,
          colsample_bytree=0.8, reg_lambda=1.0, nthread=8, tree_method='hist')

# 3-fold: train 3 models with different splits, average test predictions
test_rul_accum = np.zeros(len(te_raw))
test_soh_accum = np.zeros(len(te_raw))
for fold, seed in enumerate([42, 123, 456]):
    print(f"Fold {fold+1}/3 (seed {seed})...", flush=True)
    gss = GroupShuffleSplit(n_splits=1, test_size=30, random_state=seed)
    tri, vai = next(gss.split(Xn, groups=groups))
    # RUL
    dtr = lgb.Dataset(Xn.iloc[tri], label=y_rul[tri], categorical_feature=cat_cols)
    dva = lgb.Dataset(Xn.iloc[vai], label=y_rul[vai], categorical_feature=cat_cols)
    ml = lgb.train(lp, dtr, num_boost_round=1500, valid_sets=[dva],
                   callbacks=[lgb.early_stopping(50, verbose=False)])
    mx = xgb.train(xp, xgb.DMatrix(Xxn.iloc[tri], label=y_rul[tri]), num_boost_round=1500,
                   evals=[(xgb.DMatrix(Xxn.iloc[vai], label=y_rul[vai]), 'v')],
                   early_stopping_rounds=50, verbose_eval=False)
    pv = (ml.predict(Xn.iloc[vai]) + mx.predict(xgb.DMatrix(Xxn.iloc[vai]))) / 2
    mae = np.mean(np.abs(y_rul[vai] - pv))
    print(f"  RUL MAE={mae:.1f}", flush=True)
    # Retrain on full for test
    ml_f = lgb.train(lp, lgb.Dataset(Xn, label=y_rul, categorical_feature=cat_cols), num_boost_round=ml.best_iteration)
    mx_f = xgb.train(xp, xgb.DMatrix(Xxn, label=y_rul), num_boost_round=mx.best_iteration)
    # SOH (need for ratio) - quick single model
    feat_soh = [c for c in feat if c != 'rul_ratio']
    cat_soh = [c for c in cat_cols if c in feat_soh]
    ms = lgb.train(lp, lgb.Dataset(Xn[feat_soh].iloc[tri], label=y_soh[tri], categorical_feature=cat_soh),
                   num_boost_round=800,
                   valid_sets=[lgb.Dataset(Xn[feat_soh].iloc[vai], label=y_soh[vai], categorical_feature=cat_soh)],
                   callbacks=[lgb.early_stopping(50, verbose=False)])
    ms_f = lgb.train(lp, lgb.Dataset(Xn[feat_soh], label=y_soh, categorical_feature=cat_soh), num_boost_round=ms.best_iteration)
    te_base = add_features(te_raw)
    Xte_soh = te_base[feat_soh].copy()
    for c in cat_soh: Xte_soh[c] = Xte_soh[c].astype('category')
    t_soh = ms_f.predict(Xte_soh.reset_index(drop=True))
    test_soh_accum += t_soh / 3
    te2 = add_features(te_raw, soh=t_soh)
    Xte = te2[feat].copy()
    for c in cat_cols: Xte[c] = Xte[c].astype('category')
    Xtex = Xte.copy()
    for c in cat_cols: Xtex[c] = Xtex[c].cat.codes
    t_rul = (ml_f.predict(Xte.reset_index(drop=True)) + mx_f.predict(xgb.DMatrix(Xtex.reset_index(drop=True)))) / 2
    test_rul_accum += t_rul / 3

sub = pd.DataFrame({'RecordID': te_raw['RecordID'], 'RUL_cycles_to_80pct': test_rul_accum, 'SOH_pct': test_soh_accum})
sub.to_csv('output/sub_v12.csv', index=False)
print("saved output/sub_v12.csv", sub.shape, flush=True)
