"""V2: feature engineering + LightGBM/XGBoost ensemble, same fast split."""
import pandas as pd, numpy as np, lightgbm as lgb, xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit

TR = 'data/train (1).csv'; TE = 'data/test (1).csv'
TARGETS = ['RUL_cycles_to_80pct', 'SOH_pct']
ID_COLS = ['RecordID', 'BatteryID', 'VehicleID']

def add_features(df):
    d = df.copy()
    d['charge_efficiency'] = d['EnergyCharged_kWh'] / (d['ChargingDuration_min'] + 1)
    d['temp_range'] = d['MaxBatteryTemperature_C'] - d['MinBatteryTemperature_C']
    d['age_x_cycle'] = d['BatteryAgeMonths'] * d['Cycle']
    d['stress_total'] = (d['ChargingStressScore'] + d['DrivingStressScore'] + d['ThermalStressScore']
                         + d['RoadStressScore'] + d['SOCStressScore'] + d['BatteryStressScore'])
    d['discharge_rate'] = d['DepthOfDischarge_pct'] / (d['DischargeDuration_min'] + 1)
    d['speed_range'] = d['MaximumSpeed_kmph'] - d['AverageSpeed_kmph']
    d['regen_ratio'] = d['EnergyRegenerated_kWh'] / (d['EnergyConsumed_kWh'] + 1)
    d['volt_util'] = d['PackVoltage_V'] * d['PackCurrent_A'] / 1000  # kW check
    d['soc_swing'] = d['ChargingEndSOC_pct'] - d['ChargingStartSOC_pct']
    d['temp_x_crate'] = d['BatteryTemperature_C'] * d['C_Rate']
    d['fade_x_cycle'] = d['CapacityFadeRate_kWhPerCycle'] * d['Cycle']
    d['soh_decay_x_age'] = d['SOHDecayRate_pctPerCycle'] * d['BatteryAgeMonths']
    return d

print("loading...", flush=True)
tr = add_features(pd.read_csv(TR, low_memory=False))
te = add_features(pd.read_csv(TE, low_memory=False))
groups = tr['BatteryID'].values
y = {t: tr[t].values for t in TARGETS}
feat_cols = [c for c in tr.columns if c not in ID_COLS + TARGETS]
X = tr[feat_cols].copy(); Xte = te[feat_cols].copy()
cat_cols = X.select_dtypes(include='object').columns.tolist()
for c in cat_cols:
    X[c] = X[c].astype('category'); Xte[c] = Xte[c].astype('category')
# xgb needs numeric cats
Xx = X.copy(); Xtex = Xte.copy()
for c in cat_cols:
    Xx[c] = Xx[c].cat.codes; Xtex[c] = Xtex[c].cat.codes

def score_fn(yr, pr, ys, ps):
    return 0.5*(np.mean(np.abs(yr-pr))/5000) + 0.5*(np.mean(np.abs(ys-ps))/30.04)

gss = GroupShuffleSplit(n_splits=1, test_size=30, random_state=42)
tri, vai = next(gss.split(X, groups=groups))

lp = dict(objective='mae', learning_rate=0.08, num_leaves=63, min_child_samples=40,
          feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=8)
xp = dict(objective='reg:absoluteerror', learning_rate=0.08, max_depth=8, subsample=0.8,
          colsample_bytree=0.8, reg_lambda=1.0, nthread=8, tree_method='hist')

best_iters = {}
for t in TARGETS:
    dtr = lgb.Dataset(X.iloc[tri], label=y[t][tri], categorical_feature=cat_cols)
    dva = lgb.Dataset(X.iloc[vai], label=y[t][vai], categorical_feature=cat_cols)
    ml = lgb.train(lp, dtr, num_boost_round=1500, valid_sets=[dva],
                   callbacks=[lgb.early_stopping(50, verbose=False)])
    mx = xgb.train(xp, xgb.DMatrix(Xx.iloc[tri], label=y[t][tri]),
                   num_boost_round=1500,
                   evals=[(xgb.DMatrix(Xx.iloc[vai], label=y[t][vai]), 'v')],
                   early_stopping_rounds=50, verbose_eval=False)
    best_iters[t] = (ml.best_iteration, mx.best_iteration)
    pr = (ml.predict(X.iloc[vai]) + mx.predict(xgb.DMatrix(Xx.iloc[vai]))) / 2
    mae = np.mean(np.abs(y[t][vai]-pr))
    print(f"{t}: lgb_iter={ml.best_iteration} xgb_iter={mx.best_iteration} ens_MAE={mae:.3f}", flush=True)

# validation ensemble score
prs, pss = [], []
for t in TARGETS:
    pass
print("scoring ensemble...", flush=True)

def predict_full(t, n_lgb, n_xgb):
    dall = lgb.Dataset(X, label=y[t], categorical_feature=cat_cols)
    ml = lgb.train(lp, dall, num_boost_round=n_lgb)
    mx = xgb.train(xp, xgb.DMatrix(Xx, label=y[t]), num_boost_round=n_xgb)
    return ml, mx

preds = {}
val_preds = {}
for t in TARGETS:
    nl, nx = best_iters[t]
    # validation preds for scoring
    dtr = lgb.Dataset(X.iloc[tri], label=y[t][tri], categorical_feature=cat_cols)
    mlv = lgb.train(lp, dtr, num_boost_round=nl)
    mxv = xgb.train(xp, xgb.DMatrix(Xx.iloc[tri], label=y[t][tri]), num_boost_round=nx)
    val_preds[t] = (mlv.predict(X.iloc[vai]) + mxv.predict(xgb.DMatrix(Xx.iloc[vai]))) / 2
    # full-data preds
    ml, mx = predict_full(t, nl, nx)
    preds[t] = (ml.predict(Xte) + mx.predict(xgb.DMatrix(Xtex))) / 2
    print(f"final {t} done", flush=True)

s = score_fn(y[TARGETS[0]][vai], val_preds[TARGETS[0]], y[TARGETS[1]][vai], val_preds[TARGETS[1]])
print(f"V2 VALID ensemble score={s:.5f}", flush=True)
sub = pd.DataFrame({'RecordID': pd.read_csv(TE, usecols=['RecordID'])['RecordID'],
                    'RUL_cycles_to_80pct': preds[TARGETS[0]], 'SOH_pct': preds[TARGETS[1]]})
sub.to_csv('output/sub_v2_ens.csv', index=False)
print("saved output/sub_v2_ens.csv", sub.shape, flush=True)
