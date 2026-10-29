"""V4: CatBoost + LightGBM + XGBoost ensemble, V3 features, tuned for RUL."""
import pandas as pd, numpy as np, lightgbm as lgb, xgboost as xgb
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import GroupShuffleSplit

TR = 'data/train (1).csv'; TE = 'data/test (1).csv'
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
    d['soc_swing'] = d['ChargingEndSOC_pct'] - d['ChargingStartSOC_pct']
    d['temp_x_crate'] = d['BatteryTemperature_C'] * d['C_Rate']
    d['fade_x_cycle'] = d['CapacityFadeRate_kWhPerCycle'] * d['Cycle']
    d['soh_decay_x_age'] = d['SOHDecayRate_pctPerCycle'] * d['BatteryAgeMonths']
    d['cycle_sq'] = d['Cycle'] ** 2
    d['cycle_x_resistance'] = d['Cycle'] * d['InternalResistance_Ohm']
    d['cycle_x_fadecount'] = d['Cycle'] * d['FastChargeCount']
    return d

print("loading...", flush=True)
tr = add_features(pd.read_csv(TR, low_memory=False))
te = add_features(pd.read_csv(TE, low_memory=False))
y_rul = tr['RUL_cycles_to_80pct'].values; y_soh = tr['SOH_pct'].values
y_tl = tr['Cycle'].values + y_rul
feat_cols = [c for c in tr.columns if c not in ID_COLS + ['RUL_cycles_to_80pct', 'SOH_pct']]
X = tr[feat_cols].copy(); Xte = te[feat_cols].copy()
cat_cols = X.select_dtypes(include='object').columns.tolist()
cat_idx = [X.columns.get_loc(c) for c in cat_cols]
for c in cat_cols:
    X[c] = X[c].astype('category'); Xte[c] = Xte[c].astype('category')
Xx = X.copy(); Xtex = Xte.copy()
for c in cat_cols:
    Xx[c] = Xx[c].cat.codes; Xtex[c] = Xtex[c].cat.codes
Xn = X.reset_index(drop=True); Xten = Xte.reset_index(drop=True)
Xxn = Xx.reset_index(drop=True); Xtexn = Xtex.reset_index(drop=True)

def score_fn(yr, pr, ys, ps):
    return 0.5*(np.mean(np.abs(yr-pr))/5000) + 0.5*(np.mean(np.abs(ys-ps))/30.04)

gss = GroupShuffleSplit(n_splits=1, test_size=30, random_state=42)
tri, vai = next(gss.split(Xn, groups=tr['BatteryID'].values))

lp = dict(objective='mae', learning_rate=0.05, num_leaves=63, min_child_samples=40,
          feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=8)
xp = dict(objective='reg:absoluteerror', learning_rate=0.05, max_depth=8, subsample=0.8,
          colsample_bytree=0.8, reg_lambda=1.0, nthread=8, tree_method='hist')

def train3(Xtr, ytr, Xva, yva, Xtr_x, Xva_x, cat_features, nr=2000):
    # LightGBM
    dtr = lgb.Dataset(Xtr, label=ytr, categorical_feature=cat_cols)
    dva = lgb.Dataset(Xva, label=yva, categorical_feature=cat_cols)
    ml = lgb.train(lp, dtr, num_boost_round=nr, valid_sets=[dva],
                   callbacks=[lgb.early_stopping(80, verbose=False)])
    # XGBoost
    mx = xgb.train(xp, xgb.DMatrix(Xtr_x, label=ytr), num_boost_round=nr,
                   evals=[(xgb.DMatrix(Xva_x, label=yva), 'v')],
                   early_stopping_rounds=80, verbose_eval=False)
    # CatBoost
    mc = CatBoostRegressor(loss_function='MAE', learning_rate=0.05, depth=8,
                           l2_leaf_reg=3, iterations=2000, early_stopping_rounds=80,
                           verbose=False, thread_count=8, cat_features=cat_features,
                           random_seed=42)
    mc.fit(Xtr, ytr, eval_set=(Xva, yva), use_best_model=True)
    pv = (ml.predict(Xva) + mx.predict(xgb.DMatrix(Xva_x)) + mc.predict(Xva)) / 3
    return pv, (ml.best_iteration, mx.best_iteration, mc.tree_count_)

print("training SOH (3 models)...", flush=True)
pv_soh, it_soh = train3(Xn.iloc[tri], y_soh[tri], Xn.iloc[vai], y_soh[vai],
                        Xxn.iloc[tri], Xxn.iloc[vai], cat_idx)
print(f"SOH MAE={np.mean(np.abs(y_soh[vai]-pv_soh)):.4f} iters={it_soh}", flush=True)

print("training RUL-total-life (3 models)...", flush=True)
pv_tl, it_tl = train3(Xn.iloc[tri], y_tl[tri], Xn.iloc[vai], y_tl[vai],
                      Xxn.iloc[tri], Xxn.iloc[vai], cat_idx)
pv_rul_tl = np.maximum(pv_tl - Xn.iloc[vai]['Cycle'].values, 0)
print(f"RUL-tl MAE={np.mean(np.abs(y_rul[vai]-pv_rul_tl)):.1f} iters={it_tl}", flush=True)

# RUL direct with SOH stack
Xst = Xn.copy(); Xst['soh_stack'] = y_soh
Xstv = Xst.iloc[vai].copy(); Xstv['soh_stack'] = pv_soh
Xstx = Xxn.copy(); Xstx['soh_stack'] = y_soh
Xstxv = Xstx.iloc[vai].copy(); Xstxv['soh_stack'] = pv_soh
cat_idx2 = cat_idx + [Xst.columns.get_loc('soh_stack')]
print("training RUL-direct+stack (3 models)...", flush=True)
pv_r, it_r = train3(Xst.iloc[tri], y_rul[tri], Xstv, y_rul[vai],
                    Xstx.iloc[tri], Xstxv, cat_idx2)
print(f"RUL-direct MAE={np.mean(np.abs(y_rul[vai]-pv_r)):.1f} iters={it_r}", flush=True)

pv_rul = 0.5*pv_rul_tl + 0.5*pv_r
s = score_fn(y_rul[vai], pv_rul, y_soh[vai], pv_soh)
print(f"V4 VALID score={s:.5f}", flush=True)

# ---- Final full-data retrain ----
print("final retrain...", flush=True)
def final3(Xf, yf, Xf_x, cat_features, iters):
    nl, nx, nc = iters
    ml = lgb.train(lp, lgb.Dataset(Xf, label=yf, categorical_feature=cat_cols), num_boost_round=nl)
    mx = xgb.train(xp, xgb.DMatrix(Xf_x, label=yf), num_boost_round=nx)
    mc = CatBoostRegressor(loss_function='MAE', learning_rate=0.05, depth=8, l2_leaf_reg=3,
                           iterations=nc, verbose=False, thread_count=8,
                           cat_features=cat_features, random_seed=42)
    mc.fit(Xf, yf)
    return ml, mx, mc

ml_s, mx_s, mc_s = final3(Xn, y_soh, Xxn, cat_idx, it_soh)
test_soh = (ml_s.predict(Xten) + mx_s.predict(xgb.DMatrix(Xtexn)) + mc_s.predict(Xten)) / 3

ml_tl, mx_tl, mc_tl = final3(Xn, y_tl, Xxn, cat_idx, it_tl)
test_rul_tl = np.maximum((ml_tl.predict(Xten) + mx_tl.predict(xgb.DMatrix(Xtexn)) + mc_tl.predict(Xten))/3
                         - Xten['Cycle'].values, 0)

Xst_full = Xn.copy(); Xst_full['soh_stack'] = y_soh
Xst_fx = Xxn.copy(); Xst_fx['soh_stack'] = y_soh
Xst_te = Xten.copy(); Xst_te['soh_stack'] = test_soh
Xst_tex = Xtexn.copy(); Xst_tex['soh_stack'] = test_soh
ml_r, mx_r, mc_r = final3(Xst_full, y_rul, Xst_fx, cat_idx2, it_r)
test_rul_d = (ml_r.predict(Xst_te) + mx_r.predict(xgb.DMatrix(Xst_tex)) + mc_r.predict(Xst_te)) / 3

test_rul = 0.5*test_rul_tl + 0.5*test_rul_d
sub = pd.DataFrame({'RecordID': pd.read_csv(TE, usecols=['RecordID'])['RecordID'],
                    'RUL_cycles_to_80pct': test_rul, 'SOH_pct': test_soh})
sub.to_csv('output/sub_v4.csv', index=False)
print("saved output/sub_v4.csv", sub.shape, flush=True)
