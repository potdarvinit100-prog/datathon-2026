"""V3: total_life trick for RUL + SOH stacking + interactions."""
import pandas as pd, numpy as np, lightgbm as lgb, xgboost as xgb
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
groups = tr['BatteryID'].values
y_rul = tr['RUL_cycles_to_80pct'].values
y_soh = tr['SOH_pct'].values
y_tl = tr['Cycle'].values + y_rul  # total life trick

feat_cols = [c for c in tr.columns if c not in ID_COLS + ['RUL_cycles_to_80pct', 'SOH_pct']]
X = tr[feat_cols].copy(); Xte = te[feat_cols].copy()
cat_cols = X.select_dtypes(include='object').columns.tolist()
for c in cat_cols:
    X[c] = X[c].astype('category'); Xte[c] = Xte[c].astype('category')
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

def train_ens(Xtr, ytr, Xva, yva, Xfull=None, yfull=None, nr=1500):
    """Train lgb+xgb ensemble; return (valid_pred, full_model_preds or None, iters)."""
    dtr = lgb.Dataset(Xtr, label=ytr, categorical_feature=cat_cols)
    dva = lgb.Dataset(Xva, label=yva, categorical_feature=cat_cols)
    ml = lgb.train(lp, dtr, num_boost_round=nr, valid_sets=[dva],
                   callbacks=[lgb.early_stopping(50, verbose=False)])
    mx = xgb.train(xp, xgb.DMatrix(Xx.iloc[Xtr.index] if hasattr(Xtr,'index') else Xtr, label=ytr),
                   num_boost_round=nr,
                   evals=[(xgb.DMatrix(Xva if not hasattr(Xva,'iloc') else Xx.iloc[Xva.index], label=yva), 'v')],
                   early_stopping_rounds=50, verbose_eval=False)
    pv = (ml.predict(Xva) + mx.predict(xgb.DMatrix(Xva if not hasattr(Xva,'iloc') else Xx.iloc[Xva.index]))) / 2
    return pv, (ml.best_iteration, mx.best_iteration)

# Use numpy-backed frames to keep xgb indexing simple
Xn = X.reset_index(drop=True); Xten = Xte.reset_index(drop=True)
Xxn = Xx.reset_index(drop=True); Xtexn = Xtex.reset_index(drop=True)

# --- SOH first (for stacking) ---
pv_soh, it_soh = train_ens(Xn.iloc[tri], y_soh[tri], Xn.iloc[vai], y_soh[vai])
print(f"SOH ens_MAE={np.mean(np.abs(y_soh[vai]-pv_soh)):.4f} iters={it_soh}", flush=True)

# --- RUL via total_life ---
pv_tl, it_tl = train_ens(Xn.iloc[tri], y_tl[tri], Xn.iloc[vai], y_tl[vai])
pv_rul_tl = np.maximum(pv_tl - Xn.iloc[vai]['Cycle'].values, 0)
mae_rul = np.mean(np.abs(y_rul[vai]-pv_rul_tl))
print(f"RUL-via-totallife ens_MAE={mae_rul:.2f} iters={it_tl}", flush=True)

# --- RUL direct (with SOH-stack feature) ---
Xst = Xn.copy(); Xst['soh_oof'] = 0.0; Xst.iloc[vai, Xst.columns.get_loc('soh_oof')] = pv_soh
Xst_full = Xn.copy()
pv_r, it_r = train_ens(Xst.iloc[tri], y_rul[tri], Xst.iloc[vai], y_rul[vai])
print(f"RUL-direct+stack ens_MAE={np.mean(np.abs(y_rul[vai]-pv_r)):.2f} iters={it_r}", flush=True)

# blend the two RUL approaches
pv_rul_blend = 0.5*pv_rul_tl + 0.5*pv_r
print(f"RUL blend MAE={np.mean(np.abs(y_rul[vai]-pv_rul_blend)):.2f}", flush=True)

s = score_fn(y_rul[vai], pv_rul_blend, y_soh[vai], pv_soh)
print(f"V3 VALID score={s:.5f}", flush=True)

# --- Final: retrain on full data ---
def final_ens(Xf, yf, nl, nx):
    dall = lgb.Dataset(Xf, label=yf, categorical_feature=cat_cols)
    ml = lgb.train(lp, dall, num_boost_round=nl)
    mx = xgb.train(xp, xgb.DMatrix(Xxn if Xf is Xn else Xtexn, label=yf), num_boost_round=nx)
    return ml, mx

# SOH final (need soh for test too: predict test SOH first)
ml_s, mx_s = final_ens(Xn, y_soh, *it_soh)
test_soh = (ml_s.predict(Xten) + mx_s.predict(xgb.DMatrix(Xtexn))) / 2

# RUL total-life final
ml_tl, mx_tl = final_ens(Xn, y_tl, *it_tl)
test_tl = (ml_tl.predict(Xten) + mx_tl.predict(xgb.DMatrix(Xtexn))) / 2
test_rul_tl = np.maximum(test_tl - Xten['Cycle'].values, 0)

# RUL direct+stack final: add predicted soh as feature
Xst_test = Xten.copy(); Xst_test['soh_oof'] = test_soh
Xst_tr = Xn.copy()
# train stacked RUL on full train with OOF soh for train rows: approximate with actual SOH for train
Xst_tr['soh_oof'] = y_soh
dall = lgb.Dataset(Xst_tr, label=y_rul, categorical_feature=cat_cols)
ml_r = lgb.train(lp, dall, num_boost_round=it_r[0])
Xst_tr_x = Xxn.copy(); Xst_tr_x['soh_oof'] = y_soh
Xst_te_x = Xtexn.copy(); Xst_te_x['soh_oof'] = test_soh
mx_r = xgb.train(xp, xgb.DMatrix(Xst_tr_x, label=y_rul), num_boost_round=it_r[1])
test_rul_d = (ml_r.predict(Xst_test) + mx_r.predict(xgb.DMatrix(Xst_te_x))) / 2

test_rul = 0.5*test_rul_tl + 0.5*test_rul_d
sub = pd.DataFrame({'RecordID': pd.read_csv(TE, usecols=['RecordID'])['RecordID'],
                    'RUL_cycles_to_80pct': test_rul, 'SOH_pct': test_soh})
sub.to_csv('output/sub_v3.csv', index=False)
print("saved output/sub_v3.csv", sub.shape, flush=True)
