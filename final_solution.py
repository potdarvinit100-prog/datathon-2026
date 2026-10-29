"""
Datathon 2026 — Battery Health Prediction Solution

Targets:
  * RUL_cycles_to_80pct — remaining cycles until state-of-health drops to 80%
  * SOH_pct             — current state of health (%)

Pipeline: physics-informed feature engineering -> battery-grouped validation ->
LightGBM + XGBoost ensemble (RUL) / LightGBM (SOH) -> Kaggle submission file.
"""

import os

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit

# --- Config ---
# Put train and test CSVs in the SAME folder as this script
TRAIN_PATH = 'train (1).csv'
TEST_PATH = 'test (1).csv'
ID_COLS = ['RecordID', 'BatteryID', 'VehicleID']
TARGET_RUL = 'RUL_cycles_to_80pct'
TARGET_SOH = 'SOH_pct'

# --- Step 1: Feature Engineering ---
def add_features(df, soh=None):
    """Create physics-informed features."""
    d = df.copy()
    # Use provided SOH (for test) or the column (for train)
    s = soh if soh is not None else (d['SOH_pct'] if 'SOH_pct' in d.columns else None)

    # KING FEATURE: physics-based RUL estimate
    # If health is 90% dropping 0.1%/cycle: (90-80)/0.1 = 100 cycles left
    if s is not None:
        d['rul_ratio'] = ((np.asarray(s) - 80) /
                          d['SOHDecayRate_pctPerCycle'].clip(lower=1e-6)).clip(upper=20000)

    # How efficiently the battery charges (sick batteries charge poorly)
    d['charge_efficiency'] = d['EnergyCharged_kWh'] / (d['ChargingDuration_min'] + 1)
    # Thermal volatility (big swings damage batteries)
    d['temp_range'] = d['MaxBatteryTemperature_C'] - d['MinBatteryTemperature_C']
    # Total punishment across all stress types
    d['stress_total'] = (d['ChargingStressScore'] + d['DrivingStressScore'] +
                         d['ThermalStressScore'] + d['RoadStressScore'] +
                         d['SOCStressScore'] + d['BatteryStressScore'])
    # Depth of discharge (deep cycles hurt)
    d['soc_swing'] = d['ChargingEndSOC_pct'] - d['ChargingStartSOC_pct']
    # Heat + fast charging = double damage
    d['temp_x_crate'] = d['BatteryTemperature_C'] * d['C_Rate']
    # Age × wear interaction
    d['cycle_x_resistance'] = d['Cycle'] * d['InternalResistance_Ohm']
    return d

# --- Step 2: Load Data ---
print("Loading data...")
tr_raw = pd.read_csv(TRAIN_PATH, low_memory=False)
te_raw = pd.read_csv(TEST_PATH, low_memory=False)
y_rul = tr_raw[TARGET_RUL].values
y_soh = tr_raw[TARGET_SOH].values
groups = tr_raw['BatteryID'].values  # for grouped validation

tr = add_features(tr_raw)
feat = [c for c in tr.columns if c not in ID_COLS + [TARGET_RUL, TARGET_SOH]]
X = tr[feat].copy()
cat_cols = X.select_dtypes(include='object').columns.tolist()
for c in cat_cols:
    X[c] = X[c].astype('category')

# XGBoost needs numeric — encode categoricals
Xx = X.copy()
for c in cat_cols:
    Xx[c] = Xx[c].cat.codes

# --- Step 3: Battery-Grouped Validation ---
# Hold out 30 ENTIRE batteries (not random rows) to prevent leakage
gss = GroupShuffleSplit(n_splits=1, test_size=30, random_state=42)
tri, vai = next(gss.split(X, groups=groups))
print(f"Train: {len(tri)} rows | Valid: {len(vai)} rows (different batteries)")

# --- Step 4: Train Models ---
lgb_params = dict(objective='mae', learning_rate=0.08, num_leaves=63,
                  min_child_samples=40, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=8)
xgb_params = dict(objective='reg:absoluteerror', learning_rate=0.08, max_depth=8,
                  subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                  nthread=8, tree_method='hist')

def train_rul(X_lgb, X_xgb, y):
    """Train LightGBM + XGBoost for RUL, return both models."""
    # LightGBM
    dtr = lgb.Dataset(X_lgb.iloc[tri], label=y[tri], categorical_feature=cat_cols)
    dva = lgb.Dataset(X_lgb.iloc[vai], label=y[vai], categorical_feature=cat_cols)
    m_lgb = lgb.train(lgb_params, dtr, num_boost_round=1500, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
    # XGBoost
    m_xgb = xgb.train(xgb_params, xgb.DMatrix(X_xgb.iloc[tri], label=y[tri]),
                      num_boost_round=1500,
                      evals=[(xgb.DMatrix(X_xgb.iloc[vai], label=y[vai]), 'v')],
                      early_stopping_rounds=50, verbose_eval=False)
    # Validate ensemble
    p = (m_lgb.predict(X_lgb.iloc[vai]) +
         m_xgb.predict(xgb.DMatrix(X_xgb.iloc[vai]))) / 2
    print(f"  RUL valid MAE: {np.mean(np.abs(y[vai] - p)):.1f}")
    return m_lgb, m_xgb

def train_soh(X_lgb, y):
    """Train LightGBM for SOH (easy target, single model suffices)."""
    feat_soh = [c for c in feat if c != 'rul_ratio']
    cat_soh = [c for c in cat_cols if c in feat_soh]
    dtr = lgb.Dataset(X_lgb[feat_soh].iloc[tri], label=y[tri], categorical_feature=cat_soh)
    dva = lgb.Dataset(X_lgb[feat_soh].iloc[vai], label=y[vai], categorical_feature=cat_soh)
    m = lgb.train(lgb_params, dtr, num_boost_round=800, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    p = m.predict(X_lgb[feat_soh].iloc[vai])
    print(f"  SOH valid MAE: {np.mean(np.abs(y[vai] - p)):.4f}")
    return m

print("\nTraining RUL models...")
m_rul_lgb, m_rul_xgb = train_rul(X.reset_index(drop=True), Xx.reset_index(drop=True), y_rul)
print("Training SOH model...")
m_soh = train_soh(X.reset_index(drop=True), y_soh)

# --- Step 5: Retrain on Full Data & Predict ---
print("\nRetraining on full data...")
m_rul_lgb_f = lgb.train(lgb_params,
    lgb.Dataset(X, label=y_rul, categorical_feature=cat_cols),
    num_boost_round=m_rul_lgb.best_iteration)
m_rul_xgb_f = xgb.train(xgb_params, xgb.DMatrix(Xx, label=y_rul),
                        num_boost_round=m_rul_xgb.best_iteration)

feat_soh = [c for c in feat if c != 'rul_ratio']
cat_soh = [c for c in cat_cols if c in feat_soh]
m_soh_f = lgb.train(lgb_params,
    lgb.Dataset(X[feat_soh], label=y_soh, categorical_feature=cat_soh),
    num_boost_round=m_soh.best_iteration)

# Predict SOH first (needed for rul_ratio on test)
te_base = add_features(te_raw)
Xte_soh = te_base[feat_soh].copy()
for c in cat_soh:
    Xte_soh[c] = Xte_soh[c].astype('category')
test_soh = m_soh_f.predict(Xte_soh.reset_index(drop=True)).clip(70, 100)

# Predict RUL with SOH-derived ratio feature
te = add_features(te_raw, soh=test_soh)
Xte = te[feat].copy()
for c in cat_cols:
    Xte[c] = Xte[c].astype('category')
Xtex = Xte.copy()
for c in cat_cols:
    Xtex[c] = Xtex[c].cat.codes
test_rul = ((m_rul_lgb_f.predict(Xte.reset_index(drop=True)) +
             m_rul_xgb_f.predict(xgb.DMatrix(Xtex.reset_index(drop=True)))) / 2).clip(min=0)

# --- Step 6: Save Submission ---
os.makedirs('output', exist_ok=True)
sub = pd.DataFrame({'RecordID': te_raw['RecordID'],
                    'RUL_cycles_to_80pct': test_rul,
                    'SOH_pct': test_soh})
sub.to_csv('output/submission_final.csv', index=False)
print(f"\nSaved output/submission_final.csv {sub.shape}")
print("Done.")
