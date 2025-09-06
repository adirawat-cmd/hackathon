import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import shap
import matplotlib.pyplot as plt
import joblib

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("synthetic_chronic_care_data.csv")

# Convert day to datetime
df['date'] = pd.to_datetime(df['day'], unit='D', origin=pd.Timestamp('2024-01-01'))

# -----------------------------
# Aggregate weekly mean and std
# -----------------------------
df['week'] = df['day'] // 7

weekly_mean = df.groupby(['patient_id','week']).agg({
    'age':'first',
    'gender':'first',
    'bmi':'first',
    'heart_rate':'mean',
    'systolic_bp':'mean',
    'diastolic_bp':'mean',
    'oxygen_sat':'mean',
    'weight':'mean',
    'hba1c':'mean',
    'cholesterol':'mean',
    'creatinine':'mean',
    'sleep_hours':'mean',
    'exercise_score':'mean',
    'med_adherence':'mean',
    'deterioration_in_90_days':'max'
}).reset_index()

weekly_std = df.groupby(['patient_id','week']).agg({
    'bmi':'std',
    'heart_rate':'std',
    'systolic_bp':'std',
    'diastolic_bp':'std',
    'oxygen_sat':'std',
    'weight':'std',
    'hba1c':'std',
    'cholesterol':'std',
    'creatinine':'std',
    'sleep_hours':'std',
    'exercise_score':'std',
    'med_adherence':'std'
}).reset_index()

# Merge mean and std
weekly_df = pd.merge(weekly_mean, weekly_std, on=['patient_id','week'], suffixes=('_mean','_std'))

# -----------------------------
# Trend Features
# -----------------------------
def add_trend_features(df, feature_list, patient_col='patient_id', time_col='week', window=4):
    for f in feature_list:
        trend_name = f + '_trend'
        df[trend_name] = 0
        for pid in df[patient_col].unique():
            temp = df[df[patient_col]==pid][[time_col, f]].copy().tail(window)
            if len(temp) >= 2:
                slope = np.polyfit(temp[time_col], temp[f], 1)[0]
            else:
                slope = 0
            df.loc[df[patient_col]==pid, trend_name] = slope
    return df

# Compute trend features for all numeric columns (excluding id, week, gender, target)
numeric_features = [col for col in weekly_df.columns if col not in ['patient_id','week','gender','deterioration_in_90_days']]
weekly_df = add_trend_features(weekly_df, numeric_features)

# -----------------------------
# Encode categorical features
# -----------------------------
weekly_df['gender'] = LabelEncoder().fit_transform(weekly_df['gender'])

# -----------------------------
# Features & Target
# -----------------------------
target = 'deterioration_in_90_days'
features = [col for col in weekly_df.columns if col not in ['patient_id','week','deterioration_in_90_days']]

X = weekly_df[features]
y = weekly_df[target]

# Save features list for dashboard
joblib.dump(features, "xgb_features.pkl")
print("✅ Feature list saved successfully!")

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# -----------------------------
# Train XGBoost classifier
# -----------------------------
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

model.fit(X_train, y_train)

# -----------------------------
# Predictions & Metrics
# -----------------------------
y_pred_proba = model.predict_proba(X_test)[:,1]
y_pred = (y_pred_proba >= 0.5).astype(int)

roc = roc_auc_score(y_test, y_pred_proba)
auprc = average_precision_score(y_test, y_pred_proba)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"ROC-AUC: {roc:.3f}")
print(f"AUPRC: {auprc:.3f}")
print(f"Accuracy: {acc:.3f}")
print("Confusion Matrix:\n", cm)

# -----------------------------
# SHAP Explainer
# -----------------------------
explainer = shap.Explainer(model)
shap_values = explainer(X_test)

# Global feature importance
shap.summary_plot(shap_values, X_test, show=False)
plt.show()

# Local explanation example for first patient
shap.plots.waterfall(shap_values[0])

# -----------------------------
# Save model & explainer
# -----------------------------
joblib.dump(model, "xgb_chronic_model.pkl")
joblib.dump(explainer, "shap_explainer.pkl")
print("✅ Model and explainer saved successfully!")
