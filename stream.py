# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# import shap
# import plotly.express as px
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import LabelEncoder

# # -----------------------------
# # Load Data
# # -----------------------------
# df = pd.read_csv("synthetic_chronic_care_data.csv")

# # Convert day to datetime
# df['date'] = pd.to_datetime(df['day'], unit='D', origin=pd.Timestamp('2024-01-01'))

# # -----------------------------
# # Aggregate weekly mean & std
# # -----------------------------
# df['week'] = df['day'] // 7

# weekly_mean = df.groupby(['patient_id','week']).agg({
#     'age':'first',
#     'gender':'first',
#     'bmi':'first',
#     'heart_rate':'mean',
#     'systolic_bp':'mean',
#     'diastolic_bp':'mean',
#     'oxygen_sat':'mean',
#     'weight':'mean',
#     'hba1c':'mean',
#     'cholesterol':'mean',
#     'creatinine':'mean',
#     'sleep_hours':'mean',
#     'exercise_score':'mean',
#     'med_adherence':'mean',
#     'deterioration_in_90_days':'max'
# }).reset_index()

# weekly_std = df.groupby(['patient_id','week']).agg({
#     'bmi':'std',
#     'heart_rate':'std',
#     'systolic_bp':'std',
#     'diastolic_bp':'std',
#     'oxygen_sat':'std',
#     'weight':'std',
#     'hba1c':'std',
#     'cholesterol':'std',
#     'creatinine':'std',
#     'sleep_hours':'std',
#     'exercise_score':'std',
#     'med_adherence':'std'
# }).reset_index()

# weekly_df = pd.merge(weekly_mean, weekly_std, on=['patient_id','week'], suffixes=('_mean','_std'))
# weekly_df.rename(columns={'age':'age_mean'}, inplace=True)


# # -----------------------------
# # Trend features
# # -----------------------------
# def add_trend_features(df, feature_list, patient_col='patient_id', time_col='week', window=4):
#     for f in feature_list:
#         trend_name = f + '_trend'
#         df[trend_name] = 0
#         for pid in df[patient_col].unique():
#             temp = df[df[patient_col]==pid][[time_col, f]].copy().tail(window)
#             if len(temp) >= 2:
#                 slope = np.polyfit(temp[time_col], temp[f], 1)[0]
#             else:
#                 slope = 0
#             df.loc[df[patient_col]==pid, trend_name] = slope
#     return df

# numeric_features = [col for col in weekly_df.columns if col not in ['patient_id','week','gender','deterioration_in_90_days']]
# weekly_df = add_trend_features(weekly_df, numeric_features)

# # Encode gender
# weekly_df['gender'] = LabelEncoder().fit_transform(weekly_df['gender'])

# # -----------------------------
# # Load model, explainer, and features
# # -----------------------------
# model = joblib.load("xgb_chronic_model.pkl")
# explainer = joblib.load("shap_explainer.pkl")
# features = joblib.load("xgb_features.pkl")
# for f in features:
#     if f not in weekly_df.columns:
#         weekly_df[f] = 0
# # Subset data to match model features
# X = weekly_df[features]
# weekly_df['risk_score'] = model.predict_proba(X)[:,1]

# # -----------------------------
# # Streamlit App
# # -----------------------------
# st.set_page_config(page_title="Chronic Care Risk Dashboard", layout="wide")
# st.title("❤️ Chronic Care Risk Prediction Dashboard")

# # -----------------------------
# # Sidebar filters
# # -----------------------------
# st.sidebar.header("Patient & Cohort Filters")
# selected_patient = st.sidebar.selectbox("Select Patient ID", weekly_df['patient_id'].unique())
# lookback_weeks = st.sidebar.slider("Look-back period (weeks)", 1, 20, 4)

# age_min, age_max = int(weekly_df['age_mean'].min()), int(weekly_df['age_mean'].max())
# bmi_min, bmi_max = int(weekly_df['bmi_mean'].min()), int(weekly_df['bmi_mean'].max())

# age_filter = st.sidebar.slider("Filter by Age", age_min, age_max, (30, 80))
# bmi_filter = st.sidebar.slider("Filter by BMI", bmi_min, bmi_max, (18, 35))

# filtered_df = weekly_df[
#     (weekly_df['age_mean'] >= age_filter[0]) & (weekly_df['age_mean'] <= age_filter[1]) &
#     (weekly_df['bmi_mean'] >= bmi_filter[0]) & (weekly_df['bmi_mean'] <= bmi_filter[1])
# ]

# # -----------------------------
# # Cohort Risk View
# # -----------------------------
# st.subheader("Cohort Risk Scores")
# cohort = filtered_df.groupby('patient_id')['risk_score'].mean().reset_index()
# fig_cohort = px.bar(
#     cohort.sort_values('risk_score', ascending=False),
#     x='patient_id', y='risk_score',
#     color='risk_score', color_continuous_scale='reds',
#     labels={'risk_score':'Avg Risk Score'},
#     title="Average Risk Score per Patient"
# )
# st.plotly_chart(fig_cohort, use_container_width=True)

# # -----------------------------
# # Global SHAP Summary
# # -----------------------------
# st.subheader("Global Risk Drivers (Cohort)")
# shap.summary_plot(explainer.shap_values(X), X, show=False)
# st.pyplot(plt.gcf())
# plt.clf()

# # -----------------------------
# # Patient Detail View
# # -----------------------------
# st.subheader(f"Patient Details: {selected_patient}")
# patient_data = weekly_df[weekly_df['patient_id']==selected_patient].sort_values('week')
# recent_weeks = patient_data.tail(lookback_weeks)

# # Risk trend
# fig_risk = px.line(recent_weeks, x='week', y='risk_score', title=f'Weekly Risk Score Trend (Last {lookback_weeks} Weeks)')
# st.plotly_chart(fig_risk, use_container_width=True)

# # Vital & lab trends
# st.write("**Vital & Lab Trends**")
# vital_features = [
#     'heart_rate_mean','systolic_bp_mean','diastolic_bp_mean','oxygen_sat_mean','weight_mean',
#     'hba1c_mean','cholesterol_mean','creatinine_mean','sleep_hours_mean','exercise_score_mean','med_adherence_mean'
# ]
# for vf in vital_features:
#     fig = px.line(recent_weeks, x='week', y=vf, title=f"{vf.replace('_',' ').title()} Trend")
#     st.plotly_chart(fig, use_container_width=True)

# # -----------------------------
# # Local SHAP for patient
# # -----------------------------
# st.write("**Top Risk Drivers (SHAP values)**")
# patient_index = weekly_df[weekly_df['patient_id']==selected_patient].index[-1]
# shap.waterfall_plot(
#     shap.Explanation(
#         values=explainer.shap_values(X)[patient_index],
#         base_values=explainer.expected_value,
#         data=X.iloc[patient_index],
#         feature_names=X.columns
#     )
# )

# # -----------------------------
# # Recommended Actions
# # -----------------------------
# st.subheader("Recommended Actions")
# actions = []
# row = recent_weeks.iloc[-1]

# if row['hba1c_mean'] > 7: actions.append("📌 Review blood sugar management")
# if row['systolic_bp_mean'] > 140: actions.append("📌 Check blood pressure control")
# if row['bmi_mean'] > 30: actions.append("📌 Consider weight management plan")
# if row['med_adherence_mean'] < 80: actions.append("📌 Improve medication adherence")
# if row['sleep_hours_mean'] < 6: actions.append("📌 Encourage adequate sleep")
# if row['exercise_score_mean'] < 4: actions.append("📌 Increase physical activity")
# if not actions: actions.append("✅ Patient is stable. Continue current care.")

# for act in actions: st.write(act)

# # -----------------------------
# # Downloadable patient report
# # -----------------------------
# st.subheader("Download Patient Report")
# report_df = recent_weeks.copy()
# report_df['risk_score'] = report_df['risk_score'].round(3)
# csv = report_df.to_csv(index=False).encode()
# st.download_button("📥 Download CSV Report", data=csv, file_name=f"patient_{selected_patient}_report.csv", mime="text/csv")

# # -----------------------------
# # High-risk alerts
# # -----------------------------
# high_risk_threshold = 0.7
# high_risk_patients = weekly_df.groupby('patient_id')['risk_score'].mean().reset_index()
# high_risk_patients = high_risk_patients[high_risk_patients['risk_score'] >= high_risk_threshold]

# st.subheader("🚨 High-Risk Patients Alert")
# if not high_risk_patients.empty:
#     st.table(high_risk_patients)
# else:
#     st.write("No patients above high-risk threshold currently.")
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, confusion_matrix, roc_curve, roc_auc_score,
    precision_score, recall_score, f1_score
)
import seaborn as sns


# -----------------------------
# Load Data
# -----------------------------
df = pd.read_csv("synthetic_chronic_care_data.csv")
df['date'] = pd.to_datetime(df['day'], unit='D', origin=pd.Timestamp('2024-01-01'))

# -----------------------------
# Aggregate weekly mean & std
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

weekly_df = pd.merge(weekly_mean, weekly_std, on=['patient_id','week'], suffixes=('_mean','_std'))
weekly_df.rename(columns={'age':'age_mean'}, inplace=True)
y_true = weekly_df['deterioration_in_90_days']
y_pred = (weekly_df['risk_score'] >= 0.5).astype(int) 
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
roc_auc = roc_auc_score(y_true, weekly_df['risk_score'])
# -----------------------------
# Trend features
# -----------------------------
def add_trend_features(df, feature_list, patient_col='patient_id', time_col='week', window=4):
    for f in feature_list:
        trend_name = f + '_trend'
        df[trend_name] = 0
        for pid in df[patient_col].unique():
            temp = df[df[patient_col]==pid][[time_col, f]].copy().tail(window)
            slope = np.polyfit(temp[time_col], temp[f], 1)[0] if len(temp) >= 2 else 0
            df.loc[df[patient_col]==pid, trend_name] = slope
    return df

numeric_features = [col for col in weekly_df.columns if col not in ['patient_id','week','gender','deterioration_in_90_days']]
weekly_df = add_trend_features(weekly_df, numeric_features)

# Encode gender
weekly_df['gender'] = LabelEncoder().fit_transform(weekly_df['gender'])

# -----------------------------
# Load model, explainer, features
# -----------------------------
model = joblib.load("xgb_chronic_model.pkl")
explainer = joblib.load("shap_explainer.pkl")
features = joblib.load("xgb_features.pkl")
for f in features:
    if f not in weekly_df.columns:
        weekly_df[f] = 0
# -----------------------------
# Compute risk scores for all patients (for cohort charts)
# -----------------------------
X_all = weekly_df[features]
weekly_df['risk_score'] = model.predict_proba(X_all)[:,1]

# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="Chronic Care Risk Dashboard", layout="wide")
st.title("❤️ Chronic Care Risk Prediction Dashboard")

# -----------------------------
# Sidebar filters
# -----------------------------
st.sidebar.header("Patient & Cohort Filters")
selected_patient = st.sidebar.selectbox("Select Patient ID", weekly_df['patient_id'].unique())
lookback_weeks = st.sidebar.slider("Look-back period (weeks)", 1, 20, 4)

age_min, age_max = int(weekly_df['age_mean'].min()), int(weekly_df['age_mean'].max())
bmi_min, bmi_max = int(weekly_df['bmi_mean'].min()), int(weekly_df['bmi_mean'].max())

age_filter = st.sidebar.slider("Filter by Age", age_min, age_max, (30, 80))
bmi_filter = st.sidebar.slider("Filter by BMI", bmi_min, bmi_max, (18, 35))

filtered_df = weekly_df[
    (weekly_df['age_mean'] >= age_filter[0]) & (weekly_df['age_mean'] <= age_filter[1]) &
    (weekly_df['bmi_mean'] >= bmi_filter[0]) & (weekly_df['bmi_mean'] <= bmi_filter[1])
]


st.subheader(f"Patient Details: {selected_patient}")
patient_data = weekly_df[weekly_df['patient_id'] == selected_patient].sort_values('week')
recent_weeks = patient_data.tail(lookback_weeks)

# Compute patient-specific risk scores
X_patient = patient_data[features]
patient_data['risk_score'] = model.predict_proba(X_patient)[:,1]
recent_weeks = patient_data.tail(lookback_weeks)

# Display the latest risk score
latest_risk = recent_weeks['risk_score'].iloc[-1]
st.metric(label="Latest Risk Score", value=f"{latest_risk:.2f}")

# Risk Trend
fig_risk = px.line(recent_weeks, x='week', y='risk_score',
                   title=f'Weekly Risk Score Trend (Last {lookback_weeks} Weeks)')
st.plotly_chart(fig_risk, use_container_width=True)
# -----------------------------
# Cohort Risk View
# -----------------------------
st.subheader("Cohort Risk Scores")
cohort = filtered_df.groupby('patient_id')['risk_score'].mean().reset_index()
fig_cohort = px.bar(
    cohort.sort_values('risk_score', ascending=False),
    x='patient_id', y='risk_score',
    color='risk_score', color_continuous_scale='reds',
    labels={'risk_score':'Avg Risk Score'},
    title="Average Risk Score per Patient"
)
st.plotly_chart(fig_cohort, use_container_width=True)

# -----------------------------
# Patient Detail View
# -----------------------------
st.subheader(f"Patient Details: {selected_patient}")
patient_data = weekly_df[weekly_df['patient_id'] == selected_patient].sort_values('week')
recent_weeks = patient_data.tail(lookback_weeks)

# Compute patient-specific risk scores
# X_patient = patient_data[features]
# patient_data['risk_score'] = model.predict_proba(X_patient)[:,1]
# recent_weeks = patient_data.tail(lookback_weeks)

# # Risk Trend
# fig_risk = px.line(recent_weeks, x='week', y='risk_score',
#                    title=f'Weekly Risk Score Trend (Last {lookback_weeks} Weeks)')
# st.plotly_chart(fig_risk, use_container_width=True)

# Vital & Lab Trends
st.write("**Vital & Lab Trends**")
vital_features = [
    'heart_rate_mean','systolic_bp_mean','diastolic_bp_mean','oxygen_sat_mean','weight_mean',
    'hba1c_mean','cholesterol_mean','creatinine_mean','sleep_hours_mean','exercise_score_mean','med_adherence_mean'
]
for vf in vital_features:
    fig = px.line(recent_weeks, x='week', y=vf, title=f"{vf.replace('_',' ').title()} Trend")
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Patient SHAP
# -----------------------------
st.write("**Top Risk Drivers (SHAP values)**")
shap.waterfall_plot(
    shap.Explanation(
        values=explainer.shap_values(X_patient)[-1],
        base_values=explainer.expected_value,
        data=X_patient.iloc[-1],
        feature_names=X_patient.columns
    )
)

# -----------------------------
# Recommended Actions
# -----------------------------
st.subheader("Recommended Actions")
actions = []
row = recent_weeks.iloc[-1]

if row['hba1c_mean'] > 7: actions.append("📌 Review blood sugar management")
if row['systolic_bp_mean'] > 140: actions.append("📌 Check blood pressure control")
if row['bmi_mean'] > 30: actions.append("📌 Consider weight management plan")
if row['med_adherence_mean'] < 80: actions.append("📌 Improve medication adherence")
if row['sleep_hours_mean'] < 6: actions.append("📌 Encourage adequate sleep")
if row['exercise_score_mean'] < 4: actions.append("📌 Increase physical activity")
if not actions: actions.append("✅ Patient is stable. Continue current care.")

for act in actions: st.write(act)

# -----------------------------
# Downloadable Patient Report
# -----------------------------
st.subheader("Download Patient Report")
report_df = recent_weeks.copy()
report_df['risk_score'] = report_df['risk_score'].round(3)
csv = report_df.to_csv(index=False).encode()
st.download_button("📥 Download CSV Report", data=csv,
                   file_name=f"patient_{selected_patient}_report.csv", mime="text/csv")

# -----------------------------
# High-risk Alerts
# -----------------------------
high_risk_threshold = 0.7
high_risk_patients = weekly_df.groupby('patient_id')['risk_score'].mean().reset_index()
high_risk_patients = high_risk_patients[high_risk_patients['risk_score'] >= high_risk_threshold]

st.subheader("🚨 High-Risk Patients Alert")
if not high_risk_patients.empty:
    st.table(high_risk_patients)
else:
    st.write("No patients above high-risk threshold currently.")



st.subheader("📊 Model Performance Metrics")
st.metric("Accuracy", f"{acc:.2f}")
st.metric("Precision", f"{prec:.2f}")
st.metric("Recall", f"{rec:.2f}")
st.metric("F1 Score", f"{f1:.2f}")
st.metric("ROC AUC", f"{roc_auc:.2f}")
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_true, y_pred)
fig_cm, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig_cm)
st.subheader("ROC Curve")
fpr, tpr, thresholds = roc_curve(y_true, weekly_df['risk_score'])
fig_roc, ax = plt.subplots()
ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
ax.plot([0, 1], [0, 1], 'k--')
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve")
ax.legend()
st.pyplot(fig_roc)
