import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    make_scorer,
    precision_score,
    recall_score,
    f1_score
)

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(page_title="Intelligent Fishing Prediction", layout="wide")
st.title("🐟 Data-Driven Intelligent Fishing System")
st.markdown("### 🎯 Objective")
st.write("This system predicts fishing suitability using machine learning models trained on historical marine water-quality data.")

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv("waterquality.csv")

st.subheader("📌 Dataset Preview")
st.write("Shape:", df.shape)
st.dataframe(df.head(), use_container_width=True)

# ============================================================
# LABEL CREATION FUNCTION
# ============================================================

def fish_presence(row):
    conds = [
        (row["WaterTemp (C)"] >= 20) & (row["WaterTemp (C)"] <= 30),
        (row["DissolvedOxygen (mg/L)"] >= 5),
        (row["pH"] >= 7.0) & (row["pH"] <= 8.5),
        (row["SecchiDepth (m)"] >= 0.2)
    ]
    return int(sum(bool(c) for c in conds) >= 3)

# ============================================================
# FEATURE ENGINEERING
# ============================================================

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Month"] = df["Date"].dt.month

feature_cols = [
    "Salinity (ppt)",
    "DissolvedOxygen (mg/L)",
    "pH",
    "SecchiDepth (m)",
    "WaterDepth (m)",
    "WaterTemp (C)",
    "AirTemp (C)",
    "Month"
]

df["FishPresence"] = df.apply(fish_presence, axis=1)

X = df[feature_cols]
y = df["FishPresence"]

st.subheader("🎯 Class Distribution")
st.write(y.value_counts())

# ============================================================
# PREPROCESSING PIPELINE
# ============================================================

preprocess = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), feature_cols)
    ]
)

# ============================================================
# SIDEBAR SETTINGS
# ============================================================

st.sidebar.header("⚙️ Settings")
n_neighbors = st.sidebar.slider("KNN Neighbors (k)", 3, 15, 7)
rf_trees = st.sidebar.slider("Random Forest Trees", 50, 300, 150, step=50)

# ============================================================
# MODELS
# ============================================================

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "KNN": KNeighborsClassifier(n_neighbors=n_neighbors),
    "SVM (RBF)": SVC(kernel="rbf", probability=True),
    "Random Forest": RandomForestClassifier(n_estimators=rf_trees, random_state=42)
}

# ============================================================
# CROSS VALIDATION
# ============================================================

st.subheader("✅ 5-Fold Cross Validation Results")

scoring = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "roc_auc": "roc_auc"
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = []

for name, model in models.items():
    pipe = Pipeline([
        ("preprocess", preprocess),
        ("model", model)
    ])

    cvres = cross_validate(pipe, X, y, cv=cv, scoring=scoring)

    results.append({
        "Model": name,
        "Accuracy": np.mean(cvres["test_accuracy"]),
        "Precision": np.mean(cvres["test_precision"]),
        "Recall": np.mean(cvres["test_recall"]),
        "F1-score": np.mean(cvres["test_f1"]),
        "ROC-AUC": np.mean(cvres["test_roc_auc"])
    })

results_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False)
st.dataframe(results_df, use_container_width=True)

# ============================================================
# PERFORMANCE PLOTS
# ============================================================

st.subheader("📊 Model Performance Graphs")

fig, ax = plt.subplots()
ax.bar(results_df["Model"], results_df["Accuracy"])
ax.set_title("Accuracy Comparison")
ax.tick_params(axis="x", rotation=45)
st.pyplot(fig)

# ============================================================
# CONFUSION MATRIX + ROC (Random Forest)
# ============================================================

st.subheader("📌 Random Forest Confusion Matrix + ROC Curve")

rf = RandomForestClassifier(n_estimators=rf_trees, random_state=42)
rf_pipe = Pipeline([
    ("preprocess", preprocess),
    ("model", rf)
])

rf_pipe.fit(X, y)

y_pred = rf_pipe.predict(X)
y_proba = rf_pipe.predict_proba(X)[:, 1]

cm = confusion_matrix(y, y_pred)
disp = ConfusionMatrixDisplay(cm)

fig, ax = plt.subplots()
disp.plot(ax=ax)
st.pyplot(fig)

fpr, tpr, _ = roc_curve(y, y_proba)
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots()
ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
ax.plot([0, 1], [0, 1], "--")
ax.legend()
ax.set_title("ROC Curve")
st.pyplot(fig)

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

st.subheader("📊 Feature Importance (Random Forest)")

importances = rf.feature_importances_

fig, ax = plt.subplots()
ax.barh(feature_cols, importances)
ax.set_title("Feature Importance")
st.pyplot(fig)

# ============================================================
# DATASET STATISTICS
# ============================================================

st.subheader("📋 Dataset Statistics")
st.dataframe(df[feature_cols].describe(), use_container_width=True)

# ============================================================
# LIVE PREDICTION
# ============================================================

st.sidebar.markdown("---")
selected_model_name = st.sidebar.selectbox("Select Model for Prediction", list(models.keys()))

st.subheader("🔮 Live Fishing Suitability Prediction")

with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    with col1:
        sal = st.number_input("Salinity (ppt)", value=30.0)
        do = st.number_input("Dissolved Oxygen (mg/L)", value=6.0)
        ph_val = st.number_input("pH", value=7.5)
        secchi = st.number_input("Secchi Depth (m)", value=0.3)

    with col2:
        wdepth = st.number_input("Water Depth (m)", value=1.0)
        wtemp = st.number_input("Water Temp (C)", value=25.0)
        atemp = st.number_input("Air Temp (C)", value=28.0)
        month_val = st.slider("Month", 1, 12, 6)

    submit = st.form_submit_button("Predict")

if submit:

    selected_model = models[selected_model_name]

    model_pipe = Pipeline([
        ("preprocess", preprocess),
        ("model", selected_model)
    ])

    model_pipe.fit(X, y)

    input_df = pd.DataFrame([[sal, do, ph_val, secchi,
                              wdepth, wtemp, atemp, month_val]],
                            columns=feature_cols)

    prediction = model_pipe.predict(input_df)[0]

    if hasattr(model_pipe.named_steps["model"], "predict_proba"):
        probability = model_pipe.predict_proba(input_df)[0][1]
    else:
        probability = 0.5

    st.write(f"Model Used: {selected_model_name}")

    if prediction == 1:
        st.success(f"✅ Suitable for Fishing (Confidence: {probability:.2f})")
        st.info("Recommendation: Fishing trip likely efficient.")
    else:
        st.error(f"❌ Not Suitable for Fishing (Confidence: {probability:.2f})")
        st.warning("Recommendation: Consider postponing trip.")

st.success("✅ Streamlit app running successfully!")