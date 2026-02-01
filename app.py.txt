import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
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
    precision_recall_curve,
    average_precision_score,
    make_scorer,
    precision_score,
    recall_score,
    f1_score
)

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Fishing Suitability Prediction", layout="wide")
st.title("🐟 Intelligent Fishing Suitability Prediction System")
st.caption("4 ML Models + Stratified 5-Fold Cross Validation")

# -----------------------------
# LOAD DATASET (ONLINE AUTO LOAD)
# -----------------------------
df = pd.read_csv("waterquality.csv")

st.sidebar.success("✅ Dataset Loaded: waterquality.csv")

# -----------------------------
# FEATURE ENGINEERING
# -----------------------------
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

# -----------------------------
# CREATE LABEL (FishPresence)
# -----------------------------
def fish_presence(row):
    conds = [
        (row["WaterTemp (C)"] >= 20) & (row["WaterTemp (C)"] <= 30),
        (row["DissolvedOxygen (mg/L)"] >= 5),
        (row["pH"] >= 7.0) & (row["pH"] <= 8.5),
        (row["SecchiDepth (m)"] >= 0.2)
    ]
    return int(sum(bool(c) for c in conds) >= 3)

df["FishPresence"] = df.apply(fish_presence, axis=1)

X = df[feature_cols]
y = df["FishPresence"]

# -----------------------------
# SHOW DATASET
# -----------------------------
st.subheader("📌 Dataset Preview")
st.write("Shape:", df.shape)
st.dataframe(df.head(10), use_container_width=True)

st.subheader("🎯 Class Distribution (FishPresence)")
st.write(y.value_counts())

# -----------------------------
# PREPROCESSING PIPELINE
# -----------------------------
preprocess = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), feature_cols)
    ]
)

# -----------------------------
# MODEL SETTINGS (SIDEBAR)
# -----------------------------
st.sidebar.header("⚙️ Model Settings")
knn_k = st.sidebar.slider("KNN Neighbors (k)", 3, 15, 7)
rf_trees = st.sidebar.slider("Random Forest Trees", 50, 300, 150, step=50)

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "KNN": KNeighborsClassifier(n_neighbors=knn_k),
    "SVM (RBF)": SVC(kernel="rbf", probability=True),
    "Random Forest": RandomForestClassifier(n_estimators=rf_trees, random_state=42)
}

# -----------------------------
# SCORING
# -----------------------------
scoring = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "roc_auc": "roc_auc"
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# -----------------------------
# RUN CV
# -----------------------------
st.subheader("✅ 5-Fold Cross Validation Results")

results = []
for name, model in models.items():
    clf = Pipeline([
        ("preprocess", preprocess),
        ("model", model)
    ])

    cvres = cross_validate(clf, X, y, cv=cv, scoring=scoring, n_jobs=1)

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

best_model = results_df.iloc[0]["Model"]
st.success(f"🏆 Best Model: {best_model}")

# -----------------------------
# BAR GRAPHS
# -----------------------------
st.subheader("📊 Model Comparison Graphs")

c1, c2, c3 = st.columns(3)

with c1:
    fig, ax = plt.subplots()
    ax.bar(results_df["Model"], results_df["Accuracy"])
    ax.set_title("Accuracy")
    ax.tick_params(axis="x", rotation=35)
    st.pyplot(fig)

with c2:
    fig, ax = plt.subplots()
    ax.bar(results_df["Model"], results_df["F1-score"])
    ax.set_title("F1-score")
    ax.tick_params(axis="x", rotation=35)
    st.pyplot(fig)

with c3:
    fig, ax = plt.subplots()
    ax.bar(results_df["Model"], results_df["ROC-AUC"])
    ax.set_title("ROC-AUC")
    ax.tick_params(axis="x", rotation=35)
    st.pyplot(fig)

# -----------------------------
# RANDOM FOREST OUTPUT (CONFUSION MATRIX + ROC + PR)
# -----------------------------
st.subheader("📌 Best Model Detailed Output (Random Forest)")

rf_pipe = Pipeline([
    ("preprocess", preprocess),
    ("model", RandomForestClassifier(n_estimators=rf_trees, random_state=42))
])

y_proba = cross_val_predict(rf_pipe, X, y, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
y_pred = (y_proba >= 0.5).astype(int)

colA, colB = st.columns(2)

with colA:
    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Not Suitable(0)", "Suitable(1)"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(values_format="d", ax=ax)
    ax.set_title("Confusion Matrix (RF)")
    st.pyplot(fig)

with colB:
    fpr, tpr, _ = roc_curve(y, y_proba)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "--")
    ax.set_title("ROC Curve (RF)")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.legend()
    st.pyplot(fig)

st.subheader("📈 Precision–Recall Curve (Random Forest)")

prec, rec, _ = precision_recall_curve(y, y_proba)
ap = average_precision_score(y, y_proba)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(rec, prec, label=f"AP = {ap:.4f}")
ax.set_title("Precision–Recall Curve (RF)")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.legend()
st.pyplot(fig)
