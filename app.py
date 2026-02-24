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


# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Intelligent Fishing Prediction", layout="wide")
st.title("🐟 Data-Driven Intelligent Fishing System")
st.caption("Fish Presence Prediction using Water Quality Parameters (4 Models + 5-Fold CV)")


# -------------------------------
# LABEL CREATION FUNCTION
# -------------------------------
def fish_presence(row):
    conds = [
        (row["WaterTemp (C)"] >= 20) & (row["WaterTemp (C)"] <= 30),
        (row["DissolvedOxygen (mg/L)"] >= 5),
        (row["pH"] >= 7.0) & (row["pH"] <= 8.5),
        (row["SecchiDepth (m)"] >= 0.2)
    ]
    return int(sum(bool(c) for c in conds) >= 3)


# -------------------------------
# SIDEBAR SETTINGS
# -------------------------------
st.sidebar.header("⚙️ Settings")
n_neighbors = st.sidebar.slider("KNN Neighbors (k)", 3, 15, 7)
rf_trees = st.sidebar.slider("Random Forest Trees", 50, 300, 150, step=50)

st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("📂 Upload your CSV dataset", type=["csv"])


# -------------------------------
# LOAD DATA
# -------------------------------
if uploaded_file is None:
    st.warning("⬅️ Please upload the dataset (waterquality.csv) to start.")
    st.stop()

df = pd.read_csv(uploaded_file)

st.subheader("📌 Dataset Preview")
st.write("✅ Shape:", df.shape)
st.dataframe(df.head(10), use_container_width=True)


# -------------------------------
# DATE + FEATURE ENGINEERING
# -------------------------------
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

# Create Label
df["FishPresence"] = df.apply(fish_presence, axis=1)

X = df[feature_cols]
y = df["FishPresence"]

st.subheader("🎯 Output Label: FishPresence (0/1)")
st.write("✅ Class Distribution:")
st.write(y.value_counts())


# -------------------------------
# PREPROCESSING
# -------------------------------
preprocess = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), feature_cols)
    ]
)

# -------------------------------
# MODELS
# -------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "KNN": KNeighborsClassifier(n_neighbors=n_neighbors),
    "SVM (RBF)": SVC(kernel="rbf", probability=True),
    "Random Forest": RandomForestClassifier(n_estimators=rf_trees, random_state=42)
}

# -------------------------------
# SCORING
# -------------------------------
scoring = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "roc_auc": "roc_auc"
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# -------------------------------
# RUN CV
# -------------------------------
st.subheader("✅ 5-Fold Cross Validation Results (4 Algorithms)")

results = []
for name, model in models.items():
    clf = Pipeline(steps=[
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

best_model_name = results_df.iloc[0]["Model"]
st.success(f"🏆 Best Model: {best_model_name}")


# -------------------------------
# PLOTS SECTION
# -------------------------------
st.subheader("📊 Performance Graphs")

col1, col2, col3 = st.columns(3)

# Accuracy Plot
with col1:
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(results_df["Model"], results_df["Accuracy"])
    ax.set_title("Accuracy")
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=35)
    st.pyplot(fig)

# F1 Plot
with col2:
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(results_df["Model"], results_df["F1-score"])
    ax.set_title("F1-score")
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=35)
    st.pyplot(fig)

# ROC-AUC Plot
with col3:
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(results_df["Model"], results_df["ROC-AUC"])
    ax.set_title("ROC-AUC")
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=35)
    st.pyplot(fig)


# -------------------------------
# CONFUSION MATRIX + ROC CURVE FOR RANDOM FOREST
# -------------------------------
st.subheader("📌 Best Model Output: Confusion Matrix + ROC Curve")

# Use Random Forest for final output
rf = RandomForestClassifier(n_estimators=rf_trees, random_state=42)
rf_pipe = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", rf)
])

# Out-of-Fold predictions
y_true, y_pred, y_proba = [], [], []

for train_idx, test_idx in cv.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    rf_pipe.fit(X_train, y_train)

    pred = rf_pipe.predict(X_test)
    proba = rf_pipe.predict_proba(X_test)[:, 1]

    y_true.extend(y_test)
    y_pred.extend(pred)
    y_proba.extend(proba)

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_proba = np.array(y_proba)

# Confusion Matrix Plot
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["Not Suitable (0)", "Suitable (1)"])

colA, colB = st.columns(2)

with colA:
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(values_format="d", ax=ax)
    ax.set_title("Confusion Matrix (Random Forest)")
    st.pyplot(fig)

with colB:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "--")
    ax.set_title("ROC Curve (Random Forest)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    st.pyplot(fig)


st.info("✅ Streamlit demo completed successfully!")
# -------------------------------
# CUSTOM LIVE PREDICTION SECTION
# -------------------------------
st.subheader("🔮 Live Fishing Suitability Prediction")

st.write("Enter new environmental parameters to predict fishing suitability.")

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
    # Train final RF model on full dataset
    rf_final = RandomForestClassifier(
        n_estimators=rf_trees,
        random_state=42
    )

    rf_final_pipe = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", rf_final)
    ])

    rf_final_pipe.fit(X, y)

    input_df = pd.DataFrame([[
        sal, do, ph_val, secchi,
        wdepth, wtemp, atemp, month_val
    ]], columns=feature_cols)

    prediction = rf_final_pipe.predict(input_df)[0]
    probability = rf_final_pipe.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.success(f"✅ Suitable for Fishing (Confidence: {probability:.2f})")
    else:
        st.error(f"❌ Not Suitable for Fishing (Confidence: {probability:.2f})")
st.write("Model Used: Random Forest")
st.write(f"Number of Trees: {rf_trees}")