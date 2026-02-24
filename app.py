# ============================================================
# ADVANCED INTERPRETABILITY SECTION
# ============================================================

st.subheader("📊 Feature Importance (Random Forest)")

rf_full = RandomForestClassifier(n_estimators=rf_trees, random_state=42)
rf_full_pipe = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", rf_full)
])

rf_full_pipe.fit(X, y)

importances = rf_full_pipe.named_steps["model"].feature_importances_

fig, ax = plt.subplots(figsize=(6,4))
ax.barh(feature_cols, importances)
ax.set_title("Feature Importance")
st.pyplot(fig)


# ============================================================
# DATASET STATISTICS
# ============================================================

st.subheader("📋 Dataset Statistical Summary")
st.dataframe(df[feature_cols].describe(), use_container_width=True)


# ============================================================
# MODEL SELECTION FOR PREDICTION
# ============================================================

st.sidebar.markdown("---")
selected_model_name = st.sidebar.selectbox(
    "Select Model for Live Prediction",
    list(models.keys())
)

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

    model_pipe = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", selected_model)
    ])

    model_pipe.fit(X, y)

    input_df = pd.DataFrame([[
        sal, do, ph_val, secchi,
        wdepth, wtemp, atemp, month_val
    ]], columns=feature_cols)

    prediction = model_pipe.predict(input_df)[0]

    if hasattr(model_pipe.named_steps["model"], "predict_proba"):
        probability = model_pipe.predict_proba(input_df)[0][1]
    else:
        probability = 0.5  # fallback if probability not supported

    st.write(f"Model Used: {selected_model_name}")

    if prediction == 1:
        st.success(f"✅ Suitable for Fishing (Confidence: {probability:.2f})")
        st.info("Recommendation: Fishing trip likely efficient. Lower fuel risk.")
    else:
        st.error(f"❌ Not Suitable for Fishing (Confidence: {probability:.2f})")
        st.warning("Recommendation: High uncertainty. Consider postponing trip.")

    # Download option
    result_df = input_df.copy()
    result_df["Prediction"] = prediction
    result_df["Confidence"] = probability

    st.download_button(
        label="Download Prediction Result",
        data=result_df.to_csv(index=False),
        file_name="fishing_prediction_result.csv",
        mime="text/csv"
    )
st.markdown("### 🎯 Objective")
st.write("This system predicts fishing suitability using machine learning models trained on historical marine water-quality data.")