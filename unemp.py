import streamlit as st
import pandas as pd
import xgboost as xgb
import os
import joblib




# 1. Page Setup for PhD Presentation
st.set_page_config(page_title="West Africa Resilience Predictor", layout="wide")

st.title("🌍 West Africa Unemployment AI Predictor")

st.markdown("""
**"An Explainable AI (XAI) Framework for Modeling Non-Linear Labor Market Decoupling and Resilience: A Case Study of West African Emerging Economies"
""")

st.markdown("""
**Explainable AI (XAI) Framework:** This dashboard uses an XGBoost Regressor to model the 
non-linear relationship between economic recovery and labor market outcomes.
""")


# 2. Model Loading Function (Serialized JSON Model)
@st.cache_resource
def load_phd_model():
    model_path = 'resilience_model.json'
    if os.path.exists(model_path):
        model = xgb.XGBRegressor()
        model.load_model(model_path)
        return model
    else:
        st.error(f"❌ Error: {model_path} not found in the directory!")
        return None

model = load_phd_model()

# 3. Sidebar User Inputs
st.sidebar.header("🕹️ Economic Feature Inputs")

# Numeric Sliders/Inputs
year = st.sidebar.slider("Current Year", 2020, 2030, 2024)
employment = st.sidebar.number_input("Raw Employment Rate (%)", 0.0, 100.0, 75.0)
baseline_2020 = st.sidebar.number_input("2020 Baseline Shock Score", -10.0, 10.0, 0.0)
recovery_score = st.sidebar.number_input("Economic Recovery Score", -10.0, 10.0, 1.5)

st.sidebar.header("👥 Demographic Inputs")
sex = st.sidebar.selectbox("Gender Category", ["Total", "Male", "Female"])
age = st.sidebar.selectbox("Age Bracket", ["15+", "15-24", "25+"])

# 4. THE PREDICTION BLOCK (The Logic)
if st.button("🚀 Predict Unemployment Rate"):
    if model is not None:
        # A. Manual Mapping (The 'Mitsubishi' Style)
        sex_map = {"Total": 0, "Male": 1, "Female": 2}
        age_map = {"15+": 0, "15-24": 1, "25+": 2}

        # B. Feature Alignment (Exact names used in training)
        # Note: 'sex_num' and 'age_num' must match the columns in your training set
        input_data = pd.DataFrame([[
            year, 
            employment, 
            baseline_2020, 
            recovery_score, 
            sex_map[sex], 
            age_map[age]
        ]], columns=['year', 'employment', 'baseline_2020', 'recovery_score', 'sex_num', 'age_num'])

        # C. Inference
        try:
            prediction = model.predict(input_data)[0]
            
            # Display Result
            st.success(f"### Predicted Unemployment: {prediction:.2f}%")
            
            # D. The "Eureka" Threshold Analysis (PhD Findings)
            st.subheader("💡 PhD Model Interpretation")
            col1, col2 = st.columns(2)
            
            with col1:
                if recovery_score > 4.0:
                    st.warning("⚠️ **Jobless Growth Threshold:** Recovery score exceeds 4.0. The model indicates a decoupling where growth fails to absorb labor.")
                else:
                    st.info("✅ **Stable Recovery:** Economic gains are currently within the absorptive capacity of the labor market.")
            
            with col2:
                if age == "15-24":
                    st.info("📊 **Youth Bulge Effect:** Model accounting for high-sensitivity demographic volatility in the 15-24 bracket.")

        except Exception as e:
            st.error(f"Model Inference Error: {e}")
    else:
        st.warning("Please ensure 'resilience_model.json' is in the same folder as this script.")

# 5. Thesis Footer
st.markdown("---")
st.caption("Developed for PhD Thesis: 'A Multi-Method XAI Framework for Non-Linear Labor Market Resilience'")
