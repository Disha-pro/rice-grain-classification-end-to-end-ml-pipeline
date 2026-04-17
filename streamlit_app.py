import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

st.set_page_config(page_title="Rice Grain Classifier", layout="wide")
st.title("🌾 Rice Grain Classification ML Pipeline")
st.markdown("**End-to-End ML Project | Random Forest | 96%+ Accuracy**")

# Load and train model (cached so it runs only once per session)
@st.cache_resource
def train_model():
    url = "https://download.mlcc.google.com/mledu-datasets/Rice_Cammeo_Osmancik.csv"
    rice = pd.read_csv(url)
    
    # Preprocessing (exact same as your project)
    le = LabelEncoder()
    rice['Class_encoded'] = le.fit_transform(rice['Class'])
    
    features_to_drop = ['Convex_Area', 'Class', 'Class_encoded']
    X = rice.drop(columns=features_to_drop)
    y = rice['Class_encoded']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    
    # Save for later use
    os.makedirs("models", exist_ok=True)
    joblib.dump(rf_model, "models/random_forest_rice.pkl")
    joblib.dump(scaler, "models/scaler_rice.pkl")
    
    return rf_model, scaler, le, X.columns.tolist()

model, scaler, le, feature_names = train_model()

# Sidebar - User Input
st.sidebar.header("🔢 Input Grain Features")
area = st.sidebar.slider("Area", 7000, 22000, 13000)
perimeter = st.sidebar.slider("Perimeter", 250, 700, 450)
major_axis = st.sidebar.slider("Major Axis Length", 80, 300, 180)
minor_axis = st.sidebar.slider("Minor Axis Length", 40, 120, 70)
eccentricity = st.sidebar.slider("Eccentricity", 0.70, 1.0, 0.85, 0.01)
extent = st.sidebar.slider("Extent", 0.40, 0.95, 0.75, 0.01)

if st.sidebar.button("🚀 Predict Class"):
    input_data = pd.DataFrame([[
        area, perimeter, major_axis, minor_axis, eccentricity, extent
    ]], columns=feature_names)
    
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]
    
    class_name = le.inverse_transform([prediction])[0]
    confidence = max(probability) * 100
    
    st.success(f"**Predicted Class: {class_name}**")
    st.info(f"Confidence: **{confidence:.1f}%**")
    
    # Show probability bar
    st.bar_chart(pd.DataFrame({
        "Cammeo": [probability[0]],
        "Osmancik": [probability[1]]
    }), use_container_width=True)

# Tabs for professional showcase
tab1, tab2, tab3 = st.tabs(["📊 Model Performance", "📈 Pipeline", "📝 About Project"])

with tab1:
    st.write("**Final Model Results (from training)**")
    st.write("- Random Forest Accuracy: **96.5%+**")
    st.write("- Logistic Regression: 92.8%")
    st.write("- Gradient Boosting: 95.9%")
    st.caption("Full metrics, confusion matrix & ROC-AUC available in notebook")

with tab2:
    st.write("**Complete ML Pipeline Used**")
    st.markdown("""
    1. EDA + Correlation Heatmap (dropped Convex_Area due to 0.999 correlation)
    2. Label Encoding + Stratified Train-Test Split
    3. Feature Scaling (StandardScaler - train only)
    4. Three models trained (LR baseline + RF + GB)
    5. 5-Fold Cross Validation + Feature Importance
    6. Model + Scaler saved with joblib
    """)

with tab3:
    st.write("**Built by Disha** (ex-Business Analyst → GenAI & Agentic AI Leader)")
    st.write("Part of Learnbay IBM Certified GenAI & Agentic AI Program (Feb 2026 batch)")
    st.caption("This project demonstrates end-to-end ML skills required for AI-first leadership roles.")

st.markdown("---")
st.caption("👉 Deployed with ❤️ using Streamlit | GitHub: rice-grain-classification-end-to-end-ml-pipeline")
