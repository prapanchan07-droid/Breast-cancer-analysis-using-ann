import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.base import BaseEstimator, TransformerMixin

st.set_page_config(page_title="ANN Breast Tumor Classifier", page_icon="🧠", layout="wide")

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        X=X.copy()
        e=1e-9
        X["area_radius_ratio"]=X["area_mean"]/(X["radius_mean"]+e)
        X["perimeter_radius_ratio"]=X["perimeter_mean"]/(X["radius_mean"]+e)
        X["worst_mean_radius_ratio"]=X["radius_worst"]/(X["radius_mean"]+e)
        X["worst_mean_area_ratio"]=X["area_worst"]/(X["area_mean"]+e)
        return X

@st.cache_resource
def load_model():
    return joblib.load("ann_model.joblib")

model=load_model()

FEATURES=['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst']

st.title("🧠 Breast Tumor Classification using ANN")
st.write("Artificial Neural Network (Multi-Layer Perceptron) for binary classification.")
st.info("Architecture: 30 original features → 4 engineered features → StandardScaler → Dense 64 (ReLU) → Dense 32 (ReLU) → Output (Sigmoid).")
st.warning("Educational demonstration only. This is not a medical diagnostic tool.")

tab1,tab2=st.tabs(["Single Prediction","Dataset Row Prediction"])

with tab1:
    st.subheader("Enter tumor measurement features")
    st.caption("The values should be in the same format and units as the training dataset.")
    reference=pd.read_csv("data.csv").drop(columns=["diagnosis","id","Unnamed: 32"],errors="ignore")
    med=reference.median()
    values={}
    cols=st.columns(3)
    for i,f in enumerate(FEATURES):
        with cols[i%3]:
            values[f]=st.number_input(f,value=float(med[f]),format="%.6f")
    if st.button("Predict with ANN",type="primary",use_container_width=True):
        x=pd.DataFrame([values])
        p=int(model.predict(x)[0])
        pr=float(model.predict_proba(x)[0,1])
        if p==1: st.error("Prediction: MALIGNANT")
        else: st.success("Prediction: BENIGN")
        c1,c2=st.columns(2)
        c1.metric("Malignant probability",f"{pr:.2%}")
        c2.metric("Benign probability",f"{1-pr:.2%}")

with tab2:
    st.subheader("Predict an existing dataset row")
    data=pd.read_csv("data.csv")
    row_number=st.number_input("Select row number",min_value=0,max_value=len(data)-1,value=0,step=1)
    row=data.iloc[[int(row_number)]]
    st.dataframe(row,use_container_width=True)
    if st.button("Predict Selected Row",use_container_width=True):
        x=row[FEATURES]
        p=int(model.predict(x)[0])
        pr=float(model.predict_proba(x)[0,1])
        actual=row["diagnosis"].iloc[0] if "diagnosis" in row else "N/A"
        st.write(f"Actual dataset label: **{actual}**")
        st.write(f"ANN prediction: **{'Malignant' if p else 'Benign'}**")
        st.metric("Malignant probability",f"{pr:.2%}")

st.sidebar.header("ANN Architecture")
st.sidebar.write("Input: 30 features")
st.sidebar.write("Feature engineering: +4 ratios")
st.sidebar.write("Hidden Layer 1: 64 neurons, ReLU")
st.sidebar.write("Hidden Layer 2: 32 neurons, ReLU")
st.sidebar.write("Output: 1 neuron, probability")
st.sidebar.write("Optimizer: Adam")
