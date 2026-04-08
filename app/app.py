import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

model = joblib.load('../models/model.pkl')
cols = joblib.load('../models/columns.pkl')

st.set_page_config(page_title="Churn Dashboard", layout="wide")

st.title("📊 Customer Churn Prediction Dashboard")

uploaded = st.file_uploader("Upload CSV dataset", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Dataset Preview")
        st.dataframe(df.head())

    with col2:
        st.subheader("Churn Distribution")
        fig = px.histogram(df, x="Churn", color="Churn")
        st.plotly_chart(fig)

    st.subheader("Feature Relationship")
    fig2 = px.scatter(df, x="tenure", y="MonthlyCharges", color="Churn")
    st.plotly_chart(fig2)

    st.subheader("Feature Importance (Model Based)")
    importances = model.feature_importances_
    feat_df = pd.DataFrame({"Feature": cols, "Importance": importances})
    feat_df = feat_df.sort_values(by="Importance", ascending=False).head(10)
    fig3 = px.bar(feat_df, x="Importance", y="Feature", orientation='h')
    st.plotly_chart(fig3)

    st.subheader("Prediction Example")
    sample = df.iloc[0:1]
    sample = pd.get_dummies(sample)
    sample = sample.reindex(columns=cols, fill_value=0)

    pred = model.predict(sample)[0]

    if pred == 1:
        st.error("Customer likely to churn ❌")
    else:
        st.success("Customer will stay ✅")
