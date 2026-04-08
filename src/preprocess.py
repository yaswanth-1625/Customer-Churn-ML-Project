import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    df.drop('customerID', axis=1, inplace=True)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.fillna(df.median(numeric_only=True), inplace=True)
    return df

def preprocess(df):
    df['Churn'] = df['Churn'].map({'Yes':1,'No':0})
    df = pd.get_dummies(df, drop_first=True)
    return df
