import joblib
import pandas as pd

model = joblib.load('../models/model.pkl')
cols = joblib.load('../models/columns.pkl')

def predict(data_dict):
    df = pd.DataFrame([data_dict])
    df = pd.get_dummies(df)
    df = df.reindex(columns=cols, fill_value=0)
    pred = model.predict(df)[0]
    return pred
