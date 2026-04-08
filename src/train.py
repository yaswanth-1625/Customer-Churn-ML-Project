import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from preprocess import load_data, preprocess

df = load_data("C:/Users/Yash/Downloads/customer-churn-advanced/data/churn.csv")
df = preprocess(df)

X = df.drop('Churn', axis=1)
y = df['Churn']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = RandomForestClassifier(n_estimators=200)
model.fit(X_train,y_train)

pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test,pred))
print(classification_report(y_test,pred))

#joblib.dump(model,'../models/model.pkl')
import os

os.makedirs('../models', exist_ok=True)
joblib.dump(model, '../models/model.pkl')
joblib.dump(X.columns,'../models/columns.pkl')
