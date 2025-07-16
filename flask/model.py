# model_train.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings

warnings.filterwarnings("ignore")

# ---------- Load dataset ----------
df = pd.read_csv('anemia.csv')

# ---------- Data Exploration (optional) ----------
print(df.head())
print(df.info())
print(df.shape)
print(df.isnull().sum())

# ---------- Balance dataset ----------
from sklearn.utils import resample
majorclass = df[df['Result'] == 0]
minorclass = df[df['Result'] == 1]
major_downsample = resample(majorclass, replace=False, n_samples=len(minorclass), random_state=42)
df = pd.concat([major_downsample, minorclass])

print("Balanced class distribution:\n", df['Result'].value_counts())

# ---------- Feature and target split ----------
X = df.drop('Result', axis=1)
Y = df['Result']

# ---------- Train-test split ----------
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=20)

# ---------- Gradient Boosting Classifier (Final Model) ----------
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

GBC = GradientBoostingClassifier()
GBC.fit(x_train, y_train)
y_pred = GBC.predict(x_test)

# ---------- Evaluation ----------
print("Gradient Boosting Classifier Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ---------- Save trained model ----------
pickle.dump(GBC, open("model.pkl", "wb"))
print("✅ Model trained and saved as model.pkl successfully.")
