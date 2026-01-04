import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

DATA_PATH = "Data/fan_data.csv"
MODEL_DIR = "models"

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["churn"])
y = df["churn"]

imputer = SimpleImputer(strategy="mean")
scaler = StandardScaler()

X_processed = scaler.fit_transform(imputer.fit_transform(X))

model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
model.fit(X_processed, y)

joblib.dump(model, f"{MODEL_DIR}/retention_model.pkl")
joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
joblib.dump(imputer, f"{MODEL_DIR}/imputer.pkl")

print("Training complete. Models saved.")
