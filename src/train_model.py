import pandas as pd
import numpy as np
import joblib
import os
import mlflow  # [ADDED] - MLflow tracking library
import mlflow.sklearn
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, f1_score  # [ADDED] f1_score for explicit logging

# [ADDED] - Initialize the connection to your MLflow server
mlflow.set_tracking_uri("http://localhost:5050")
mlflow.set_experiment("Fan_Retention_Professional")


def train_pro_system():
    # 1. PATHING: Setup absolute paths for data and models
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "raw", "fan_data.csv")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    # [ADDED] - Start a tracking context. All code inside this block is recorded as one "Run"
    with mlflow.start_run(run_name="Professional_Pipeline"):
        # 2. LOAD
        df = pd.read_csv(data_path)
        # We drop email here because it's a string and shouldn't go into XGBoost
        X = df.drop(columns=['fan_id', 'churn'], errors='ignore')
        y = df['churn']

        # 3. PREPROCESS: Handle missing values and scale (Your original logic)
        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)

        # 4. VALIDATE: Cross-validation to ensure model reliability
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_results = cross_val_score(XGBClassifier(), X_scaled, y, cv=skf, scoring='f1')

        # [ADDED] - Log the cross-validation score to the MLflow dashboard
        avg_f1 = np.mean(cv_results)
        mlflow.log_metric("cv_avg_f1", avg_f1)
        print(f"📊 Avg Cross-Val F1-Score: {avg_f1:.2f}")

        # 5. TRAIN: Fit model with early stopping to prevent overfitting
        X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.1, random_state=42)

        # [ADDED] - Log the hyperparameters to MLflow
        params = {"n_estimators": 1000, "learning_rate": 0.05, "early_stopping_rounds": 10}
        mlflow.log_params(params)

        model = XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        # [ADDED] - Calculate and log the final validation metric
        val_preds = model.predict(X_val)
        final_f1 = f1_score(y_val, val_preds)
        mlflow.log_metric("final_val_f1", final_f1)

        # 6. EXPORT: Save artifacts locally (Original logic)
        joblib.dump(imputer, os.path.join(models_dir, "imputer.pkl"))
        joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))
        joblib.dump(model, os.path.join(models_dir, "retention_model.pkl"))

        # [ADDED] - Version the model and artifacts in MLflow
        # This allows you to track which version of imputer/scaler goes with which model
        mlflow.sklearn.log_model(model, "retention_model")
        mlflow.log_artifacts(models_dir)

        print(f"✅ Success: Model artifacts saved and tracked in MLflow")


if __name__ == "__main__":
    train_pro_system()