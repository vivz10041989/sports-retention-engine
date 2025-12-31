import joblib
import pandas as pd
import os
import mlflow # To log real-time predictions
from fastapi import FastAPI
from pydantic import BaseModel, EmailStr  # Added EmailStr for validation
from agent_engine import RetentionAgent

# [ADDED] - Connect to the same tracking server used in training
mlflow.set_tracking_uri("http://localhost:5050")
mlflow.set_experiment("Fan_Retention_Monitoring") # Separate experiment for live calls
app = FastAPI(title="Sports Retention Agentic API")
agent = RetentionAgent()

# Path logic for models
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_PATH = os.path.join(BASE_DIR, "models")

# Load XGBoost Brain
model = joblib.load(os.path.join(MODELS_PATH, "retention_model.pkl"))
scaler = joblib.load(os.path.join(MODELS_PATH, "scaler.pkl"))
imputer = joblib.load(os.path.join(MODELS_PATH, "imputer.pkl"))


class FanRequest(BaseModel):
    email: EmailStr  # The agent will use this
    matches_watched_loss: int
    app_opens_month: int
    membership_years: int


@app.post("/v1/predict")
def run_retention_engine(fan: FanRequest):
    # Data cleaning for XGBoost (XGBoost doesn't like strings like email)
    data = fan.model_dump()
    target_email = data.pop('email')

    input_df = pd.DataFrame([data])
    processed_data = scaler.transform(imputer.transform(input_df))

    # 1. ML Prediction
    prob = model.predict_proba(processed_data)[0][1]
    prediction = int(model.predict(processed_data)[0])

    # [ADDED] - MLOps Monitoring Block
    # Every time this API is hit, we log the inputs and the probability to MLflow
    with mlflow.start_run(run_name="API_Inference_Event"):
        mlflow.log_param("target_fan", target_email)
        mlflow.log_metric("churn_probability", float(prob))
        mlflow.log_metric("prediction_label", prediction)

    # 2. Agent Action (Only if risk > 50%)
    agent_output = "No action taken. Fan is currently stable."

    if prob > 0.5:
        # We pass the full fan data back to include the email for the agent
        agent_output = agent.decide_and_act(fan.model_dump(), float(prob))

        # [ADDED] - Log the Agent's reasoning so we can audit it later
        with mlflow.start_run(run_name="Agent_Reasoning_Log", nested=True):
            mlflow.set_tag("agent_decision", agent_output[:250])

    return {
        "fan_email": target_email,
        "risk_score": round(float(prob), 4),
        "agent_response": agent_output.replace("\n", " ")
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)