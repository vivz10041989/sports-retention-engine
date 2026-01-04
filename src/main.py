import os
import joblib
import pandas as pd
import mlflow

from fastapi import FastAPI
from pydantic import BaseModel, EmailStr
from contextlib import asynccontextmanager

from src.agent_engine import RetentionAgent


# -------- Lifespan (CONFIG ONLY) --------
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        mlflow.set_tracking_uri(
            os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5050")
        )
        mlflow.set_experiment("Fan_Retention_Monitoring")
        print("[STARTUP] MLflow configured")
    except Exception as e:
        # IMPORTANT: never crash FastAPI on MLflow issues
        print(f"[WARNING] MLflow init failed: {e}")

    yield



app = FastAPI(
    title="Sports Retention Agentic API",
    lifespan=lifespan
)

# -------- Load ML artifacts --------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_PATH = os.path.join(BASE_DIR, "models")

model = joblib.load(os.path.join(MODELS_PATH, "retention_model.pkl"))
scaler = joblib.load(os.path.join(MODELS_PATH, "scaler.pkl"))
imputer = joblib.load(os.path.join(MODELS_PATH, "imputer.pkl"))

agent = RetentionAgent()


class FanRequest(BaseModel):
    email: EmailStr
    matches_watched_loss: int
    app_opens_month: int
    membership_years: int


@app.post("/v1/predict")
def run_retention_engine(fan: FanRequest):
    data = fan.model_dump()
    target_email = data.pop("email")

    input_df = pd.DataFrame([data])
    processed_data = scaler.transform(imputer.transform(input_df))

    prob = model.predict_proba(processed_data)[0][1]
    prediction = int(model.predict(processed_data)[0])

    try:
        with mlflow.start_run(run_name="API_Inference_Event"):
            mlflow.log_param("target_fan", target_email)
            mlflow.log_metric("churn_probability", float(prob))
            mlflow.log_metric("prediction_label", prediction)
    except Exception as e:
        print(f"[WARNING] MLflow logging failed: {e}")

    agent_output = "No action taken. Fan is currently stable."

    if prob > 0.5:
        agent_output = agent.decide_and_act(fan.model_dump(), float(prob))
        try:
            with mlflow.start_run(run_name="Agent_Reasoning_Log", nested=True):
                mlflow.set_tag("agent_decision", agent_output[:250])
        except Exception as e:
            print(f"[WARNING] Agent MLflow logging failed: {e}")

    return {
        "fan_email": target_email,
        "risk_score": round(float(prob), 4),
        "agent_response": agent_output.replace("\n", " ")
    }
