:

🏆 Sports Fan Retention Engine (Agentic MLOps)
A hybrid AI system that combines Classical Machine Learning (XGBoost) with Agentic Generative AI (Llama 3.1) to predict fan churn and execute automated, personalized retention strategies.

🚀 The Architecture
This project demonstrates a full-stack AI lifecycle:

Predictive Layer: XGBoost classifies fans as "High Risk" based on behavioral data.

Reasoning Layer: An Agent (Llama 3.1 via Ollama) analyzes the risk and decides on a personalized discount strategy.

Action Layer: The Agent dispatches real emails to fans using the Resend API.

MLOps Layer: MLflow tracks every training experiment and every live API inference for full auditability.

🛠️ Tech Stack
Language: Python 3.13

ML Framework: XGBoost, Scikit-Learn

LLM Engine: Ollama (Llama 3.1)

API Framework: FastAPI (Uvicorn)

MLOps: MLflow + SQLite

Communication: Resend API

🏃‍♂️ How to Run

1. Environment Setup

Bash

_Install dependencies_ -

pip install -r requirements.txt

_Start Ollama (Ensure llama3.1 is pulled)_

ollama run llama3.1

2. Start MLOps Tracking Server
Bash

mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --port 5050

3. Train the Model (with Tracking)
Bash

python src/train_model.py
View metrics at http://localhost:5050

4. Launch the Agentic API

Bash

python src/main.py

Test the system at http://127.0.0.1:8000/docs

📊 MLOps Insights

We use MLflow to ensure the Agent's decisions are transparent. Every time the API is called:

The Churn Probability is logged as a metric.

The Agent's Reasoning is logged as a tag.

The Target Email is recorded for auditing.

📧 Example Response

JSON

{
  "email_targeted": "vivek.katariyajewels@gmail.com",
  "churn_probability": 0.9213,
  "agent_action_summary": "High risk detected. Dispatched 'LOYAL50' discount via Resend. Reason: Fan has watched 9 losses with low app engagement."
}

NOTE: MLFLOW DOES NOT WORK IN THE LATEST PUSH