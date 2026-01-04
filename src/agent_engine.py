import os


class RetentionAgent:
    """
    Agentic AI for fan retention.
    Business logic preserved.
    Docker-safe lazy initialization.
    """

    def __init__(self):
        self.llm = None
        self.resend_client = None

    # -------- Lazy LLM init --------
    def _get_llm(self):
        if self.llm is None:
            from langchain_ollama import OllamaLLM

            self.llm = OllamaLLM(
                model="llama3.1:latest",
                base_url=os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
            )
        return self.llm

    # -------- Lazy Resend init --------
    def _get_resend(self):
        if self.resend_client is None:
            import resend

            resend.api_key = os.getenv("RESEND_API_KEY")
            if not resend.api_key:
                raise ValueError("RESEND_API_KEY missing in environment")

            self.resend_client = resend
        return self.resend_client

    # -------- Core logic (UNCHANGED) --------
    def decide_and_act(self, fan_data: dict, churn_prob: float) -> str:
        try:
            llm = self._get_llm()
            resend_client = self._get_resend()

            prompt = f"""
            You are a Sports Marketing Agent.
            Target Fan: {fan_data['email']}
            Tenure: {fan_data['membership_years']} years
            Churn Risk: {churn_prob:.2f}

            Task: Write a personalized, 1-line subject and a 1-line body.
            The fan loves the team but is discouraged by recent losses.
            Offer a {'50% discount code: LOYAL50' if churn_prob > 0.8 else '20% discount code: STAY20'}.

            Return the output in this format:
            SUBJECT: <subject>
            BODY: <body>
            """

            raw_response = llm.invoke(prompt)
            clean_response = raw_response.replace("\n", " | ")

            subject = raw_response.split("SUBJECT:")[1].split("BODY:")[0].strip()
            body = raw_response.split("BODY:")[1].strip()

            params = {
                "from": "Retention AI <onboarding@resend.dev>",
                "to": [fan_data["email"]],
                "subject": subject,
                "html": f"<strong>{body}</strong>",
            }

            resend_client.Emails.send(params)
            print(f"[AGENT] Email sent to {fan_data['email']}")

            return clean_response

        except Exception as e:
            return f"Agent Logic Failure: {str(e)}"
