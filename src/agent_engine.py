import os
import resend  # The Transactional Email Tool
from langchain_ollama import OllamaLLM

# 1. CONFIGURE THE TOOL
# Replace 're_12345' with your actual key from Resend
resend.api_key = "re_5bMp7gDn_Cc9uaSrWZoz6ATR9gschM2xN"


class RetentionAgent:
    def __init__(self):
        # Using llama3.1 as it's better at following instructions than 3.2:1b
        self.llm = OllamaLLM(model="llama3.1:latest")

    def decide_and_act(self, fan_data, churn_prob):
        try:
            # Tell Llama 3 to act as a Marketing Executive
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

            raw_response = self.llm.invoke(prompt)

            # Clean newlines for the API response
            clean_response = raw_response.replace("\n", " | ")

            # 2. THE TOOL EXECUTION: Sending to any email address
            params = {
                "from": "Retention AI <onboarding@resend.dev>",  # Resend provides this default sender
                "to": [fan_data['email']],
                "subject": raw_response.split("SUBJECT:")[1].split("BODY:")[0].strip(),
                "html": f"<strong>{raw_response.split('BODY:')[1].strip()}</strong>",
            }

            email_response = resend.Emails.send(params)
            print(f"🚀 [AGENT] Real email dispatched to {fan_data['email']} via Resend API.")

            return clean_response

        except Exception as e:
            return f"Agent Logic Failure: {str(e)}"