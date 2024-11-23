import requests
api_key="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"
class HuggingFaceModel:
    def __init__(self, api_key):
        self.api_key = api_key
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.sentiment_url = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment"
        self.cbt_url = "https://api-inference.huggingface.co/models/google/flan-t5-large"

    def analyze_sentiment(self, text):
        payload = {"inputs": text}
        try:
            response = requests.post(self.sentiment_url, headers=self.headers, json=payload)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": f"Sentiment analysis error: {str(e)}"}

    