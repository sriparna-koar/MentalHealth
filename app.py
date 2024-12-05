

from flask import Flask, render_template, request
from transformers import pipeline
import requests

app = Flask(__name__)

# Hugging Face API details
API_URL = "https://api-inference.huggingface.co/models/"
headers = {"Authorization": f"Bearer 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"}

# Helper function to query Hugging Face API
def query_huggingface_api(model, payload):
    try:
        response = requests.post(f"{API_URL}{model}", headers=headers, json=payload)
        response.raise_for_status()  # Raise an error for HTTP errors
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error querying the Hugging Face API: {e}")
        return {"error": str(e)}

# Use transformer pipeline for local inference
def query_huggingface_local(model_name, input_text):
    try:
        pipe = pipeline("text-classification", model=model_name, tokenizer=model_name)
        return pipe(input_text)
    except Exception as e:
        print(f"Error loading the model locally: {e}")
        return {"error": str(e)}

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/analyze', methods=['POST'])
def analyze():
    task = request.form.get('task')
    input_text = request.form.get('input_text')
    context = request.form.get('context') if task == "query" else None

    result = "Invalid task selected."
    if task == "symptoms":
        model_name = "Zabihin/Symptom_to_Diagnosis"
        try:
            payload = {"inputs": input_text}
            response = query_huggingface_api(model_name, payload)

            if "error" in response:
                print(f"API Error: {response['error']}")
                response = query_huggingface_local(model_name, input_text)

            # Validate and parse response structure
            if isinstance(response, list) and len(response) > 0 and all(isinstance(item, dict) for item in response[0]):
                predictions = response[0]  # Assumes response is a nested list with dict elements
                sorted_predictions = sorted(predictions, key=lambda x: x['score'], reverse=True)

                # Display all predictions
                results_list = [
                    f"Diagnosis: {prediction['label']}, Confidence: {prediction['score']:.2f}"
                    for prediction in sorted_predictions
                ]
                result = "\n".join(results_list)  # Format for HTML rendering

            else:
                result = f"Unexpected response structure: {response}"
        except Exception as e:
            result = f"Error processing the request: {e}"
    elif task == "summarize":
        model_name = "Falconsai/medical_summarization"
        try:
            payload = {"inputs": input_text, "parameters": {"max_length": 512, "min_length": 150, "do_sample": False}}
            response = query_huggingface_api(model_name, payload)

            if "error" in response:
                print(f"API Error: {response['error']}")
                result = f"Error summarizing medical report: {response['error']}"
            elif "summary_text" in response:
                result = response["summary_text"]
            else:
                result = f" {response}"
        except Exception as e:
            result = f"Error processing the request: {e}"

    elif task == "query":
        model_name = "deepset/roberta-base-squad2"  # Example model for question answering
        try:
            payload = {"inputs": {"question": input_text, "context": context}}
            response = query_huggingface_api(model_name, payload)

            if "error" in response:
                print(f"API Error: {response['error']}")
                result = f"Error answering query: {response['error']}"
            elif "answer" in response:
                result = response["answer"]
            else:
                result = "Unable to process the query."
        except Exception as e:
            result = f"Error processing the request: {e}"

    return render_template('result.html', task=task.capitalize(), result=result)

if __name__ == '__main__':
    app.run(debug=True)








