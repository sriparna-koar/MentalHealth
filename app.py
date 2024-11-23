# from flask import Flask, render_template, request, jsonify, redirect, url_for
# from pymongo import MongoClient
# from models.HuggingFaceAPI import HuggingFaceModel
# import datetime

# app = Flask(__name__)

# # MongoDB setup
# client = MongoClient("mongodb+srv://koarsk03:bCfmp6zruE29urxj@cluster0.jzro4.mongodb.net/")
# db = client["mental_health_db"]
# moods_collection = db["moods"]
# journals_collection = db["journals"]

# # Hugging Face API
# hugging_face_model = HuggingFaceModel(api_key="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")

# @app.route("/")
# def home():
#     return render_template("index.html")

# @app.route("/mood-tracker", methods=["GET", "POST"])
# def mood_tracker():
#     if request.method == "POST":
#         mood = request.form.get("mood")
#         date = datetime.datetime.now()
#         analysis = hugging_face_model.analyze_sentiment(mood)
#         moods_collection.insert_one({"mood": mood, "date": date, "analysis": analysis})
#         return jsonify({"status": "success", "message": "Mood recorded successfully!"})
#     moods = list(moods_collection.find())
#     return render_template("mood_tracker.html", moods=moods)

# @app.route("/journal", methods=["GET", "POST"])
# def journal():
#     if request.method == "POST":
#         entry = request.form.get("entry")
#         date = datetime.datetime.now()
#         analysis = hugging_face_model.analyze_sentiment(entry)
#         journals_collection.insert_one({"entry": entry, "date": date, "analysis": analysis})
#         return jsonify({"status": "success", "message": "Journal entry saved!"})
#     entries = list(journals_collection.find())
#     return render_template("journal.html", entries=entries)
# if __name__ == "__main__":
#     app.run(debug=True)



# from flask import Flask, render_template, request
# from transformers import pipeline
# import requests

# app = Flask(__name__)

# # Hugging Face API details
# API_URL = "https://api-inference.huggingface.co/models/"
# headers = {"Authorization": f"Bearer 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"}

# # Helper function to query Hugging Face API
# def query_huggingface_api(model, payload):
#     try:
#         response = requests.post(f"{API_URL}{model}", headers=headers, json=payload)
#         response.raise_for_status()  # Raise an error for HTTP errors
#         return response.json()
#     except requests.exceptions.RequestException as e:
#         print(f"Error querying the Hugging Face API: {e}")
#         return {"error": str(e)}

# # Use transformer pipeline for local inference
# def query_huggingface_local(model_name, input_text):
#     try:
#         pipe = pipeline("text-classification", model=model_name)
#         return pipe(input_text)
#     except Exception as e:
#         print(f"Error loading the model locally: {e}")
#         return {"error": str(e)}

# @app.route('/')
# def index():
#     return render_template('index.html')
# @app.route('/analyze', methods=['POST'])
# def analyze():
#     task = request.form.get('task')
#     input_text = request.form.get('input_text')
#     context = request.form.get('context') if task == "query" else None

#     if task == "symptoms":
#         model_name = "Zabihin/Symptom_to_Diagnosis"
#         try:
#             # Attempt API query
#             payload = {"inputs": input_text}
#             response = query_huggingface_api(model_name, payload)

#             # Fallback to local inference if API fails
#             if "error" in response:
#                 response = query_huggingface_local(model_name, input_text)

#             # Validate response structure
#             if isinstance(response, list) and len(response) > 0 and "label" in response[0]:
#                 result = str(response)  # Convert to string format for output
#             else:
#                 result = "Unexpected response format. Unable to process."
#         except Exception as e:
#             result = f"Error processing the request: {e}"

#     elif task == "summarize":
#         model = "t5-small"
#         payload = {"inputs": input_text}
#         response = query_huggingface_api(model, payload)
#         result = response.get("summary_text", "Unable to summarize.") if isinstance(response, dict) else "Error."

#     elif task == "query":
#         model = "deepset/roberta-base-squad2"
#         payload = {"inputs": {"question": input_text, "context": context}}
#         response = query_huggingface_api(model, payload)
#         result = response.get("answer", "No answer found.") if isinstance(response, dict) else "Error."

#     else:
#         result = "Invalid task selected."

#     return render_template('result.html', task=task.capitalize(), result=result)

# # @app.route('/analyze', methods=['POST'])
# # def analyze():
# #     task = request.form.get('task')
# #     input_text = request.form.get('input_text')
# #     context = request.form.get('context') if task == "query" else None

# #     if task == "symptoms":
# #         model_name = "Zabihin/Symptom_to_Diagnosis"
# #         try:
# #             # Attempt API query
# #             payload = {"inputs": input_text}
# #             response = query_huggingface_api(model_name, payload)
# #             if "error" in response:
# #                 # Fall back to local inference
# #                 response = query_huggingface_local(model_name, input_text)
# #             result = response[0]["label"] if isinstance(response, list) else "Unable to process."
# #         except Exception as e:
# #             result = f"Error processing the request: {e}"

# #     elif task == "summarize":
# #         model = "t5-small"
# #         payload = {"inputs": input_text}
# #         response = query_huggingface_api(model, payload)
# #         result = response.get("summary_text", "Unable to summarize.") if isinstance(response, dict) else "Error."

# #     elif task == "query":
# #         model = "deepset/roberta-base-squad2"
# #         payload = {"inputs": {"question": input_text, "context": context}}
# #         response = query_huggingface_api(model, payload)
# #         result = response.get("answer", "No answer found.") if isinstance(response, dict) else "Error."

# #     else:
# #         result = "Invalid task selected."

# #     return render_template('result.html', task=task.capitalize(), result=result)

# if __name__ == '__main__':
#     app.run(debug=True)

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

    if task == "symptoms":
        model_name = "Zabihin/Symptom_to_Diagnosis"
        try:
            payload = {"inputs": input_text}
            response = query_huggingface_api(model_name, payload)

            if "error" in response:
                print(f"API Error: {response['error']}")
                response = query_huggingface_local(model_name, input_text)

            if isinstance(response, list) and len(response) > 0 and "label" in response[0]:
                predicted_label = response[0]['label']
                predicted_score = response[0]['score']
                result = f"Predicted Diagnosis: {predicted_label} (Confidence: {predicted_score:.2f})"
            else:
                result = f"Unexpected response format: {response}"
        except Exception as e:
            result = f"Error processing the request: {e}"
    else:
        result = "Invalid task selected."

    return render_template('result.html', task=task.capitalize(), result=result)

# @app.route('/analyze', methods=['POST'])
# def analyze():
#     task = request.form.get('task')
#     input_text = request.form.get('input_text')
#     context = request.form.get('context') if task == "query" else None

#     if task == "symptoms":
#         model_name = "Zabihin/Symptom_to_Diagnosis"
#         try:
#             pipe = pipeline("text-classification", model=model_name, tokenizer=model_name)
#             response = pipe(input_text)
#             if "error" in response:
#                 response = query_huggingface_local(model_name, input_text)
#             if isinstance(response, list) and len(response) > 0 and "label" in response[0]:
#                 predicted_label = response[0]['label']
#                 predicted_score = response[0]['score']
#                 result = f"Predicted Diagnosis: {predicted_label} (Confidence: {predicted_score:.2f})"
#             else:
#                 result = f"Unexpected response format: {response}"
#         except Exception as e:
#             result = f"Error processing the request: {e}"

#     elif task == "summarize":
#         model = "t5-small"
#         payload = {"inputs": input_text}
#         response = query_huggingface_api(model, payload)
#         result = response.get("summary_text", "Unable to summarize.") if isinstance(response, dict) else "Error."

#     elif task == "query":
#         model = "deepset/roberta-base-squad2"
#         payload = {"inputs": {"question": input_text, "context": context}}
#         response = query_huggingface_api(model, payload)
#         result = response.get("answer", "No answer found.") if isinstance(response, dict) else "Error."

#     else:
#         result = "Invalid task selected."

#     return render_template('result.html', task=task.capitalize(), result=result)

if __name__ == '__main__':
    app.run(debug=True)











# from flask import Flask, render_template, request
# import requests

# app = Flask(__name__)

# # Hugging Face API details
# API_URL = "https://api-inference.huggingface.co/models/"
# headers = {"Authorization": f"Bearer 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"}

# # Helper function to query Hugging Face models
# # def query_huggingface(model, payload):
# #     response = requests.post(f"{API_URL}{model}", headers=headers, json=payload)
# #     return response.json()
# def query_huggingface(model, payload):
#     try:
#         response = requests.post(f"{API_URL}{model}", headers=headers, json=payload)
#         response.raise_for_status()  # Raise an error for HTTP errors
#         return response.json()
#     except requests.exceptions.RequestException as e:
#         print(f"Error querying the Hugging Face API: {e}")
#         return {"error": str(e)}

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/analyze', methods=['POST'])
# def analyze():
#     # Get form inputs
#     task = request.form.get('task')
#     input_text = request.form.get('input_text')
#     context = request.form.get('context') if task == "query" else None

#     # Model selection and input processing
#     if task == "symptoms":
#         model = "Zabihin/Symptom_to_Diagnosis"
#         payload = {"inputs": input_text}
#         response = query_huggingface(model, payload)
#         # result = response[0]['label'] if isinstance(response, list) else "Unable to process."
#         result = response.get("label", "Unable to process.") if isinstance(response, dict) else "Error."
#         print(f"Model: {model}")
#         print(f"Payload: {payload}")

        
#     elif task == "summarize":
#         model = "t5-small"
#         payload = {"inputs": input_text}
#         response = query_huggingface(model, payload)
#         result = response.get("summary_text", "Unable to summarize.") if isinstance(response, dict) else "Error."

#     elif task == "query":
#         model = "deepset/roberta-base-squad2"
#         payload = {"inputs": {"question": input_text, "context": context}}
#         response = query_huggingface(model, payload)
#         result = response.get("answer", "No answer found.") if isinstance(response, dict) else "Error."

#     else:
#         result = "Invalid task selected."

#     return render_template('result.html', task=task.capitalize(), result=result)

# if __name__ == '__main__':
#     app.run(debug=True)
