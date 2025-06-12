from flask import Flask, request, jsonify
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import os

app = Flask(__name__)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load FAQs (do not embed all at once)
df = pd.read_csv("faqs.csv")
questions = df['Question'].tolist()
answers = df['Answer'].tolist()

@app.route('/')
def home():
    return "Chatbot is live!"

@app.route('/chatbot', methods=['POST'])
def chatbot():
    user_input = request.json.get('message')
    if not user_input:
        return jsonify({'response': "Please provide a question."})

    # Dynamically embed all questions (more CPU, less memory)
    question_embeddings = model.encode(questions, convert_to_tensor=True)
    input_embedding = model.encode(user_input, convert_to_tensor=True)
    scores = util.cos_sim(input_embedding, question_embeddings)
    best_idx = scores.argmax().item()
    confidence = scores[0][best_idx].item()

    if confidence > 0.6:
        return jsonify({'response': answers[best_idx]})
    else:
        return jsonify({'response': "Sorry, I don't know the answer to that."})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
