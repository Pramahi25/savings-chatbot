from flask import Flask, request, jsonify
import pandas as pd
import os
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load FAQs from CSV
df = pd.read_csv('faqs.csv')
questions = df['Question'].tolist()
answers = df['Answer'].tolist()
embeddings = model.encode(questions, convert_to_tensor=True)

@app.route('/')
def home():
    return "Chatbot is running."

@app.route('/chatbot', methods=['POST'])
def chatbot():
    user_input = request.json.get('message')
    if not user_input:
        return jsonify({'response': "Please provide a question."})

    input_embedding = model.encode(user_input, convert_to_tensor=True)
    scores = util.cos_sim(input_embedding, embeddings)
    best_index = scores.argmax().item()
    confidence = scores[0][best_index].item()

    if confidence > 0.6:
        return jsonify({'response': answers[best_index]})
    else:
        return jsonify({'response': "Sorry, I don’t know the answer to that."})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
