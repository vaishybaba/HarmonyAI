import logging
logging.basicConfig(level=logging.DEBUG)
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import random
from joblib import load
import os

app = Flask(__name__, static_folder="static")
CORS(app)

model = load(os.path.join("backend", "mood_model.joblib"))
vectorizer = load(os.path.join("backend", "vectorizer.joblib"))

# Load songs dataset
songs_df = pd.read_csv(os.path.join("backend","songs.csv"))

@app.route("/ping", methods=["GET"])
def ping():
    return "pong"

# --- Serve frontend ---
@app.route("/")
def serve_index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/recommend", methods=["POST"])
def recommend_song():
    try:
        data = request.json
        print("📥 Received data:", data)
        return jsonify({"message": "Backend is working!", "received": data})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5500)
