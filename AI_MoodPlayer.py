import pandas as pd
import pywhatkit as kit
import pickle
import random

# --- Load AI model + vectorizer ---
from joblib import load
model = load("backend/mood_model.joblib")
vectorizer = load("backend/vectorizer.joblib")


# --- Load song dataset ---
songs_df = pd.read_csv("backend/songs.csv")

print("🎧 Welcome to the AI Mood-Based Music Recommender 🎧")
print("Tell me how you're feeling today (e.g., 'I feel happy', 'I'm tired', 'Feeling low'): ")

# --- Step 1: Get user message ---
user_input = input("You: ")

# --- Step 2: Predict mood using AI model ---
input_tfidf = vectorizer.transform([user_input])
predicted_mood = model.predict(input_tfidf)[0]
print(f"\n🤖 Detected mood: {predicted_mood}")

# --- Step 3: Ask for language ---
print("\nWhich language would you prefer? (Tamil / Hindi / English)")
language = input("Language: ").strip().lower()

# --- Step 4: Filter dataset for mood + language ---
filtered = songs_df[(songs_df["mood"] == predicted_mood) & (songs_df["language"].str.lower() == language)]

if not filtered.empty:
   song = random.choice(filtered["song"].tolist())
   print(f"\n🎶 Based on your {predicted_mood} mood, playing: {song} ({language.title()})")
   kit.playonyt(song)
else:
   print("\n😅 Sorry, no songs found for that mood and language.")
