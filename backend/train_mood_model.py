# train_mood_model.py
import os
import pandas as pd
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

# ---------- CONFIG ----------
CSV_PATH = "emotion_dataset.csv"   # change if your file name is different
MODEL_OUT = "mood_model.joblib"
VECT_OUT = "vectorizer.joblib"
RANDOM_STATE = 42
TEST_SIZE = 0.15
# If your dataset has different label names, map them to our target moods:
LABEL_MAP = {
    # example mapping if dataset labels are like 'joy','sadness','anger' etc.
    "joy": "happy",
    "happiness": "happy",
    "happy": "happy",
    "sadness": "sad",
    "sad": "sad",
    "anger": "angry",
    "angry": "angry",
    "love": "romantic",
    "romantic": "romantic",
    "relaxed": "relaxed",
    "neutral": "relaxed",
    "surprise": "energetic",
    "fear": "sad",  # example (you can change)
    # add more if needed
}
# ----------------------------

def read_dataset(path):
    df = pd.read_csv(path)
    # try to find text + label columns
    text_col = None
    label_col = None
    candidates = [c.lower() for c in df.columns]
    for c in df.columns:
        cl = c.lower()
        if cl in ("text", "content", "sentence", "utterance", "message"):
            text_col = c
        if cl in ("label", "emotion", "sentiment", "class"):
            label_col = c
    # fallback: try first two columns
    if text_col is None:
        text_col = df.columns[0]
    if label_col is None:
        if len(df.columns) > 1:
            label_col = df.columns[1]
        else:
            raise ValueError("Cannot detect label column in dataset. Provide CSV with at least 2 columns or rename them to 'text' and 'label'.")
    df = df[[text_col, label_col]].dropna()
    df.columns = ["text", "label"]
    return df

def clean_text(t):
    t = str(t).lower()
    t = re.sub(r"http\S+|www\S+|https\S+", "", t)   # remove urls
    t = re.sub(r"[^a-zA-Z0-9\s']", " ", t)          # keep letters, numbers, apostrophe
    t = re.sub(r"\s+", " ", t).strip()
    # remove stopwords (english)
    stops = set(stopwords.words("english"))
    tokens = [w for w in t.split() if w not in stops]
    return " ".join(tokens)

def map_label(l):
    l = str(l).lower().strip()
    if l in LABEL_MAP:
        return LABEL_MAP[l]
    # try direct common replacements
    if l in ("joyful", "joyous"):
        return "happy"
    return l  # keep as-is if unknown; you can further map manually

def main():
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: dataset file not found at {CSV_PATH}. Put your CSV there or change CSV_PATH in the script.")
        return
    df = read_dataset(CSV_PATH)
    print("Loaded dataset. Example rows:")
    print(df.head(5))

    # map labels
    df['label'] = df['label'].apply(map_label)
    # optional: keep only a set of target moods
    allowed = {"happy","sad","angry","relaxed","romantic","energetic"}
    df = df[df['label'].isin(allowed)]
    print("After mapping/filtering label value counts:")
    print(df['label'].value_counts())

    # clean text
    df['text_clean'] = df['text'].apply(clean_text)

    # train-test split
    X = df['text_clean']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    # vectorize + model
    vect = TfidfVectorizer(ngram_range=(1,2), min_df=3, max_df=0.9)
    X_train_tfidf = vect.fit_transform(X_train)

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train_tfidf, y_train)

    # evaluate
    X_test_tfidf = vect.transform(X_test)
    preds = model.predict(X_test_tfidf)
    print("Test accuracy:", accuracy_score(y_test, preds))
    print("Classification report:")
    print(classification_report(y_test, preds))

    # save
    joblib.dump(model, MODEL_OUT)
    joblib.dump(vect, VECT_OUT)
    print(f"Saved model -> {MODEL_OUT} and vectorizer -> {VECT_OUT}")

if __name__ == "__main__":
    main()
