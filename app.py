import sys
print("Python version in Streamlit Cloud:", sys.version)

print("Python version:", sys.version)

import streamlit as st
import json
import uuid
import numpy as np
import matplotlib.pyplot as plt
import nltk
import tempfile
from nltk.sentiment import SentimentIntensityAnalyzer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from sklearn.preprocessing import LabelEncoder
import random
import base64

# Download required NLTK data
nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

# Load ML components
model = load_model("trained_modelrev.h5")

with open("tokenizerrev.json") as f:
    tokenizer = tokenizer_from_json(f.read())

with open("label_encoderrev.json") as f:
    data = json.load(f)
lbl_encoder = LabelEncoder()
lbl_encoder.classes_ = np.array(data["classes"])

with open("MHResilience.json") as f:
    intents = json.load(f)

responses = {i["tag"]: i.get("responses", i.get("response")) for i in intents["intents"]}
max_len = 20
padding_type = 'post'

fallback_tags = {
    "breathing": "resilience_breathing_step1",
    "gratitude": "resilience_gratitude_step1",
    "grounding": "resilience_grounding_step1",
    "done with breathing 1": "resilience_breathing_step2",
    "done with breathing 2": "resilience_breathing_step3",
    "done with breathing 3": "resilience_breathing_end",
    "done with gratitude 1": "resilience_gratitude_step2",
    "done with gratitude 2": "resilience_gratitude_end",
    "done with grounding 1": "resilience_grounding_step2",
    "done with grounding 2": "resilience_grounding_step3",
    "done with grounding 3": "resilience_grounding_step4",
    "done with grounding 4": "resilience_grounding_step5",
    "done with grounding 5": "resilience_grounding_end"
}

# Initialize session state
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []

if "emotion_log" not in st.session_state:
    st.session_state.emotion_log = []

def detect_emotion(text):
    score = sia.polarity_scores(text)["compound"]
    if score >= 0.3:
        return "positive", score
    elif score <= -0.2:
        return "negative", score
    else:
        return "neutral", score

def chat(user_input):
    emotion, score = detect_emotion(user_input)
    st.session_state.emotion_log.append((emotion, score))

    inp_clean = user_input.lower().strip()
    if inp_clean in fallback_tags:
        tag = fallback_tags[inp_clean]
    else:
        seq = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(seq, maxlen=max_len, padding=padding_type, truncating=padding_type)
        pred = model.predict(padded)[0]
        if np.max(pred) < 0.4:
            tag = "unclear_input"
        else:
            tag = lbl_encoder.inverse_transform([np.argmax(pred)])[0]

    response = random.choice(responses.get(tag, ["I'm here to listen."]))
    st.session_state.chat_log.append((user_input, response, emotion))
    return response

def show_mood_chart():
    if len(st.session_state.emotion_log) < 2:
        return None, "Not enough data yet for chart."

    scores = [s for _, s in st.session_state.emotion_log]
    emotions = [e for e, _ in st.session_state.emotion_log]
    x = list(range(1, len(scores) + 1))

    plt.figure(figsize=(8, 4))
    plt.plot(x, scores, marker="o", color="blue", label="Sentiment Score")
    plt.axhline(0, color="gray", linestyle="--")
    for i, label in enumerate(emotions):
        plt.text(x[i], scores[i] + 0.05, label, fontsize=8, ha="center")
    plt.title("Your Emotional Trend")
    plt.xlabel("Message Count")
    plt.ylabel("Sentiment Score")
    plt.ylim(-1, 1)
    plt.grid(True)

    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    plt.savefig(temp.name)
    plt.close()

    return temp.name, "Here's your mood chart!"

# Streamlit App UI
st.set_page_config(page_title="ResiliBot", page_icon="🧠")
st.title("🧠 ResiliBot — Your Mental Health Companion")
st.markdown("Talk to the bot about your feelings. It understands and supports you.")

# Chat interface
user_input = st.text_input("How are you feeling today?", key="user_input")
if st.button("Send") and user_input:
    bot_reply = chat(user_input)
    st.markdown(f"**ResiliBot says:** {bot_reply}")

# Chat history
if st.session_state.chat_log:
    st.markdown("### 💬 Chat History")
    for user, bot, emo in st.session_state.chat_log:
        st.markdown(f"**You:** {user}")
        st.markdown(f"**ResiliBot:** {bot}  \n*Detected emotion: {emo}*")

# Mood chart generation
if st.button("Generate Mood Chart"):
    chart_path, msg = show_mood_chart()
    if chart_path:
        st.image(chart_path)
        with open(chart_path, "rb") as f:
            st.download_button("📥 Download Mood Chart", f, file_name="mood_chart.png")
    else:
        st.warning(msg)
