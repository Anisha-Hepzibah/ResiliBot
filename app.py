import gradio as gr
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

# Load components
nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()
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

# Session memory
chat_log = []
emotion_log = []

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
    emotion_log.append((emotion, score))

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
    chat_log.append((user_input, response, emotion))
    return response

def show_mood_chart():
    if len(emotion_log) < 2:
        return None, "Not enough data yet for chart."

    scores = [s for _, s in emotion_log]
    emotions = [e for e, _ in emotion_log]
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

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 🧠 ResiliBot — Your Mental Health Companion")
    chatbox = gr.Textbox(lines=2, placeholder="How are you feeling today?")
    output = gr.Textbox(label="ResiliBot says...")
    send_btn = gr.Button("Send")
    img_output = gr.Image(label="Mood Chart", visible=False)
    download_btn = gr.Button("Download Mood Chart")

    file_output = gr.File(label="Download Chart File", visible=False)
    message = gr.Textbox(visible=False)

    def handle_send(text):
        reply = chat(text)
        return reply

    def handle_download():
        chart_path, msg = show_mood_chart()
        return chart_path, chart_path if chart_path else None, msg

    send_btn.click(handle_send, inputs=chatbox, outputs=output)
    download_btn.click(handle_download, outputs=[img_output, file_output, message])

demo.launch()
