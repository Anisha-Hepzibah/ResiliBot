import streamlit as st
import json
import uuid
import random
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from tensorflow.keras.preprocessing.text import tokenizer_from_json  
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import os

# ✅ Ensure local user_data folder exists before anything
os.makedirs("user_data", exist_ok=True)

# --- Setup: Emotion model ---
nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

# --- Load model and encoders ---
model = load_model("trained_modelrev.h5")

with open("label_encoderrev.json") as f:
    label_data = json.load(f)
lbl_encoder = LabelEncoder()
lbl_encoder.classes_ = np.array(label_data["classes"])

with open("tokenizerrev.json") as f:
    tokenizer_json = f.read()
tokenizer = tokenizer_from_json(tokenizer_json)

with open("MHResilience.json") as f:
    intents = json.load(f)
responses = {i["tag"]: i.get("responses", i.get("response")) for i in intents["intents"]}

# --- Constants ---
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

# --- Emotion Detection ---
def detect_emotion(text):
    score = sia.polarity_scores(text)
    compound = score["compound"]
    if compound >= 0.3:
        emotion = "positive"
    elif compound <= -0.2:
        emotion = "negative"
    else:
        emotion = "neutral"
    return emotion, compound

# --- User Session Setup ---
def get_user_session():
    if "user_data" not in st.session_state:
        st.session_state.user_data = {}

    user_data = st.session_state.user_data

    if "nickname" not in user_data or "ref" not in user_data:
        st.markdown("### Hi there! Have you used ResiliBot before? (yes/no)")
        user_choice = st.text_input(">", key="user_check")

        if user_choice.lower().strip() == "no":
            nickname = st.text_input("What nickname should I use for you?", key="nickname_input")
            if nickname:
                ref = str(uuid.uuid4())[:8]
                st.session_state.user_data = {
                    "nickname": nickname,
                    "ref": ref,
                    "chat_history": [],
                    "emotion_log": []
                }
                # ✅ Save immediately so file exists
                with open(f"user_data/{ref}.json", "w") as f:
                    json.dump(st.session_state.user_data, f, indent=2)
                st.success(f"Thanks, {nickname}. Your reference number is: {ref} (save this!)")

        elif user_choice.lower().strip() == "yes":
            ref = st.text_input("Enter your reference number:", key="ref_input")
            if ref:
                user_file = f"user_data/{ref}.json"
                if os.path.exists(user_file):
                    with open(user_file) as f:
                        st.session_state.user_data = json.load(f)
                    st.success(f"Welcome back, {st.session_state.user_data['nickname']}! Your ref: {ref}")
                else:
                    st.warning("Sorry, that reference number wasn't found. Please try again.")
                    st.stop()

    return st.session_state.user_data

# --- Initialize Session ---
user = get_user_session()

# --- UI ---
st.title("🧠 ResiliBot: Talk. Breathe. Heal.")
if "nickname" in user:
    st.caption(f"Welcome, **{user['nickname']}** 🌿")

# --- Chat Form ---
with st.form("chat_form", clear_on_submit=True):
    if "nickname" in user:
        user_input = st.text_input(f"{user['nickname']}:", placeholder="Type your message here...")
    else:
        user_input = st.text_input("Say something:", placeholder="Type your message here...")
    submitted = st.form_submit_button("Send")

if submitted and user_input:
    emotion, score = detect_emotion(user_input)
    user["emotion_log"].append({"emotion": emotion, "score": score})

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

    bot_response = random.choice(responses.get(tag, ["I'm here to listen. Can you say that again?"]))
    user["chat_history"].append((user_input, bot_response, emotion))

    # ✅ Save after every message
    if "ref" in user:
        with open(f"user_data/{user['ref']}.json", "w") as f:
            json.dump(user, f, indent=2)

# --- Display Chat Log ---
for user_msg, bot_msg, em in user.get("chat_history", []):
    st.markdown(f"**{user['nickname']}**: {user_msg}")
    st.markdown(f" *({em})*")
    st.markdown(f"**ResiliBot:** {bot_msg}")
    st.markdown("---")

# --- Quit & Mood Chart ---
if st.button("Quit and Show Mood Chart 📊"):
    if len(user.get("emotion_log", [])) < 2:
        st.warning("Not enough conversation yet to generate a chart.")
    else:
        scores = [e["score"] for e in user["emotion_log"]]
        emotions = [e["emotion"] for e in user["emotion_log"]]
        x = list(range(1, len(scores) + 1))

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, scores, marker='o', linestyle='-', color='purple')
        ax.axhline(0, color='gray', linestyle='--')
        for i, (x_pt, label) in enumerate(zip(x, emotions)):
            ax.text(x_pt, scores[i] + 0.05, label, ha='center', fontsize=8)
        ax.set_ylim(-1, 1)
        ax.set_title(f"{user['nickname']}'s Emotional Trend")
        ax.set_xlabel("Message Count")
        ax.set_ylabel("Sentiment Score (-1 to 1)")
        ax.grid(True)
        st.pyplot(fig)
 streamlit as st
import json
import uuid
import random
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from tensorflow.keras.preprocessing.text import tokenizer_from_json  
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import os


# --- Setup: Emotion model ---
nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

# --- Load model and encoders ---
model = load_model("trained_modelrev.h5")

with open("label_encoderrev.json") as f:
    label_data = json.load(f)
lbl_encoder = LabelEncoder()
lbl_encoder.classes_ = np.array(label_data["classes"])

with open("tokenizerrev.json") as f:
    tokenizer_json = f.read()
tokenizer = tokenizer_from_json(tokenizer_json)

with open("MHResilience.json") as f:
    intents = json.load(f)
responses = {i["tag"]: i.get("responses", i.get("response")) for i in intents["intents"]}

# --- Constants ---
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

# --- Emotion Detection ---
def detect_emotion(text):
    score = sia.polarity_scores(text)
    compound = score["compound"]
    if compound >= 0.3:
        emotion = "positive"
    elif compound <= -0.2:
        emotion = "negative"
    else:
        emotion = "neutral"
    return emotion, compound

# --- User Session Setup ---
def get_user_session():
    if "user_data" not in st.session_state:
        st.session_state.user_data = {}

    user_data = st.session_state.user_data

    if "nickname" not in user_data or "ref" not in user_data:
        st.markdown("### Hi there! Have you used ResiliBot before? (yes/no)")
        user_choice = st.text_input(">", key="user_check")
        if user_choice.lower().strip() == "no":
            nickname = st.text_input("What nickname should I use for you?", key="nickname_input")
            if nickname:
                ref = str(uuid.uuid4())[:8]
                st.session_state.user_data = {
                    "nickname": nickname,
                    "ref": ref,
                    "chat_history": [],
                    "emotion_log": []
                }
                st.success(f"Thanks, {nickname}. Your reference number is: {ref} (save this!)")
        elif user_choice.lower().strip() == "yes":
             ref = st.text_input("Enter your reference number:", key="ref_input")
             if ref:
                 user_file = f"user_data/{ref}.json"
                 if os.path.exists(user_file):
                      with open(user_file) as f:
                          st.session_state.user_data = json.load(f)
                      st.success(f"Welcome back, {st.session_state.user_data['nickname']}! Your ref: {ref}")
                 else:
                      st.warning("Sorry, that reference number wasn't found. Please try again.")
                      st.stop()

    return st.session_state.user_data

# --- Initialize Session ---
user = get_user_session()

# --- UI ---
st.title("🧠 ResiliBot: Talk. Breathe. Heal.")
st.caption(f"Welcome, **{user['nickname']}** 🌿")

with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input(f"{user['nickname']}:", placeholder="Type your message here...")
    submitted = st.form_submit_button("Send")

if submitted and user_input:
    emotion, score = detect_emotion(user_input)
    user["emotion_log"].append({"emotion": emotion, "score": score})

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

    bot_response = random.choice(responses.get(tag, ["I'm here to listen. Can you say that again?"]))
    user["chat_history"].append((user_input, bot_response, emotion))
# Save user session to file after every interaction
if "user_data" in st.session_state:
    ref = st.session_state.user_data.get("ref")
    if ref:
        with open(f"user_data/{ref}.json", "w") as f:
            json.dump(st.session_state.user_data, f, indent=2)

# --- Display Chat Log ---
for user_msg, bot_msg, em in user.get("chat_history", []):
    st.markdown(f"**{user['nickname']}**: {user_msg}")
    st.markdown(f" *({em})*")
    st.markdown(f"**ResiliBot:** {bot_msg}")
    st.markdown("---")

# --- Quit & Mood Chart ---
if st.button("Quit and Show Mood Chart 📊"):
    if len(user["emotion_log"]) < 2:
        st.warning("Not enough conversation yet to generate a chart.")
    else:
        scores = [e["score"] for e in user["emotion_log"]]
        emotions = [e["emotion"] for e in user["emotion_log"]]
        x = list(range(1, len(scores)+1))

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, scores, marker='o', linestyle='-', color='purple')
        ax.axhline(0, color='gray', linestyle='--')
        for i, (x_pt, label) in enumerate(zip(x, emotions)):
            ax.text(x_pt, scores[i] + 0.05, label, ha='center', fontsize=8)
        ax.set_ylim(-1, 1)
        ax.set_title(f"{user['nickname']}'s Emotional Trend")
        ax.set_xlabel("Message Count")
        ax.set_ylabel("Sentiment Score (-1 to 1)")
        ax.grid(True)
        st.pyplot(fig)
