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

#  LIME
from lime.lime_text import LimeTextExplainer

USER_FOLDER = "user_data"

if not os.path.exists(USER_FOLDER):
    try:
        os.makedirs(USER_FOLDER)
    except FileExistsError:
        pass
elif not os.path.isdir(USER_FOLDER):
    os.rename(USER_FOLDER, USER_FOLDER + "_old")
    os.makedirs(USER_FOLDER)

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
explainer = LimeTextExplainer(class_names=list(lbl_encoder.classes_))

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

# --- Helper: Model prediction wrapper for LIME ---
def lime_predict_proba(texts):
    seqs = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(seqs, maxlen=max_len, padding=padding_type, truncating=padding_type)
    return model.predict(padded)

# --- Helper: Run LIME on a single message ---
from lime.lime_text import LimeTextExplainer

def explain_with_lime(user_text, num_features=5):
    """Run LIME explainability for a single user_text."""

    # 1️ Prediction wrapper for LIME
    def predictor(texts):
        seqs = tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(seqs, maxlen=max_len, padding=padding_type, truncating=padding_type)
        return model.predict(padded)

    # 2️ Improved LimeTextExplainer for short healthcare text
    explainer = LimeTextExplainer(
        class_names=list(lbl_encoder.classes_),   # all intent classes
        bow=False,                               # preserve sequence context
        split_expression=r"\W+",                 # split words properly
        kernel_width=5                           # makes LIME more sensitive
    )

    # 3️ Get explanation for the top predicted class
    exp = explainer.explain_instance(
        user_text,
        predictor,
        num_features=num_features,
        top_labels=1
    )

    # 4️ Get the top predicted label
    top_label = exp.top_labels[0]

    # 5️ Return feature weights for that label
    return exp.as_list(label=top_label)

# --- Helper: Generate explanation text ---
def generate_explanation(top3_conf, impactful_words):
    exp = []
    if top3_conf:
        exp.append("**Top intent predictions:**")
        for intent, conf in top3_conf:
            exp.append(f"- {intent} ({conf}%)")
    if impactful_words:
        exp.append(
            f"**Impactful words** that influenced my response: {', '.join(impactful_words)}"
        )
    exp.append(" *Based on these signals, I selected a supportive response for you.*")
    return "\n".join(exp)

def make_json_safe(obj):
    """Recursively convert NumPy / tuple / non-JSON types to plain Python types."""
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    elif isinstance(obj, tuple):
        return [make_json_safe(v) for v in obj]
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    else:
        return obj

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
                # Save immediately
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
    # Detect emotion
    emotion, score = detect_emotion(user_input)
    user["emotion_log"].append({"emotion": emotion, "score": score})

    # Predict intent + confidence
    inp_clean = user_input.lower().strip()
    top3_conf = []
    impactful_words = []

    if inp_clean in fallback_tags:
        tag = fallback_tags[inp_clean]
    else:
        seq = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(seq, maxlen=max_len, padding=padding_type, truncating=padding_type)
        pred = model.predict(padded)[0]
        sorted_conf = sorted(
            dict(zip(lbl_encoder.classes_, pred)).items(),
            key=lambda x: x[1],
            reverse=True
        )
        if np.max(pred) < 0.4:
            tag = "unclear_input"
        else:
            tag = sorted_conf[0][0]
            top3_conf = [
                [intent, f"{score * 100:.2f}"] for intent, score in sorted_conf[:3]
            ]

        # Find impactful negative words
        word_scores = {w: sia.polarity_scores(w)["compound"] for w in user_input.split()}
        impactful_words = [str(w) for w, s in word_scores.items() if s <= -0.2]

    # Generate bot reply
    bot_response = random.choice(
        responses.get(tag, ["I'm here to listen. Can you say that again?"])
    )

    # JSON-safe chat record
    user["chat_history"].append({
        "user_msg": str(user_input),
        "bot_msg": str(bot_response),
        "emotion": str(emotion),
        "top3_conf": top3_conf,
        "impactful_words": impactful_words
    })

    # Make user JSON-safe and save
    safe_user = make_json_safe(user)
    if "ref" in user:
        with open(f"user_data/{user['ref']}.json", "w") as f:
            json.dump(safe_user, f, indent=2)

# --- Display Chat Log ---
for i, chat in enumerate(user.get("chat_history", [])):
    st.markdown(f"**{user['nickname']}**: {chat['user_msg']}")
    st.markdown(f" *({chat['emotion']})*")
    st.markdown(f"**ResiliBot:** {chat['bot_msg']}")

    # Basic explanation
    with st.expander("💡 Why this response?"):
        explanation_text = generate_explanation(chat.get("top3_conf", []), chat.get("impactful_words", []))
        st.markdown(explanation_text)

        #  NEW: Deep LIME Button
        if st.button(f"🔍 Run Deep Explainability (LIME) for: '{chat['user_msg'][:20]}...'",key=f"lime_button_{i}"):

            with st.spinner("Running LIME... please wait ⏳"):
                lime_results = explain_with_lime(chat['user_msg'])

                # Sort for better visualization
                lime_results_sorted = sorted(lime_results, key=lambda x: abs(x[1]), reverse=True)

                # 1️.Textual Explanation
                st.subheader("Top words influencing this prediction:")
                for w, weight in lime_results_sorted:
                    st.write(f"- **{w}** → {round(weight, 3)}")

                # 2️.Colored Horizontal Bar Chart
                st.subheader("Contribution visualization")
                words = [w for w, _ in lime_results_sorted]
                weights = [w_val for _, w_val in lime_results_sorted]
                fig, ax = plt.subplots(figsize=(6, 3))
                colors = ["green" if w > 0 else "red" for w in weights]
                ax.barh(words, weights, color=colors)
                ax.axvline(0, color="black", linewidth=1)
                ax.set_xlabel("Influence on Prediction")
                ax.set_ylabel("Words")
                plt.tight_layout()
                st.pyplot(fig)

                # 3️. Highlighted Sentence
                st.subheader("Highlighted sentence")
                highlighted_sentence = []
                word_dict = {w: val for w, val in lime_results}
                for token in chat['user_msg'].split():
                    if token in word_dict:
                        weight = word_dict[token]
                        color = "rgba(0, 200, 0, 0.3)" if weight > 0 else "rgba(255, 0, 0, 0.3)"
                        highlighted_sentence.append(
                            f"<span style='background-color:{color}; padding:2px'>{token}</span>"
                        )
                    else:
                        highlighted_sentence.append(token)
                st.markdown(" ".join(highlighted_sentence), unsafe_allow_html=True)

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
