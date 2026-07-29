import json
import pickle
import random

import numpy as np
import streamlit as st
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

# NLTK resources setup
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

lemmatizer = WordNetLemmatizer()

# Load saved assets
with open('intents.json') as f:
    intents = json.load(f)

with open('words.pkl', 'rb') as f:
    words = pickle.load(f)

with open('classes.pkl', 'rb') as f:
    classes = pickle.load(f)

model = load_model('chatbot_model.keras')


def clean_up_sentence(sentence):
    tokens = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(t.lower()) for t in tokens]


def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [1 if w in sentence_words else 0 for w in words]
    return np.array([bag], dtype=np.float32)


def predict_intent(sentence, confidence_threshold=0.5):
    bag = bag_of_words(sentence)
    prediction = model.predict(bag, verbose=0)[0]

    best_index = int(np.argmax(prediction))
    best_confidence = float(prediction[best_index])

    if best_confidence < confidence_threshold:
        return None, best_confidence

    return classes[best_index], best_confidence


def get_response(tag):
    for intent in intents['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "I'm not sure how to respond to that, but here's a free smile: 🙂"


# Streamlit Page Setup
st.set_page_config(
    page_title="GiggleBot - Joke Generator",
    page_icon="🤡",
)

st.title("🤡 GiggleBot")
st.write("An AI chatbot that delivers dad jokes, programming puns, knock-knock jokes, and quick one-liners — powered by a neural network trained with Deep Learning!")

st.divider()

# Session State Initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Hi! I'm GiggleBot 🤖. Ask me for a dad joke, programming humor, puns, or a knock-knock joke!"}
    ]

# Display Chat History
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User Input Handling
user_input = st.chat_input("Ask for a joke (e.g., 'Tell me a dad joke')...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    tag, confidence = predict_intent(user_input)

    if tag is None:
        reply = "Hmm, my sense of humor didn't catch that. Try asking for a dad joke, a programming joke, a pun, or a knock-knock joke!"
    else:
        reply = get_response(tag)

    st.session_state.chat_history.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.write(reply)
        st.caption(f"Detected intent category: `{tag}` (confidence: {confidence:.2f})" if tag else f"Low confidence: {confidence:.2f}")

st.divider()

# Info Sidebar / Expander
with st.expander("ℹ️ How GiggleBot works behind the scenes"):
    st.markdown("""
    1. **Joke Knowledge Base (`intents.json`)** — Example prompts grouped into funny categories (e.g., `dad_jokes`, `programming_jokes`, `puns`).
    2. **NLP Preprocessing** — Sentences are split into tokens and lemmatized (e.g., "jokes" → "joke") using NLTK.
    3. **Bag-of-Words Vectorization** — Converts the input sentence into a binary array indicating word presence.
    4. **Neural Network Prediction** — A Keras Dense Neural Network analyzes the vector and classifies the joke type requested.
    5. **Joke Delivery** — The chatbot picks a random funny response matching your requested category!
    """)