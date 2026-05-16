# app.py
# Author: Divya Modi
# Description: Sentiment Analysis Web App using DistilBERT Transformers

import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# ─── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="SentimentIQ",
    page_icon="🧠",
    layout="centered"
)

# ─── Custom Styling ───────────────────────────────────────────────────────────
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

        html, body, [class*="css"] {
            font-family: 'DM Sans', sans-serif;
            background-color: #0f0f0f;
            color: #f0f0f0;
        }

        .title-block {
            text-align: center;
            padding: 2rem 0 1rem 0;
        }

        .title-block h1 {
            font-family: 'Space Mono', monospace;
            font-size: 2.8rem;
            color: #e2ff00;
            letter-spacing: -1px;
            margin-bottom: 0.2rem;
        }

        .title-block p {
            color: #888;
            font-size: 1rem;
            font-weight: 300;
        }

        .result-card {
            border-radius: 16px;
            padding: 1.5rem 2rem;
            margin-top: 1.5rem;
            text-align: center;
            font-family: 'Space Mono', monospace;
        }

        .result-positive {
            background: linear-gradient(135deg, #1a3a1a, #0d260d);
            border: 1px solid #4caf50;
        }

        .result-negative {
            background: linear-gradient(135deg, #3a1a1a, #260d0d);
            border: 1px solid #f44336;
        }

        .result-neutral {
            background: linear-gradient(135deg, #1a1a3a, #0d0d26);
            border: 1px solid #5c6bc0;
        }

        .result-label {
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 0.4rem;
        }

        .result-sub {
            font-size: 0.85rem;
            color: #aaa;
        }

        .stTextArea textarea {
            background-color: #1a1a1a !important;
            color: #f0f0f0 !important;
            border: 1px solid #333 !important;
            border-radius: 10px !important;
            font-family: 'DM Sans', sans-serif !important;
            font-size: 0.95rem !important;
        }

        div.stButton > button {
            background-color: #e2ff00;
            color: #0f0f0f;
            font-family: 'Space Mono', monospace;
            font-weight: 700;
            border: none;
            border-radius: 10px;
            padding: 0.6rem 2.5rem;
            font-size: 0.95rem;
            cursor: pointer;
            width: 100%;
            transition: opacity 0.2s ease;
        }

        div.stButton > button:hover {
            opacity: 0.85;
        }

        .prob-bar-label {
            font-size: 0.8rem;
            color: #aaa;
            margin-bottom: 0.2rem;
        }
    </style>
""", unsafe_allow_html=True)


# ─── Model Loading ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "assemblyai/distilbert-base-uncased-sst2"
    )
    model.eval()
    return tokenizer, model


# ─── Prediction Logic ─────────────────────────────────────────────────────────
def predict_sentiment(text, tokenizer, model):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1).squeeze()

    positive = probs[1].item()
    negative = probs[0].item()

    if positive >= 0.65:
        label = "POSITIVE"
    elif negative >= 0.65:
        label = "NEGATIVE"
    else:
        label = "NEUTRAL"

    return label, positive, negative


# ─── UI ───────────────────────────────────────────────────────────────────────
def main():
    st.markdown("""
        <div class="title-block">
            <h1>🧠 SentimentIQ</h1>
            <p>Transformer-powered sentiment detection · DistilBERT · SST-2</p>
        </div>
    """, unsafe_allow_html=True)

    st.divider()

    tokenizer, model = load_model()

    input_text = st.text_area(
        "Your Text",
        placeholder="Paste a review, tweet, or any text here...",
        height=150,
        label_visibility="collapsed"
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze = st.button("→ Analyze Sentiment")

    if analyze:
        if not input_text.strip():
            st.warning("⚠️ Please enter some text before analyzing.")
        else:
            with st.spinner("Analyzing..."):
                label, pos_prob, neg_prob = predict_sentiment(input_text, tokenizer, model)

            # Result card
            card_class = {
                "POSITIVE": "result-positive",
                "NEGATIVE": "result-negative",
                "NEUTRAL": "result-neutral"
            }[label]

            emoji = {"POSITIVE": "✅", "NEGATIVE": "❌", "NEUTRAL": "➖"}[label]
            color = {"POSITIVE": "#4caf50", "NEGATIVE": "#f44336", "NEUTRAL": "#7986cb"}[label]

            st.markdown(f"""
                <div class="result-card {card_class}">
                    <div class="result-label" style="color:{color}">{emoji} {label}</div>
                    <div class="result-sub">Confidence: Positive {pos_prob:.1%} · Negative {neg_prob:.1%}</div>
                </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Probability bars
            st.markdown('<div class="prob-bar-label">Positive probability</div>', unsafe_allow_html=True)
            st.progress(pos_prob)
            st.markdown('<div class="prob-bar-label">Negative probability</div>', unsafe_allow_html=True)
            st.progress(neg_prob)

    st.divider()
    st.markdown(
        "<p style='text-align:center;color:#444;font-size:0.8rem;'>Built with 🤗 Transformers · Streamlit · DistilBERT SST-2</p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()