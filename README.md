# SentimentIQ — Sentiment Analysis with Transformers

A sleek, transformer powered sentiment analysis web app built with **DistilBERT** and **Streamlit**. Enter any text and instantly get a **Positive**, **Negative**, or **Neutral** prediction with confidence scores.

# Features

- Real-time sentiment prediction (Positive / Negative / Neutral)
- Confidence probability bars for each prediction
- Custom dark-themed UI with modern design
- Fast inference using DistilBERT (lightweight BERT variant)
- Runs fully locally no API keys needed

# Tech Stack

| Transformers | Pre-trained DistilBERT model |
| PyTorch | Model inference backend |
| Streamlit | Web app framework |
| Python 3.10+ | Core language |

# Project Structure

SentimentIQ/
│
├── app.py                  # Main Streamlit application
├── requirements.txt        # Project dependencies
├── app_screens/            # UI screenshots
│   ├── home.PNG
│   ├── positive.PNG
│   ├── negative.PNG
│   └── neutral.PNG
└── README.md               # Project documentation

# Installation & Setup

# 1. Clone the repository
git clone https://github.com/your-username/SentimentIQ.git
cd SentimentIQ

# 2. Create a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt


# 4. Run the app
streamlit run app.py

Then open your browser at `http://localhost:8501`

# Example Inputs

**Positive:**
> "I absolutely loved this product. It exceeded every expectation!"

**Negative:**
> "Terrible experience. The product broke within a day and support was useless."

**Neutral:**
> "It works fine. Nothing impressive but gets the job done."

# Model Details

- **Model:** `assemblyai/distilbert-base-uncased-sst2`
- **Base:** DistilBERT (distilled version of BERT — 40% smaller, 60% faster)
- **Trained on:** Stanford Sentiment Treebank v2 (SST-2)
- **Task:** Binary sentiment classification → extended with Neutral threshold logic

# Screenshots

# Home Page
![Home](app_screens/home.PNG)

# Positive Prediction
![Positive](app_screens/positive.PNG)

# Negative Prediction
![Negative](app_screens/negative.PNG)

# Neutral Prediction
![Neutral](app_screens/neutral.PNG)

# Notes

- First run will download the model (~250MB) from Hugging Face automatically
- Model is cached after first load for faster subsequent runs
- Neutral label is triggered when neither Positive nor Negative confidence exceeds 65%

# License

This project is open-source and available under the [MIT License](LICENSE).