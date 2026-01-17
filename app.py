import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Page Config
st.set_page_config(page_title="SpamGuard AI", page_icon="🛡️", layout="centered")

# Custom CSS for Premium Look
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stApp {
        background-color: #0f172a;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #2e77d0;
        color: white;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #1e5bad;
        border: none;
        color: white;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin-top: 20px;
    }
    .spam { background-color: #ffdadb; color: #cc0000; border: 2px solid #cc0000; }
    .ham { background-color: #d4edda; color: #155724; border: 2px solid #155724; }
    </style>
    """, unsafe_allow_html=True)

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [i for i in tokens if i.isalnum()]
    tokens = [i for i in tokens if i not in stopwords.words('english')]
    tokens = [ps.stem(i) for i in tokens]
    return " ".join(tokens)

# Load Model and Vectorizer
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model files not found! Please run train_model.py first.")

# UI Elements
st.markdown("<h1 style='text-align: center; color: #1e3a8a;'>🛡️ SpamGuard: SMS AI Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #64748b;'>Paste your message below to check if it's Safe or Spam</p>", unsafe_allow_html=True)

input_sms = st.text_area("", placeholder="Enter SMS message here...", height=150)

if st.button('Analyze Message'):

    if input_sms.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        # 1. Preprocess
        transformed_sms = transform_text(input_sms)
        # 2. Vectorize
        vector_input = tfidf.transform([transformed_sms]).toarray()
        # 3. Predict
        result = model.predict(vector_input)[0]

        # 4. Display
        if result == 1:
            st.markdown('<div class="result-box spam">🚨 SPAM DETECTED</div>', unsafe_allow_html=True)
            st.warning("Be careful! This message looks like a scam or promotion.")
        else:
            st.markdown('<div class="result-box ham">✅ SAFE / HAM</div>', unsafe_allow_html=True)
            st.success("This message appears to be safe.")

st.markdown("---")
st.markdown("<p style='text-align: center; font-size: 12px; color: #94a3b8;'>Project developed by Jatin</p>", unsafe_allow_html=True)
