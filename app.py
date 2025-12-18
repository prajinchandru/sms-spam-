import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

# App title
st.title("📩 SMS Spam Classifier (SVM)")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("spam.csv", encoding="latin-1")[["v1", "v2"]]
    df.columns = ["label", "message"]
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    return df

df = load_data()

# Train model
@st.cache_resource
def train_model():
    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("svm", LinearSVC())
    ])
    model.fit(df["message"], df["label"])
    return model

model = train_model()

# User input
msg = st.text_area("Enter SMS message")

if st.button("Predict"):
    prediction = model.predict([msg])[0]
    if prediction == 1:
        st.error("🚫 SPAM MESSAGE")
    else:
        st.success("✅ NOT SPAM")
