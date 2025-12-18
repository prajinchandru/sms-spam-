import streamlit as st
import pickle

# Load model
model = pickle.load(open('spam_svm_model.pkl','rb'))

st.title("📩 SMS Spam Classifier")
st.write("Type a message below and click *Predict*:")

user_msg = st.text_area("Enter SMS text:")

if st.button("Predict"):
    result = model.predict([user_msg])[0]
    label = "Spam 🚫" if result == 1 else "Not Spam ✅"
    st.success(f"Prediction: {label}")
