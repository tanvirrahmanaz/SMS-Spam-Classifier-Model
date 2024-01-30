import streamlit as st
import pickle
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import datetime

nltk.download('punkt')  # Download necessary NLTK resources
nltk.download('stopwords')

ps = PorterStemmer()  # Instantiate stemmer outside functions for efficiency

def preprocess_text(text):
    """Preprocesses text for spam classification."""
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token.isalnum()]
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    tokens = [ps.stem(token) for token in tokens]
    return ' '.join(tokens)

def load_models():
    """Loads the trained TF-IDF vectorizer and spam classification model."""
    with open('vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    return tfidf, model

def main():
    st.title("SMS Spam Classifier and User Feedback")

    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose a feature:", ["SMS Spam Classifier", "Complaint Box"])

    if app_mode == "SMS Spam Classifier":
        tfidf, model = load_models()

        input_sms = st.text_input("Enter the message")
        if st.button('Predict'):
            transform_sms = preprocess_text(input_sms)
            vector_input = tfidf.transform([transform_sms])
            result = model.predict(vector_input)[0]
            st.header("Spam" if result == 1 else "Not Spam")

    else:
        with st.form("complaint_form"):
            st.write("Share your thoughts, report issues, or suggest improvements below:")
            complaint_text = st.text_area("Type your complaint here:")
            submit_button = st.form_submit_button("Submit Complaint")

        if submit_button and complaint_text:
            save_complaint(complaint_text)
            st.success("Complaint submitted successfully! Thank you for your feedback.")

def save_complaint(complaint_text):
    """Saves a complaint to a CSV file with a timestamp."""
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Get current time
    df = pd.DataFrame({"Complaints": [complaint_text], "Timestamp": [current_time]})
    df.to_csv("complaints.csv", mode="a", header=False, index=False)

if __name__ == "__main__":
    main()
