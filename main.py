import pickle
import streamlit as st
import numpy as np

# Load the saved model and vectorizer
try:
    with open("Sentimental Analysis.pkl", "rb") as f:
        saved_data = pickle.load(f)

    sentiment_model = saved_data['model']
    tfidf_vectorizer = saved_data['tfidf_vectorizer']

except FileNotFoundError:
    st.error("Model file not found. Please make sure the model file is available.")

# Title and sidebar
st.title("Sentiment Analysis App")
st.sidebar.header("About")
st.sidebar.write(
    "This app analyzes the sentiment of text using a pre-trained model. "
    "Simply enter some text and click the 'Analyze Sentiment' button."
)

# User input and analysis
st.header("Enter Text for Sentiment Analysis")

user_text = st.text_area("Type your text here:")
st.write(f"You Entered:\n {user_text}")

if st.button("Analyze Sentiment"):
    if user_text:
        text = [user_text]
        new_data = tfidf_vectorizer.transform(text)

        # Ensure the feature dimensions match the model's expectations
        n = 10000 - new_data.shape[1]
        zero_padding = np.zeros((1, n))
        data = np.hstack((new_data.toarray(), zero_padding))

        prediction = sentiment_model.predict(data)
        st.success(f"Sentiment: {prediction[0]}")
    else:
        st.warning("Please enter some text before analyzing sentiment.")


# Disclaimer
st.subheader("Disclaimer")
st.write(
    "The sentiment analysis results are based on a pre-trained model and are provided for "
    "demonstration purposes only. The accuracy of the analysis may vary based on the input text."
)
