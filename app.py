import streamlit as st
from transformers import pipeline

st.title("Movie Review Analyzer🎬")

# Model 1: Sentiment 
sentiment_model = pipeline(
    "text-classification",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

# Model 2: Zero-shot 
topic_model = pipeline(
    "zero-shot-classification",
    model="Horizon-Labs/multilingual-zeroshot-small"
)

classes = [
    "Acting", "Plot and Story","Cinematography and Visual Effects",
    "Music and Soundtrack","Themes and Message", "Pacing and Editing"
]
#ask user input
review = st.text_area("Enter your movie review")

if st.button("Check"):
    if review:
        #sentiment prediction
        sentiment = sentiment_model( review, truncation=True, max_length=512)[0]

        #topic prediction
        topic = topic_model(review, candidate_labels=classes, multi_label=False)

        #display results
        st.write("Sentiment:", sentiment["label"])
        st.write("Sentiment Confidence:", sentiment["score"])
        st.write("Category:", topic["labels"][0])
        st.write("Category Score:", topic["scores"][0])
    else:
        st.warning("Please enter a review.")
