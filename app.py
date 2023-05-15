import streamlit as st
import joblib
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

model = joblib.load('transformer.joblib')
embeddings = joblib.load('embeddings.pkl')
sentences = joblib.load('sentences.pkl')

st.title('Enter the Paper you want to read today...')
paper_name = st.text_input("Input paper name")

def predict():
    encoded_name = model.encode(paper_name)

    cosine_scores = cosine_similarity([encoded_name], embeddings)
    top_similar_papers = np.argsort(cosine_scores, axis=1)[0][-5:][::-1]

    for i in top_similar_papers:
        st.write(sentences[i])

if st.button('Display'):
    predict()
