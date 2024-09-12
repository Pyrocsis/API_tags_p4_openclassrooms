from flask import Flask, request, jsonify
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import joblib
import re
from bs4 import BeautifulSoup
import spacy
import os

# Create a Flask app
app = Flask(__name__)

import subprocess
import spacy

# Ensure the en_core_web_sm model is downloaded
def download_spacy_model():
    try:
        spacy.load("en_core_web_sm")
    except OSError:
        print("Downloading en_core_web_sm model...")
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        print("Download completed!")

# Call the function to download the model if not present
download_spacy_model()

# Load the model for use
nlp = spacy.load("en_core_web_sm")


# Function to clean text
def clean_text(text):
    if not isinstance(text, str):
        return text
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.lower()

# Function to preprocess text using spaCy
def preprocess_text_spacy(text):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if token.pos_ in {'NOUN', 'PROPN'}]
    return tokens

# Function to remove single letters from tokens
def post_process_tokens(tokens):
    processed_tokens = [token for token in tokens if len(token) > 1]
    return processed_tokens

# Combined function to process a sentence
def process_sentence(sentence):
    cleaned_text = clean_text(sentence)
    preprocessed_tokens = preprocess_text_spacy(cleaned_text)
    processed_tokens = post_process_tokens(preprocessed_tokens)
    return processed_tokens

# Function to dynamically load the best model, PCA, and MultiLabelBinarizer from MLflow
def load_mlflow_model(tag_type):
    client = MlflowClient()

    model_uri = f"model_{tag_type}.pkl"
    pca_uri = f"pca_{tag_type}.pkl"
    mlb_artifact_path = f"mlb_{tag_type}.pkl"


    # Load the model and PCA from MLflow
    # Load the model, PCA, and MLB from binary files
    model = joblib.load(model_uri)  # Load the binary model
    pca = joblib.load(pca_uri)      # Load the PCA object
    mlb = joblib.load(mlb_artifact_path)  # Load the MultiLabelBinarizer

    return model, mlb, pca

# API route to get predicted tags for a given sentence
@app.route('/predict', methods=['POST'])
def predict_tags():
    data = request.get_json()
    sentence = data.get('sentence', '')
    tag_type = data.get('tag_type', 'top_15')  # Default to 'top_15'
    num_tags = int(data.get('num_tags', 10))  # Default to top 10 tags

    if not sentence:
        return jsonify({'error': 'No sentence provided'}), 400

    # Process the sentence to obtain the feature vector
    processed_tokens = process_sentence(sentence)

    if not processed_tokens:
        return jsonify({'error': 'No valid tokens extracted from the sentence.'}), 400

    # Convert processed tokens into a format for PCA (e.g., embeddings or one-hot encoding)
    processed_vector = np.mean([nlp(token).vector for token in processed_tokens], axis=0).reshape(1, -1)

    # Load the appropriate model, PCA, and MLB based on the tag_type
    model, mlb, pca = load_mlflow_model(tag_type)

    # Apply PCA to the processed vector
    pca_vector = pca.transform(processed_vector)

    # Predict the tag probabilities
    predicted_probabilities = model.predict_proba(pca_vector)[0]

    # Get the indices of the top predicted tags
    top_indices = np.argsort(predicted_probabilities)[::-1][:num_tags]

    # Get the tags corresponding to the top indices
    predicted_tags_binary = np.zeros_like(predicted_probabilities)
    predicted_tags_binary[top_indices] = 1

    predicted_tags = mlb.inverse_transform([predicted_tags_binary])

    return jsonify({'predicted_tags': predicted_tags[0]})

if __name__ == '__main__':
    app.run(debug=True)
