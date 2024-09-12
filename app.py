from flask import Flask, request, jsonify
import numpy as np
import joblib
import re
from bs4 import BeautifulSoup
import subprocess
import spacy
import os
import shutil
# Create a Flask app
app = Flask(__name__)

# Define the vectorizer directory
vectorizer_save_path = "vectorizers"




# Fonction pour obtenir l'embedding USE
def get_use_embedding(text, model):
    """
    Obtenir l'embedding Universal Sentence Encoder (USE) pour un texte donné.

    Paramètres :
    - text : Liste de tokens représentant le texte.
    - model : Modèle USE.

    Retourne :
    - Embedding USE sous forme de vecteur numpy.
    """
    return model([' '.join(text)]).numpy()[0]  # Joindre les tokens en une seule chaîne de caractères

# Fonction pour obtenir l'embedding Word2Vec
def get_word2vec_embedding(text, model):
    """
    Obtenir l'embedding Word2Vec pour un texte donné.

    Paramètres :
    - text : Liste de tokens représentant le texte.
    - model : Modèle Word2Vec.

    Retourne :
    - Embedding Word2Vec sous forme de vecteur numpy.
    """
    word_vectors = [model[word] for word in text if word in model]
    if len(word_vectors) == 0:
        return np.zeros(100)  # Supposons des vecteurs GloVe de 100 dimensions
    return np.mean(word_vectors, axis=0)

# # Effacer le cache de TensorFlow Hub si nécessaire
# tfhub_cache_dir = os.path.expanduser('~/.cache/tfhub_modules')
# if os.path.exists(tfhub_cache_dir):
#     shutil.rmtree(tfhub_cache_dir)

# # Télécharger manuellement le modèle Universal Sentence Encoder (USE)
# use_model_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
# use_model_path = os.path.join(os.getcwd(), "universal-sentence-encoder")

# if not os.path.exists(use_model_path):
#     print("Téléchargement du modèle USE...")
#     os.makedirs(use_model_path)
#     tf.keras.utils.get_file(
#         fname=os.path.join(use_model_path, "use_model.tar.gz"),
#         origin=use_model_url + "?tf-hub-format=compressed"
#     )
#     shutil.unpack_archive(
#         os.path.join(use_model_path, "use_model.tar.gz"),
#         use_model_path
#     )



# Helper function to load vectorizer by identifying the name
def load_vectorizer_by_name(file_name):
    vectorizer_map = {
        'BoW': 'bow_vectorizer.pkl',
        'TF-IDF': 'tfidf_vectorizer.pkl',
        'Word2Vec': 'word2vec_model.pkl',
        'Doc2Vec': 'doc2vec_model.pkl',
        'USE': 'use_model',
        'BERT': ('bert_tokenizer', 'bert_model')
    }
    
    for key, value in vectorizer_map.items():
        if key.lower() in file_name.lower():
            if key == 'BERT':
                tokenizer_path = os.path.join(vectorizer_save_path, value[0])
                model_path = os.path.join(vectorizer_save_path, value[1])
                return joblib.load(tokenizer_path), joblib.load(model_path)
            else:
                return joblib.load(os.path.join(vectorizer_save_path, value))
    return None


import joblib

def load_mlflow_model(tag_type):
    """
    Load the model, PCA, MLB, and vectorizer for the specified tag_type (e.g., top_50 or top_15).
    
    Params:
    - tag_type: The type of tag (e.g., 'top_50', 'top_15').
    
    Returns:
    - model: The classification model.
    - mlb: The MultiLabelBinarizer.
    - pca: The PCA model.
    - vectorizer: The vectorizer (TF-IDF in this case).
    """

    # Define paths based on the tag_type
    model_path = f"model_{tag_type}.pkl"
    pca_path = f"pca_{tag_type}.pkl"
    mlb_path = f"mlb_{tag_type}.pkl"

    # Load the models
    model = joblib.load(model_path)
    pca = joblib.load(pca_path)
    mlb = joblib.load(mlb_path)

    # Hardcoded to use TF-IDF for now
    vectorizer_path = "vectorizers/tfidf_vectorizer.pkl"
    vectorizer = joblib.load(vectorizer_path)

    return model, mlb, pca, vectorizer



def vectorize_sentence(sentence, feature_name, vectorizer):
    """
    Vectorizes the input sentence based on the feature_name and vectorizer.
    
    Params:
    - sentence: The preprocessed tokens from the sentence.
    - feature_name: The name of the feature used for vectorization (e.g., 'TF-IDF', 'BoW', etc.).
    - vectorizer: The corresponding vectorizer to use for vectorization.
    
    Returns:
    - vectorized_sentence: The vectorized sentence.
    """
    if feature_name == 'BoW':
        # Bag of Words
        return vectorizer.transform([' '.join(sentence)]).toarray()
    
    elif feature_name == 'TF-IDF':
        # TF-IDF
        return vectorizer.transform([' '.join(sentence)]).toarray()

    elif feature_name == 'Word2Vec':
        # Word2Vec
        return np.mean([vectorizer[word] for word in sentence if word in vectorizer], axis=0).reshape(1, -1)
    
    elif feature_name == 'Doc2Vec':
        # Doc2Vec
        return vectorizer.infer_vector(sentence).reshape(1, -1)


    elif feature_name == 'USE':
        # USE embeddings
        return get_use_embedding(sentence, vectorizer).reshape(1, -1)

    else:
        raise ValueError(f"Unknown feature name: {feature_name}")

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



# API route to get predicted tags for a given sentence
@app.route('/predict', methods=['POST'])
def predict_tags():
    data = request.get_json()
    sentence = data.get('sentence', '')
    tag_type = data.get('tag_type', 'top_15')  # Default to 'top_15'
    num_tags = int(data.get('num_tags', 10))  # Default to top 10 tags

    feature_name="TF-IDF"
    
    if not sentence:
        return jsonify({'error': 'No sentence provided'}), 400

    # Process the sentence to obtain the feature vector
    processed_tokens = process_sentence(sentence)

    if not processed_tokens:
        return jsonify({'error': 'No valid tokens extracted from the sentence.'}), 400

    # Convert processed tokens into a format for PCA (e.g., embeddings or one-hot encoding)
    processed_vector = np.mean([nlp(token).vector for token in processed_tokens], axis=0).reshape(1, -1)
    
    # Load the appropriate model, PCA, and MLB based on the tag_type
    model, mlb, pca,vectorizer = load_mlflow_model(tag_type)

    vectorize_sentence(processed_vector, feature_name, vectorizer)
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
