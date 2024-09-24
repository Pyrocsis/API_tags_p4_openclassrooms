from flask import Flask, request, jsonify
import numpy as np
import joblib
import re
from bs4 import BeautifulSoup
import subprocess
import spacy
import os
import shutil
import cloudpickle

# Créer une application Flask
app = Flask(__name__)

# Définir le répertoire du vectorizer
vectorizer_save_path = "vectorizers"

import pickle

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
        return np.zeros(100)  # Supposons des vecteurs Word2Vec de 100 dimensions
    return np.mean(word_vectors, axis=0)


import joblib

# Fonction pour charger le modèle, la PCA, MLB et le vectorizer en fonction du type d'étiquette (par exemple, top_50 ou top_15)
def load_mlflow_model(tag_type, feature_name):
    """
    Charger le modèle, PCA, MLB et le vectorizer pour le type d'étiquette spécifié (ex : top_50 ou top_15).
    
    Paramètres :
    - tag_type : Le type d'étiquette (par exemple, 'top_50', 'top_15').
    
    Retourne :
    - model : Le modèle de classification.
    - mlb : Le MultiLabelBinarizer.
    - pca : Le modèle PCA.
    - vectorizer : Le vectorizer (TF-IDF dans ce cas).
    """
    vectorizer_map = {
        'BoW': 'bow_vectorizer.pkl',
        'TF-IDF': 'tfidf_vectorizer.pkl',
        'Word2Vec': 'word2vec_model.pkl',
        'Doc2Vec': 'doc2vec_model.pkl',
        'USE': 'use_model',
        'BERT': ('bert_tokenizer', 'bert_model')
    }
    # Définir les chemins en fonction du type d'étiquette
    model_path = f"model_{tag_type}.pkl"
    pca_path = f"pca_{tag_type}.pkl"
    mlb_path = f"mlb_{tag_type}.pkl"

    # Charger les modèles
    model = joblib.load(model_path)
    pca = joblib.load(pca_path)
    mlb = joblib.load(mlb_path)
    
    vectorizer_path = "vectorizers/" + vectorizer_map.get(feature_name)
    
    # Charger le vectorizer TF-IDF avec cloudpickle
    if feature_name == 'TF-IDF':
        with open(vectorizer_path, 'rb') as f:
            vectorizer = cloudpickle.load(f)
    else:
        vectorizer = joblib.load(vectorizer_path)
    
    return model, mlb, pca, vectorizer


# Fonction pour vectoriser une phrase en fonction du feature_name et du vectorizer
def vectorize_sentence(sentence, feature_name, vectorizer):
    """
    Vectoriser la phrase en entrée en fonction du feature_name et du vectorizer.
    
    Paramètres :
    - sentence : Les tokens prétraités de la phrase.
    - feature_name : Le nom de la fonctionnalité utilisée pour la vectorisation (par ex. 'TF-IDF', 'BoW', etc.).
    - vectorizer : Le vectorizer correspondant à utiliser pour la vectorisation.
    
    Retourne :
    - vectorized_sentence : La phrase vectorisée.
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
        # Embeddings USE
        return get_use_embedding(sentence, vectorizer).reshape(1, -1)

    else:
        raise ValueError(f"Nom de fonctionnalité inconnu : {feature_name}")

# S'assurer que le modèle en_core_web_sm est téléchargé
def download_spacy_model():
    try:
        spacy.load("en_core_web_sm")
    except OSError:
        print("Téléchargement du modèle en_core_web_sm...")
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        print("Téléchargement terminé !")

# Appeler la fonction pour télécharger le modèle s'il n'est pas présent
download_spacy_model()

# Charger le modèle pour l'utiliser
nlp = spacy.load("en_core_web_sm")

# Fonction pour nettoyer le texte
def clean_text(text):
    if not isinstance(text, str):
        return text
    text = BeautifulSoup(text, "html.parser").get_text()  # Supprimer les balises HTML
    text = re.sub(r'\s+', ' ', text)  # Remplacer les espaces multiples par un seul espace
    text = re.sub(r'[^\w\s]', '', text)  # Supprimer les caractères non alphanumériques
    return text.lower()  # Convertir en minuscules

# Fonction pour prétraiter le texte avec spaCy
def preprocess_text_spacy(text):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if token.pos_ in {'NOUN', 'PROPN'}]  # Lemmatizer les noms et les noms propres
    return tokens

# Fonction pour supprimer les lettres seules des tokens
def post_process_tokens(tokens):
    processed_tokens = [token for token in tokens if len(token) > 1]  # Supprimer les tokens d'une seule lettre
    return processed_tokens

# Fonction combinée pour traiter une phrase
def process_sentence(sentence):
    cleaned_text = clean_text(sentence)  # Nettoyer le texte
    preprocessed_tokens = preprocess_text_spacy(cleaned_text)  # Appliquer le prétraitement spaCy
    processed_tokens = post_process_tokens(preprocessed_tokens)  # Post-traiter les tokens
    return processed_tokens


# Route API pour obtenir les étiquettes prédites pour une phrase donnée
@app.route('/predict', methods=['POST'])
def predict_tags():
    data = request.get_json()  # Récupérer les données JSON envoyées
    sentence = data.get('sentence', '')  # Obtenir la phrase à traiter
    tag_type = data.get('tag_type', 'top_15')  # Par défaut, utiliser 'top_15'
    num_tags = int(data.get('num_tags', 10))  # Par défaut, retourner les 10 meilleures étiquettes

    feature_name = "TF-IDF"  # Utiliser TF-IDF par défaut
    
    if not sentence:
        return jsonify({'error': 'Aucune phrase fournie'}), 400  # Si la phrase est vide, retourner une erreur

    # Traiter la phrase pour obtenir le vecteur de caractéristiques
    processed_tokens = process_sentence(sentence)

    if not processed_tokens:
        return jsonify({'error': 'Aucun token valide extrait de la phrase.'}), 400  # Si aucun token valide, retourner une erreur

    # Charger le modèle, la PCA et le vectorizer en fonction du type d'étiquette
    model, mlb, pca, vectorizer = load_mlflow_model(tag_type, feature_name)
    
    # Vectoriser la phrase traitée
    processed_vector = vectorize_sentence(processed_tokens, feature_name, vectorizer)
    
    # Appliquer la PCA au vecteur traité
    pca_vector = pca.transform(processed_vector)

    # Prédire les probabilités des étiquettes
    predicted_probabilities = model.predict_proba(pca_vector)[0]

    # Obtenir les indices des étiquettes les mieux prédites
    top_indices = np.argsort(predicted_probabilities)[::-1][:num_tags]

    # Obtenir les étiquettes correspondant aux meilleurs indices
    predicted_tags_binary = np.zeros_like(predicted_probabilities)
    predicted_tags_binary[top_indices] = 1
    predicted_tags_binary = predicted_tags_binary.reshape(1, -1)

    # Transformer en étiquettes inverses
    predicted_tags = mlb.inverse_transform(predicted_tags_binary)
    
    return jsonify({'predicted_tags': predicted_tags[0]})  # Retourner les étiquettes prédites sous forme de JSON

# Point d'entrée pour démarrer l'application Flask
if __name__ == '__main__':
    app.run(debug=True)  # Démarrer l'application en mode débogage"# Test GitHub Actions trigger" 
"# Test GitHub Actions trigger" 
"# Test GitHub Actions trigger" 
"# Test GitHub Actions trigger" 
