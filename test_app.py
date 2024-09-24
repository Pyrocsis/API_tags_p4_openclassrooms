import pytest
from app import app  # Importer l'application Flask
import json

@pytest.fixture
def client():
    # Configurer l'application Flask en mode test
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_predict_tags_valid(client):
    # Envoyer une requête POST valide à l'API
    response = client.post('/predict', json={
        'sentence': 'This is a test sentence.',
        'tag_type': 'top_15',
        'num_tags': 5
    })
    
    # Vérifier que la réponse est OK (code HTTP 200)
    assert response.status_code == 200

    # Vérifier que les étiquettes prédites sont renvoyées
    data = response.get_json()
    assert 'predicted_tags' in data

def test_predict_tags_no_sentence(client):
    # Tester l'API avec une requête vide
    response = client.post('/predict', json={})
    
    # Vérifier que la réponse est une erreur (code HTTP 400)
    assert response.status_code == 400

    # Vérifier que le message d'erreur est correct
    data = response.get_json()
    assert 'error' in data
    assert data['error'] == 'Aucune phrase fournie'
