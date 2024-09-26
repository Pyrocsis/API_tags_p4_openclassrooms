import pytest
from app import app  # Importer l'application Flask
import json
import cloudpickle


@pytest.fixture
def client():
    app.config['TESTING'] = True  # Enable test mode
    app.config['DEBUG'] = False  # Disable debug mode (optional)
    with app.test_client() as client:
        yield client

def test_predict_tags_valid(client):
    # Send a valid POST request to the API
    response = client.post('/predict', json={
        'sentence': 'This is a test sentence.',
        'tag_type': 'top_15',
        'num_tags': 5
    })

    # Check if the response status is OK
    assert response.status_code == 200

    # Print response JSON for debugging
    data = response.get_json()
    print("Response Data:", data)  # Add this for debugging

    # Ensure 'predicted_tags' exists in the response
    assert 'predicted_tags' in data

def test_predict_tags_no_sentence(client):
    # Test the API with an empty request
    response = client.post('/predict', json={})

    # Ensure the response status is an error (400)
    assert response.status_code == 400

    # Print error response for debugging
    data = response.get_json()
    print("Error Response Data:", data)  # Add this for debugging

    # Ensure error message exists in the response
    assert 'error' in data
    assert data['error'] == 'Aucune phrase fournie'
