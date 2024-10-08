import pytest
from app import app  # Importer l'application Flask
import json
import cloudpickle
import requests

@pytest.fixture
def client():
    app.config['TESTING'] = True  # Enable test mode
    app.config['DEBUG'] = False  # Disable debug mode (optional)
    with app.test_client() as client:
        yield client


def test_request(client):


    # The URL of the Flask API
    # url = 'https://api-project-4-tag-3c6eddb4da51.herokuapp.com/predict'
    url = '/predict'
    
    sentence="""Good morning eveyone


    i have this dataframe using:

    data_T =[(folder, folder.split('/')[-2]) for folder in subfolders]

    Can somebody help me to create this new dataframe pandas and how i could use regex to look into description field if there is break line inside the field.

    Thank you so much

    Regards """


    data = {
        "sentence": sentence,
        "tag_type": "top_15",  
        "num_tags": 5  }   

    response = client.post(url, json=data)
    # response = requests.post(url, json=data)

    # Check if the response status is OK
    assert response.status_code == 200

    # Print response JSON for debugging
    data = response.get_json()
    # data = response.json()

    print("Response Data:", data)  # Add this for debugging

    # Ensure 'predicted_tags' exists in the response
    assert 'predicted_tags' in data




# def test_predict_tags_valid(client):
#     # Send a valid POST request to the API
#     sentence="""Good morning eveyone


#     i have this dataframe using:

#     data_T =[(folder, folder.split('/')[-2]) for folder in subfolders]

#     Can somebody help me to create this new dataframe and how i could use regex to look into description field if there is break line inside the field.

#     Thank you so much

#     Regards """

#     response = client.post('/predict', json={
#         'sentence': sentence,
#         'tag_type': 'top_15',
#         'num_tags': 5
#     })

#     # Check if the response status is OK
#     assert response.status_code == 200

#     # Print response JSON for debugging
#     data = response.get_json()
#     print("Response Data:", data)  # Add this for debugging

#     # Ensure 'predicted_tags' exists in the response
#     assert 'predicted_tags' in data

# def test_predict_tags_no_sentence(client):
#     # Test the API with an empty request
#     response = client.post('/predict', json={})

#     # Ensure the response status is an error (400)
#     assert response.status_code == 400

#     # Print error response for debugging
#     data = response.get_json()
#     print("Error Response Data:", data)  # Add this for debugging

#     # Ensure error message exists in the response
#     assert 'error' in data
#     assert data['error'] == 'Aucune phrase fournie'
