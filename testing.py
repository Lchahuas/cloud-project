from fastapi.testclient import TestClient
from faker import Faker
from app import app  # Assuming your FastAPI app instance is named 'app'

# Create a TestClient for testing the FastAPI app
client = TestClient(app)

# Create a Faker instance for generating fake data
fake = Faker()

# Test for the health check endpoint
def test_health_check():
    # Send a GET request to /health
    response = client.get("/health")

    # Assert that the response status code is 200 (OK)
    assert response.status_code == 200

    # Assert that the response JSON matches the expected health check response
    assert response.json() == {"status": "OK"}

# Test for the predict endpoint with correct data
def test_predict():
    # Example of correct data generated using Faker
    data = {
        "Age": fake.random_int(min=25, max=80),
        "Sex": fake.random_element(elements=("M", "F")),
        "ChestPainType": fake.random_element(elements=("ATA", "NAP", "ASY", "TA")),
        "RestingBP": fake.random_int(min=0, max=210),
        "Cholesterol": fake.random_int(min=0, max=607),
        "FastingBS": fake.random_int(min=0, max=1),
        "RestingECG": fake.random_element(elements=("Normal", "LVH", "ST")),
        "MaxHR": fake.random_int(min=55, max=210),
        "ExerciseAngina": fake.random_element(elements=("N", "Y")),
        "Oldpeak": fake.pyfloat(min_value=-4.0, max_value=7.0),
        "ST_Slope": fake.random_element(elements=("Up", "Flat", "Down")),
    }

    # Send a POST request to /predict with the generated data
    response = client.post("/predict", json=data)

    # Assert that the response status code is 200 (OK)
    assert response.status_code == 200

    # Assert that the response JSON contains the 'prediction' key
    assert "prediction" in response.json()