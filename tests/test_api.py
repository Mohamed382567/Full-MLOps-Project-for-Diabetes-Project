from fastapi.testclient import TestClient
from src.app.main import app 

client = TestClient(app)

def test_predict_endpoint():
    """
    Test if the /predict endpoint is reachable and returns a valid JSON response.
    """
    payload = {
        "Pregnancies": 1, "Glucose": 100, "BloodPressure": 70,
        "SkinThickness": 20, "Insulin": 50, "BMI": 23.0,
        "DiabetesPedigreeFunction": 0.3, "Age": 25
    }
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 200
    assert "prediction" in response.json()
