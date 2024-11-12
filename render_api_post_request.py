"""
Script for sending a POST request to Render API
"""
import requests
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")


features = {
    "age": 38,
    "workclass": "Federal-gov",
    "fnlgt": 125933,
    "education": "Masters",
    "education_num": 14,
    "marital_status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital_gain": 0,
    "capital_loss": 0,
    "hours_per_week": 40,
    "native_country": "Iran"
}

app_url = "https://deploying-ml-model-to-cloud-application.onrender.com/predict"

r = requests.post(app_url, json=features)

logging.info("Sending POST request to Render app")
logging.info(f"Status code: {r.status_code}")
logging.info(f"Response: {r.json()}")