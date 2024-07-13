from fastapi import FastAPI
from joblib import load
import numpy as np
import pandas as pd

# Load the model
model = load("model.joblib")
scaler = load("scaler.joblib")

# Create the FastAPI app
app = FastAPI()


# Define a POST method for the /predict endpoint
@app.post("/predict")
async def predict(data: dict):
    """
    This function makes a prediction based on the data passed in the request. the data should be a dictionary with a key
    "features" that contains a list of the following features in the following order:
    "Vehicle_value(USD)", "Length (meters)", "Weight(kg)", "Age(Years)", "Driver_experience(Years)".
    :param data: The data passed in the request. It should be a dictionary with a key "features" that contains a list
    of features.
    eg: {"features": [38900, 4.2, 1100, 32, 12]}
    :return:  A dictionary with a key "prediction" that contains the prediction.
    eg: {"prediction": 1}
    """
    column_names = ['Value_vehicle', 'Length', 'Weight', 'Age', 'Driving_experience']

    # Extract the features from the request
    input_data = np.array([data["features"]])
    features = pd.DataFrame(input_data, columns=column_names)
    features_scaled = scaler.transform(features)

    # Make a prediction
    prediction = model.predict(features_scaled)

    # Return the prediction
    return {"prediction": prediction[0]}
