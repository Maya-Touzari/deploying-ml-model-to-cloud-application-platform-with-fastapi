# Script to train machine learning model.
import joblib
import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from ml import process_data, train_model, inference, compute_model_metrics


import os
cwd = os.getcwd()
print(cwd)


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")

# Add the necessary imports for the starter code.

def go():
    # Add code to load in the data.
    data = pd.read_csv("data/census_clean.csv")

    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    train, test = train_test_split(data, test_size=0.20, random_state=42)

    cat_features = [
        "workclass",
        "education", # may be remove, duplicate of education-num
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    
    print(X_train)
    print(y_train)
    # Process the test data with the process_data function.
    X_test, y_test, _, _ = process_data(
        test, categorical_features=[], label="salary", training=False, encoder=encoder, lb=lb
    )

    model = train_model(X_train, y_train)
    
    y_pred = inference(model, X_test)
    
    scores = compute_model_metrics(y_test, y_pred)
    logging.info(scores)
    
    # joblib.dump(model, 'model/model.pkl')
    # joblib.dump(encoder, 'model/encoder.pkl')
    # joblib.dump(lb, 'model/lb.pkl')

if __name__=="__main__":
    go()