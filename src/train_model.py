# Script to train machine learning model.
import joblib
import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, SimpleImputer, LabelBinarizer
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, precision_score, f1_score

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")

# Add the necessary imports for the starter code.

def process_data(X, categorical_features, label="salary", training=True, encoder=None, lb=None):

    # ordinal_categorical = ["room_type"]
    # ordinal_categorical_preproc = OrdinalEncoder()

    X_train = X.copy()
    y_train = X_train.pop(label) if (label and label in X_train.columns) else None
        
    if training:
        
        numerical_features = set(X_train.columns).difference(set(categorical_features))
        encoder = ColumnTransformer(
            transformers=[
                ("cat", 
                make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder()), 
                categorical_features),
                ("num", 
                make_pipeline(SimpleImputer(strategy="median")), 
                numerical_features),
            ],
            remainder="drop", 
        )
        lb = LabelBinarizer()

        encoder = encoder.fit(X_train)
        lb = lb.fit(y_train)

    X_train = encoder.transform(X_train)
    y_train = lb.transform(y_train) if y_train else None
        
    return X_train, y_train, encoder, lb

def train_model(X_train, y_train, n_estimators=100, max_depth=20):

    model = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, n_jobs=-1, random_state=42)

    model.fit(X_train, y_train)

    return(model)

def compute_metrics(y, y_pred):

    summary = {
        "f1": f1_score(y, y_pred, average="macro"),
        "precision": precision_score(y, y_pred, average="macro"),
        "recall": recall_score(y, y_pred, average="macro")
    }

    return summary

def go():
    # Add code to load in the data.
    data = pd.read_csv("../data/census_clean.csv")

    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    train, test = train_test_split(data, test_size=0.20)

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
    
    # Process the test data with the process_data function.
    X_test, y_test, _, _ = process_data(
        test, categorical_features=[], label="salary", training=False, encoder=encoder, lb=lb
    )

    model = train_model(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    scores = compute_metrics(y_test, y_pred)
    logging.info(scores)
    
    joblib.dump(model, 'model/model.pkl')
    joblib.dump(encoder, 'model/encoder.pkl')
    joblib.dump(lb, 'model/lb.pkl')

if __name__=="__main__":
    go()