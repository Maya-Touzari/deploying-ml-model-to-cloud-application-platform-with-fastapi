"""
Script to train machine learning model.
"""
import logging

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from ml import import_data, process_data, train_model, inference, compute_model_metrics


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")


def go():
    logging.info(f"Importing data.")
    data = import_data("data/census_clean.csv")

    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    logging.info(f"Splitting data.")
    train, test = train_test_split(
        data, test_size=0.20, random_state=42, stratify=data["salary"])
    
    logging.info(f"Processing data.")
    cat_features = [
        "workclass",
        "education",  # may be remove, duplicate of education-num
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
        test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )
    
    logging.info(f"Training model.")
    model = train_model(X_train, y_train)

    y_pred = inference(model, X_test)
    
    logging.info(f"Computing metrics on test set.")
    precision, recall, fbeta = compute_model_metrics(y_test, y_pred)
    cl_report = classification_report(y_test, y_pred)
    logging.info(
        f"Precision: {precision: .2f}. Recall: {recall: .2f}. Fbeta: {fbeta: .2f}")
    logging.info(cl_report)


if __name__ == "__main__":
    go()
