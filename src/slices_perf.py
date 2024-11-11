import joblib
import logging

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from ml import import_data, process_data, inference, compute_model_metrics


logging.basicConfig(level=logging.INFO, format="%(message)s")


def slice_perf():
    data = import_data("data/census_clean.csv")
    _, X = train_test_split(data, test_size=0.20,
                            random_state=42, stratify=data["salary"])

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

    lb = joblib.load("model/lb.pkl")
    encoder = joblib.load("model/encoder.pkl")
    model = joblib.load("model/model.pkl")

    # Process the test data with the process_data function.
    X, y, _, _ = process_data(
        data, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )

    y_pred = inference(model, X)

    for feature in cat_features:
        logging.info(feature)
        for cls in data[feature].dropna().unique():
            logging.info(cls)
            idx = data[data[feature] == cls].index.to_numpy()
            y_cls = y[idx]
            y_pred_cls = y_pred[idx]
            print(len(idx), len(y_pred_cls), len(y_cls))

            precision, recall, fbeta = compute_model_metrics(y_cls, y_pred_cls)
            cl_report = classification_report(y_cls, y_pred_cls)
            logging.info(
                f"Precision: {precision: .2f}. Recall: {recall: .2f}. Fbeta: {fbeta: .2f}")
            logging.info(cl_report)


if __name__ == "__main__":
    slice_perf()
