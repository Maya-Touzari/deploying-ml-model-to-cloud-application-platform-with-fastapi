"""
Module for unit testing the ml module
"""
import os
from os import path
import logging
import joblib
import pytest

from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

from src.ml import import_data, process_data, train_model

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")

# Create temporary directory
PATH = 'test_temp_dir'
if not os.path.exists(PATH):
    os.mkdir(PATH)


@pytest.fixture()
def data():
    """
    Fixture - returns dataset
    """

    try:
        df = import_data("data/census_clean.csv")
        logging.info("Data fixture: SUCCESS")

    except FileNotFoundError as err:
        logging.error("Testing data fixture: The file wasn't found")
        raise err

    return df


@pytest.fixture()
def processed_data(data):
    """
    Fixture - returns processed dataset
    """

    cat_features = [
        "workclass",
        "education",  # may be remove, duplicate of education-num
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country"
    ]
    try:
        X, y, encoder, lb = process_data(
            data, categorical_features=cat_features, label="salary", training=True, dir=PATH
        )
        logging.info("Processed process_data fixture: SUCCESS")

    except Exception as e:
        logging.error("Testing process_data fixture: FAILED")
        raise e

    return X, y, encoder, lb


def test_import_data(data):
    """
    Test data import - this example is completed for you to assist with the other test functions
    """

    try:
        assert data.shape[0] > 0
        assert data.shape[1] > 0
        logging.info("Testing import_data: SUCCESS")

    except AssertionError as err:
        logging.error(
            "Testing import_data: FAILED The file doesn't appear to have rows or columns")
        raise err


def test_data_process(processed_data):
    """ 
    Test data processing 
    """

    X, y, encoder, lb = processed_data
    try:

        assert isinstance(encoder, OneHotEncoder)
        assert isinstance(lb, LabelBinarizer)

        joblib.load(path.join(PATH, 'lb.pkl'))
        joblib.load(path.join(PATH, 'encoder.pkl'))

        assert len(X) == len(y)

        logging.info("Testing process_data: SUCCESS")

    except Exception as e:
        logging.info("Testing process_data: FAILED")
        raise e


def test_train_model(processed_data):
    """ 
    Test model training 
    """

    X, y, _, _ = processed_data
    try:
        model = train_model(X, y, dir=PATH)

        assert isinstance(model, RandomForestClassifier)

        joblib.load(path.join(PATH, 'model.pkl'))

        os.remove(path.join(PATH, 'model.pkl'))
        os.remove(path.join(PATH, 'lb.pkl'))
        os.remove(path.join(PATH, 'encoder.pkl'))
        os.rmdir(PATH)

        logging.info("Testing train_model: SUCCESS")

    except Exception as e:
        logging.info("Testing train_model: FAILED")
        raise e
