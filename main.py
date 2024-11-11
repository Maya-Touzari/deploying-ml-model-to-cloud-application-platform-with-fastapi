# Import Union since our Item object will have tags that can be strings or a list.
import joblib
from typing import Literal

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI
# BaseModel from Pydantic is used to define data objects.
from pydantic import BaseModel, ConfigDict

from src.ml import process_data, inference


app = FastAPI()


def hyphenize(field: str):
    return field.replace("_", "-")


class ModelInput(BaseModel):
    model_config = ConfigDict(alias_generator=hyphenize)

    age: int
    workclass: Literal['State-gov', 'Self-emp-not-inc', 'Private',
                       'Federal-gov', 'Local-gov', 'Self-emp-inc', 'Without-pay', 'Never-worked']
    fnlgt: int
    education: Literal['Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college',
                       'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school',
                       '5th-6th', '10th', '1st-4th', 'Preschool', '12th']
    education_num: int
    marital_status: Literal['Never-married', 'Married-civ-spouse', 'Divorced',
                            'Married-spouse-absent', 'Separated', 'Married-AF-spouse', 'Widowed']
    occupation: Literal['Adm-clerical', 'Exec-managerial', 'Handlers-cleaners',
                        'Prof-specialty', 'Other-service', 'Sales', 'Craft-repair',
                        'Transport-moving', 'Farming-fishing', 'Machine-op-inspct',
                        'Tech-support', 'Protective-serv', 'Armed-Forces', 'Priv-house-serv']
    relationship: Literal['Not-in-family', 'Husband',
                          'Wife', 'Own-child', 'Unmarried', 'Other-relative']
    race: Literal['White', 'Black', 'Asian-Pac-Islander',
                  'Amer-Indian-Eskimo', 'Other']
    sex: Literal['Male', 'Female']
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: Literal['United-States', 'Cuba', 'Jamaica', 'India', 'Mexico',
                            'South', 'Puerto-Rico', 'Honduras', 'England', 'Canada',
                            'Germany', 'Iran', 'Philippines', 'Italy', 'Poland',
                            'Columbia', 'Cambodia', 'Thailand', 'Ecuador', 'Laos',
                            'Taiwan', 'Haiti', 'Portugal', 'Dominican-Republic',
                            'El-Salvador', 'France', 'Guatemala', 'China', 'Japan',
                            'Yugoslavia', 'Peru', 'Outlying-US(Guam-USVI-etc)',
                            'Scotland' 'Trinadad&Tobago', 'Greece', 'Nicaragua',
                            'Vietnam', 'Hong', 'Ireland', 'Hungary', 'Holand-Netherlands']


@app.get("/")
async def say_hello():
    return {"greeting": "Hello, this app predicts income (<=50K, >50K)."}


@app.post("/predict")
async def predict(input: ModelInput):
    features = [
        "age",
        "workclass",
        "fnlgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
    ]

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

    input_dict = input.model_dump(by_alias=True)
    input_df = pd.DataFrame(data=np.array([[input_dict.get(feature) for feature in features]]),
                            columns=features)

    print(input_df)

    lb = joblib.load("model/lb.pkl")
    encoder = joblib.load("model/encoder.pkl")
    model = joblib.load("model/model.pkl")

    # Process the test data with the process_data function.
    X, _, _, _ = process_data(
        input_df, categorical_features=cat_features, training=False, encoder=encoder, lb=lb
    )

    prediction = lb.inverse_transform(inference(model, X))[0]
    return {"prediction": prediction}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
