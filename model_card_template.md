# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf.

## Model 

The model is Random Forest classifier with max_depth=35 and default values for other parameters.

## Intended Use

The model is a binary classifier that predicts if the salary is >50K or <=50K.

## Training Data

Information about training data can be found [here](https://archive.ics.uci.edu/dataset/20/census+income). 

## Evaluation Data

80-20 split is used to get training and validation data.

## Metrics

Evaluation based on three metrics: Precision:  0.71. Recall:  0.68. Fbeta:  0.70.

## Ethical Considerations

Data contains demographical information such as sex, race, etc. Consider investigating model bias.

## Caveats and Recommendations

Hyperparameter optimization and a better preprocessing of the data.
