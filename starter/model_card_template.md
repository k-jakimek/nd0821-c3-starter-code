# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

 - Person developing model: For dev training
 – Model date: 2026.09.26
 – Model version:1.0.0
 – License: Open

## Intended Use
- Intended to be used for fun applications
- Mainly used for test purposes

## Training Data
- Fake ./starter/data/census.csv training data split.

## Evaluation Data
- Chosen as a basic proof-of-concept.

## Metrics
_Please include the metrics used and your model's performance on those metrics._
- Precision=0.73 - measures how many predicted positive cases were truly positive - High precision means that when the model predicts positive, it is usually correct, minimizing false alarms.
- Recall=0.62 - ability of the model to find all positive instances
- fbeta=0.67 - score is a weighted harmonic mean of precision and recall

## Ethical Considerations
- Faces and annotations based on fake persons.

## Caveats and Recommendations
- Does not capture race or skin type, which has been reported as a source of disproportionate errors
- Used gender is binary