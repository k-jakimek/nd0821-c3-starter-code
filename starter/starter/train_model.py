# Script to train machine learning model.
from sklearn.model_selection import train_test_split
from ml import model
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


# Add code to load in the data.
df = model.import_data()
# Optional enhancement:
# use K-fold cross validation instead of a train-test split.
train, test = train_test_split(df, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

x_train, x_test, y_train, y_test, encoder, lb = model.process_test_train_data(
    train, test, cat_features)

# Train and save a model.
train_model = model.train_model(x_train, y_train)
predictions = model.inference(train_model, x_test)
precision, recall, fbeta = model.compute_model_metrics(y_test, predictions)
logging.info(f"Metrics: precision={precision} recall={recall} fbeta={fbeta}")

model.save_model(train_model)
model.evaluate_slices(train_model, df, encoder, lb, cat_features)
