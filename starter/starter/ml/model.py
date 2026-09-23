from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
import joblib
import pandas as pd
from ml import data

DATA_PATH = "./starter/data/census.csv"
MODEL_DUMP_PATH = "./starter/model/dump"


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.
    Inputs
    ------
    X_train : np.ndarray
        Training data.
    y_train : np.ndarray
        Labels.
    Returns
    -------
    model : RandomForestClassifier
        Trained machine learning model.
    """
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision,
    recall, and F1.

    Inputs
    ------
    y : np.ndarray
        Known labels, binarized.
    preds : np.ndarray
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.
    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    X : np.ndarray
        Data used for prediction.
    Returns
    -------
    preds : np.ndarray
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds


def process_test_train_data(train, test, categorical_features):
    """ Run model inferences and return the predictions.
    Inputs
    ------
    train : Array - train inputs
    test : Array - test inputs
    categorical_features: list[str]
            List containing the names of the categorical features (default=[])
    Returns
    -------
    output:
              x_train, x_test, y_train, y_test, encoder, lb
    """

    x_train, y_train, encoder, lb = data.process_data(
        train, categorical_features=categorical_features, label="salary",
        training=True)

    x_test, y_test, encoder, lb = data.process_data(
        test, categorical_features=categorical_features, label="salary",
        training=False, encoder=encoder, lb=lb)
    return x_train, x_test, y_train, y_test, encoder, lb


def save_model(model):
    """ Saves model in selected location.
    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    filePath : string
        Output file location
    Returns
    -------
    None
    """
    joblib.dump(model, MODEL_DUMP_PATH)


def import_data():
    """
    Prepares data frame from DATA_PATH csv file

    Inputs
    ----------
    NONE

    Returns
    -------
    data - data frame object after strip strip column names
    """
    data = pd.read_csv(DATA_PATH)
    data.columns = [col.strip() for col in data.columns]
    return data


def evaluate_slices(train_model, df, encoder, lb, categorical_features):
    """
    Outputs the performance of the model on slices of the data for each
    categorical feature.

    Inputs
    ----------
    model : RandomForestClassifier
        Trained machine learning model.
    df : pandas.DataFrame
        The dataset including features.
    categorical_features : list of strings
        List of categorical feature column names to slice on.

    Returns
    -------
    None
    """
    for feature in categorical_features:
        print(f"Performance on slices of categorical feature: '{feature}'")
        unique_values = df[feature].unique()
        for value in unique_values:
            slice_df = df[df[feature] == value]
            x_test, y_test, l_encoder, l_lb = data.process_data(
                slice_df, categorical_features=categorical_features,
                label="salary", training=False, encoder=encoder, lb=lb)
            predictions = inference(train_model, x_test)
            precision, recall, fbeta = compute_model_metrics(
                y_test, predictions)
            print(f"Feature:{feature} value:{value} precision:{precision} "
                  + f"recall:{recall} fbeta:{fbeta}")
        print()  # Blank line for readability
