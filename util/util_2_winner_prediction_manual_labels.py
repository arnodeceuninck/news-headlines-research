# Note: some of those functions are based on code written in 2a Naive Bayes (but were applicable for more models)
from sklearn import metrics
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from .util_0_introduction import get_preprocessed_dataset
from .util_1_label_classification import get_label_columns

from sklearn import model_selection


def get_winners_only(df):
    return df[df["Winner"] == True]


def get_x_y_from_ids(df, ids, full_y=False):
    filtered_df = df[df["Test"].isin(ids)]

    x = filtered_df.drop(columns=["Winner"])

    if not full_y:
        y = get_winners_only(filtered_df)
    else:
        y = filtered_df[["Test", "Headline ID", "Winner"]]

    return x, y


def merge_x_y(x, y):
    return x.merge(y, on=['Test', 'Headline ID'], how='inner')


def get_wpm_train_test(x_train_features_only=True, full_y_train=True, full_y_test=False, df=None, include_groups=False):
    if df is None:
        df = get_preprocessed_dataset()

    tests_ids = df.Test.unique()

    train_ids, test_ids = model_selection.train_test_split(tests_ids, test_size=0.2, random_state=42)

    train_x, train_y = get_x_y_from_ids(df, train_ids, full_y=full_y_train)
    test_x, test_y = get_x_y_from_ids(df, test_ids, full_y=full_y_test)

    groups = train_x['Test'] if include_groups else None

    if x_train_features_only:
        train_x = get_manually_labeled_features(train_x)

    if include_groups:
        return train_x, train_y, test_x, test_y, groups
    else:
        return train_x, train_y, test_x, test_y


def get_manually_labeled_features(df):
    return df[get_label_columns()]


def drop_labels(df):
    return df.drop(columns=get_label_columns(), axis=1)


def evaluate_wp(target, predicted):
    assert len(target) == len(predicted)

    target_winner = target.reset_index(drop=True)
    predicted_winner = predicted.reset_index(drop=True)

    # test_y_pred = pd.merge(target_winner, predicted_winner, on=["Test"], suffixes=("", " Predicted"))
    # Filter the rows where Headline ID == Headline ID Predicted
    # test_y_pred_correct = test_y_pred[test_y_pred["Headline ID"] == test_y_pred["Headline ID Predicted"]]

    # n_correct = len(test_y_pred_correct)
    # n_total = len(target_winner)

    accuracy_pd = metrics.accuracy_score(target_winner["Headline ID"], predicted_winner["Headline ID"])
    # accuracy = n_correct / n_total
    # assert accuracy == accuracy_pd

    return accuracy_pd


def print_wp_evaluation(target, predicted, return_acc=False):
    accuracy = evaluate_wp(target, predicted)

    total_tests = len(target)
    n_correct = int(accuracy * total_tests)

    print(f"Accuracy: {100 * accuracy:.2f}% ({n_correct}/{total_tests})")

    return accuracy if return_acc else None


def predict_wp(model, test_x, proba=True, features=None, multiple_class_names=None):
    if multiple_class_names is None:
        multiple_class_names = proba
    if features is None:
        features = get_label_columns()
    if proba:
        predicted_probs = model.predict_proba(test_x[features])
    else:
        predicted_probs = model.predict(test_x[features])

    class_names = list(
        range(len(model.classes_))) if multiple_class_names else 1  # if no prediction per class, only winner

    test_x_predictions = test_x.reset_index(drop=True, inplace=False)
    test_x_predictions[class_names] = pd.DataFrame(predicted_probs)

    predicted_winners = test_x_predictions.groupby('Test').apply(lambda x: x.sort_values(by=1, ascending=False).head(1))

    assert len(predicted_winners) == len(test_x.Test.unique())
    return predicted_winners


def fit_predict_print_wp(model, train_x, train_y, test_x, test_y, proba=True, return_acc=False, groups=None,
                         multiple_class_names=None, silent=False, features=None):

    if features is None:
        features = get_label_columns()

    # Fit
    if groups is None:
        model.fit(train_x[features], train_y['Winner'])
    else:
        model.fit(train_x[features], train_y['Winner'], groups)

    # Predict
    predicted_winners = predict_wp(model, test_x, proba, multiple_class_names=multiple_class_names, features=features)

    # Evaluate
    if not silent:
        acc_or_none = print_wp_evaluation(test_y, predicted_winners, return_acc)
    else:
        acc_or_none = evaluate_wp(test_y, predicted_winners)

    return acc_or_none


class RandomPredictor(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.classes_ = [0, 1]

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.random.randint(2, size=len(X))

    def predict_proba(self, X):
        return np.random.rand(len(X), 2)


def get_random_predictor_model():
    return RandomPredictor()
