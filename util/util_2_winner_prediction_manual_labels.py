# The functions used here are based on code written in "0 Introduction.ipynb",
#  look at the notebook for extra explanation.
from .util_0_introduction import get_preprocessed_dataset
from .util_1_label_classification import get_label_columns

from sklearn import model_selection


def get_x_y_from_ids(df, ids, full_y=False):
    filtered_df = df[df["Test"].isin(ids)]

    x = filtered_df.drop(columns=["Winner"])

    if not full_y:
        y = filtered_df[filtered_df["Winner"] == True][["Test", "Headline ID"]]
    else:
        y = filtered_df[["Test", "Headline ID", "Winner"]]

    return x, y


def get_wpm_train_test(full_y_train=True, full_y_test=False):
    df = get_preprocessed_dataset()

    tests_ids = df.Test.unique()

    train_ids, test_ids = model_selection.train_test_split(tests_ids, test_size=0.2, random_state=42)

    train_x, train_y = get_x_y_from_ids(df, train_ids, full_y=full_y_train)
    test_x, test_y = get_x_y_from_ids(df, tests_ids, full_y=full_y_test)

    return train_x, train_y, test_x, test_y


def get_manually_labeled_features(df):
    return df[get_label_columns()]
