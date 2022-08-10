# The functions used here are based on code written in "0 Introduction.ipynb",
#  look at the notebook for extra explanation.
from .util_0_introduction import get_preprocessed_dataset

from sklearn import model_selection


def get_all_y_columns(df):
    return df[['Actief', 'Lang', 'Vragen', 'Interpunctie', 'Tweeledigheid', 'Emotie', 'Voorwaartse Verwijzing',
               'Signaalwoorden', 'Lidwoorden', 'Adjectieven', 'Eigennamen', 'Betrekking', 'Voor+Achternaam',
               'Cijfers', 'Quotes']]


def get_cls_train_test(column_name=None):
    # Returns the train and test dataframes for the column with given name
    # @param column_name: str The name of the column you want to predict

    df = get_preprocessed_dataset()

    # Train test split
    df_per_test = df.groupby("Test").apply(lambda x: x.sample(1, random_state=42))

    if column_name is not None:
        train_x, test_x, train_y, test_y = model_selection.train_test_split(df_per_test["Headline"],
                                                                            df_per_test[column_name], random_state=42)
    else:
        train, test = model_selection.train_test_split(df_per_test, random_state=42)

        train_x = train[["Headline"]]
        train_y = get_all_y_columns(train)

        test_x = test[["Headline"]]
        test_y = get_all_y_columns(test)

    return train_x, train_y, test_x, test_y
