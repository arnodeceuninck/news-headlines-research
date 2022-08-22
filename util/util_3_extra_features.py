from .util_2a_naive_bayes import get_naive_bayes_model_wp
from .util_2e_gradient_boosting_classifier import get_xgboost_model_wp
from .util_2_winner_prediction_manual_labels import predict_wp, print_wp_evaluation


def add_length_to_dataframe(df):
    modified_df = df.copy()
    modified_df["Length"] = df["Headline"].apply(len)
    return modified_df


# Actually from 3b, but makes more sense here
def fit_predict_evaluate_extra_features(train_x_new, train_y, test_x_new, test_y, feature_columns, train_groups):
    model_nb = get_naive_bayes_model_wp()

    model_nb.fit(train_x_new[feature_columns], train_y["Winner"])
    predicted_winners = predict_wp(model_nb, test_x_new, features=feature_columns)
    print_wp_evaluation(predicted_winners, test_y)

    model_xgb = get_xgboost_model_wp()

    model_xgb.fit(train_x_new[feature_columns], train_y["Winner"], train_groups)
    predicted_winners = predict_wp(model_xgb, test_x_new, features=feature_columns)
    print_wp_evaluation(predicted_winners, test_y)
    return model_nb, model_xgb
