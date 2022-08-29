from sklearn.ensemble import RandomForestClassifier


def get_random_forest_model_wp():
    best_params = {'criterion': 'gini',
                   'max_depth': 3,
                   'max_features': 'sqrt',
                   'n_estimators': 1500}
    model = RandomForestClassifier(**best_params)
    return model
