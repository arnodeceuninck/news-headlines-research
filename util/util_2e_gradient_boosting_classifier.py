from xgboost import XGBClassifier
import xgboost as xgb
import matplotlib.pyplot as plt


def get_xgboost_model_wp():
    # Gives the untrained model
    best_params = {'colsample_bytree': 0.8376842762481432,
                   'gamma': 1.0100982566020316,
                   'learning_rate': 0.10560571568287097,
                   'max_depth': 5,
                   'reg_alpha': 41.0,
                   'reg_lambda': 0.46191366472424383}

    return XGBClassifier(n_estimators=500, random_state=42, **best_params)


def get_xgboost_importance(model, max_num_features=10):
    xgb.plot_importance(model, max_num_features=max_num_features)
    plt.figure(figsize=(16, 12))
    plt.show()
