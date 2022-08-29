from sklearn.neural_network import MLPClassifier


def get_mlp_model_wp():
    model = MLPClassifier(max_iter=1000)
    return model
