from sklearn.base import BaseEstimator, TransformerMixin
from spicy import sparse


class FunctionTransformer():
    def __init__(self, func):
        super().__init__()
        self.func = func

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        return sparse.csr_matrix(X_.apply(self.func)).transpose()
