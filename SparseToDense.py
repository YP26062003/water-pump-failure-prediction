from scipy.sparse import issparse
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.decomposition import TruncatedSVD

class SparseToDenseTransformer:
    def __init__(self, model):
        self.model = model
        self.svd = None

    def fit(self, X, y=None):
        # Fit TruncatedSVD if the model is HistGradientBoostingClassifier and the data is sparse
        if isinstance(self.model, HistGradientBoostingClassifier) and issparse(X):
            print("Fitting TruncatedSVD for HistGradientBoostingClassifier")
            self.svd = TruncatedSVD(n_components=100, random_state=42)
            self.svd.fit(X)
        return self

    def transform(self, X):
        # Apply TruncatedSVD for HistGradientBoostingClassifier if the data is sparse
        if isinstance(self.model, HistGradientBoostingClassifier) and issparse(X):
            print("Applying TruncatedSVD for HistGradientBoostingClassifier")
            if self.svd is None:
                # Initialize TruncatedSVD with a reasonable number of components
                self.svd = TruncatedSVD(n_components=100, random_state=42)
                self.svd.fit(X)  # Fit on the training data
            X_transformed = self.svd.transform(X)  # Transform the data
            return X_transformed
        return X

