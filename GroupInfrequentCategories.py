from sklearn.base import BaseEstimator, TransformerMixin

class GroupInfrequentCategories(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=100, cat_threshold=5):
        self.threshold = threshold  # Cardinality threshold for high-cardinality features
        self.cat_threshold = cat_threshold  # Minimum category frequency threshold
        self.high_cardinality_features_ = []  # Will store high-cardinality column names
        self.infrequent_categories_ = {}  # Will store infrequent categories for each column

    def fit(self, X, y=None):
        # Find features with high cardinality
        cardinality = X.nunique()
        self.high_cardinality_features_ = cardinality[cardinality > self.threshold].index.tolist()

        # Identify infrequent categories for each high-cardinality feature
        for col in self.high_cardinality_features_:
            freq = X[col].value_counts()
            self.infrequent_categories_[col] = freq[freq < self.cat_threshold].index.tolist()

        return self

    def transform(self, X):
        X = X.copy()  # Avoid modifying the original dataframe

        for col in self.high_cardinality_features_:
            if col in self.infrequent_categories_:
                X[col] = X[col].replace(self.infrequent_categories_[col], 'Other')

        return X
