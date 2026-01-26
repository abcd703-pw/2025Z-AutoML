from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


class Preprocessor():
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.numeric_cols = None
        self.categorical_cols = None
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.medians = None
        self.modes = None

    def fit(self, X):
        # Categorical and numerical columns
        self.numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
        self.categorical_cols = X.select_dtypes(exclude=['number']).columns.tolist()
        
        # Medians and modes for missing values
        if self.numeric_cols:
            self.medians = {col: X[col].median() for col in self.numeric_cols}
        if self.categorical_cols:
            self.modes = {col: X[col].mode()[0] for col in self.categorical_cols}

        X = X.copy()

        # Categorical
        if self.categorical_cols:
            # Missing values
            X[self.categorical_cols] = X[self.categorical_cols].fillna(self.modes)
            # One-hot encoder
            self.encoder.fit(X[self.categorical_cols])

        # Numerical
        if self.numeric_cols:
            # Missing values
            X[self.numeric_cols] = X[self.numeric_cols].fillna(self.medians)
            # Standard Scaler
            self.scaler.fit(X[self.numeric_cols])

        return self


    def transform(self, X):
        X = X.copy()

        # Numeric
        if self.numeric_cols:
            X[self.numeric_cols] = X[self.numeric_cols].fillna(self.medians)
            X[self.numeric_cols] = self.scaler.transform(X[self.numeric_cols])

        # Categorical
        if self.categorical_cols:
            X[self.categorical_cols] = X[self.categorical_cols].fillna(self.modes)
            encoded = self.encoder.transform(X[self.categorical_cols])
            encoded_df = pd.DataFrame(
                encoded,
                columns=self.encoder.get_feature_names_out(self.categorical_cols),
                index=X.index
            )

            X = X.drop(columns=self.categorical_cols)
            X = pd.concat([X, encoded_df], axis=1)

        return X
    

    def fit_transform(self, X):
        return self.fit(X).transform(X)
