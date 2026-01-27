from boruta import BorutaPy
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import (
    chi2, mutual_info_classif, mutual_info_regression,
    SelectFromModel, SelectKBest
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier, XGBRegressor


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
        A modular feature selection transformer compatible with scikit-learn.

        This class provides a unified interface for various feature selection
        strategies, ranging from simple variance filters to advanced iterative
        algorithms like Boruta. It is designed to work with preprocessed
        pandas DataFrames.

        Parameters
        ----------
        method : str, default='variance'
            The feature selection strategy to be used. Available options:
            - 'variance': Removes features with low variability.
            - 'mutual_info': Selects features based on Mutual Information scores.
            - 'chi2': Uses the Chi-Squared statistical test (requires non-negative data).
            - 'random_forest': Selects features based on Random Forest importance.
            - 'xgboost': Selects features based on XGBoost 'gain' or 'weight'.
            - 'boruta': An all-relevant selection method using shadow features.
            - 'permutation': Selects features based on the drop in model performance.
            - 'all': Pass-through method that keeps all original features.

        params : dict, optional (default=None)
            A dictionary of hyperparameters for the chosen method.
            If None, default values for each method are applied.
        """
    def __init__(self, method='variance', selected_fraction=1, params=None, verbose=False, random_state=42):
        """
            Initializes the FeatureSelector with specific method parameters.

            Generic params keys used across multiple methods:
            ------------------------------------------------
            task : str ('classification' or 'regression')
                Required for: 'mutual_info', 'random_forest', 'xgboost', 'boruta', 'permutation'.
            threshold : float or str
                Used as a cutoff value. In model-based methods, can be 'mean' or 'median'.

            Detailed Method-Specific Parameters:
            -----------------------------------
            method='variance'
                - threshold (float): Minimum variance required to keep a feature.
                  Default: 0.0 (removes only constant columns).

            method='mutual_info'
                - k (int): Number of top features to select. Default: 10.
                - n_neighbors (int): Number of neighbors for MI estimation. Default: 3.
                - task (str): 'classification' or 'regression'. Default: 'classification'.

            method='chi2'
                - k (int): Number of top features to select. Default: 10.
                Note: Data must be non-negative (e.g., after MinMaxScaler).

            method='random_forest'
                - n_estimators (int): Number of trees in the forest. Default: 100.
                - max_depth (int): Maximum depth of trees. Default: None.
                - threshold (str/float): Cutoff for importance (e.g., 'mean', 'median', or 0.01).
                  Default: 'mean'.
                - task (str): 'classification' or 'regression'. Default: 'classification'.

            method='xgboost'
                - importance_type (str): 'gain', 'weight', or 'cover'. Default: 'gain'.
                - threshold (str/float): Cutoff for importance. Default: 'mean'.
                - task (str): 'classification' or 'regression'. Default: 'classification'.

            method='boruta'
                - n_estimators (int/str): Number of trees or 'auto'. Default: 'auto'.
                - perc (int): Strictness level (0-100). Higher is more strict. Default: 100.
                - alpha (float): Statistical significance level (p-value). Default: 0.05.
                - max_iter (int): Maximum number of iterations. Default: 100.
                - task (str): 'classification' or 'regression'. Default: 'classification'.

            method='permutation'
                - n_repeats (int): Number of times to shuffle each feature. Default: 5.
                - threshold (float): Minimum score drop required to keep feature. Default: 0.001.
                - task (str): 'classification' or 'regression'. Default: 'classification'.
        """
        self.method = method
        self.params = params if params is not None else {}
        self.selected_columns_ = None
        self.original_columns_ = None
        self.verbose = verbose
        self.random_state = random_state

        if self.method == 'variance' and 'threshold' not in self.params:
            self.params['threshold'] = 0.0

        self.selected_fraction = selected_fraction

    def fit(self, X, y=None):
        """
        Fits the feature selector by identifying relevant features based on the chosen method.

        This method validates the input data, stores the original column names,
        and dispatches the feature selection logic to the specific internal
        method defined during initialization.

        Args:
            X (pd.DataFrame): The input feature matrix. Must be a pandas DataFrame
                to ensure column names are preserved and accessible.
            y (np.ndarray, optional): The target variable. Required for supervised
                methods (e.g., 'boruta', 'xgboost', 'mutual_info'). Defaults to None.

        Returns:
            self: The fitted FeatureSelector instance.

        Raises:
            ValueError: If X is not an instance of pandas.DataFrame.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be pandas.DataFrame object.")

        self.original_columns_ = X.columns.tolist()


        if self.method == 'variance':
            self.selected_columns_ = self._variance_filter(X)

        elif self.method == 'mutual_info':
            self.selected_columns_ = self._mutual_info_selection(X, y)

        elif self.method == 'all':
            self.selected_columns_ = self.original_columns_

        elif self.method == 'chi2':
            self.selected_columns_ = self._chi2_selection(X, y)

        elif self.method == 'random_forest':
            self.selected_columns_ = self._rf_selection(X, y)

        elif self.method == 'xgboost':
            self.selected_columns_ = self._xgboost_selection(X, y)

        elif self.method == 'boruta':
            self.selected_columns_ = self._boruta_selection(X, y)

        elif self.method == 'permutation':
            self.selected_columns_ = self._permutation_selection(X, y)

        else:
            self.selected_columns_ = self.original_columns_

        return self

    def _variance_filter(self, X):
        """
        Filters features based on their individual variance.

        This unsupervised method calculates the variance of each column and
        retains only those that exceed the specified 'threshold'. It is
        primarily used to remove constant or near-constant features that
        provide little to no predictive power.

        Parameters
        ----------
        X : pd.DataFrame
            The input feature matrix.

        Returns
        -------
        list
            A list of column names with variance strictly greater than the threshold.
        """
        variances = X.var()
        selected = variances[variances > self.params.get('threshold', 0.0)].index.tolist()

        return selected

    def _mutual_info_selection(self, X, y):
        """
        Selects the top K features based on Mutual Information (MI) scores.

        Mutual Information measures the statistical dependence between each
        feature and the target variable. Unlike linear correlation, it can
        capture non-linear relationships. It uses K-Nearest Neighbors
        methods for entropy estimation.

        Parameters
        ----------
        X : pd.DataFrame
            The input feature matrix.
        y : pd.Series or np.array
            The target variable.

        Returns
        -------
        list
            A list of the top 'k' column names with the highest Mutual
            Information scores.

        Raises
        ------
        ValueError
            If the target variable 'y' is not provided.
        """
        if y is None:
            raise ValueError("The 'mutual_info' method requires a target (y) to be provided in the fit method.")

        k = int(np.ceil(X.shape[1] * self.selected_fraction))
        task = self.params.get('task', 'classification')
        n_neighbors = self.params.get('n_neighbors', 3)

        mi_func = mutual_info_classif if task == 'classification' else mutual_info_regression

        def mi_wrapper(X_input, y_input):
            return mi_func(X_input, y_input, n_neighbors=n_neighbors, random_state=42)

        selector = SelectKBest(score_func=mi_wrapper, k=k)
        selector.fit(X, y)

        selected_idx = selector.get_support(indices=True)
        selected = X.columns[selected_idx].tolist()

        return selected

    def _chi2_selection(self, X, y):
        """
        Performs feature selection using the Chi-Squared (χ2) statistical test.

        This method evaluates the independence between each feature and the target
        variable. It is highly effective for categorical data or binned numerical
        features. The test requires all input values to be non-negative.

        Parameters
        ----------
        X : pd.DataFrame
            The input feature matrix. Must contain non-negative values.
        y : pd.Series or np.array
            The target variable (typically discrete/categorical).

        Returns
        -------
        list
            A list of the top 'k' column names with the highest Chi-Squared statistics.

        Raises
        ------
        ValueError
            If the target variable 'y' is not provided or if the input matrix 'X'
            contains negative values.
        """
        if y is None:
            raise ValueError("The 'chi2' method requires the target variable (y) to be provided.")

        if (X < 0).any().any():
            raise ValueError(
                "The Chi-Square test requires non-negative values. "
                "Ensure that you have used MinMaxScaler instead of StandardScaler."
            )

        k = int(np.ceil(X.shape[1] * self.selected_fraction))

        selector = SelectKBest(score_func=chi2, k=k)
        selector.fit(X, y)

        selected_idx = selector.get_support(indices=True)
        selected = X.columns[selected_idx].tolist()

        return selected

    def _rf_selection(self, X: pd.DataFrame, y: np.ndarray) -> list:
        """
        Selects features based on Random Forest importance scores.

        This method trains a Random Forest model (Classifier or Regressor) to
        rank features using Mean Decrease in Impurity (MDI). It then uses
        scikit-learn's `SelectFromModel` to extract the top-performing features
        based on the specified 'selected_fraction'.

        Args:
            X (pd.DataFrame): The input feature matrix.
            y (np.ndarray): The target variable. Required for supervised importance
                calculation.

        Returns:
            list: A list of column names representing the selected features.

        Raises:
            ValueError: If the target variable 'y' is not provided.
        """
        if y is None:
            raise ValueError("The 'random_forest' method requires a target variable (y).")

        k = int(np.ceil(X.shape[1] * self.selected_fraction))
        task = self.params.get('task', 'classification')
        n_estimators = self.params.get('n_estimators', 100)
        max_depth = self.params.get('max_depth', None)

        if task == 'classification':
            rf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=self.random_state,
                n_jobs=-1)
        else:
            rf = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=self.random_state,
                n_jobs=-1)

        selector = SelectFromModel(estimator=rf, threshold=-np.inf, max_features=k)
        selector.fit(X, y)

        selected_idx = selector.get_support(indices=True)
        selected = X.columns[selected_idx].tolist()

        return selected

    def _xgboost_selection(self, X, y):
        """
        Selects features based on XGBoost feature importance scores.

        This method utilizes a Gradient Boosting model to evaluate feature
        significance. It leverages the 'SelectFromModel' meta-transformer
        to discard features with importance scores below a specified threshold.
        XGBoost is particularly effective at handling sparse data and
        capturing complex interactions.

        Parameters
        ----------
        X : pd.DataFrame
            The input feature matrix.
        y : pd.Series or np.array
            The target variable.

        Returns
        -------
        list
            A list of column names selected based on the XGBoost importance criteria.

        Raises
        ------
        ValueError
            If the target variable 'y' is not provided.
        """
        if y is None:
            raise ValueError("The 'xgboost' method requires a target variable (y).")

        task = self.params.get('task', 'classification')
        importance_type = self.params.get('importance_type', 'gain')
        k = int(np.ceil(X.shape[1] * self.selected_fraction))

        X_tmp = X.copy()
        cols_to_convert = X_tmp.select_dtypes(exclude=[np.number]).columns
        X_tmp[cols_to_convert] = X_tmp[cols_to_convert].astype('category')

        if task == 'classification':
            xgb = XGBClassifier(
                n_estimators=100,
                importance_type=importance_type,
                random_state=self.random_state,
                n_jobs=-1
                ,enable_categorical=True
                ,tree_method='hist'
            )
        else:
            xgb = XGBRegressor(
                n_estimators=100,
                importance_type=importance_type,
                random_state=self.random_state,
                n_jobs=-1
                ,enable_categorical=True
                ,tree_method='hist'
            )

        selector = SelectFromModel(estimator=xgb, threshold=-np.inf, max_features=k)
        selector.fit(X, y)

        selected_idx = selector.get_support(indices=True)
        selected = X.columns[selected_idx].tolist()

        return selected

    def _boruta_selection(self, X, y):
        """
        Selects features using the Boruta all-relevant feature selection algorithm.

        Boruta is an iterative method that identifies all features that are
        statistically significantly better than "shadow features" (randomly
        shuffled copies of the original features). It provides a robust way
        to distinguish between truly informative variables and noise.

        Parameters
        ----------
        X : pd.DataFrame
            The input feature matrix.
        y : pd.Series or np.array
            The target variable.

        Returns
        -------
        list
            A list of column names confirmed as significant by Boruta. If no
            features are confirmed, features with 'weak support' are returned.

        Raises
        ------
        ValueError
            If the target variable 'y' is not provided.
        """
        if y is None:
            raise ValueError("Metoda 'boruta' wymaga podania celu (y).")

        task = self.params.get('task', 'classification')
        n_estimators = self.params.get('n_estimators', 'auto')
        perc = self.params.get('perc', 100)
        alpha = self.params.get('alpha', 0.05)
        max_iter = self.params.get('max_iter', 100)


        if task == 'classification':
            rf = RandomForestClassifier(n_estimators=100 if n_estimators == 'auto' else n_estimators,
                                        n_jobs=-1, max_depth=5, random_state=self.random_state)
        else:
            rf = RandomForestRegressor(n_estimators=100 if n_estimators == 'auto' else n_estimators,
                                       n_jobs=-1, max_depth=5, random_state=self.random_state)

        feat_selector = BorutaPy(
            rf,
            n_estimators=n_estimators,
            perc=perc,
            alpha=alpha,
            max_iter=max_iter,
            random_state=self.random_state,
            verbose=0
        )

        feat_selector.fit(X.values, y.values)
        selected_idx = np.where(feat_selector.support_)[0]

        if len(selected_idx) == 0:
            selected_idx = np.where(feat_selector.support_weak_)[0]
            print("Boruta: No confirmed features found, selecting 'weak support' features.")

        selected = X.columns[selected_idx].tolist()
        return selected

    def _permutation_selection(self, X, y):
        """
        Selects features based on Permutation Importance scores.

        This method trains a baseline Random Forest model and measures the
        importance of each feature by calculating the drop in model performance
        when the feature's values are randomly shuffled. Features that cause
        a performance drop greater than the 'threshold' are retained.

        Parameters
        ----------
        X : pd.DataFrame
            The input feature matrix.
        y : pd.Series or np.array
            The target variable.

        Returns
        -------
        list
            A list of column names that passed the permutation importance threshold.

        Raises
        ------
        ValueError
            If the target variable 'y' is not provided.
        """
        if y is None:
            raise ValueError("The 'permutation' method requires the target variable (y) to be provided.")

        task = self.params.get('task', 'classification')
        n_repeats = self.params.get('n_repeats', 5)
        k = int(np.ceil(X.shape[1] * self.selected_fraction))


        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if task == 'classification' else None
        )


        if task == 'classification':
            model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
        else:
            model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)

        model.fit(X_train, y_train)

        result = permutation_importance(
            model, X_val, y_val, n_repeats=n_repeats, random_state=42, n_jobs=-1
        )

        importance_series = pd.Series(result.importances_mean, index=X.columns)
        selected = importance_series.sort_values(ascending=False).head(k).index.tolist()

        return selected


    def preprocess(self, X):
        """
        Reduces the input DataFrame to the subset of features selected during fit.

        This method applies the feature selection results to a new dataset.
        It ensures that the input is a pandas DataFrame and validates that
        all previously selected columns are present before performing the slice.

        Parameters
        ----------
        X : {array-like, sparse matrix, pd.DataFrame} of shape (n_samples, n_features)
            The new input samples to be transformed.

        Returns
        -------
        pd.DataFrame
            A copy of the input DataFrame containing only the selected features.

        Raises
        ------
        RuntimeError
            If the 'fit' method has not been called before preprocessing.
        """
        if self.selected_columns_ is None:
            raise RuntimeError("The 'fit' method must be called before 'preprocess' (or 'transform').")

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.original_columns_)

        return X[self.selected_columns_].copy()

    def transform(self, X):
        """
        Reduces X to the selected features.

        Parameters
        ----------
        X : pd.DataFrame or array-like
            The data to transform.

        Returns
        -------
        pd.DataFrame
            Selected features from X.
        """
        return self.preprocess(X)

    def fit_transform(self, X, y=None, **fit_params):
        """
        Fit the selector to the data and transform it in a single step.

        Parameters
        ----------
        X : pd.DataFrame
            The training input samples.
        y : pd.Series or np.array, optional
            The target variable.
        **fit_params : dict
            Additional fit parameters.

        Returns
        -------
        pd.DataFrame
            The dataset containing only the selected features.
        """
        return self.fit(X, y).transform(X)