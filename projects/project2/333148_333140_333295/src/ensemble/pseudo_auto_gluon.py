import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.base import clone, BaseEstimator


class PseudoAutoGluon(BaseEstimator):
    """
    A multi-layer stacking ensemble estimator inspired by AutoGluon.

    This class implements a multi-layer stacking architecture where each layer
    contains multiple estimators. Out-of-fold (OOF) predictions from one layer
    are concatenated with the original features to serve as input for the
    subsequent layers. The final predictions are made by a meta-estimator
    trained on the accumulated stacked features.

    Attributes:
        fitted_layers (list): Storage for trained fold models for each layer.
        fitted_final_model (BaseEstimator): The trained meta-estimator.
        n_classes_ (int): Number of unique classes (only for classification).
    """

    def __init__(
        self,
        layers_estimators: list[list[tuple[str, BaseEstimator]]],
        final_estimator: BaseEstimator,
        task: str = "classification",
        n_folds: int = 5,
        random_state: int = 42,
    ):
        """
        Initializes the PseudoAutoGluon stacking ensemble.

        Args:
            layers_estimators (list[list[tuple[str, BaseEstimator]]]): A list of layers,
                where each layer is a list of (name, estimator) tuples.

            final_estimator (BaseEstimator): The meta-estimator used for the final
                prediction on the stacked features.

            task (str, optional): The machine learning task, either 'classification'
                or 'regression'. Defaults to "classification".

            n_folds (int, optional): Number of cross-validation folds for
                out-of-fold prediction generation. Defaults to 5.

            random_state (int, optional): Controls the randomness of the
                KFold/StratifiedKFold splits. Defaults to 42.
        """
        self.layers_estimators = layers_estimators
        self.final_estimator = final_estimator
        self.task = task
        self.n_folds = n_folds
        self.random_state = random_state

        self.fitted_layers = []
        self.fitted_final_model = None
        self.n_classes_ = None

    def _get_cv(self, y: np.ndarray):
        """
        Constructs the cross-validation splitter based on the task type.

        Returns a StratifiedKFold splitter for classification to preserve
        class distributions, or a standard KFold splitter for regression.

        Args:
            y (np.ndarray): The target variable used to determine
                stratification splits in classification tasks.

        Returns:
            Union[StratifiedKFold, KFold]: A scikit-learn cross-validation
                splitter object configured with the instance's n_folds
                and random_state.
        """
        if self.task == "classification":
            return StratifiedKFold(
                n_splits=self.n_folds,
                shuffle=True,
                random_state=self.random_state,
            )

        return KFold(
            n_splits=self.n_folds,
            shuffle=True,
            random_state=self.random_state,
        )

    def _flatten_layers(self) -> list[tuple[str, BaseEstimator]]:
        """
        Flattens the multi-layer estimator structure into a single list.

        Iterates through each layer in `layers_estimators` and extracts all
        individual model tuples into a flat representation. This is primarily
        used for easier iteration or inspection of all base learners.

        Returns:
            list[tuple[str, BaseEstimator]]: A flat list containing all
                (name, model) tuples from all layers in their original order.
        """
        flat = []
        for layer in self.layers_estimators:
            for model in layer:
                flat.append(model)
        return flat

    def fit(self, X_list, y):
        """
        Fits the multi-layer ensemble using Out-Of-Fold (OOF) prediction strategy.

        The method trains each model in each layer using cross-validation. For every model,
        it generates OOF predictions that are concatenated with predictions from previous
        layers. These accumulated 'stacked features' serve as augmented input for
        subsequent layers and the final meta-estimator.

        Args:
            X_list (list[np.ndarray]): A list of input datasets. The order of datasets
                must strictly correspond to the order of models defined in
                `layers_estimators`.

            y (np.ndarray): The target vector for training.

        Returns:
            self: The fitted PseudoAutoGluon instance.

        Note:
            - In classification, if `n_classes > 2`, full probability distributions
              are used for stacking.
            - For binary classification, only the probability of the positive class
              is propagated.
            - For regression, raw predictions are used.
        """
        X_list = [
            X.to_numpy(dtype=float, na_value=np.nan)
            if isinstance(X, pd.DataFrame)
            else np.asarray(X, dtype=float)
            for X in X_list
        ]
        y = y.values if isinstance(y, (pd.Series, pd.DataFrame)) else y

        self.fitted_layers = []

        if self.task == "classification":
            self.n_classes_ = len(np.unique(y))

        stacked_features = None
        X_ptr = 0

        for layer in self.layers_estimators:
            layer_oof_preds = []
            layer_models = []

            for _, model in layer:
                X = X_list[X_ptr]
                X_ptr += 1

                cv = self._get_cv(y)

                if self.task == "classification":
                    if self.n_classes_ > 2:
                        oof = np.zeros((len(X), self.n_classes_))
                    else:
                        oof = np.zeros((len(X), 1))
                else:
                    oof = np.zeros((len(X), 1))

                fold_models = []

                for tr, val in cv.split(X, y):
                    X_tr = X[tr, :]
                    X_val = X[val, :]
                    y_tr = y[tr]

                    if stacked_features is not None:
                        X_tr = np.hstack([X_tr, stacked_features[tr]])
                        X_val = np.hstack([X_val, stacked_features[val]])

                    m = clone(model)
                    m.fit(X_tr, y_tr)

                    if self.task == "classification":
                        p = m.predict_proba(X_val)
                        if self.n_classes_ == 2:
                            p = p[:, 1].reshape(-1, 1)
                    else:
                        p = m.predict(X_val)

                    oof[val] = p
                    fold_models.append(m)

                layer_oof_preds.append(oof)
                layer_models.append((X_ptr - 1, fold_models))

            new_stack = np.hstack(layer_oof_preds)
            stacked_features = new_stack if stacked_features is None else np.hstack([stacked_features, new_stack])

            self.fitted_layers.append(layer_models)

        self.fitted_final_model = clone(self.final_estimator)
        self.fitted_final_model.fit(stacked_features, y)

        return self

    def _stack_predict(self, X_list: list[pd.DataFrame]) -> np.ndarray:
        """
        Transforms input data into stacked features through the trained layers.

        This internal method processes the input datasets layer by layer. For each
        model in a layer, it computes predictions using all previously trained
        fold models and averages them. These averaged predictions are then
        concatenated with previous stacked features to form the input for
        the next layer.

        Args:
            X_list (list[np.ndarray]): A list of input datasets corresponding
                to the indices mapped during the fitting process.

        Returns:
            np.ndarray: The final matrix of accumulated stacked features,
                ready to be processed by the final meta-estimator.

        Note:
            The method ensures that the dimensionality of features matches the
            expected input shape of each layer by performing the same
            horizontal stacking (`np.hstack`) as seen in the `fit` method.
        """
        X_list = [
            X.to_numpy(dtype=float, na_value=np.nan)
            if isinstance(X, pd.DataFrame)
            else np.asarray(X, dtype=float)
            for X in X_list
        ]
        stacked_features = None

        for layer in self.fitted_layers:
            layer_preds = []

            for X_idx, fold_models in layer:
                X = X_list[X_idx]
                preds = []

                for m in fold_models:
                    X_in = X if stacked_features is None else np.hstack([X, stacked_features])

                    if self.task == "classification":
                        p = m.predict_proba(X_in)
                        if self.n_classes_ == 2:
                            p = p[:, 1].reshape(-1, 1)
                    else:
                        p = m.predict(X_in).reshape(-1, 1)

                    preds.append(p)

                layer_preds.append(np.mean(preds, axis=0))

            new_stack = np.hstack(layer_preds)
            stacked_features = new_stack if stacked_features is None else np.hstack([stacked_features, new_stack])

        return stacked_features

    def predict(self, X_list):
        """
            Predicts target values for the given input datasets.

            First, it transforms the input data into a stacked feature set by
            passing it through the fitted ensemble layers. Then, it uses the
            final meta-estimator to produce the ultimate predictions.

            Args:
                X_list (list[pd.DataFrame]): A list of input datasets corresponding
                    to the models in the ensemble.

            Returns:
                np.ndarray: Predicted class labels (for classification) or
                    continuous values (for regression).
            """
        stacked = self._stack_predict(X_list)
        return self.fitted_final_model.predict(stacked)

    def predict_proba(self, X_list: list[pd.DataFrame]) -> np.ndarray:
        """
        Predicts class probabilities for the given input datasets.

        This method generates stacked features using the multi-layer architecture
         and then applies the final meta-estimator's probability estimation logic.

        Args:
            X_list (list[np.ndarray]): A list of input datasets corresponding
                to the models in the ensemble.

        Returns:
            np.ndarray: Probability of each class for each sample,
                typically of shape (n_samples, n_classes).

        Raises:
            AttributeError: If the ensemble was initialized with task='regression',
                as probability estimation is not defined for regression.
        """
        if self.task != "classification":
            raise AttributeError("predict_proba only for classification")

        stacked = self._stack_predict(X_list)
        return self.fitted_final_model.predict_proba(stacked)
