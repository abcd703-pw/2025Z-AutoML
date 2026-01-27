import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


class Stacker(BaseEstimator):
    """
    A two-level stacking ensemble regressor/classifier.

    This class implements a classic stacking architecture where Level-0 base
    estimators provide predictions that serve as input features for a Level-1
    meta-estimator. It supports multi-view learning by allowing each base
    estimator to be trained on a different dataset.

    Attributes:
        estimators (list of tuples): List of (name, model) tuples for Level-0.
        final_estimator (BaseEstimator): The Level-1 meta-model.
        task (str): Nature of the task ('classification' or 'regression').
    """

    def __init__(
            self,
            estimators: list[tuple[str, BaseEstimator]],
            final_estimator: BaseEstimator,
            task: str = 'classification'
    ):
        """
        Initializes the Stacker ensemble.

        Args:
            estimators (list of tuples): A list of (name, model) tuples representing
                the base learners (Level-0). Each model should follow the
                scikit-learn API.

            final_estimator (BaseEstimator): The meta-estimator (Level-1) that
                combines the predictions of the base learners.

            task (str, optional): The type of machine learning task.
                Supported values: 'classification', 'regression'.
                Defaults to 'classification'.
        """
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.task = task

    def _get_base_predictions(self, Xs: list[pd.DataFrame]) -> np.ndarray:
        """
        Collects predictions from all base models and assembles them into a meta-feature matrix.

        This method acts as a feature extractor for the meta-model. For classification
        tasks, it prefers using class probabilities (`predict_proba`) as features.
        In binary classification, it simplifies the output by taking only the
        probability of the positive class. For regression or models without
        probability support, it falls back to raw predictions.

        Args:
            Xs (list[np.ndarray]): A list of datasets, where each dataset
                corresponds to an estimator in `self.estimators`.

        Returns:
            np.ndarray: A horizontally stacked matrix of predictions (meta-features)
                with shape (n_samples, n_extracted_features).

        Note:
            The number of columns in the returned matrix depends on the task
            and the number of classes. For multi-class classification using
            probabilities, each model contributes `n_classes` columns.
        """
        meta_features = []
        for i, (name, model) in enumerate(self.estimators):
            if self.task == 'classification':
                if hasattr(model, "predict_proba"):
                    preds = model.predict_proba(Xs[i])

                    if preds.shape[1] == 2:
                        preds = preds[:, 1].reshape(-1, 1)
                else:
                    preds = model.predict_proba(Xs[i])
            else:
                preds = model.predict(Xs[i]).reshape(-1, 1)

            meta_features.append(preds)

        return np.hstack(meta_features)

    def fit(self, Xs: list[pd.DataFrame], y: np.ndarray, refit_base: bool = True):
        """
        Trains the stacking ensemble by fitting base models and then the meta-model.

        The training process follows two steps:
        1. Optionally train all Level-0 estimators on their respective datasets.
        2. Generate meta-features by collecting predictions (probabilities or labels)
           from Level-0 models and use them to train the Level-1 final estimator.

        Args:
            Xs (list[np.ndarray]): List of input datasets for each base estimator.
            y (np.ndarray): Target values for training both levels.
            refit_base (bool, optional): If True, trains the base estimators from
                scratch. If False, assumes base estimators are already fitted.
                Defaults to True.

        Returns:
            self: The fitted Stacker instance.

        Raises:
            ValueError: If the number of provided datasets in `Xs` does not match
                the number of estimators.
        """

        if len(Xs) != len(self.estimators):
            raise ValueError(f"The number of data frames in Xs ({len(Xs)}) must match the number of estimators ({len(self.estimators)}).")

        if refit_base:
            for i, (name, model) in enumerate(self.estimators):
                model.fit(Xs[i], y)

        meta_features = self._get_base_predictions(Xs)

        self.final_estimator.fit(meta_features, y)

        return self

    def predict(self, Xs: list[pd.DataFrame]) -> np.ndarray:
        """
        Generates final predictions by routing input through the two-level stacking architecture.

        The process consists of:
        1. Level-0: Extracting base predictions from all estimators to form meta-features.
        2. Level-1: Passing these meta-features into the final estimator to obtain
           the ultimate result.

        Args:
            Xs (list[pd.DataFrame]): A list of input datasets (one for each base estimator).

        Returns:
            np.ndarray: Predicted class labels or regression values, depending on the task.
        """
        meta_features = self._get_base_predictions(Xs)

        return self.final_estimator.predict(meta_features)

    def predict_proba(self, Xs: list[pd.DataFrame]) -> np.ndarray:
        """
        Computes class probabilities from the meta-model.

        This method generates meta-features by collecting Level-0 predictions
        and then uses the final estimator to predict class probabilities.
        It is only applicable when the task is set to 'classification'.

        Args:
            Xs (list[np.ndarray]): A list of input datasets, one for each base estimator.

        Returns:
            np.ndarray: Probability estimates for each class, typically of
                shape (n_samples, n_classes).

        Raises:
            AttributeError: If the task is set to 'regression', as probability
                estimation is not supported for continuous targets.
        """
        if self.task != 'classification':
            raise AttributeError("predict_proba is only available for classification tasks.")

        meta_features = self._get_base_predictions(Xs)
        return self.final_estimator.predict_proba(meta_features)