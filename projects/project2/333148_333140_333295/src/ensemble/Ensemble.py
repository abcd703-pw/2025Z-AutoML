import numpy as np
from scipy.stats import mode
from sklearn.base import BaseEstimator


class Ensemble(BaseEstimator):
    """
    A meta-estimator for combining multiple heterogeneous models into an ensemble.

    This class implements a multi-model ensemble wrapper that supports both
    classification and regression tasks. It allows passing different input data
    to different base estimators, enabling the construction of complex multi-view
    learning pipelines. For classification, it supports both hard (majority)
    and soft (probability-based) voting.

    Attributes:
        estimators (list of tuples): List of (name, model) tuples representing the base learners.
        task (str): The nature of the machine learning task ('classification' or 'regression').
        voting (str): The voting strategy for classification ('hard' or 'soft').
    """

    def __init__(self, estimators, task='classification', voting='hard'):
        """
        Initializes the Ensemble meta-estimator.

        Args:
            estimators (list of tuples): A list of (name, estimator) tuples where
                'name' is a unique string and 'estimator' is a model instance
                following the scikit-learn API.
            task (str, optional): Defines the task type. Must be either
                'classification' or 'regression'. Defaults to 'classification'.
            voting (str, optional): Specifies the voting logic. Use 'hard' for
                majority class voting or 'soft' for averaging predicted
                probabilities. Defaults to 'hard'.

        Raises:
            ValueError: If an unsupported task type or voting strategy is provided.
        """
        self.estimators = estimators
        self.task = task
        self.voting = voting

    def fit(self, Xs, y=None, refit=True):
        """
        Fits each base estimator to its corresponding dataset.

        This method iterates through the provided list of datasets and trains
        the internal models. Each model in the ensemble is mapped to exactly
        one dataset in the input list based on its index.


        Args:
            Xs (list of array-like): A list containing datasets (DataFrames or arrays)
                corresponding to each estimator in `self.estimators`.
                The length of this list must match the number of estimators.

            y (array-like, optional): The target values (class labels in classification,
                real numbers in regression). Defaults to None.

            refit (bool, optional): If True, all models will be trained from scratch.
                If False, the training step is skipped, assuming models are already
                pre-trained. Defaults to True.

        Returns:
            self: Returns the instance of the ensemble.

        Raises:
            ValueError: If the length of `Xs` does not match the number of
                defined estimators.
        """
        if not refit:
            return self

        if len(Xs) != len(self.estimators):
            raise ValueError(f"Liczba ramek danych ({len(Xs)}) nie pasuje do liczby modeli ({len(self.estimators)}).")

        for i, (name, model) in enumerate(self.estimators):
            model.fit(Xs[i], y)

        return self

    def predict_proba(self, Xs):
        """
        Zwraca uśrednione prawdopodobieństwa (tylko dla task='classification').
        """
        if self.task != 'classification':
            raise AttributeError("The predict_proba method is only available for classification tasks.")

        if len(Xs) != len(self.estimators):
            raise ValueError(f"The number of datasets ({len(Xs)}) does not match the number of estimators ({len(self.estimators)}).")

        probas = []
        for i, (name, model) in enumerate(self.estimators):
            if not hasattr(model, "predict_proba"):
                raise AttributeError(f"Model '{name}' does not have a 'predict_proba' method.")
            probas.append(model.predict_proba(Xs[i]))

        return np.mean(probas, axis=0)

    def predict(self, Xs):
        """
        Computes average class probabilities for classification tasks.

        This method collects probability estimates from all base estimators
        and calculates their arithmetic mean. It requires each underlying
        model to support the `predict_proba` method.

        Args:
            Xs (list of array-like): A list of datasets corresponding to each
                estimator. Must have the same length as `self.estimators`.

        Returns:
            numpy.ndarray: An array of shape (n_samples, n_classes) containing
                the averaged probability of each class for each sample.

        Raises:
            AttributeError: If the task is not set to 'classification' or if
                one of the base estimators does not support probability estimation.
            ValueError: If the length of `Xs` does not match the number of
                defined estimators.
        """
        if len(Xs) != len(self.estimators):
            raise ValueError(f"The number of data frames ({len(Xs)}) does not match the number of models ({len(self.estimators)}).")

        # --- REGRESSION ---
        if self.task == 'regression':
            preds = []
            for i, (name, model) in enumerate(self.estimators):
                preds.append(model.predict(Xs[i]))
            return np.mean(preds, axis=0)

        # --- CLASSIFICATION ---
        elif self.task == 'classification':
            if self.voting == 'soft':
                avg_proba = self.predict_proba(Xs)
                return np.argmax(avg_proba, axis=1)

            elif self.voting == 'hard':
                preds = []
                for i, (name, model) in enumerate(self.estimators):
                    preds.append(model.predict(Xs[i]))

                preds_stack = np.vstack(preds)
                major_vote, _ = mode(preds_stack, axis=0, keepdims=False)

                return major_vote

        else:
            raise ValueError("Unknown task. Choose either 'classification' or 'regression'.")