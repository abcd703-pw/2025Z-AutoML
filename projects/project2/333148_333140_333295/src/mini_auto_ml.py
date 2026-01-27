from statistics import LinearRegression
from typing import Iterable, Union

import numpy as np
import pandas as pd
from sklearn import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

from preprocessing.preprocess import Preprocessor
from feature_selection.select_features import FeatureSelector
from model_selector_v3.model_selector import ModelSelectorV3
from model_selector_v3._utils import DataMapper

from ensemble.Ensemble import Ensemble
from ensemble.pseudo_auto_gluon import PseudoAutoGluon
from ensemble.Stacker import Stacker

def balanced_accuracy(y_true: Union[np.ndarray, pd.Series, Iterable[int]],
                      y_pred: Union[np.ndarray, pd.Series, Iterable[int]],
                      sample_weight: Union[np.ndarray, pd.Series, Iterable[float], None] = None,
                      ) -> float:
    """
    Calculate balanced classification accuracy.

    This function is a wrapper around ``sklearn.metrics.balanced_accuracy_score``.

    Balanced accuracy is defined as the average of recall obtained on each class.
    It is particularly useful for imbalanced classification problems.

    Parameters
    ----------
    y_true : np.ndarray or pd.Series or iterable of int
        Ground truth (correct) target labels.

    y_pred : np.ndarray or pd.Series or iterable of int
        Predicted target labels.

    sample_weight : np.ndarray or pd.Series or iterable of float, default=None
        Sample weights. If ``None``, all samples are weighted equally.

    Returns
    -------
    score : float
        Balanced accuracy score.

    See Also
    --------
    sklearn.metrics.balanced_accuracy_score
    """

    return balanced_accuracy_score(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)


class MiniAutoML:
    """
    Parameters
    ----------

    models_config: str, default=None
        A local filesystem path to a JSON file containing the list of models.

    columns_to_ignore : list[str], default=[]
        List of feature column names to ignore during preprocessing.

    cols_to_ignore_destiny : {"drop", "preserve"}, default="drop"
        Defines how ignored columns are handled:
        - "drop": remove ignored columns from the dataset.
        - "preserve": keep ignored columns unchanged.

    date_use : bool, default=False
        Whether to enable date feature detection and processing.

    drop_uninformative_cols : bool, default=True
        Whether to automatically drop columns considered uninformative

    enable_advanced_auto_typing : bool, default=True
        Whether to enable advanced automatic feature type inference.

    na_col_uninformative_threshold : float, default=0.5
        Maximum allowed fraction of missing values in a column before it
        is considered uninformative and dropped.

    cat_variability_threshold : float, default=1
        Fractional threshold used to determine whether a categorical column
        is informative. A column is considered uninformative if::

            df[col].nunique() <= cat_variability_threshold * number_of_samples

    num_variability_threshold : float, default=0.1
        Minimum variance required for a numerical column to be considered
        informative.

    if_encode : bool, default=True
        Whether to apply categorical feature encoding.

    test_size : float, default=0.2
        Fraction of the dataset to use as the test set when splitting data.
        Applies only when ``X_test`` in ``Preprocessor.fit_transform()`` is None.

    ordinal_cols_enc : object or None, default=None
        Encoder used for ordinal categorical features.
        If None, an ``sklearn.preprocessing.OrdinalEncoder`` is created as::

            OrdinalEncoder(
                handle_unknown=ordinal_enc_handle_unknown,
                unknown_value=ordinal_enc_unknown_value
            )

        If provided, the object must implement a sklearn ``fit_transform`` method.

    ordinal_enc_handle_unknown : str, default="use_encoded_value"
        Value passed to ``OrdinalEncoder(handle_unknown=...)`` when
        ``ordinal_cols_enc`` is None.

    ordinal_enc_unknown_value : int, default=-1
        Value passed to ``OrdinalEncoder(unknown_value=...)`` when
        ``ordinal_cols_enc`` is None.

    nominal_cols_enc : object or None, default=None
        Encoder used for nominal categorical features.
        If None, an ``sklearn.preprocessing.OneHotEncoder`` is created as::

            OneHotEncoder(handle_unknown=nominal_enc_handle_unknown)

        If provided, the object must implement a sklearn ``fit_transform`` method.

    nominal_enc_handle_unknown : str, default="ignore"
        Value passed to ``OneHotEncoder(handle_unknown=...)`` when
        ``nominal_cols_enc`` is None.

    random_state : int, default=0
        Random seed used for reproducibility of operations such as data
        splitting.

    if_scale : bool, default=True
        Whether to apply feature scaling to numerical features.

    scaler_type : str or object, default="power_transform"
        Scaling strategy to use. If a string, the following mappings apply:

        - ``"power_transform"`` → ``PowerTransformer(method="yeo-johnson")``
        - ``"standardize"`` → ``StandardScaler()``
        - ``"normalize"`` → ``MinMaxScaler()``

        If an object is provided, it must implement a sklearn ``fit_transform`` method
        (e.g. ``PowerTransformer(method="box-cox")``).

    if_scale_cat_not_enc : bool, default=False
        Whether to scale categorical features that are not encoded.

    if_impute : bool, default=True
        Whether to perform missing-value imputation.

    num_imputation_type : str or object, default="median"
        Imputation strategy for numerical features. If a string, the following
        mappings apply:

        - ``"median"`` → ``SimpleImputer(strategy="median")``
        - ``"mean"`` → ``SimpleImputer(strategy="mean")``

        If an object is provided, it must implement a sklearn ``fit_transform`` method
        (e.g. ``SimpleImputer(strategy="most_frequent")``).

    date_imputation_type : str or object, default="most_frequent"
        Imputation strategy for date features. If a string, the following
        mappings apply:

        - ``"most_frequent"`` → ``SimpleImputer(strategy="most_frequent")``

        If an object is provided, it must implement a sklearn ``fit_transform`` method.

    if_impute_cat_not_enc : bool, default=False
        Whether to impute categorical features that are not encoded.

    if_impute_not_scaled : bool, default=False
        Whether to impute features that are not scaled.

    Attributes
    ----------
    random_state : int
        Random seed used throughout the pipeline.
    """

    def __init__(self,
                 task: str = 'classification',
                 verbose: bool = False,

                 models_config: str | None = None,

                 columns_to_ignore: list[str] = [],
                 cols_to_ignore_destiny: str = "drop",
                 date_use: bool = False,
                 drop_uninformative_cols: bool = True,
                 enable_advanced_auto_typing: bool = False,
                 na_col_uninformative_threshold: float = 0.5,
                 cat_variability_threshold: float = 1.0,
                 num_variability_threshold: float = 0.1,

                 if_encode: bool = True,
                 test_size: float = 0.2,
                 ordinal_cols_enc: object | None = None,
                 ordinal_enc_handle_unknown: str = "use_encoded_value",
                 ordinal_enc_unknown_value: int = -1,
                 nominal_cols_enc: object | None = None,
                 nominal_enc_handle_unknown: str = "ignore",
                 random_state: int = 0,

                 if_scale: bool = True,
                 scaler_type: str | object = "power_transform",
                 if_scale_cat_not_enc: bool = False,

                 if_impute: bool = True,
                 num_imputation_type: str | object = "median",
                 date_imputation_type: str | object = "most_frequent",
                 if_impute_cat_not_enc: bool = False,
                 if_impute_not_scaled: bool = False


                ################### FEATURE SELECTION
                 ,method='xgboost'
                 ,params=None
                 ,selected_fraction=0.9,


                 ################## MODELS SELECTION
                 time_budget: int = 150,
                 cv_folds: int = 5,



                 ################## ENSEMBLE
                 ensemble_type = 'ensemble'
                 ,voting='soft'
                 ,autogluon_n_folds = 5
                 ,autogluon_n_layers = 3
                 ,final_estimator = None
                 ):

        self.verbose = verbose
        self.estimators = None
        self.model_ = None
        self.y_train = None
        self.X_train = None
        self.models_config = "models.json" if models_config is None else models_config

        self.ensemble_type = ensemble_type
        self.voting = voting
        self.autogluon_n_folds = autogluon_n_folds

        if final_estimator is None and ensemble_type != 'ensemble':
            if task == 'classification':
                self.final_estimator = LogisticRegression(random_state=random_state, penalty='l2', C=0.1)
            else:
                self.final_estimator = LinearRegression(random_state=random_state, penalty='l2', C=0.1)
        else:
            self.final_estimator = final_estimator
        self.autogluon_n_layers = autogluon_n_layers



        self.cols_to_ignore = columns_to_ignore
        self.cols_to_ignore_destiny = cols_to_ignore_destiny
        self.date_use = date_use
        self.drop_uninformative_cols = drop_uninformative_cols
        self.enable_advanced_auto_typing = enable_advanced_auto_typing

        self.na_col_uninformative_threshold = na_col_uninformative_threshold
        self.cat_variability_threshold = cat_variability_threshold
        self.num_variability_threshold = num_variability_threshold

        self.if_encode = if_encode
        self.test_size = test_size
        self.ordinal_cols_enc = ordinal_cols_enc
        self.ordinal_enc_handle_unknown = ordinal_enc_handle_unknown
        self.ordinal_enc_unknown_value = ordinal_enc_unknown_value
        self.nominal_cols_enc = nominal_cols_enc
        self.nominal_enc_handle_unknown = nominal_enc_handle_unknown
        self.random_state = random_state

        self.if_scale = if_scale
        self.scaler_type = scaler_type
        self.if_scale_cat_not_enc = if_scale_cat_not_enc

        self.if_impute = if_impute
        self.num_imputation_type = num_imputation_type
        self.date_imputation_type = date_imputation_type
        self.if_impute_cat_not_enc = if_impute_cat_not_enc
        self.if_impute_not_scaled = if_impute_not_scaled

        self.preprocessor = Preprocessor(
            self.cols_to_ignore, self.cols_to_ignore_destiny, self.date_use, self.drop_uninformative_cols,self.enable_advanced_auto_typing,

        self.na_col_uninformative_threshold, self.cat_variability_threshold, self.num_variability_threshold,

        self.if_encode, self.test_size, self.ordinal_cols_enc, self.ordinal_enc_handle_unknown, self.ordinal_enc_unknown_value,
        self.nominal_cols_enc, self.nominal_enc_handle_unknown,self.random_state,

        self.if_scale, self.scaler_type, self.if_scale_cat_not_enc,

        self.if_impute, self.num_imputation_type, self.date_imputation_type,self.if_impute_cat_not_enc, self.if_impute_not_scaled
        )


        ############################## FEATURE SELECTION
        self.method = method
        default_params = {
            'xgboost': {
                'task': 'classification',
                'importance_type': 'gain',
                'threshold': 'mean'
            },
            'boruta': {
                'task': 'classification',
                'n_estimators': 'auto',
                'perc': 100,
                'max_iter': 100
            },
            'random_forest': {
                'task': 'classification',
                'threshold': 'mean',
                'n_estimators': 100
            }
        }
        self.selected_fraction = selected_fraction

        # Jeśli użytkownik nie podał params, bierzemy defaulty dla wybranej metody
        if params is None:
            self.params = default_params.get(self.method, {'task' : task})
        else:
            self.params = params

        self.selected_columns_ = None
        self.original_columns_ = None

        # Specyficzny fix dla wariancji (tak jak miałeś w kodzie)
        if self.method == 'variance' and 'threshold' not in self.params:
            self.params['threshold'] = 0.0




        ######################################## MODELS SELECTION
        self.time_budget = time_budget
        self.cv_folds = cv_folds
        self.task = task

        self.final_models = []
        self.selector = None
        self.mapper = None
        self.X_train_dict_cached = None
        self.y_train_enc_cached = None


    def __preprocess(self, X_train, y_train, X_train_or_test, if_return_cols_types_dict):
        preprocessor = self.preprocessor
        return preprocessor.fit_transform(X_train, y_train, X_train_or_test,
                                          if_return_cols_types_dict=if_return_cols_types_dict)

########################################### FEATURE SELECTION ##################################################
    def __select_features(self, X_train_dict, y_train_dict, col_types_dict):
        """
        Fits feature selectors to the training data and returns selected feature sets.

        This method iterates through different data configurations (e.g., encoded
        vs. non-encoded), initializes a `FeatureSelector` for each, and performs
        supervised or unsupervised selection. The fitted selectors are stored in
        `self.feature_selectors` to be applied later to test or validation sets.

        Args:
            X_train_dict (dict): A dictionary where keys are configuration names
                and values are pandas DataFrames containing training features.
            y_train_dict (dict): A dictionary containing the target variable.
                Must include the key 'enc' for the encoded target.
            col_types_dict (dict): A dictionary mapping configuration names to
                their respective column type metadata.

        Returns:
            tuple: (X_train_selected_dict, y_train_dict, final_col_types)
                - X_train_selected_dict: Dictionary of DataFrames with reduced features.
                - y_train_dict: The original target dictionary.
                - final_col_types: Updated column type metadata reflecting the selection.

        Raises:
            ValueError: If `y_train_dict` does not contain the required 'enc' key.
        """
        y_train = y_train_dict.get('enc')

        if y_train is None:
            raise ValueError("The y_train_dict must contain the key 'enc'.")


        y_train_flat = y_train.values.ravel() if isinstance(y_train, pd.DataFrame) else y_train
        self.feature_selectors = {}
        X_train_selected_dict = {}

        if self.verbose:
            print("\n--- Starting Feature Selection Fitting (Train) ---", flush=True)

        for config_name, X_train_df in X_train_dict.items():
            if X_train_df is None or X_train_df.empty:
                continue

            if self.verbose:
                print(f"Fitting selector for: {config_name} (Total Features: {X_train_df.shape[1]})", flush=True)

            try:
                selector = FeatureSelector(
                    method=self.method,
                    params=self.params,
                    selected_fraction=self.selected_fraction
                )

                X_for_fit = X_train_df.copy()
                if 'not_enc' in config_name or self.method == 'xgboost':
                    non_num = X_for_fit.select_dtypes(exclude=['number']).columns
                    X_for_fit[non_num] = X_for_fit[non_num].astype('category')

                selector.fit(X_for_fit, y_train_flat)
                self.feature_selectors[config_name] = selector
                X_selected = selector.transform(X_train_df)
                X_train_selected_dict[config_name] = X_selected

                if self.verbose:
                    print(f"  -> Selected {X_selected.shape[1]} features.", flush=True)

            except Exception as e:
                if self.verbose:
                    print(f"  ERROR in configuration {config_name}: {e}", flush=True)
                self.feature_selectors[config_name] = None
                X_train_selected_dict[config_name] = X_train_df

        final_col_types = self._filter_col_types(col_types_dict)
        self.col_types = final_col_types

        return X_train_selected_dict, y_train_dict, final_col_types

    def _filter_col_types(self, col_types_dict: dict) -> dict:
        """
        Updates the column type metadata structure for each configuration.

        This method synchronizes the global column type dictionary with the results
        of the feature selection process. It ensures that only the features that
        passed the selection are retained in the type mapping, preserving the
        underlying metadata structure.

        Args:
            col_types_dict (dict): The original column type mapping, typically in the
                format {type_name: [list_of_columns]}.

        Returns:
            dict: A nested dictionary in the format {config_name: {type_name: [kept_columns]}},
                reflecting the state after feature selection.

        Note:
            This method requires `self.feature_selectors` to be populated (usually
            by running `__select_features` first).
        """
        updated_types_dict = {}

        for config_name, selector in self.feature_selectors.items():
            if selector is None or selector.selected_columns_ is None:
                updated_types_dict[config_name] = col_types_dict  # Fallback do oryginału
                continue

            selected_set = set(selector.selected_columns_)
            config_types = {}

            for type_name, cols_list in col_types_dict.items():
                kept_cols = [col for col in cols_list if col in selected_set]
                config_types[type_name] = kept_cols

            updated_types_dict[config_name] = config_types

            total_kept = sum(len(v) for v in config_types.values())

            if self.verbose:
                print(f"[{config_name}] Metadata updated: {total_kept} variables remaining in col_types.", flush=True)

        self.col_types = updated_types_dict.copy()
        return updated_types_dict

    def __transform_features(self, X_dict: dict) -> dict:
        """
        Applies the previously fitted selectors to transform input datasets.

        This method retrieves the specific `FeatureSelector` for each configuration
        from `self.feature_selectors` and uses it to filter the input DataFrame,
        ensuring that only the features selected during training are present.

        Args:
            X_dict (dict): A dictionary of dataframes where keys are configuration
                names and values are the pandas DataFrames to be transformed.

        Returns:
            dict: A dictionary of transformed DataFrames containing only
                the selected features.

        Note:
            If no selectors are found in the object state, the original data is
            returned with a warning.
        """
        if not hasattr(self, 'feature_selectors') or not self.feature_selectors:
            if self.verbose:
                print("WARNING: No fitted selectors found. Returning original data.", flush=True)
            return X_dict

        X_transformed_dict = {}
        if self.verbose:
            print("\n--- Starting Data Transformation (Feature Selection) ---", flush=True)

        for config_name, df in X_dict.items():
            if df is None or df.empty:
                continue

            selector = self.feature_selectors.get(config_name)

            if selector is not None:
                try:
                    X_new = selector.transform(df)

                    X_transformed_dict[config_name] = X_new

                    diff = df.shape[1] - X_new.shape[1]

                    if self.verbose:
                        print(f"[{config_name}] Dropped {diff} features (Remaining: {X_new.shape[1]}).", flush=True)

                except Exception as e:
                    if self.verbose:
                        print(f"  ERROR during transformation of {config_name}: {e}")
                        print("  -> Returning original data as a fallback.")
                    X_transformed_dict[config_name] = df

            else:
                if self.verbose:
                    print(f"[{config_name}] No selector found (skipping transformation).")
                X_transformed_dict[config_name] = df

        return X_transformed_dict
###############################################    ModelSelector    ########################################################

    def _train_selector_and_refit(self, X_train_dict, y_train_dict):
        """
        Metoda wewnętrzna:
        1. Uruchamia ModelSelectorV3.
        2. Zapisuje i ZWRACA predykcje OOF oraz wybrane modele.
        3. Wykonuje Refit (dotrenowanie) na pełnym zbiorze.

        Returns:
            tuple: (final_models, oof_preds)
        """
        print(f"[MiniAutoML] Krok 3: Szukanie Modeli (Budżet: {self.time_budget}s)...")

        # Cache danych (potrzebny do stackingu)
        self.X_train_dict_cached = X_train_dict
        self.y_train_enc_cached = y_train_dict['enc']

        # Inicjalizacja i start selektora
        self.selector = ModelSelectorV3()
        self.selector.time_budget = self.time_budget
        self.selector.random_state = self.random_state

        # Pobieramy modele i ich predykcje OOF
        # UWAGA: selector.fit zwraca (models, oof_preds)
        self.final_models, self.oof_preds = self.selector.fit(X_train_dict, self.y_train_enc_cached, task=self.task)

        if not self.final_models:
            print("!!! OSTRZEŻENIE: Nie znaleziono żadnego modelu.")
            return [], None

        # Refit (Dotrenowanie na pełnych danych)
        print(f"[MiniAutoML] Krok 4: Refit ({len(self.final_models)} modeli)...")
        self.mapper = DataMapper(self.selector.config)
        y_flat = np.ravel(self.y_train_enc_cached)
        data_frames =[]
        for model in self.final_models:
            m_type = model.__class__.__name__
            X_subset = self.mapper.get_X_for_model(m_type, X_train_dict)

            data_frames.append(X_subset)

            # Parametry specyficzne dla CatBoosta
            fit_params = {}
            if 'catboost' in m_type.lower() and hasattr(X_subset, 'columns'):
                cat_cols = [c for c in X_subset.columns if X_subset[c].dtype.name in ['object', 'category']]
                if cat_cols: fit_params['cat_features'] = cat_cols

            try:
                model.fit(X_subset, y_flat, **fit_params)
            except Exception as e:
                print(f"  Błąd refitu dla {m_type}: {e}")

        # ZWRACAMY WYNIKI
        return self.final_models, self.oof_preds, data_frames

    def _predict_stacking_matrix(self, X_test):
        """
        Wewnętrzna metoda generująca predykcje Level 0.
        """
        if not hasattr(self, 'final_models') or not self.final_models:
            raise RuntimeError("Model nie jest wytrenowany. Najpierw fit()!")

        # 1. Preprocessing & Transformacja
        X_test_dict_raw, _, _ = self.__preprocess(self.X_train, self.y_train, X_test, if_return_cols_types_dict=True)
        X_test_dict = {k: v[1] if len(v) > 1 else v[0] for k, v in X_test_dict_raw.items()}
        X_test_dict = self.__transform_features(X_test_dict)

        if self.selector:
            self.selector._sanitize_catboost_data(X_test_dict)

        # 2. Generowanie macierzy
        preds_matrix = []
        data_frames = []
        for model in self.final_models:
            m_type = model.__class__.__name__
            X_in = self.mapper.get_X_for_model(m_type, X_test_dict)
            data_frames.append(X_in)
            try:
                p = model.predict_proba(X_in)
                # Bierzemy kolumnę 1 (prawdopodobieństwo klasy pozytywnej)
                col = 1 if p.shape[1] == 2 else 0
                preds_matrix.append(p[:, col])
            except Exception:
                # W razie błędu wstawiamy zera
                preds_matrix.append(np.zeros(X_in.shape[0]))

        return np.column_stack(preds_matrix), data_frames

################################################### koniec Selektora #####################################################

    def fit(self, X_train, y_train):

        """
        Standard ``fit`` function following sklearn API

        Parameters
        ----------
        X_train : pd.DataFrame or str
            Feature dataset. A pandas DataFrame (without metadata like frames e.g. from OpenML) containing feature columns.

        y_train : pd.DataFrame or str
            Target dataset. A pandas DataFrame (without metadata like frames e.g. from OpenML) with exactly one column representing the target.
        """

        self.X_train, self.y_train = X_train, y_train
        X_dict, y_dict, col_types_dict = self.__preprocess(X_train, y_train, X_train, if_return_cols_types_dict = True)

        for key, val in X_dict.items():
            X_dict[key] = val[0]
        y_dict['enc'] = y_dict['enc'][0]

        X_dict, y_dict, col_types_dict = self.__select_features(X_dict, y_dict, col_types_dict)

        tmp = self._train_selector_and_refit(X_dict, y_dict)

        estimators, oof_preds, data_frames = tmp
        self.estimators = estimators
        estimators = [(f"{model.__class__.__name__.lower()}_{i}", model) for i, model in enumerate(estimators)]

        if self.ensemble_type == "ensemble":
            self.model_ = Ensemble(
                estimators=estimators,
                task=self.task,
                voting=self.voting
            )

            # modele są już dopasowane
            self.model_.fit(data_frames, y_train, refit=False)

        elif self.ensemble_type == "stacker":
            self.model_ = Stacker(
                estimators=estimators,
                final_estimator=clone(self.final_estimator),
                task=self.task,
            )

            # modele bazowe są już fitted
            self.model_.fit(
                data_frames,
                y_train,
                refit_base=False,
            )


        elif self.ensemble_type == "pseudo_autogluon":

            layers_estimators = [

                [(name, clone(est)) for name, est in estimators]

                for _ in range(self.autogluon_n_layers)

            ]

            self.model_ = PseudoAutoGluon(

                layers_estimators=layers_estimators,

                final_estimator=clone(self.final_estimator),

                task=self.task,

                n_folds=self.autogluon_n_folds,

                random_state=42,

            )

            # POPRAWKA: Powielamy listę data_frames tyle razy, ile jest warstw

            # Zamiast: data_frames = [data_frames[0]] * n_models

            data_frames = data_frames * self.autogluon_n_layers

            self.model_.fit(data_frames, y_train)

            n_models = sum(len(layer) for layer in layers_estimators)
            data_frames = [data_frames[0]] * n_models
            self.model_.fit(data_frames, y_train)

        else:
            raise ValueError(
                f"Invalid ensemble_type: {self.ensemble_type}"
            )

        return self

    def predict(self, X_test):

        """
        Standard ``predict`` function following sklearn API

        Parameters
        ----------
        X_test : pd.DataFrame or str
            Feature dataset. A pandas DataFrame (without metadata like frames e.g. from OpenML) containing feature columns.
        """

        X_dict, y_dict, col_types_dict = self.__preprocess(self.X_train, self.y_train, X_test, if_return_cols_types_dict = True)

        for key, val in X_dict.items():
            X_dict[key] = val[1]

        X_dict = self.__transform_features(X_dict)

        data_frames = []
        for model in self.estimators:
            m_type = model.__class__.__name__
            data_frames.append(self.mapper.get_X_for_model(m_type, X_dict))

        if self.ensemble_type == "pseudo_autogluon":
            n_models = len(self.estimators) * self.autogluon_n_layers
            data_frames = [data_frames[0]] * n_models

        return self.model_.predict(data_frames)

    def predict_proba(self, X_test):

        """
        Standard ``predict_proba`` function following sklearn API

        Parameters
        ----------
        X_test : pd.DataFrame or str
            Feature dataset. A pandas DataFrame (without metadata like frames e.g. from OpenML) containing feature columns.
        """

        X_dict, y_dict, col_types_dict = self.__preprocess(self.X_train, self.y_train, X_test, if_return_cols_types_dict = True)
        # naprawianie rozjebanych typów:
        for key, val in X_dict.items():
            X_dict[key] = val[1]


        X_dict = self.__transform_features(X_dict)

        data_frames = []
        for model in self.estimators:
            m_type = model.__class__.__name__
            data_frames.append(self.mapper.get_X_for_model(m_type, X_dict))

        if self.ensemble_type == "pseudo_autogluon":
            n_models = len(self.estimators) * self.autogluon_n_layers
            data_frames = [data_frames[0]] * n_models
            
        return self.model_.predict_proba(data_frames)
    

    def fit_predict(self, X_train, y_train, X_test):

        """
        Standard ``fit_predict`` function following sklearn API

        Parameters
        ----------
        X_train : pd.DataFrame or str
            Feature dataset. A pandas DataFrame (without metadata like frames e.g. from OpenML) containing feature columns.

        y_train : pd.DataFrame or str
            Target dataset. A pandas DataFrame (without metadata like frames e.g. from OpenML) with exactly one column representing the target.

        X_test : pd.DataFrame or str
            Feature dataset. A pandas DataFrame (without metadata like frames e.g. from OpenML) containing feature columns.
        """

        return self.fit(X_train, y_train).predict(X_test)

    def fit_predict_proba(self, X_train, y_train, X_test):

        """
        Standard ``fit_predict_proba`` function following sklearn API

        Parameters
        ----------
        X_train : pd.DataFrame or str
            Feature dataset. A pandas DataFrame (without metadata like frames e.g. from OpenML) containing feature columns.

        y_train : pd.DataFrame or str
            Target dataset. A pandas DataFrame (without metadata like frames e.g. from OpenML) with exactly one column representing the target.

        X_test : pd.DataFrame or str
            Feature dataset. A pandas DataFrame (without metadata like frames e.g. from OpenML) containing feature columns.
        """

        return self.fit(X_train, y_train).predict_proba(X_test)














