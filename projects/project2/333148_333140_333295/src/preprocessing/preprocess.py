from typing import Union, Any

import pandas as pd

from ._infer_types import Typer
from ._encode import Encoder
from ._transform_scale import Scaler
from ._handle_na import NAHandler

# X_dict, y_dict, d = (Preprocessor(**return_default_params()).fit_transform(X_train, y_train, X_test, if_return_cols_types_dict=True))

class Preprocessor:
    """
    End-to-end preprocessing pipeline for tabular data including typing,
    filtering, encoding, scaling, and imputation.

    This class accepts either in-memory pandas DataFrames or local paths
    to CSV files for both features and target. It supports automatic
    column typing, uninformative column removal, categorical encoding,
    numerical scaling, and missing-value imputation with extensive
    configuration.

    Parameters
    ----------

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
    #     Attributes
    #     ----------
    #     X : pd.DataFrame or str
    #         Input features as provided by the user.
    #
    #     y : pd.DataFrame or str
    #         Target variable as provided by the user.

    def __init__(self, columns_to_ignore: list[str], cols_to_ignore_destiny: str, date_use: bool, drop_uninformative_cols: bool,
                 enable_advanced_auto_typing: bool, na_col_uninformative_threshold: float,
                 cat_variability_threshold: float, num_variability_threshold: float,

                 if_encode: bool, test_size: float, ordinal_cols_enc: object | None, ordinal_enc_handle_unknown: str,
                 ordinal_enc_unknown_value: int, nominal_cols_enc: object | None, nominal_enc_handle_unknown: str, random_state: int,

                 if_scale: bool, scaler_type: str | object, if_scale_cat_not_enc: bool,

                 if_impute: bool, num_imputation_type: str | object, date_imputation_type: str | object,
                 if_impute_cat_not_enc: bool, if_impute_not_scaled: bool):

        # self.X = X_train
        # self.y = y_train
        # self.X_test = X_test
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

    def fit_transform(self, X_train: "pd.DataFrame | str", y_train: "pd.DataFrame | str", X_test: "pd.DataFrame | str | None" = None,
                      if_return_cols_types_dict: bool = False,
                      operations_abbreviations_dict: dict[str, str] =
                   {'raw_data_abb': 'raw', 'cat_abb': 'cat', 'enc_abb': 'enc',
                    'not_enc_abb': 'not_enc', 'scaled_abb': 'scaled', 'without_na_abb': 'without_na'}
                      ) -> Union[
        tuple[dict[str, list[pd.DataFrame]], dict[str, list[pd.DataFrame]],],
        tuple[dict[str, list[pd.DataFrame]], dict[str, list[pd.DataFrame]], dict[str, list[str]],],
    ]:
        """
        Perform data preprocessing and generate multiple versions of feature and target datasets.

        The method applies combinations of preprocessing operations (e.g., categorical handling,
        encoding, scaling, NA removal) and returns the resulting datasets grouped by operation
        abbreviations. Each dataset variant contains both training and test splits.

        Parameters
        ----------
        X_train : pd.DataFrame or str
            Feature dataset. Either:

            - a pandas DataFrame (without metadata like frames e.g. from OpenML) containing feature columns, or

            - a local filesystem path to a CSV file containing the features.

        y_train : pd.DataFrame or str
            Target dataset. Either:

            - a pandas DataFrame (without metadata like frames e.g. from OpenML) with exactly one column representing the target,

            - or a local filesystem path to a CSV file containing the target.

        X_test : pd.DataFrame or str, default=None
            Feature dataset. Either:

            - a pandas DataFrame (without metadata like frames e.g. from OpenML) containing feature columns,

            - or a local filesystem path to a CSV file containing the features,

            - or None

            If equal to None, data splitting is not performed.

        if_return_cols_types_dict : bool, default=False
            Whether to include the column types dictionary (`col_types_dict`) in the returned tuple.

        operations_abbreviations_dict : dict[str, str], default={
            'raw_data_abb': 'raw',
            'cat_abb': 'cat',
            'enc_abb': 'enc',
            'not_enc_abb': 'not_enc',
            'scaled_abb': 'scaled',
            'without_na_abb': 'without_na'
        }
            Mapping of preprocessing operation identifiers to their string abbreviations.
            These abbreviations are used to construct keys in the returned dictionaries
            (e.g., ``"cat_enc_scaled"``).

            The semantics of each abbreviation are:

            - ``raw_data_abb`` : Raw, unprocessed data.
            - ``cat_abb`` : Categorical feature handling.
            - ``enc_abb`` : Encoded features.
            - ``not_enc_abb`` : Features that are not encoded.
            - ``scaled_abb`` : Scaled numerical features.
            - ``without_na_abb`` : Data with missing values removed.

        Returns
        -------
        X_dict : dict[str, list[pd.DataFrame]]
            Dictionary mapping preprocessing operation combinations (constructed from
            `operations_abbreviations_dict` values) to a list of feature datasets.

            When X_test is None each value is a list of length 2:

            - index 0: preprocessed training features (`X_train`)

            - index 1: preprocessed test features (`X_test`)

            Otherwise, each value is a list of length 1 with preprocessed training features (`X_train`)

        y_dict : dict[str, list[pd.DataFrame]]
            Dictionary with the same keys and structure as `X_dict`, containing the
            corresponding target datasets.

            When X_test is None each value is a list of length 2:

            - index 0: training targets (`y_train`)

            - index 1: test targets (`y_test`)

            Otherwise, each value is a list of length 1 with preprocessed training features (`y_train`)

        col_types_dict : dict[str, list[str]], optional
            Dictionary mapping column type identifiers (e.g., ``"bin"``, ``"num"``, ``"cat"``)
            to lists of column names belonging to each type.

            Returned only if `if_return_cols_types_dict=True`.


        Notes
        -----
        The keys of `X_dict` and `y_dict` are dynamically created by combining the values
        of `operations_abbreviations_dict` according to the preprocessing steps applied.
        """

        raw_data_abb = operations_abbreviations_dict['raw_data_abb']
        presplitted = X_test is not None

        if type(X_train) == str:
            X_train = pd.read_csv(X_train)
        if type(y_train) == str:
            y_train = pd.read_csv(y_train)
        if type(X_test) == str:
            X_test = pd.read_csv(X_test)
        if set(self.cols_to_ignore) == set(X_train.columns):
            X_dict = {f'{raw_data_abb}': [X_train, X_test]}
            y_dict = {f'{raw_data_abb}': y_train}
            return (X_dict, y_dict, {}) if if_return_cols_types_dict else (X_dict, y_dict)

        if isinstance(y_train, pd.Series):
            y_train = y_train.to_frame()

        if self.cols_to_ignore_destiny == 'drop':
            X_train.drop(columns=self.cols_to_ignore, inplace=True)
            if presplitted:
                X_test.drop(columns=self.cols_to_ignore, inplace=True)
            self.cols_to_ignore = []


        X, y, col_types_dict = Typer(X_train, y_train, self.cols_to_ignore,
                                     self.date_use, self.drop_uninformative_cols, self.enable_advanced_auto_typing,
                                     self.na_col_uninformative_threshold, self.cat_variability_threshold,
                                     self.num_variability_threshold).infer_types()

        cat_abb = operations_abbreviations_dict['cat_abb']
        enc_abb = operations_abbreviations_dict['enc_abb']
        not_enc_abb = operations_abbreviations_dict['not_enc_abb']
        scaled_abb = operations_abbreviations_dict['scaled_abb']
        without_na_abb = operations_abbreviations_dict['without_na_abb']

        X_dict = {f'{cat_abb}_{enc_abb}': [], f'{cat_abb}_{not_enc_abb}': []}
        y_dict = {}
        if_classification = False
        if self.if_encode:
            X_list_cat_enc, X_list_cat_not_enc, y_list, col_types_dict, if_classification = (
                Encoder(X, y, X_test, self.date_use, col_types_dict, self.test_size, self.ordinal_cols_enc,
                        self.ordinal_enc_handle_unknown, self.ordinal_enc_unknown_value,
                        self.nominal_cols_enc, self.nominal_enc_handle_unknown, self.random_state, presplitted).encode()
            )
            X_dict.update({f'{cat_abb}_{enc_abb}': X_list_cat_enc,
                           f'{cat_abb}_{not_enc_abb}': X_list_cat_not_enc})
            y_dict.update({f'{enc_abb}': y_list})

        num_cols = col_types_dict['num']

        if self.if_scale:
            X_dict, y_dict = Scaler(X_dict, y_dict, num_cols, self.scaler_type, self.if_scale_cat_not_enc,
                                   not_enc_abb, if_classification).transform_scale(scaled_abb)

        date_cols = col_types_dict['date'] if self.date_use else []
        # print('after scaling', X_dict)

        if self.if_impute:
            X_dict = NAHandler(X_dict, self.num_imputation_type, self.date_imputation_type, num_cols, date_cols,
                         self.if_impute_cat_not_enc, self.if_impute_not_scaled, not_enc_abb, scaled_abb).impute(without_na_abb)

        if not (self.if_encode or self.if_scale or self.if_impute):
            X_dict = {f'{raw_data_abb}': X}
            y_dict = {f'{raw_data_abb}': y}

        return (X_dict, y_dict, col_types_dict) if if_return_cols_types_dict else (X_dict, y_dict)


def return_default_params() -> dict[str, Any]:
    """
    Returns the dict with default values of all parameters required by an instance of the Preprocessor class.
    """

    return {"columns_to_ignore": [], "cols_to_ignore_destiny": "drop",
            "date_use": False,
            "drop_uninformative_cols": True,
            "enable_advanced_auto_typing": False,

            "na_col_uninformative_threshold": 0.5, "cat_variability_threshold": 1, "num_variability_threshold": 0.1,

            "if_encode": True, "test_size": 0.2, "ordinal_cols_enc": None,
            "ordinal_enc_handle_unknown": "use_encoded_value", "ordinal_enc_unknown_value": -1,
            "nominal_cols_enc": None, "nominal_enc_handle_unknown": "ignore", "random_state": 0,

            "if_scale": True, "scaler_type": "power_transform", "if_scale_cat_not_enc": False,

            "if_impute": True, "num_imputation_type": "median", "date_imputation_type": "most_frequent",
            "if_impute_cat_not_enc": False, "if_impute_not_scaled": False}
