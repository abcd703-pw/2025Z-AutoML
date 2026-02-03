import warnings

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_integer_dtype, is_string_dtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, TargetEncoder, LabelEncoder
from sklearn.impute import SimpleImputer

class Encoder:

    def __init__(self, X, y, X_test, date_use, cols_types_dict, test_size, ordinal_cols_enc,
                 ordinal_enc_handle_unknown, ordinal_enc_unknown_value, nominal_cols_enc, nominal_enc_handle_unknown,
                 random_state, presplitted):
        self.X = X
        self.y = y
        self.X_test = X_test
        self.date_use = date_use
        self.cols_types_dict = cols_types_dict
        self.test_size = test_size

        self.ordinal_cols_enc = ordinal_cols_enc
        self.ordinal_enc_handle_unknown = ordinal_enc_handle_unknown
        self.ordinal_enc_unknown_value = ordinal_enc_unknown_value
        self.nominal_cols_enc = nominal_cols_enc
        self.nominal_enc_handle_unknown = nominal_enc_handle_unknown
        self.random_state = random_state
        self.presplitted = presplitted

    def __cos_transform(self, value, period):
        return np.cos(2 * np.pi * value / period)

    def __sin_transform(self, value, period):
        return np.sin(2 * np.pi * value / period)

    def __is_y_bool_or_int(self, s):
        s = s.iloc[:, 0] # y is treated as df, not series
        return (is_bool_dtype(s) or is_integer_dtype(s) or (is_string_dtype(s) and s.nunique() == 2) or
                s.apply(lambda x: pd.isna(x) or isinstance(x, bool) or isinstance(x, int)).all())

    def encode(self):
        X_train = self.X
        X_test = self.X_test
        y_train = self.y
        y_test = None
        if not self.presplitted:
            X_train, X_test, y_train, y_test = (
                train_test_split(self.X, self.y, stratify=self.y, test_size=self.test_size, random_state=self.random_state))

        X_train_cat_not_enc = X_train.copy()
        X_test_cat_not_enc = X_test.copy()

        X_list_cat_enc = [X_train, X_test]
        X_list_cat_not_enc = [X_train_cat_not_enc, X_test_cat_not_enc]
        X_list = X_list_cat_enc + X_list_cat_not_enc
        y_list = [y_train, y_test] if y_test is not None else [y_train]

        cat_cols_encoding_dtype = 'float64'

        sklearn_encoding_cols = list(set().union(
            *(self.cols_types_dict.get(col_type, []) for col_type in ["bin", "ordinal", 'nominal', 'drop'])))
        missing_values = np.nan
        for X in X_list_cat_enc:
            X[sklearn_encoding_cols] = X[sklearn_encoding_cols].astype(str).where(X[sklearn_encoding_cols].notna(), missing_values)

        imputer = SimpleImputer(missing_values=missing_values, strategy='constant',fill_value=self.ordinal_enc_unknown_value)
        for cols_type, cols in self.cols_types_dict.items():
            match cols_type:
                case "date":
                    for col in cols:
                        for i, X in enumerate(X_list):
                            # timestamp_sec = X[col].map(pd.Timestamp.timestamp)
                            # day = 24 * 60 * 60
                            # month = day * 30.4 # average month length
                            # year = day * 365.25
                            #
                            # X_list[i][f'{col}_year'] = self.__cos_transform(timestamp_sec, year)
                            # X_list[i][f'{col}_month'] = self.__cos_transform(timestamp_sec, month)
                            # X_list[i][f'{col}_day'] = self.__cos_transform(timestamp_sec, day)

                            seconds_in_day = 24 * 60 * 60
                            seconds = (X[col].dt.hour * 3600 + X[col].dt.minute * 60 + X[col].dt.second)
                            X_list[i][f"{col}_day_cos"] = self.__cos_transform(seconds, seconds_in_day)
                            X_list[i][f"{col}_day_sin"] = self.__sin_transform(seconds, seconds_in_day)

                            day_of_month = X[col].dt.day - 1  # zero-based
                            days_in_month = X[col].dt.days_in_month
                            X_list[i][f"{col}_month_cos"] = self.__cos_transform(day_of_month, days_in_month)
                            X_list[i][f"{col}_month_sin"] = self.__sin_transform(day_of_month, days_in_month)

                            day_of_year = X[col].dt.dayofyear - 1  # zero-based
                            days_in_year = X[col].dt.is_leap_year.map({True: 366, False: 365})
                            X_list[i][f"{col}_year_cos"] = self.__cos_transform(day_of_year, days_in_year)
                            X_list[i][f"{col}_year_sin"] = self.__sin_transform(day_of_year, days_in_year)

                            if self.date_use:
                                self.cols_types_dict['date'].remove(col)
                                self.cols_types_dict['date'].extend([f"{col}_day_cos", f"{col}_day_sin",
                                                                f"{col}_month_cos", f"{col}_month_sin",
                                                                f"{col}_year_cos", f"{col}_year_sin"])

                            X_list[i].drop(columns=col, inplace=True)

                case "bin":
                    if cols:
                        oe = OrdinalEncoder(handle_unknown=self.ordinal_enc_handle_unknown,
                                            unknown_value=self.ordinal_enc_unknown_value,
                                            encoded_missing_value=self.ordinal_enc_unknown_value-1)
                        X_list_cat_enc[0][cols] = oe.fit_transform(X_list_cat_enc[0][cols]).astype(cat_cols_encoding_dtype)
                        X_list_cat_enc[1][cols] = oe.transform(X_list_cat_enc[1][cols]).astype(cat_cols_encoding_dtype)

                case "ordinal":
                    if cols:
                        if self.ordinal_cols_enc is None:
                            oe = OrdinalEncoder(handle_unknown=self.ordinal_enc_handle_unknown,
                                                unknown_value=self.ordinal_enc_unknown_value,
                                                encoded_missing_value=self.ordinal_enc_unknown_value-1)
                            X_list_cat_enc[0][cols] = oe.fit_transform(X_list_cat_enc[0][cols]).astype(cat_cols_encoding_dtype)
                            X_list_cat_enc[1][cols] = oe.transform(X_list_cat_enc[1][cols]).astype(cat_cols_encoding_dtype)
                        else:
                            X_list_cat_enc[0][cols] = self.ordinal_cols_enc.fit_transform(X_list_cat_enc[0][cols]).astype(
                                cat_cols_encoding_dtype)
                            X_list_cat_enc[1][cols] = self.ordinal_cols_enc.transform(X_list_cat_enc[1][cols]).astype(
                                cat_cols_encoding_dtype)

                case "nominal":
                    if cols:
                        # for i, X in enumerate(X_list_cat_enc):
                        #     X[cols] = X[cols].astype(str).where(X[cols].notna(), np.nan)
                            # if self.ordinal_cols_enc is None:
                            #     n_unique_categories = X[cols].nunique()
                            #     threshold = np.min(X.shape[0] / self.num_cols_per_nomial_cat_threshold,
                            #                        self.num_cols_per_nomial_cat_threshold * X.shape[1])
                            #     high_cardinality_features = n_unique_categories[n_unique_categories > threshold].index
                            #     low_cardinality_features = n_unique_categories[n_unique_categories <= threshold].index

                            # X[cols] = ((OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit_transform(X[cols]).astype(cat_cols_encoding_dtype))
                            #            if self.ordinal_cols_enc is None
                            #            else self.ordinal_cols_enc.fit_transform(X[cols]).astype(cat_cols_encoding_dtype))


                        if self.nominal_cols_enc is None:
                            ohe = OneHotEncoder(handle_unknown=self.nominal_enc_handle_unknown, sparse_output=False, dtype=cat_cols_encoding_dtype)
                            X_list_cat_enc[0] = X_list_cat_enc[0].drop(columns=cols).join(
                                pd.DataFrame(ohe.fit_transform(X_list_cat_enc[0][cols]),
                                             columns=ohe.get_feature_names_out(cols), index=X_list_cat_enc[0].index))
                            X_list_cat_enc[1] = X_list_cat_enc[1].drop(columns=cols).join(
                                pd.DataFrame(ohe.transform(X_list_cat_enc[1][cols]),
                                             columns=ohe.get_feature_names_out(cols), index=X_list_cat_enc[1].index))


                        else:
                            X_list_cat_enc[0] = X_list_cat_enc[0].drop(columns=cols).join(
                                pd.DataFrame(self.nominal_cols_enc.fit_transform(X_list_cat_enc[0][cols]),
                                             columns=self.nominal_cols_enc.get_feature_names_out(cols), index=X_list_cat_enc[0].index))
                            X_list_cat_enc[1] = X_list_cat_enc[1].drop(columns=cols).join(
                                pd.DataFrame(self.nominal_cols_enc.transform(X_list_cat_enc[1][cols]),
                                             columns=self.nominal_cols_enc.get_feature_names_out(cols), index=X_list_cat_enc[1].index))


                        X_list_cat_enc[1] = X_list_cat_enc[1].reindex(columns=X_list_cat_enc[0].columns, fill_value=0)


                case "drop":
                    if cols:
                        # for i, (X, y) in enumerate(zip(X_list_cat_enc, y_list)):
                        te = TargetEncoder(random_state=self.random_state)
                        X_list_cat_enc[0][cols] = te.fit_transform(X_list_cat_enc[0][cols], y_list[0].values.ravel()).astype(cat_cols_encoding_dtype)
                        X_list_cat_enc[0][cols] = imputer.fit_transform(X_list_cat_enc[0][cols]).astype(cat_cols_encoding_dtype)
                        X_list_cat_enc[1][cols] = te.transform(X_list_cat_enc[1][cols]).astype(cat_cols_encoding_dtype)
                        X_list_cat_enc[1][cols] = imputer.transform(X_list_cat_enc[1][cols]).astype(cat_cols_encoding_dtype)


        if all(self.__is_y_bool_or_int(y) for y in y_list):
            if_classification = True
            le = LabelEncoder().fit(y_list[0].copy().to_numpy().ravel())
            for i, y in enumerate(y_list):
                # y = LabelEncoder().fit_transform(y.copy().to_numpy().ravel()).astype(cat_cols_encoding_dtype)
                y_list[i] = pd.DataFrame(le.transform(y.copy().to_numpy().ravel()), columns=y.columns, index=y.index)
        else:
            if_classification = False
            warnings.warn("Target column is unsuitable for a classification task.", RuntimeWarning)

        for df in X_list:
            float64_cols = df.select_dtypes(include="Float64").columns
            df[float64_cols] = df[float64_cols].astype(cat_cols_encoding_dtype)
            df.where(df.notna(), missing_values, inplace=True)

        return X_list_cat_enc, X_list_cat_not_enc, y_list, self.cols_types_dict, if_classification






