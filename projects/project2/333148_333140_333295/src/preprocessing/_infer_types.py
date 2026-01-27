from collections.abc import Iterable
import numpy as np
from pandas.api.types import is_string_dtype, is_object_dtype
import pandas as pd
# pd.set_option('future.infer_string', True)

class Typer:

    def __init__(self, X, y, cols_to_ignore, date_use, drop_uninformative_cols, enable_advanced_auto_typing,
                 na_col_uninformative_threshold,cat_variability_threshold, num_variability_threshold):
        self.X = X
        self.y = y
        self.cols_to_ignore = cols_to_ignore
        self.date_use = date_use
        self.drop_uninformative_cols = drop_uninformative_cols
        self.enable_advanced_auto_typing = enable_advanced_auto_typing

        self.na_col_uninformative_threshold = na_col_uninformative_threshold
        self.cat_variability_threshold = cat_variability_threshold
        self.num_variability_threshold = num_variability_threshold

        self.cols_types = None

        self.date_cols = []
        self.cat_bin_cols = []
        self.cat_ordinal_cols = []
        self.cat_nominal_cols = []
        self.cat_cols = None
        self.num_cols = []

        self.drop_cols = []

    def infer_types(self):
        X = self.X.replace([np.inf, -np.inf], np.nan)
        y = self.y.replace([np.inf, -np.inf], np.nan)

        na_inds, _ = np.where(y.isna())
        y.dropna(inplace=True)
        X.drop(index=na_inds, inplace=True)
        self.X = X

        self.cat_bin_cols += [col for col in self.X.columns if self.X[col].nunique() == 2 and col not in self.cols_to_ignore]

        y = y.convert_dtypes()
        df = self.X.convert_dtypes()
        df_copy = df.copy(deep=True)

        n = df.shape[0]
        # df = df.where(df.notna(), np.nan)
        # col_names = list(df.select_dtypes(include='number').columns)
        # df[col_names] = df[col_names].astype(float)

        if self.date_use:
            df_date = (df.drop(columns=self.cols_to_ignore).select_dtypes(include=['datetime', 'datetimetz'])
                               .dropna(axis=1,thresh=self.na_col_uninformative_threshold * n))
            if not df_date.empty:
                self.date_cols.extend(df_date.columns)

        df_int = df.select_dtypes(include='int')
        self.cat_ordinal_cols += [col for col in df_int.columns if df_int[col].nunique() <= self.num_variability_threshold * n
                                  and col not in self.cat_bin_cols and col not in self.cols_to_ignore and
                                  df_int[col].isna().mean() <= self.na_col_uninformative_threshold  * n]

        num_cols_assigned = []
        for col in df.columns:
            first_el = df[col].iloc[0]
            currency_list = ['$', '€', '£', '¥']
            if (col not in self.cols_to_ignore and col not in self.cat_bin_cols and col not in self.cat_ordinal_cols
                and df[col].isna().mean() <= self.na_col_uninformative_threshold * n and isinstance(first_el, Iterable) and
                    (is_string_dtype(df[col]) or (is_object_dtype(df[col]) and type(first_el) is str)) and
                    (any(el in first_el for el in currency_list) or df[col].str.contains(",").any())):
                df[col] = (df[col].replace(currency_list, "").replace(",", "").astype("float"))

                if df[col].nunique() <= self.num_variability_threshold * n:
                    self.cat_ordinal_cols.append(col)
                else:
                    self.num_cols.append(col)
                    num_cols_assigned.append(col)

        df_float = (df.drop(columns=num_cols_assigned+self.cols_to_ignore).select_dtypes(include='float')
                    .dropna(axis=1,thresh=self.na_col_uninformative_threshold * n))
        self.cat_nominal_cols += [col for col in df_float.columns if df_float[col].nunique() <= self.num_variability_threshold * n
                                  and col not in self.cat_bin_cols and col not in self.cat_ordinal_cols]


        self.num_cols += list(set(df_float.columns) - set(self.cat_ordinal_cols) - set(self.cat_nominal_cols) - set(self.date_cols))

        # temp_series = (df.dtypes == "string[python]")
        # string_cols = temp_series[temp_series == True].index.tolist()
        # df[string_cols] = df[string_cols].astype(object)
        df_obj = df.drop(columns=self.cols_to_ignore).select_dtypes(include='object')
        self.cat_nominal_cols += [col for col in df.columns if (col in df_obj.columns or is_string_dtype(df[col])) and
                                  df[col].nunique() <= self.cat_variability_threshold * n and col not in self.cols_to_ignore
                                  and col not in self.cat_bin_cols and col not in self.cat_ordinal_cols
                                  and df[col].isna().mean() <= self.na_col_uninformative_threshold * n]

        self.cat_cols = np.concatenate([self.cat_bin_cols, self.cat_ordinal_cols, self.cat_nominal_cols]).flatten()

        self.drop_cols += list(set(df.columns) - set(self.cat_cols) - set(self.num_cols) -
                               set(self.date_cols) - set(self.cols_to_ignore))

        # print("bin", self.cat_bin_cols)
        # print("ord", self.cat_ordinal_cols)
        # print("nom", self.cat_nominal_cols)
        # print("cat", self.cat_cols)
        # print("num", self.num_cols)
        # print("drop_cols", self.drop_cols)

        self.cols_types = {
            "bin": self.cat_bin_cols,
            "ordinal": self.cat_ordinal_cols,
            "nominal": self.cat_nominal_cols,
            "num": self.num_cols
        }

        if self.date_use:
            self.cols_types["date"] =  self.date_cols
        if not self.drop_uninformative_cols:
            self.cols_types["drop"] = self.drop_cols
        if self.cols_to_ignore:
            self.cols_types["ignored"] = self.cols_to_ignore

        self.X = df_copy.drop(columns=self.drop_cols) if self.drop_uninformative_cols else df_copy
        self.y = y

        # more advanced options possible
        if self.enable_advanced_auto_typing:
            self.X = self.__auto_type_advanced(self.X)


        return self.X, self.y, self.cols_types
















































    def __auto_type_advanced(self, X):
        df = X.copy()

        for cols in [self.num_cols, self.date_cols]:
            X_pair = X[cols].copy()
            X_concat = pd.concat([X, X_pair], axis=1)
            X_pair[0][cols] = X_concat.iloc[:len(X_pair[0])][cols]
            X_pair[1][cols] = X_concat.iloc[len(X_pair[0]):][cols]

        X_test_dict_raw, _, _ = self.__preprocess(self.X_train, self.y_train, X,
                                                  if_return_cols_types_dict=True)
        X_test_dict = {k: v[1] if len(v) > 1 else v[0] for k, v in X_test_dict_raw.items()}
        X_test_dict = self.__transform_features(X_test_dict)


        preds_matrix = []
        data_frames = []
        for model in self.final_models:
            m_type = model.__class__.__name__
            X_in = self.mapper.get_X_for_model(m_type, X_test_dict)
            data_frames.append(X_in)
            try:
                p = model.predict_proba(X_in)
                preds_matrix.append(p[:, self.col])
                df = data_frames
            except Exception:
                preds_matrix.append(np.zeros(X_in.shape[0]))

        return df





