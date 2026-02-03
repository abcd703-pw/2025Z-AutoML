from collections.abc import Iterable

import numpy as np
from pandas.api.types import is_string_dtype, is_object_dtype
import pandas as pd
# pd.set_option('future.infer_string', True)
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder
from sklearn.impute import SimpleImputer

class Typer:

    def __init__(self, X, y, cols_to_ignore, date_use, drop_uninformative_cols, enable_advanced_auto_typing,
                 na_col_uninformative_threshold,cat_variability_threshold, num_variability_threshold, random_state):
        self.X = X
        self.y = y
        self.cols_to_ignore = cols_to_ignore
        self.date_use = date_use
        self.drop_uninformative_cols = drop_uninformative_cols
        self.enable_advanced_auto_typing = enable_advanced_auto_typing

        self.na_col_uninformative_threshold = na_col_uninformative_threshold
        self.cat_variability_threshold = cat_variability_threshold
        self.num_variability_threshold = num_variability_threshold
        self.random_state = random_state

        self.cols_types = None

        self.date_cols = []
        self.cat_bin_cols = []
        self.cat_ordinal_cols = []
        self.cat_nominal_cols = []
        self.cat_cols = None
        self.num_cols = []

        self.drop_cols = []

    def infer_types(self):
        X = self.X
        y = self.y

        na_inds, _ = np.where(y.isna())
        y.dropna(inplace=True)
        X.drop(index=na_inds, inplace=True)
        self.X = X

        self.cat_bin_cols += [col for col in self.X.columns if self.X[col].nunique() == 2 and col not in self.cols_to_ignore
                              and self.X[col].isna().mean() <= self.na_col_uninformative_threshold  * self.X.shape[0]]

        y = y.convert_dtypes()
        df = self.X.convert_dtypes()
        df_copy = df.copy(deep=True)

        n = df.shape[0]
        self.encoding_dtype = 'float64'
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

        memory_threshold = 1000
        df_float = (df.drop(columns=num_cols_assigned+self.cols_to_ignore).select_dtypes(include='float')
                    .dropna(axis=1,thresh=self.na_col_uninformative_threshold * n))
        self.cat_nominal_cols += [col for col in df_float.columns if df_float[col].nunique() <= self.num_variability_threshold * n
                                  and col not in self.cat_bin_cols and col not in self.cat_ordinal_cols and col not in self.cols_to_ignore
                                  and df_float[col].isna().mean() <= self.na_col_uninformative_threshold * n and
                                  df_float[col].nunique() <= min(memory_threshold, 1/10 * df_float.shape[0])]


        self.num_cols += list(set(df_float.columns) - set(self.cat_ordinal_cols) - set(self.cat_nominal_cols) - set(self.date_cols))

        # temp_series = (df.dtypes == "string[python]")
        # string_cols = temp_series[temp_series == True].index.tolist()
        # df[string_cols] = df[string_cols].astype(object)
        df_obj = df.drop(columns=self.cols_to_ignore).select_dtypes(include='object')
        self.cat_nominal_cols += [col for col in df.columns if (col in df_obj.columns or is_string_dtype(df[col])) and
                                  df[col].nunique() <= self.cat_variability_threshold * n and col not in self.cols_to_ignore
                                  and col not in self.cat_bin_cols and col not in self.cat_ordinal_cols
                                  and df[col].isna().mean() <= self.na_col_uninformative_threshold * n and
                                  df[col].nunique() <= min(memory_threshold, 1/10 * df.shape[0])]

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
            self.__auto_type_advanced(self.X, self.y, subsample = 100000, random_state= self.random_state)

        return self.X, self.y, self.cols_types



   # Advanced auto-typing

   # ginis

    def __gini(self, y_true, y_pred):
        y_true = y_true[np.argsort(y_pred)]
        n = len(y_true)
        return (np.cumsum(y_true).sum() / y_true.sum() - (n + 1) / 2) / n if y_true.sum() != 0 else 0.0

    def __gini_normalized_helper(self, y_true, y_pred):
        denom = self.__gini(y_true, y_true)
        return self.__gini(y_true, y_pred) / denom if denom != 0 else 0.0

    def __gini_normalized(self, y_true, x_col):
        if isinstance(y_true, pd.DataFrame):
            y_true = y_true.iloc[:, 0].values

        elif isinstance(y_true, pd.Series):
            y_true = y_true.values

        mask = np.isfinite(x_col)
        if mask.sum() == 0:
            return 0.0

        yt = y_true[mask]
        xp = x_col[mask]

        return abs(self.__gini_normalized_helper(yt, xp))

    def __column_gini_scores(self, X, y):
        scores = np.zeros(X.shape[1])
        for i in range(X.shape[1]):
            scores[i] = self.__gini_normalized(y, X.iloc[:, i])
        return scores

    # null scores

    def __get_null_scores(self, X, y, cols_to_check, subsample,random_state):

        if cols_to_check is not None:
            X = X[cols_to_check]

        n_samples = len(X)
        if subsample is not None and n_samples > subsample:
            idx = np.random.RandomState(random_state).choice(n_samples, subsample, replace=False)
            X = X.iloc[idx]
            y_sub = y.iloc[idx].to_numpy()
        else:
            y_sub = y.to_numpy()

        X_np = X.to_numpy(dtype=self.encoding_dtype)
        scores = np.zeros(X_np.shape[1], dtype=self.encoding_dtype)

        for i in range(X_np.shape[1]):
            col_mask = (~np.isfinite(X_np))[:, i]
            if col_mask.all() or (~col_mask).all():
                scores[i] = 0.0
            else:
                scores[i] = self.__gini_normalized(y_sub, col_mask.astype(self.encoding_dtype))

        return pd.Series(scores, index=X.columns, name="max_score")

    # num cols

    def __numeric_feature_stats(self, df, y, num_cols, subsample, random_state):
        if not num_cols:
            return pd.DataFrame()

        X = df[num_cols]#.to_numpy(dtype=self.encoding_dtype)
        # n_samples = len(X)
        # rng = np.random.RandomState(random_state)
        # if subsample is not None and n_samples > subsample:
        #     idx = rng.choice(n_samples, subsample, replace=False)
        #     X = X[idx]
        #     y_sub = y[idx]
        # else:
        #     y_sub = y

        y_sub = y
        raw_scores = self.__column_gini_scores(X, y)

        X = SimpleImputer(missing_values=np.nan, strategy='median').fit_transform(X)
        X_binned = pd.DataFrame(KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="quantile").fit_transform(X))
        binned_scores = self.__column_gini_scores(X_binned, y)

        X_encoded = np.zeros_like(X)
        X_freq = np.zeros_like(X)
        for i in range(X.shape[1]):
            col = X[:, i]
            mask = np.isfinite(col)

            le = LabelEncoder()
            col_enc = np.full_like(col, np.nan)
            col_enc[mask] = le.fit_transform(col[mask].astype(str))
            X_encoded[:, i] = col_enc

            counts = pd.Series(col[mask]).value_counts()
            col_freq = np.full_like(col, np.nan)
            col_freq[mask] = pd.Series(col[mask]).map(counts).to_numpy()
            X_freq[:, i] = col_freq

        encoded_scores = self.__column_gini_scores(pd.DataFrame(X_encoded), y_sub)
        freq_scores = self.__column_gini_scores(pd.DataFrame(X_freq), y_sub)

        unique_counts = np.array([len(np.unique(X[:, i][np.isfinite(X[:, i])])) for i in range(X.shape[1])])
        # no negative elements allowed in numpy bincount, sugestions:
        # values, counts = np.unique(a, return_counts=True)
        #
        # ind = np.argmax(counts)
        # print(values[ind])  # prints the most frequent element
        #
        # ind = np.argpartition(-counts, kth=10)[:10]
        # print(values[ind])  # prints the 10 most frequent elements
        top_freqs = np.array([np.max(np.bincount(X[:, i][np.isfinite(X[:, i])].astype(int))) for i in range(X.shape[1])])
        unique_rate = unique_counts / X.shape[0]
        nan_rate = (~np.isfinite(X)).mean(axis=0)

        stats = pd.DataFrame({ "flg_manual": [False] * len(num_cols), "unique": unique_counts, "unique_rate": unique_rate,
            "top_freq_values": top_freqs, "raw_scores": raw_scores, "binned_scores": binned_scores, "encoded_scores": encoded_scores,
            "freq_scores": freq_scores, "nan_rate": nan_rate}, index=num_cols)

        scores_stat = stats[["raw_scores", "binned_scores", "encoded_scores", "freq_scores"]].values
        top_encodings = scores_stat.argsort(axis=1)[:, ::-1]
        sorted_scores = np.take_along_axis(scores_stat, top_encodings, axis=1)
        stats["max_to_3rd_rate"] = sorted_scores[:, 0] / sorted_scores[:, 2]
        stats["max_to_2rd_rate"] = sorted_scores[:, 0] / sorted_scores[:, 1]
        stats["max_score"] = scores_stat.max(axis=1)
        stats["max_score_rate"] = stats["max_score"] / stats["max_score"].max()

        # Rules mentioned in paper
        stats["rule_0"] = top_encodings[:, 0] == 0 # best score is raw score
        stats["rule_1"] = top_encodings[:, :1].sum(axis=1) == 1 # top 1 score is binned, top 2 score is raw
        stats["rule_2"] = stats["unique"] <= 2
        stats["rule_3"] = stats["unique_rate"] > self.cat_variability_threshold # too many unique values
        stats["rule_4"] = stats["max_to_3rd_rate"] < 1.1 # encoding type have no impact
        stats["rule_5"] = (top_encodings[:, 0] == 1) & (stats["max_to_3rd_rate"] > 2) # binning encode wins with high rate
        stats["rule_6"] = (top_encodings[:, 1] == 0) & (stats["max_to_2rd_rate"] < 1.1) # raw encoding looses to top with very small rate
        stats["rule_7"] = (stats["max_score_rate"] < 0.2) | (stats["max_score"] < 0.05) # uninformative col
        stats["rule_8"] = stats["flg_manual"]

        stats["discrete_rule"] = (~stats["rule_7"]) & ((stats["binned_scores"] / stats["raw_scores"]) > 2)

        return stats

    # def __numeric_roles_from_stats(self, stats):
    #     roles_dict = {}
    #     for col, row in stats.iterrows():
    #         roles_dict[col] = {
    #             "dtype": self.encoding_dtype,
    #             "discretization": bool(row["discrete_rule"])
    #         }
    #     return roles_dict

    def __rule_based_num_handler(self, stats):

        numbers = stats[stats[[x for x in stats.columns if "rule_" in x]].any(axis=1)].copy()
        categories = stats.drop(numbers.index)

        roles_dict = {}

        numbers["discrete_rule"] = (~numbers["rule_7"]) & ((numbers["binned_scores"] / numbers["raw_scores"]) > 2)

        categories["int_rule"] = categories["unique"] < 10
        categories["freq_rule"] = (categories["freq_scores"] / categories["encoded_scores"]) > 1.3
        categories["ord_rule"] = categories["unique_rate"] > 0.03

        # discrete num cols
        for f in numbers[numbers["discrete_rule"]].index:
            roles_dict[f] = {"dtype": self.encoding_dtype, "discretization": True}

        # other num cols
        for f in numbers[~numbers["discrete_rule"]].index:
            roles_dict[f] = {"dtype": self.encoding_dtype, "discretization": False}


        # low cardinality cats
        feats = categories[categories["int_rule"]].index
        ordinal = categories["ord_rule"][categories["int_rule"]].values
        for f, ord_flag in zip(feats, ordinal):
            roles_dict[f] = {"dtype": self.encoding_dtype, "encoding_type": "int", "ordinal": bool(ord_flag)}

        # freq enc features
        ordinal = categories["ord_rule"][categories["freq_rule"]].values
        for f, ord_flag in zip(categories[categories["freq_rule"]].index, ordinal):
            roles_dict[f] = {"dtype": self.encoding_dtype, "encoding_type": "freq", "ordinal": bool(ord_flag)}

        # other cats
        mask = (~categories["freq_rule"]) & (~categories["int_rule"])
        ordinal = categories["ord_rule"][mask].values
        for f, ord_flag in zip(categories[mask].index, ordinal):
            roles_dict[f] = {"dtype": self.encoding_dtype, "encoding_type": "auto", "ordinal": bool(ord_flag)}

        return roles_dict


    # cat cols

    def __categorical_feature_stats(self, df, cat_cols):
        if not cat_cols:
            return pd.DataFrame()

        rows = []
        n_rows = len(df)
        for col in cat_cols:
            vc = df[col].value_counts(dropna=False)
            unique = vc.size
            top_freq = vc.iloc[0]
            freq_ratio = top_freq / n_rows

            rows.append({ "feature": col, "dtype": str(df[col].dtype), "freq_rule_dominate": freq_ratio > 0.8,
                "freq_rule_low_unique": unique <= 5, "auto_rule_high_unique": unique > 50, "auto_rule_balanced": (vc / n_rows).std() < 0.1,
                "ord_rule_ordered": unique <= 10 and vc.sort_values(ascending=False).diff().abs().mean() < 0.05,
            })


        cat_stats =  pd.DataFrame(rows).set_index("feature")

        # scores_stat = cat_stats[["encoded_scores", "freq_scores", "ord_scores"]].values #mismatch of names
        # top_encodings = scores_stat.argsort(axis=1)[:, ::-1]
        # cat_stats["max_score"] = scores_stat.max(axis=1)
        #
        # cat_stats["ord_rule_1"] = top_encodings[:, 0] == 2 # top 1 is ordinal
        # cat_stats["ord_rule_2"] = cat_stats["unique"] <= 2
        # cat_stats["freq_rule_1"] = top_encodings[:, 0] == 1
        # cat_stats["auto_rule_1"] = top_encodings[:, 0] == 0

        return cat_stats

    def __rule_based_cat_handler(self, stats):
        roles_dict = {}
        freqs = stats[stats[[x for x in stats.columns if "freq_rule_" in x]].any(axis=1)]
        autos = stats[stats[[x for x in stats.columns if "auto_rule_" in x]].any(axis=1)]
        ordinals = stats[stats[[x for x in stats.columns if "ord_rule_" in x]].any(axis=1)]

        for enc_type, st in zip(["freq", "auto", "ord"], [freqs, autos, ordinals]):
            ordinal = (enc_type == "ord")
            if enc_type == "ord":
                enc_type = "auto"
            for col in st.index:
                roles_dict[col] = {"dtype": self.encoding_dtype, "encoding_type": enc_type, "ordinal": ordinal}
        return roles_dict


    # main function

    def __auto_type_advanced(self, df, y, subsample, random_state):
        numeric_cols = self.cols_types.get("num", [])
        cat_cols = self.cols_types.get("bin", []) + self.cols_types.get("ordinal", []) + self.cols_types.get("nominal", [])

        if numeric_cols:
            num_stats = self.__numeric_feature_stats(df, y, numeric_cols, subsample=subsample, random_state=random_state)
            numeric_roles = self.__rule_based_num_handler(num_stats)
        else:
            numeric_roles = {}

        if cat_cols:
            cat_stats = self.__categorical_feature_stats(df, cat_cols)
            categorical_roles = self.__rule_based_cat_handler(cat_stats)
        else:
            categorical_roles = {}


        all_roles = {**numeric_roles, **categorical_roles}
        self.cols_types["num"] = []
        self.cols_types["bin"] = []
        self.cols_types["ordinal"] = []
        self.cols_types["nominal"] = []

        for col, role in all_roles.items():
            enc_type = role.get("encoding_type", None)
            discret = role.get("discretization", False)
            # ordinal = role.get("ordinal", False)

            if discret:
                self.cols_types["num"].append(col)
            elif enc_type == "int":
                if df[col].nunique(dropna=True) == 2:
                    self.cols_types["bin"].append(col)
                else:
                    self.cols_types["ordinal"].append(col)
            elif enc_type == "freq":
                self.cols_types["nominal"].append(col)
            elif enc_type == "auto":
                self.cols_types["ordinal"].append(col)
            else:
                self.cols_types["num"].append(col)

        if not self.drop_uninformative_cols:
            drop_cols = [c for c in df.columns if c not in (all_roles or self.date_cols or self.cols_to_ignore)]

            if numeric_cols:
                null_scores = self.__get_null_scores(df, y, numeric_cols, subsample=subsample, random_state=random_state)
                drop_cols.extend(null_scores[null_scores <= 0.01].index.tolist())

            self.cols_types["drop"] = drop_cols

        return all_roles










