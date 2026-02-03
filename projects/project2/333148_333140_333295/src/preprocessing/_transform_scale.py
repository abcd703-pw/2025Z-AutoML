import numbers
import warnings

import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import PowerTransformer, StandardScaler, MinMaxScaler



class Scaler:

    def __init__(self, X_dict, y_dict, num_cols, scaler_type, if_scale_cat_not_enc, not_enc_abb, if_classification):
        self.X_dict = X_dict
        self.y_dict = y_dict
        self.num_cols = num_cols
        self.scaler_type = scaler_type

        self.if_scale_cat_not_enc = if_scale_cat_not_enc
        self.not_enc_abb = not_enc_abb
        self.if_classification = if_classification

    def __is_y_numeric(self, s):
        s = s.iloc[:, 0]  # y is treated as df, not series
        return (
                is_numeric_dtype(s) or
                s.apply(lambda x: pd.isna(x) or isinstance(x, numbers.Number)).all()
        )

    def __transform(self, df_dict, df_train, df_test, num_cols, scaled_abb):
        if isinstance(num_cols, str):
            num_cols = [num_cols]

        scaler = None
        if type(self.scaler_type) != str:
            scaler = self.scaler_type
        else:
            match self.scaler_type:
                case 'power_transform':
                    scaler = PowerTransformer(method="yeo-johnson")
                case 'standardize':
                    scaler = StandardScaler()
                case 'normalize':
                    scaler = MinMaxScaler()

        df_train.loc[:, num_cols] = scaler.fit_transform(df_train.loc[:, num_cols])
        if df_test is not None:
            df_test.loc[:, num_cols] = scaler.transform(df_test.loc[:, num_cols])

        for key, X_pair in list(df_dict.items()):
            if (not self.if_scale_cat_not_enc) and (self.not_enc_abb in key):
                continue

            if df_test is not None:
                df_dict[f"{key}_{scaled_abb}"] = [df_train, df_test]
            else:
                df_dict[f"{key}_{scaled_abb}"] = [df_train]
        return df_dict

    def transform_scale(self, scaled_abb):

        # for key, X_pair in list(self.X_dict.items()):
        #     if (not self.if_scale_cat_not_enc) and (self.not_enc_abb in key):
        #         continue
        #     X_concat = pd.concat(X_pair, axis=0)
        #     if type(self.scaler_type) != str:
        #         X_concat[self.num_cols] = self.scaler_type.fit_transform(X_concat[self.num_cols])
        #     else:
        #         match self.scaler_type:
        #             case 'power_transform':
        #                 X_concat[self.num_cols] = PowerTransformer(method="yeo-johnson").fit_transform(X_concat[self.num_cols])
        #             case 'standardize':
        #                 X_concat[self.num_cols] = StandardScaler().fit_transform(X_concat[self.num_cols])
        #             case 'normalize':
        #                 X_concat[self.num_cols] = MinMaxScaler().fit_transform(X_concat[self.num_cols])
        #
        #     X_pair[0][self.num_cols] = X_concat.iloc[:len(X_pair[0])][self.num_cols]
        #     X_pair[1][self.num_cols] = X_concat.iloc[len(X_pair[0]):][self.num_cols]
        #
        #     self.X_dict[f"{key}_{scaled_abb}"] = X_pair


        # in all pairs X_train, X_test num_cols are the same - differences only in cat_cols
        # so better to apply transform on one pair, e.g. with index 0, and change num_cols in every pair
        if any(a != [] for a in self.X_dict.values()):
            X_train, X_test = list(self.X_dict.values())[0]
            self.X_dict = self.__transform(self.X_dict, X_train, X_test, self.num_cols, scaled_abb)

        if (not self.if_classification) and any(a != [] for a in self.y_dict.values()):
            if  all(self.__is_y_numeric(y) for y in list(self.y_dict.values())[0]):
                y_vals = list(self.y_dict.values())[0]

                y_train = y_vals[0]
                y_train[y_train.columns[0]] = y_train[y_train.columns[0]].astype('float64')

                if len(y_vals) == 2:
                    y_test = y_vals[1]
                    y_test[y_test.columns[0]] = y_test[y_test.columns[0]].astype('float64')
                self.y_dict = self.__transform(self.y_dict, y_train, y_test if len(y_vals) == 2 else None,
                                               y_train.columns[0], scaled_abb)
            else:
                warnings.warn("Target column is unsuitable for a regression task.", RuntimeWarning)


        # if type(self.scaler_type) != str:
        #     X_concat[self.num_cols] = self.scaler_type.fit_transform(X_concat[self.num_cols])
        # else:
        #     match self.scaler_type:
        #         case 'power_transform':
        #             X_concat[self.num_cols] = PowerTransformer(method="yeo-johnson").fit_transform(X_concat[self.num_cols])
        #         case 'standardize':
        #             X_concat[self.num_cols] = StandardScaler().fit_transform(X_concat[self.num_cols])
        #         case 'normalize':
        #             X_concat[self.num_cols] = MinMaxScaler().fit_transform(X_concat[self.num_cols])
        #
        # for key, X_pair in list(self.X_dict.items()):
        #     if (not self.if_scale_cat_not_enc) and (self.not_enc_abb in key):
        #         continue
        #     self.X_dict[f"{key}_{scaled_abb}"] = [X_concat.iloc[:len(X_pair[0])][self.num_cols],
        #                                           X_concat.iloc[len(X_pair[0]):][self.num_cols]]



        return self.X_dict, self.y_dict