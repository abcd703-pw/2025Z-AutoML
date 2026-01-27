import warnings

import numpy as np
from sklearn.impute import SimpleImputer

class NAHandler:

    def __init__(self, X_dict, num_imputation_type, date_imputation_type, num_cols, date_cols,
                 if_impute_cat_not_enc, if_impute_not_scaled, not_enc_abb, scaled_abb):
        self.X_dict = X_dict

        self.num_imputation_type = num_imputation_type
        self.date_imputation_type = date_imputation_type
        self.num_cols = num_cols
        self.date_cols = date_cols

        self.if_impute_cat_not_enc = if_impute_cat_not_enc
        self.if_impute_not_scaled = if_impute_not_scaled
        self.not_enc_abb = not_enc_abb
        self.scaled_abb = scaled_abb

    def impute(self, without_na_abb):

        # warning_issued = False
        # for key, X_pair in list(self.X_dict.items()):
        #     if (not self.if_impute_cat_not_enc) and any(abb in key for abb in (self.not_enc_abb, self.scaled_abb)):
        #         continue
        #     X_concat = pd.concat(X_pair, axis=0)
        #     missing_values = pd.NA
        #
        #     # num_cols
        #     if self.num_cols and X_concat[self.num_cols].isnull().values.any():
        #         if type(self.num_imputation_type) != str:
        #             X_concat[self.num_cols] = self.num_imputation_type.fit_transform(X_concat[self.num_cols])
        #         else:
        #             match self.num_imputation_type:
        #                 case 'median':
        #                     X_concat[self.num_cols] = SimpleImputer(missing_values=missing_values, strategy='median'
        #                                                             ).fit_transform(X_concat[self.num_cols])
        #                 case 'mean':
        #                     X_concat[self.num_cols] = SimpleImputer(missing_values=missing_values, strategy='mean'
        #                                                             ).fit_transform(X_concat[self.num_cols])
        #
        #     # date_cols
        #     if self.date_cols and X_concat[self.date_cols].isnull().values.any():
        #         if not warning_issued:
        #             warnings.warn("There are missing values in columns with type date", RuntimeWarning)
        #
        #         X_concat[self.date_cols] = (self.date_imputation_type.fit_transform(X_concat[self.date_cols])
        #                                                     if type(self.date_imputation_type) != str else
        #                                     SimpleImputer(missing_values=missing_values, strategy='most_frequent'
        #                                                   ).fit_transform(X_concat[self.date_cols]))
        #
        #     for cols in [self.num_cols, self.date_cols]:
        #         X_pair[0][cols] = X_concat.iloc[:len(X_pair[0])][cols]
        #         X_pair[1][cols] = X_concat.iloc[len(X_pair[0]):][cols]
        #
        #     self.X_dict[f"{key}_{without_na_abb}"] = X_pair


        # in all pairs X_train, X_test (not scaled) and X_train_scaled, X_test_scaled num_cols are the same - differences only in cat_cols
        # so better to apply transform on one pair, e.g. with index 0, and change num_cols in every pair
        key_scaled = [k for k in self.X_dict if self.scaled_abb in k][0]
        key_unscaled = [k for k in self.X_dict if self.scaled_abb not in k][0] if self.if_impute_not_scaled else None

        X_pairs = [self.X_dict[key_scaled]]
        if key_unscaled is not None:
            X_pairs.append(self.X_dict[key_unscaled])

        warning_issued = False
        for i, (X_train, X_test) in enumerate(X_pairs):
            missing_values = np.nan

            # num_cols
            if self.num_cols and X_train[self.num_cols].isnull().values.any():
                imputer = None
                if type(self.num_imputation_type) != str:
                    imputer = self.num_imputation_type
                else:
                    match self.num_imputation_type:
                        case 'median':
                            imputer = SimpleImputer(missing_values=missing_values, strategy='median')
                        case 'mean':
                            imputer = SimpleImputer(missing_values=missing_values, strategy='mean')

                X_train[self.num_cols] = imputer.fit_transform(X_train[self.num_cols])
                X_test[self.num_cols] = imputer.transform(X_test[self.num_cols])

            # date_cols
            if self.date_cols and X_train[self.date_cols].isnull().values.any():
                if not warning_issued:
                    warnings.warn("There are missing values in columns with type date", RuntimeWarning)
                    warning_issued = True

                imputer = (self.date_imputation_type if type(self.date_imputation_type) != str else
                           SimpleImputer(missing_values=missing_values, strategy='most_frequent'))

                X_train[self.date_cols] = imputer.fit_transform(X_train[self.date_cols])
                X_test[self.date_cols] = imputer.transform(X_test[self.date_cols])

            for key, X_pair in list(self.X_dict.items()):
                if ( ((not self.if_impute_cat_not_enc) and self.not_enc_abb in key) or
                        ((not self.if_impute_not_scaled) and self.scaled_abb not in key) ):
                    continue
                if (self.scaled_abb in key and i == 0) or (self.scaled_abb not in key and i == 1):
                    # BŁĄD BYŁ TUTAJ (wycinanie kolumn):
                    # self.X_dict[f"{key}_{without_na_abb}"] = [X_train[self.num_cols + self.date_cols],
                    #                                           X_test[self.num_cols + self.date_cols]]

                    # POPRAWKA (zapisujemy całe przetworzone ramki danych):
                    self.X_dict[f"{key}_{without_na_abb}"] = [X_train, X_test]

            return self.X_dict






