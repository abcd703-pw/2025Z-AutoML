from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import warnings
import importlib

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, IsolationForest
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, ClassifierMixin

class SingleModelRunner:
    """
    Fit and use a single model defined by a config dict.
    """

    def __init__(self, model_config: Dict[str, Any]):
        self.model_config = model_config
        self.name = model_config["name"]
        self.model = None

    @staticmethod
    def _load_class(path: str):
        try:
            module_name, class_name = path.rsplit(".", 1)
            module = importlib.import_module(module_name)
            return getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Could not load class {path}: {e}")

    def fit(self, X, y):
        self.model = self._build_estimator()
        self.model.fit(X, y)
        return self

    def make_estimator(self):
        """Return a fresh, unfitted estimator instance defined by the config."""
        return self._build_estimator()

    def _build_estimator(self):
        class_path = self.model_config["class"]
        params = self.model_config.get("params", {})
        ModelClass = self._load_class(class_path)
        est = ModelClass(**params)
        if hasattr(est, "random_state"):
            est.set_params(random_state=42)
        return est

    def predict(self, X) -> np.ndarray:
        self._check_is_fitted()
        return self.model.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        self._check_is_fitted()
        if not hasattr(self.model, "predict_proba"):
            raise AttributeError(f"Model '{self.name}' does not support predict_proba().")
        return self.model.predict_proba(X)

    def get_model(self):
        return self.model

    def _check_is_fitted(self):
        if self.model is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

class MiniAutoML(BaseEstimator, ClassifierMixin):

    def __init__(self, models_config: list[dict[str, Any]]) -> None:
        self.models_config = models_config
        self.fitted_models = []
        self.score = None
        
        # State variables
        self.scaler = None
        self.selected_features_ = None 
        self.positive_class = None
        self.negative_class = None
        self._imputer_is_fit = False
    

    def __repr__(self) -> str:
        status = "Fitted" if self.fitted_models else "Not Fitted"
        n_models = len(self.models_config)
        best_score = f"{self.score:.4f}" if self.score else "N/A"
        return (
            f"<MiniAutoML(status='{status}', "
            f"candidates={n_models}, "
            f"best_cv_score={best_score})>"
        )
    @staticmethod
    def _is_tree_model(model_class_path: str) -> bool:
        path = model_class_path.lower()
        return any(
            k in path
            for k in (
                "lightgbm",
                "xgboost",
                "randomforest",
                "extratrees",
                "catboost"
            )
        )


    def fit(self, X_train, y_train):

        print("\n========== MiniAutoML: START FIT ==========")

        # 0. Basic Validation
        print("[STEP 0] Input validation")
        if not isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame(X_train)

        if isinstance(y_train, pd.DataFrame):
            y_train = y_train.iloc[:, 0]

        if not isinstance(y_train, pd.Series):
            y_train = pd.Series(np.ravel(y_train))

        print(f"  Samples: {X_train.shape[0]}, Raw features: {X_train.shape[1]}")

        # 1. Handle Missing Targets
        if y_train.isnull().values.any():
            warnings.warn("y_train contains missing values; removing corresponding rows.")
            where_na = y_train.isna()
            X_train = X_train.loc[~where_na].reset_index(drop=True)
            y_train = y_train.loc[~where_na].reset_index(drop=True)

        print(f"[STEP 1] Target cleaned | Samples left: {X_train.shape[0]}")
        print("[STEP 1b] Fitting imputer")
        self._fit_imputer(X_train)
        X_train = self._transform_imputer(X_train)

        # 2. OUTLIER REMOVAL
        print("[STEP 2] Detecting outliers (IsolationForest)")
        X_train = X_train.apply(pd.to_numeric, errors='coerce').fillna(0)

        iso = IsolationForest(contamination=0.02, random_state=42, n_jobs=-1)
        outliers = iso.fit_predict(X_train)
        mask = outliers != -1

        removed = np.sum(~mask)
        if removed > 0:
            print(f"  Removed outliers: {removed}")
            X_train = X_train[mask].reset_index(drop=True)
            y_train = y_train[mask].reset_index(drop=True)

        print(f"  Samples after outlier removal: {X_train.shape[0]}")

        # 3. FEATURE SELECTION
        print("[STEP 3] Feature selection")

        print(f"  Start features: {X_train.shape[1]}")

        # A) VarianceThreshold
        sel_var = VarianceThreshold(threshold=0)
        sel_var.fit(X_train)
        curr_cols = X_train.columns[sel_var.get_support()]
        X_train = X_train[curr_cols]

        print(f"  After VarianceThreshold: {X_train.shape[1]}")

        # B) Correlation filter
        corr_matrix = X_train.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [c for c in upper.columns if any(upper[c] > 0.95)]

        if to_drop:
            print(f"  Dropping correlated features: {len(to_drop)}")
            X_train = X_train.drop(columns=to_drop)

        print(f"  Final selected features: {X_train.shape[1]}")

        self.selected_features_ = X_train.columns.tolist()

        # 4. Scaling
        print("[STEP 4] Scaling features (MinMaxScaler)")
        self.scaler = MinMaxScaler().fit(X_train)

        X_train_scaled = self.scaler.transform(X_train)
        X_train_raw = X_train.values
        

        # 5. Target encoding
        print("[STEP 5] Target encoding")
        y_train = np.ravel(y_train)
        unique_classes = np.unique(y_train)

        if len(unique_classes) == 2:
            self.negative_class = unique_classes[0]
            self.positive_class = unique_classes[1]
        else:
            self.negative_class = unique_classes[0]
            self.positive_class = unique_classes[0]

        print(f"  Classes: {unique_classes}")
        print(f"  Negative: {self.negative_class}, Positive: {self.positive_class}")

        # 6. MODEL TRAINING
        print("[STEP 6] Model training (CV)")
        fitted_models = []
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        print(f"  Candidates: {len(self.models_config)}")

        for model_cfg in self.models_config:
            runner = SingleModelRunner(model_cfg)
            print(f"  -> Training model: {runner.name}")

            current = {'name': runner.name, 'score': 0.0, 'runner': None}
            try:
                estimator = runner.make_estimator()
                is_tree = self._is_tree_model(model_cfg["class"])

                X_used = X_train_raw if is_tree else X_train_scaled

                scores = cross_val_score(
                    estimator,
                    X_used,
                    y_train,
                    cv=cv,
                    scoring="balanced_accuracy",
                    n_jobs=-1
                )

                current['score'] = float(np.mean(scores))
                current['runner'] = runner
                fitted_models.append(current)

                print(f"     CV balanced_accuracy: {current['score']:.4f}")

            except Exception as exc:
                warnings.warn(f"Failed to train model '{runner.name}': {exc}")

        # 7. Select Top Models
        print("[STEP 7] Selecting top models")
        self.fitted_models = sorted(
            fitted_models,
            key=lambda d: d["score"],
            reverse=True
        )[:5]

        if not self.fitted_models:
            raise RuntimeError("Brak wytrenowanych modeli - sprawdź konfigurację.")

        for i, m in enumerate(self.fitted_models, 1):
            print(f"  {i}. {m['name']} | CV score: {m['score']:.4f}")

        # 8. Refit best models
        print("[STEP 8] Refitting top models on full data")
        for d in self.fitted_models:
            is_tree = self._is_tree_model(d['runner'].model_config["class"])
            X_used = X_train_raw if is_tree else X_train_scaled

            d['runner'].fit(X_used, y_train)

        self.score = float(np.mean([m['score'] for m in self.fitted_models]))

        print(f"[STEP 9] Training finished | Ensemble CV score: {self.score:.4f}")
        print("========== MiniAutoML: END FIT ==========\n")

        return self

    def _impute_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        
        self._fit_imputer(X, random_state=42)
        return self._transform_imputer(X)

    def predict_proba(self, X_test):
        if len(self.fitted_models) == 0:
            raise RuntimeError("No fitted models available. Call fit() first.")
            
        # 1. Impute
        X_test = self._transform_imputer(X_test)
        
        # 2. Feature Selection (Apply same mask as in fit)
        if self.selected_features_ is not None:
            missing = set(self.selected_features_) - set(X_test.columns)
            if missing:
                for c in missing:
                    X_test[c] = 0 
            X_test = X_test[self.selected_features_]
            
        # 3. Scale
        X_test_scaled = self.scaler.transform(X_test)
        X_test_raw = X_test.values
        
        # 4. Predict
        # Soft Voting
        all_probs = []
        for m in self.fitted_models:
            model = m['runner'].get_model()
            if hasattr(model, "predict_proba"):
                is_tree = self._is_tree_model(m['runner'].model_config["class"])
                X_used = X_test_raw if is_tree else X_test_scaled
                p = model.predict_proba(X_used)
                all_probs.append(p[:, 1])
            elif hasattr(model, "decision_function"):
                scores = model.decision_function(X_test)
                probs = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
                all_probs.append(probs)
            else:
                raise RuntimeError(f"Model {model} has no usable output")

                
        probs = np.mean(all_probs, axis=0)
        return probs

    def predict(self, X_test):
        probs = self.predict_proba(X_test)
        y_pred = (probs >= 0.5).astype(int)
        
        final_preds = np.empty(y_pred.shape, dtype=type(self.positive_class))
        final_preds[y_pred == 0] = self.negative_class
        final_preds[y_pred == 1] = self.positive_class
        return final_preds

    def _fit_imputer(
        self,
        X: pd.DataFrame,
        *,
        n_estimators: int = 50, 
        random_state: int = 42,
        n_jobs: int = -1,
        min_features_non_missing: int = 1,
    ) -> "MiniAutoML":
        """
        Fit (freeze) imputation artifacts on TRAIN only.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")

        self._impute_random_state = int(random_state)
        self._impute_n_estimators = int(n_estimators)
        self._impute_n_jobs = int(n_jobs)
        self._impute_min_features = int(min_features_non_missing)

        X_train = X.copy(deep=True)

        self._impute_cat_cols = [
            c for c in X_train.columns
            if pd.api.types.is_object_dtype(X_train[c])
            or pd.api.types.is_categorical_dtype(X_train[c])
            or pd.api.types.is_bool_dtype(X_train[c])
        ]
        self._impute_num_cols = [c for c in X_train.columns if c not in self._impute_cat_cols]
        self._impute_columns = list(X_train.columns)

        self._impute_fully_missing_num = [c for c in self._impute_num_cols if X_train[c].isna().all()]
        self._impute_fully_missing_cat = [c for c in self._impute_cat_cols if X_train[c].isna().all()]

        for c in self._impute_fully_missing_num:
            X_train[c] = 0.0
        for c in self._impute_fully_missing_cat:
            X_train[c] = "__MISSING__"

        self._impute_num_imp = None
        self._impute_cat_imp = None

        if self._impute_num_cols:
            self._impute_num_imp = SimpleImputer(strategy="median")
            X_train[self._impute_num_cols] = self._impute_num_imp.fit_transform(X_train[self._impute_num_cols])

        if self._impute_cat_cols:
            self._impute_cat_imp = SimpleImputer(strategy="most_frequent")
            X_train[self._impute_cat_cols] = self._impute_cat_imp.fit_transform(X_train[self._impute_cat_cols])

        self._impute_cat_maps = {}
        self._impute_cat_inv_maps = {}
        self._impute_cat_one_hot_cols = {}

        for c in self._impute_cat_cols:
            s = X_train[c].astype(str)
            uniq = pd.Index(s.unique())
            if "__UNK__" not in uniq:
                uniq = uniq.append(pd.Index(["__UNK__"]))
            mapping = {k: i for i, k in enumerate(uniq)}
            inv = {i: k for k, i in mapping.items()}
            self._impute_cat_maps[c] = mapping
            self._impute_cat_inv_maps[c] = inv
            ordered_categories = [inv[i] for i in range(len(inv))]
            self._impute_cat_one_hot_cols[c] = [f"{c}__{cat}" for cat in ordered_categories]

        X_enc = X_train.copy()
        for c in self._impute_cat_cols:
            m = self._impute_cat_maps[c]
            X_enc[c] = X_train[c].astype(str).map(lambda v: m.get(v, m["__UNK__"])).astype(float)

        self._impute_models = {}
        self._impute_target_is_cat = {}

        rng = np.random.RandomState(self._impute_random_state)

        for target in self._impute_columns:
            y_raw = X[target]
            obs_mask = ~y_raw.isna().to_numpy()

            if obs_mask.sum() == 0: continue

            feature_cols = [c for c in self._impute_columns if c != target]
            if len(feature_cols) < self._impute_min_features: continue

            X_train_feat = X_enc.loc[obs_mask, feature_cols]

            if target in self._impute_cat_cols:
                self._impute_target_is_cat[target] = True
                y_enc = X_enc.loc[obs_mask, target].astype(int).to_numpy()
                if np.unique(y_enc).size < 2: continue

                model = RandomForestClassifier(
                    n_estimators=self._impute_n_estimators,
                    random_state=rng.randint(0, 2**31 - 1),
                    n_jobs=self._impute_n_jobs,
                    class_weight="balanced_subsample",
                )
                model.fit(X_train_feat, y_enc)
                self._impute_models[target] = model

            else:
                self._impute_target_is_cat[target] = False
                y = X_enc.loc[obs_mask, target].astype(float).to_numpy()
                if np.nanstd(y) == 0.0: continue

                model = RandomForestRegressor(
                    n_estimators=self._impute_n_estimators,
                    random_state=rng.randint(0, 2**31 - 1),
                    n_jobs=self._impute_n_jobs,
                )
                model.fit(X_train_feat, y)
                self._impute_models[target] = model

        self._impute_output_columns = list(self._impute_num_cols)
        for c in self._impute_cat_cols:
            self._impute_output_columns.extend(self._impute_cat_one_hot_cols.get(c, []))

        self._imputer_is_fit = True
        return self

    def _transform_imputer(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply frozen imputation artifacts to any dataset (train/valid/test).
        """
        if not getattr(self, "_imputer_is_fit", False):
            raise RuntimeError("Imputer is not fit. Call _fit_imputer(X_train) first.")

        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")

        X0 = X[self._impute_columns].copy(deep=True)
        orig_missing = {c: X0[c].isna().to_numpy() for c in self._impute_columns}

        for c in self._impute_fully_missing_num:
            X0.loc[:, c] = X0[c].astype(float)
            X0.loc[orig_missing[c], c] = 0.0
        for c in self._impute_fully_missing_cat:
            X0.loc[:, c] = X0[c].astype(object)
            X0.loc[orig_missing[c], c] = "__MISSING__"

        if self._impute_num_cols and self._impute_num_imp is not None:
            X0[self._impute_num_cols] = self._impute_num_imp.transform(X0[self._impute_num_cols])

        if self._impute_cat_cols and self._impute_cat_imp is not None:
            X0[self._impute_cat_cols] = self._impute_cat_imp.transform(X0[self._impute_cat_cols])

        X_enc = X0.copy()
        for c in self._impute_cat_cols:
            m = self._impute_cat_maps[c]
            unk = m["__UNK__"]
            X_enc[c] = X0[c].astype(str).map(lambda v: m.get(v, unk)).astype(float)

        for target, model in self._impute_models.items():
            miss = orig_missing[target]
            if not miss.any(): continue

            feature_cols = [c for c in self._impute_columns if c != target]
            X_feat = X_enc.loc[miss, feature_cols]

            if self._impute_target_is_cat.get(target, False):
                pred = model.predict(X_feat).astype(int)
                inv = self._impute_cat_inv_maps[target]
                X0.loc[miss, target] = pd.Series(pred, index=X0.index[miss]).map(inv).to_numpy()
                X_enc.loc[miss, target] = pred.astype(float)
            else:
                pred = model.predict(X_feat).astype(float)
                X0.loc[miss, target] = pred
                X_enc.loc[miss, target] = pred

        num_out = X_enc[self._impute_num_cols].copy()
        for c in self._impute_num_cols:
            if pd.api.types.is_integer_dtype(X[c].dtype):
                vals = num_out[c].to_numpy()
                if np.all(np.isfinite(vals)) and np.all(np.isclose(vals, np.round(vals))):
                    num_out[c] = np.round(num_out[c]).astype(X[c].dtype)

        cat_frames = []
        for c in self._impute_cat_cols:
            inv = self._impute_cat_inv_maps[c]
            codes = X_enc[c].astype(int)
            labels = pd.Series(codes, index=X_enc.index).map(inv)
            dummies = pd.get_dummies(labels, prefix=c, prefix_sep="__")
            expected_cols = self._impute_cat_one_hot_cols.get(c, [])
            dummies = dummies.reindex(columns=expected_cols, fill_value=0)
            cat_frames.append(dummies.astype(float))

        if cat_frames:
            out = pd.concat([num_out] + cat_frames, axis=1)
        else:
            out = num_out

        if hasattr(self, "_impute_output_columns"):
            out = out.reindex(columns=self._impute_output_columns, fill_value=0)

        return out