
import json
import time
import importlib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder


class _IdentityTransformer:
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


class MiniAutoML:
    """
    Mini AutoML for binary classification on tabular data.

    Fits a pre-set portfolio of 50 models to a given binary classification problem; chooses best model(s) and outputs
    their predictions.
    """

    def __init__(self, model_configs, time_limit=1200, use_ensemble=False, random_state=42,
                 max_ensemble_size=5, max_per_family=2, screening_top_k=15):
        self.models = self.load_models_from_json(model_configs)
        self.random_state = random_state

        self.best_model = None
        self.best_model_name = None

        self.use_ensemble = use_ensemble
        self.best_ensemble = []
        self.best_ensemble_names = []

        self.time_limit = time_limit
        self.label_encoder = LabelEncoder()

        self.threshold_ = 0.5  # learned in fit via OOF preds (balanced accuracy)

        self.max_ensemble_size = int(max_ensemble_size)
        self.max_per_family = int(max_per_family)
        self.screening_top_k = int(screening_top_k)

    def load_models_from_json(self, path):
        """Loads model objects from JSON. Keeps the (name, model) pairing."""
        with open(path, "r") as f:
            configs = json.load(f)

        models = []
        for cfg in configs:
            import_path = cfg["class"]
            params = cfg.get("params", {}) or {}
            name = cfg.get("name", import_path)

            module_path, class_name = import_path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            model_class = getattr(module, class_name)

            model = model_class(**params)
            models.append((name, model))

        return models

    def _family_name(self, model):
        """Model-family name used to enforce diversity in ensembles."""
        cls = model.__class__.__name__.lower()
        mod = model.__class__.__module__.lower()

        if "catboost" in mod or "catboost" in cls:
            return "catboost"
        if "lightgbm" in mod or "lgbm" in cls:
            return "lightgbm"
        if "xgboost" in mod or "xgb" in cls:
            return "xgboost"
        if "randomforest" in cls:
            return "random_forest"
        if "extratrees" in cls:
            return "extra_trees"
        if "histgradientboosting" in cls:
            return "hist_gb"
        if "gradientboosting" in cls:
            return "gb"
        if "logisticregression" in cls:
            return "logreg"
        if "svc" in cls:
            return "svc"
        if "kneighbors" in cls or "knn" in cls:
            return "knn"
        if "mlp" in cls:
            return "mlp"
        if "sgd" in cls:
            return "sgd"
        return "other"

    def _is_catboost(self, model):
        mod = model.__class__.__module__.lower()
        cls = model.__class__.__name__.lower()
        return ("catboost" in mod) or ("catboost" in cls)

    def _is_tree_native_no_ohe(self, model):
        """Models that are better without OHE."""
        fam = self._family_name(model)
        return fam in {"lightgbm", "xgboost", "random_forest", "extra_trees", "hist_gb", "gb"}

    def _cat_feature_indices(self, X: pd.DataFrame):
        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        return [X.columns.get_loc(c) for c in cat_cols], cat_cols

    def _get_proba_or_score(self, fitted_pipeline, X):
        """Return a continuous score for ROC-AUC (proba[:,1] preferred; else decision_function)."""
        if hasattr(fitted_pipeline, "predict_proba"):
            proba = fitted_pipeline.predict_proba(X)
            if proba.ndim == 2 and proba.shape[1] >= 2:
                return proba[:, 1]
            # fallback
            return proba.ravel()
        if hasattr(fitted_pipeline, "decision_function"):
            return fitted_pipeline.decision_function(X)
        # last resort: predicted labels
        return fitted_pipeline.predict(X)

    def _build_preprocessor_linear(self):
        """Preprocessing for linear/SVC/kNN/MLP: OHE + scaling."""
        num_pipeline = Pipeline(steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", MinMaxScaler()),
        ])
        cat_pipeline = Pipeline(steps=[
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        bool_pipeline = Pipeline(steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("scale", MinMaxScaler()),
        ])
        return ColumnTransformer(
            transformers=[
                ("num", num_pipeline, make_column_selector(dtype_include=[np.number])),
                ("cat", cat_pipeline, make_column_selector(dtype_include=["object", "category"])),
                ("bool", bool_pipeline, make_column_selector(dtype_include=[bool])),
            ],
            remainder="drop"
        )

    def _build_preprocessor_tree_native(self):
        """Preprocessing for tree models: ordinal encoding for categoricals (no OHE, no scaling)."""
        num_pipeline = Pipeline(steps=[
            ("impute", SimpleImputer(strategy="median")),
        ])
        cat_pipeline = Pipeline(steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ord", OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1
            )),
        ])
        bool_pipeline = Pipeline(steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
        ])
        return ColumnTransformer(
            transformers=[
                ("num", num_pipeline, make_column_selector(dtype_include=[np.number])),
                ("cat", cat_pipeline, make_column_selector(dtype_include=["object", "category"])),
                ("bool", bool_pipeline, make_column_selector(dtype_include=[bool])),
            ],
            remainder="drop"
        )

    def get_model_pipeline(self, model, X_reference=None):
        """
        Build a model + preprocessing pipeline.
        - catboost: returns model wrapped in an empty preprocessing pipeline; expects raw df
        - tree models: ordinal encoding preprocessor (no OHE).
        - Others: OHE + scaling preprocessor.
        """
        model = clone(model)

        # catboost handles categoricals on its own
        # so its just wrapped in an empty pipeline
        if self._is_catboost(model):
            if hasattr(model, "set_params"):
                try:
                    params = model.get_params()
                    if "random_state" in params and params.get("random_state", None) is None:
                        model.set_params(random_state=self.random_state)
                except Exception:
                    pass

            return Pipeline(steps=[
                ("preprocessing", _IdentityTransformer()),
                ("model", model),
            ])

        # tree models - no OHE, ordinal encoding
        if self._is_tree_native_no_ohe(model):
            pre = self._build_preprocessor_tree_native()
        else:
            pre = self._build_preprocessor_linear()

        return Pipeline(steps=[
            ("preprocessing", pre),
            ("model", model)
        ])

    def fit(self, X_train, y_train):
        """
        Select best model (or ensemble) via ROC-AUC and fit on full training set.
        """
        if not isinstance(X_train, pd.DataFrame):
            # require df bc we select columns based on datatypes
            X_train = pd.DataFrame(X_train)

        y_enc = self.label_encoder.fit_transform(np.asarray(y_train))

        start_time = time.time()
        # heuristic: fewer folds when dataset is large
        n_splits_cv = 5 if len(y_enc) < 10000 else 3
        cv = StratifiedKFold(n_splits=n_splits_cv, shuffle=True, random_state=self.random_state)

        # --- Stage 1: quick screening (1 split) ---
        screen_scores = []
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=self.random_state)

        screen_train_idx, screen_val_idx = next(sss.split(X_train, y_enc))
        X_tr0, X_val0 = X_train.iloc[screen_train_idx], X_train.iloc[screen_val_idx]
        y_tr0, y_val0 = y_enc[screen_train_idx], y_enc[screen_val_idx]

        for name, model in self.models:
            if time.time() - start_time > self.time_limit * 0.35:
                break

            try:
                pipe = self.get_model_pipeline(model, X_reference=X_train)
                fitted = clone(pipe)
                if self._is_catboost(model):
                    cat_idx, _ = self._cat_feature_indices(X_tr0)
                    try:
                        fitted.named_steps["model"].set_params(cat_features=cat_idx)
                    except Exception:
                        pass

                fitted.fit(X_tr0, y_tr0)
                scores = self._get_proba_or_score(fitted, X_val0)

                auc = roc_auc_score(y_val0, scores)
                screen_scores.append((auc, name, model))
            except Exception:
                continue

        if not screen_scores:
            raise ValueError("No models could be evaluated during screening.")

        # keep top k models for full CV
        screen_scores.sort(key=lambda x: x[0], reverse=True)
        candidates = screen_scores[:max(1, min(self.screening_top_k, len(screen_scores)))]

        # --- Stage 2: CV evaluation ---
        best_score = -np.inf
        best_name = None
        best_model_obj = None

        # for ensemble: keep (score, name, model)
        best_ensemble = []
        family_counts = {}

        models_evaluated = 0

        for _, name, model in candidates:
            if time.time() - start_time > self.time_limit * 0.85:
                break

            try:
                oof_scores = np.full(shape=(len(y_enc),), fill_value=np.nan, dtype=float)
                fold_aucs = []
                completed_folds = 0

                for train_idx, val_idx in cv.split(X_train, y_enc):
                    if time.time() - start_time > self.time_limit * 0.85:
                        break

                    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                    y_tr, y_val = y_enc[train_idx], y_enc[val_idx]

                    pipe = self.get_model_pipeline(model, X_reference=X_train)
                    fitted = clone(pipe)
                    if self._is_catboost(model):
                        cat_idx, _ = self._cat_feature_indices(X_tr)
                        try:
                            fitted.named_steps["model"].set_params(cat_features=cat_idx)
                        except Exception:
                            pass
                    fitted.fit(X_tr, y_tr)
                    val_scores = self._get_proba_or_score(fitted, X_val)

                    oof_scores[val_idx] = val_scores
                    fold_aucs.append(roc_auc_score(y_val, val_scores))
                    completed_folds += 1

                if completed_folds == 0:
                    continue

                avg_auc = float(np.mean(fold_aucs))
                models_evaluated += 1

                if not self.use_ensemble:
                    if avg_auc > best_score:
                        best_score = avg_auc
                        best_name = name
                        best_model_obj = model
                        best_oof_scores = oof_scores.copy()
                else:
                    fam = self._family_name(model)
                    entry = (avg_auc, name, model, oof_scores)

                    if len(best_ensemble) < self.max_ensemble_size:
                        best_ensemble.append(entry)
                        family_counts[fam] = family_counts.get(fam, 0) + 1

                    else:
                        scores = [e[0] for e in best_ensemble]

                        # global worst
                        global_worst_idx = int(np.argmin(scores))
                        global_worst_score = best_ensemble[global_worst_idx][0]
                        global_worst_fam = self._family_name(best_ensemble[global_worst_idx][2])

                        # indices of models from the same family
                        fam_indices = [
                            i for i, e in enumerate(best_ensemble)
                            if self._family_name(e[2]) == fam
                        ]

                        fam_is_full = family_counts.get(fam, 0) >= self.max_per_family

                        replace_idx = None

                        if fam_is_full:
                            # compare against the worst model within the same family
                            fam_worst_idx = min(fam_indices, key=lambda i: best_ensemble[i][0])
                            if avg_auc > best_ensemble[fam_worst_idx][0]:
                                replace_idx = fam_worst_idx
                        else:
                            # family not full = compare against global worst
                            if avg_auc > global_worst_score:
                                replace_idx = global_worst_idx

                        if replace_idx is not None:
                            old_fam = self._family_name(best_ensemble[replace_idx][2])

                            # update ensemble
                            best_ensemble[replace_idx] = entry

                            # update family counts
                            family_counts[old_fam] = max(0, family_counts.get(old_fam, 1) - 1)
                            family_counts[fam] = family_counts.get(fam, 0) + 1

            except Exception:
                continue

        # --- Fit final model(s) on full data ---
        if not self.use_ensemble:
            if best_model_obj is None:
                # fallback: best from screening
                _, best_name, best_model_obj = screen_scores[0]
                best_oof_scores = None

            self.best_model_name = best_name

            pipe = self.get_model_pipeline(best_model_obj, X_reference=X_train)
            if self._is_catboost(best_model_obj):
                cat_idx, _ = self._cat_feature_indices(X_train)
                try:
                    pipe.named_steps["model"].set_params(cat_features=cat_idx)
                except Exception:
                    pass
            pipe.fit(X_train, y_enc)
            self.best_model = pipe

            # learn threshold from OOF scores if available
            if best_oof_scores is not None and np.isfinite(best_oof_scores).any():
                self.threshold_ = self._best_threshold_balanced_accuracy(y_enc, best_oof_scores)
            else:
                self.threshold_ = 0.5

        else:
            if not best_ensemble:
                # fallback: take top from screening with diversity ignored
                best_ensemble = [(s, n, m, None) for (s, n, m) in screen_scores[:self.max_ensemble_size]]

            self.best_ensemble = []
            self.best_ensemble_names = []
            oof_matrix = []

            for (auc, name, model, oof_scores) in best_ensemble:
                try:
                    pipe = self.get_model_pipeline(model, X_reference=X_train)
                    if self._is_catboost(model):
                        cat_idx, _ = self._cat_feature_indices(X_train)
                        try:
                            pipe.named_steps["model"].set_params(cat_features=cat_idx)
                        except Exception:
                            pass
                    pipe.fit(X_train, y_enc)
                    self.best_ensemble.append(pipe)

                    self.best_ensemble_names.append(name)
                    if oof_scores is not None and np.isfinite(oof_scores).any():
                        oof_matrix.append(oof_scores)
                except Exception:
                    continue

            # learn threshold from averaged OOF scores (if available)
            if oof_matrix:
                avg_oof = np.nanmean(np.vstack(oof_matrix), axis=0)
                self.threshold_ = self._best_threshold_balanced_accuracy(y_enc, avg_oof)
            else:
                self.threshold_ = 0.5

        return self

    def _best_threshold_balanced_accuracy(self, y_true, scores):
        """Choose threshold that maximizes balanced accuracy on provided scores."""
        finite_scores = scores[np.isfinite(scores)]
        if finite_scores.size == 0:
            return 0.5

        qs = np.linspace(0.05, 0.95, 19)
        thresholds = np.quantile(finite_scores, qs)

        best_thr = thresholds[0]
        best_ba = -np.inf
        for thr in thresholds:
            pred = (scores >= thr).astype(int)
            ba = balanced_accuracy_score(y_true, pred)
            if ba > best_ba:
                best_ba = ba
                best_thr = thr
        return float(best_thr)

    def predict_proba(self, X_test):
        """Returns P(class=1) for each row (best model or soft-voting ensemble)."""
        if not isinstance(X_test, pd.DataFrame):
            X_test = pd.DataFrame(X_test)

        if not self.use_ensemble:
            if self.best_model is None:
                raise ValueError("Model not fitted yet, call fit() first.")

            model = self.best_model
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_test)
                return proba[:, 1]
            # decision_function fallback
            if hasattr(model, "decision_function"):
                scores = model.decision_function(X_test)
                # map to (0,1) via logistic (rough)
                return 1.0 / (1.0 + np.exp(-scores))
            # last resort
            return model.predict(X_test).astype(float)

        else:
            if not self.best_ensemble:
                raise ValueError("Model not fitted yet, call fit() first.")

            probas = []
            for model in self.best_ensemble:
                if hasattr(model, "predict_proba"):
                    probas.append(model.predict_proba(X_test)[:, 1])
                elif hasattr(model, "decision_function"):
                    scores = model.decision_function(X_test)
                    probas.append(1.0 / (1.0 + np.exp(-scores)))
                else:
                    probas.append(model.predict(X_test).astype(float))

            return np.mean(np.vstack(probas), axis=0)

    def predict(self, X_test):
        """Returns predicted class labels using learned threshold."""
        probas = self.predict_proba(X_test)
        y_pred_enc = (probas >= self.threshold_).astype(int)
        return self.label_encoder.inverse_transform(y_pred_enc)
