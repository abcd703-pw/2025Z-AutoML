import re
import json
from pathlib import Path
from pathlib import Path
import pandas as pd
import numpy as np
import math
import argparse
import importlib
import pickle
import pickle
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from typing import Any, Dict, List, Tuple
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score


class ModelConfig:
    def __init__(self, name: str, model_class: str, params: Dict[str, Any]):
        self.name = name
        self.model_class = model_class
        self.params = params

    def parse(self) -> Dict[str, Any]:
        try:
            module_path, class_name = self.model_class.rsplit(".", 1)
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
        except (ValueError, ImportError, AttributeError) as e:
            raise ValueError(f"Invalid model_class '{self.model_class}'") from e

        return {
            "name": self.name,
            "model": cls(**self.params),
        }

    def __repr__(self) -> str:
        return (
            f"ModelConfig(name={self.name!r}, model_class={self.model_class!r}, "
            f"params={self.params!r})"
        )

    def __str__(self) -> str:
        try:
            params_str = json.dumps(self.params, ensure_ascii=False)
        except Exception:
            params_str = repr(self.params)
        return f"ModelConfig {self.name}: class={self.model_class}, params={params_str}"


class MiniAutoML:
    def __init__(self, models_config: List[ModelConfig]):
        self.classifiers = {}

        for model_cfg in models_config:
            classifier = model_cfg.parse()
            self.classifiers[classifier["name"]] = classifier["model"]

        self.cross_val_scores = {}
        self.best_model = None
        self.statistics = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame):
        X_train_preprocessed, y_train_preprocessed, self.statistics = preprocess(X_train, y_train)

        for name, model in self.classifiers.items():
            print(f"[MiniAutoML] [Info] Training {name} model")
            cvs = cross_val_score(model, X_train_preprocessed, y_train_preprocessed, cv=5, scoring='balanced_accuracy')
            self.cross_val_scores[name] = cvs.mean()
            print(f"[MiniAutoML] [Info] Trained {name} model with mean cross-validation score: {self.cross_val_scores[name]:.4f}")

        # wybór najlepszych modeli z poszczególnych rodzin
        best_keys = select_best_from_families(self.cross_val_scores)

        # wyświetlenie informacji o wybranych najlepszych modelach przed ensemblingiem
        print("\n[MiniAutoML] [Info] Best models before ensembling:")
        for family, model_name in best_keys.items():
            score = self.cross_val_scores.get(model_name, None)
            print(f"[MiniAutoML] [Info] Selected best {family} model: {model_name} with mean cross-validation score: {score:.4f}")
            

        best_model_name, self.best_model = select_best(self.classifiers, self.cross_val_scores, X_train_preprocessed, y_train_preprocessed)
        print(f"\n[MiniAutoML] [Info] Selected best model before ensembling: {best_model_name} with mean cross-validation score: {self.cross_val_scores[best_model_name]:.4f}\n")
        
        # pierwszy stacking
        base_estimators = [
            (fam, self.classifiers[best_keys[fam]])
            for fam in ["gbm", "rf", "cat", "lr", "knn"]
            if best_keys[fam] is not None
        ]

        # cv wewnątrz StackingClassifier
        cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        stacking_clf = StackingClassifier(
            estimators=base_estimators,
            final_estimator=self.classifiers[best_keys["xgb"]],
            cv=cv_strategy,
            n_jobs=-1
        )

        print("[MiniAutoML] [Info] Training first stacking model with xgb final estimator")
        self.classifiers["stacking_xgb_final"] = stacking_clf
        cvs = cross_val_score(stacking_clf, X_train_preprocessed, y_train_preprocessed, cv=5, scoring='balanced_accuracy')
        self.cross_val_scores["stacking_xgb_final"] = cvs.mean()
        print(f"[MiniAutoML] [Info] Trained first stacking model with mean cross-validation score: {self.cross_val_scores['stacking_xgb_final']:.4f}")

        # drugi stacking
        base_estimators = [
            (fam, self.classifiers[best_keys[fam]])
            for fam in ["gbm", "cat", "xgb", "rf", "knn"]
            if best_keys[fam] is not None
        ]

        # cv wewnątrz StackingClassifier
        cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        stacking_clf = StackingClassifier(
            estimators=base_estimators,
            final_estimator=self.classifiers[best_keys["lr"]],
            cv=cv_strategy,
            n_jobs=-1
        )

        print("[MiniAutoML] [Info] Training second stacking model with lr final estimator")
        self.classifiers["stacking_lr_final"] = stacking_clf
        cvs = cross_val_score(stacking_clf, X_train_preprocessed, y_train_preprocessed, cv=5, scoring='balanced_accuracy')
        self.cross_val_scores["stacking_lr_final"] = cvs.mean()
        print(f"[MiniAutoML] [Info] Trained second stacking model with mean cross-validation score: {self.cross_val_scores['stacking_lr_final']:.4f}")

        # soft voting
        soft_voting_estimators = [
            (fam, self.classifiers[best_keys[fam]])
            for fam in ["gbm", "cat", "rf", "xgb"]
            if best_keys.get(fam) is not None
        ]

        soft_voting_clf = VotingClassifier(
            estimators=soft_voting_estimators,
            voting="soft",
            n_jobs=-1
        )

        print("[MiniAutoML] [Info] Training soft voting model")
        self.classifiers["soft_voting"] = soft_voting_clf
        cvs = cross_val_score(soft_voting_clf, X_train_preprocessed, y_train_preprocessed, cv=5, scoring='balanced_accuracy')
        self.cross_val_scores["soft_voting"] = cvs.mean()
        print(f"[MiniAutoML] [Info] Trained soft voting model with mean cross-validation score: {self.cross_val_scores['soft_voting']:.4f}")

        best_model_name, self.best_model = select_best(self.classifiers, self.cross_val_scores, X_train_preprocessed, y_train_preprocessed)

        print(f"\n[MiniAutoML] [Info] Selected best model after ensembling: {best_model_name} with mean cross-validation score: {self.cross_val_scores[best_model_name]:.4f}")

        return self.best_model
    

    def predict(self, X_test: pd.DataFrame) -> pd.DataFrame:
        if self.best_model is None:
            raise ValueError("The best_model has not been set. Please call fit() before predict().")
        X_test_preprocessed = preprocess_test(X_test, self.statistics)
        return self.best_model.predict(X_test_preprocessed)

    def predict_proba(self, X_test: pd.DataFrame) -> pd.DataFrame:
        if self.best_model is None:
            raise ValueError("The best_model has not been set. Please call fit() before predict().")
        X_test_preprocessed = preprocess_test(X_test, self.statistics)
        return self.best_model.predict_proba(X_test_preprocessed)

    def save(self, file_path: str) -> None:
        """Serialize the MiniAutoML instance to a file."""
        if self.best_model is None:
            raise ValueError("No trained model to save. Please call fit() before save().")
        
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(file_path: str):
        """Load a MiniAutoML instance from a file."""
        with open(file_path, "rb") as f:
            return pickle.load(f)

    def save(self, file_path: str) -> None:
        """Serialize the MiniAutoML instance to a file."""
        if self.best_model is None:
            raise ValueError("No trained model to save. Please call fit() before save().")
        
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(file_path: str):
        """Load a MiniAutoML instance from a file."""
        with open(file_path, "rb") as f:
            return pickle.load(f)


def read_configuration(file_path: str) -> List[ModelConfig]:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(
            "Configuration file must contain a list of model configurations"
        )

    configs: List[ModelConfig] = []
    for entry in data:
        if not isinstance(entry, dict):
            raise ValueError("Each model configuration must be a JSON object")

        name = entry.get("name")
        model_class = entry.get("class")
        params = entry.get("params", {})

        if name is None or model_class is None:
            raise ValueError("Each model config must include 'name' and 'class' fields")

        if params is None:
            params = {}

        configs.append(ModelConfig(name=name, model_class=model_class, params=params))

    return configs


def read_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path, header=0)


def preprocess(X: pd.DataFrame, y: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    data = pd.concat([X, y], axis=1)
    y_name = y.columns[0]
    data.dropna(axis=0, how="all", inplace=True) # drop rows where all values are NaN
    data.dropna(axis=1, how="all", inplace=True) # drop columns where all values are NaN (optymistyczne zalozenie, ze y to nie same NaN?)
    data.dropna(subset=[y_name], inplace=True) # drop rows where target is NaN
    data.dropna(thresh=math.ceil(0.7*X.shape[1]), inplace=True) # drop rows where over 70% of values are NaN (maybe different threshold?, what about such columns?)
    data.reset_index(drop=True, inplace=True)

    # remove "<" or "[ ]" from column names for xgboost
    regex = re.compile(r"(<=|>=|<|>|\[|\])")
    data.columns = [regex.sub("_", str(col)) for col in data.columns]

    X_cleaned = data.drop(columns=[y_name])
    y_cleaned = data[y_name]
    numerical_columns = X_cleaned.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = X_cleaned.columns.drop(numerical_columns).tolist()

    statistics = {}
    statistics["columns_to_keep"] = X_cleaned.columns.tolist()
    statistics["numerical_columns"] = numerical_columns
    statistics["categorical_columns"] = categorical_columns

    label_encoder = LabelEncoder()
    y_final = label_encoder.fit_transform(y_cleaned)

    def preprocess_categorical(X_cleaned: pd.DataFrame, categorical_columns: List[str], statistics: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        # Imputation
        categorical_imputer = SimpleImputer(strategy="most_frequent")
        X_imputed_categorical = categorical_imputer.fit_transform(X_cleaned[categorical_columns])
        X_imputed_categorical = pd.DataFrame(X_imputed_categorical, columns=categorical_columns)
        statistics["categorical_imputer"] = categorical_imputer

        # OneHotEncoding
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first") # TODO: check drop parameter
        X_categorical_encoded = encoder.fit_transform(X_imputed_categorical)
        statistics["encoder"] = encoder

        cat_feature_names = encoder.get_feature_names_out(input_features=X_imputed_categorical.columns)
        X_categorical_encoded = pd.DataFrame(X_categorical_encoded, columns=cat_feature_names,index=X_imputed_categorical.index)

        # remove "<" or "[ ]" from column names for xgboost
        X_categorical_encoded.columns = [regex.sub("_", str(col)) for col in X_categorical_encoded.columns]

        return X_categorical_encoded, statistics

    def preprocess_numerical(X_cleaned: pd.DataFrame, numerical_columns: List[str], statistics: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        # Imputation
        numeric_imputer = SimpleImputer(strategy="median") 
        X_imputed_numerical = numeric_imputer.fit_transform(X_cleaned[numerical_columns])
        statistics["numeric_imputer"] = numeric_imputer

        # Scaling
        scaler = StandardScaler()
        X_scaled_numerical = scaler.fit_transform(X_imputed_numerical)
        statistics["scaler"] = scaler

        return pd.DataFrame(X_scaled_numerical, columns=numerical_columns), statistics

    if len(categorical_columns) == 0 and len(numerical_columns) == 0:
        raise ValueError("No data to preprocess.")
    
    X_parts = []
    
    if categorical_columns:
        X_preprocessed, statistics = preprocess_categorical(X_cleaned, categorical_columns, statistics)
        X_parts.append(pd.DataFrame(X_preprocessed))
    if numerical_columns:
        X_preprocessed, statistics = preprocess_numerical(X_cleaned, numerical_columns, statistics)
        X_parts.append(pd.DataFrame(X_preprocessed))
    
    return pd.concat(X_parts, axis = 1), y_final, statistics

def preprocess_test(X: pd.DataFrame, statistics: Dict[str, Any]) -> pd.DataFrame:
    X = X[[c for c in statistics["columns_to_keep"] if c in X.columns]].copy() # Keep only columns seen during training
    
    for col in statistics["columns_to_keep"]: # Add missing columns with NaN values
        if col not in X.columns:
            X[col] = np.nan

    X = X[statistics["columns_to_keep"]] # Ensure the same column order as during training
    
    # remove "<" or "[ ]" from column names for xgboost
    regex = re.compile(r"(<=|>=|<|>|\[|\])")
    X.columns = [regex.sub("_", str(col)) for col in X.columns]

    X_parts = []

    def transform_numerical(X: pd.DataFrame, statistics: Dict[str, Any]) -> pd.DataFrame:
        num_cols = statistics["numerical_columns"]
        X_imputed = statistics["numeric_imputer"].transform(X[num_cols])
        X_scaled = statistics["scaler"].transform(X_imputed)

        return pd.DataFrame(X_scaled, columns=num_cols)
    
    def transform_categorical(X: pd.DataFrame, statistics: Dict[str, Any]) -> pd.DataFrame:
        cat_cols = statistics["categorical_columns"]
        
        X_imputed = statistics["categorical_imputer"].transform(X[cat_cols])
        X_imputed = pd.DataFrame(X_imputed, columns=cat_cols)
        
        X_encoded = statistics["encoder"].transform(X_imputed)
        cat_feature_names = statistics["encoder"].get_feature_names_out(input_features=cat_cols)
        X_encoded = pd.DataFrame(X_encoded, columns=cat_feature_names)
        # remove "<" or "[ ]" from column names for xgboost
        regex = re.compile(r"(<=|>=|<|>|\[|\])")
        X_encoded.columns = [regex.sub("_", str(col)) for col in X_encoded.columns]
        return X_encoded
    
    if statistics["categorical_columns"]:
        X_cat = transform_categorical(X, statistics)
        X_parts.append(X_cat)

    if statistics["numerical_columns"]:
        X_num = transform_numerical(X, statistics)
        X_parts.append(X_num)

    return pd.concat(X_parts, axis=1)


def select_best_from_families(scores: Dict[str, float]) -> Dict[str, str]:
    families = ["gbm", "cat", "xgb", "rf", "lr", "knn"]
    best_keys = {}

    for fam in families:
        items = {
            k: v for k, v in scores.items()
            if fam in k
        }
        best_keys[fam] = max(items, key=items.get) if items else None
    
    return best_keys


def select_best(classifiers: Dict[str, Any], scores: Dict[str, float], X: pd.DataFrame, y: pd.Series) -> Tuple[str, Any]:
    max_key = max(
    (k for k in scores if "knn" not in k),  # knn nie może być bo ma być predict_proba
    key=scores.get
    )

    best_model = classifiers[max_key]
    best_model.fit(X, y)
    return max_key, best_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mini AutoML System")
    parser.add_argument(
        "--models_config",
        "-m",
        type=str,
        default="models.json",
        help="Path to models configuration file in json format",
    )
    parser.add_argument(
        "--X",
        "-X",
        type=str,
        default="data/X.csv",
        help="Path to feature data in csv format",
    )
    parser.add_argument(
        "--y",
        "-y",
        type=str,
        default="data/y.csv",
        help="Path to target data in csv format",
    )
    args = parser.parse_args()

    models_config = read_configuration(args.models_config)
    X_train = read_data(args.X)
    y_train = read_data(args.y)

    automl = MiniAutoML(models_config)
    automl.fit(X_train, y_train)

    output_model_folder = "models"
    Path(output_model_folder).mkdir(parents=True, exist_ok=True)
    automl.save(f"{output_model_folder}/example_model.pkl")


    output_model_folder = "models"
    Path(output_model_folder).mkdir(parents=True, exist_ok=True)
    automl.save(f"{output_model_folder}/example_model.pkl")
