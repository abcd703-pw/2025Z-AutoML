import traceback
import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.preprocessing import MinMaxScaler
from .select_features import FeatureSelector


# ===============================
# Utils
# ===============================

def run_test(name, func):
    print(f"\n🔹 TEST: {name}")
    try:
        func()
        print("✅ RESULT: OK")
    except Exception as e:
        print("❌ RESULT: FAIL")
        print(f"   Error: {e}")
        traceback.print_exc()


def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


# ===============================
# Data loaders
# ===============================

def load_classification():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    return X, y


def load_classification_non_negative():
    X, y = load_classification()
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns
    )
    return X_scaled, y


def load_regression():
    data = load_diabetes()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    return X, y


# ===============================
# Tests
# ===============================

def test_variance():
    X, _ = load_classification()
    fs = FeatureSelector(method="variance", params={"threshold": 0.01})
    X_sel = fs.fit_transform(X)

    assert_true(isinstance(X_sel, pd.DataFrame), "Output is not DataFrame")
    assert_true(0 < X_sel.shape[1] <= X.shape[1], "Invalid number of features")


def test_mutual_info_classification():
    X, y = load_classification()
    fs = FeatureSelector(
        method="mutual_info",
        params={"k": 5, "task": "classification"}
    )
    X_sel = fs.fit_transform(X, y)

    assert_true(X_sel.shape[1] == 5, "Expected exactly 5 features")


def test_mutual_info_regression():
    X, y = load_regression()
    fs = FeatureSelector(
        method="mutual_info",
        params={"k": 5, "task": "regression"}
    )
    X_sel = fs.fit_transform(X, y)

    assert_true(X_sel.shape[1] == 5, "Expected exactly 5 features")


def test_chi2_negative_error():
    X, y = load_classification()

    # Wymuszamy wartości ujemne
    X = X.copy()
    X.iloc[0, 0] = -1.0

    fs = FeatureSelector(method="chi2", params={"k": 5})

    try:
        fs.fit(X, y)
        raise AssertionError("Expected ValueError for negative values")
    except ValueError:
        pass



def test_chi2_negative_error():
    X, y = load_classification()
    fs = FeatureSelector(method="chi2", params={"k": 5})

    try:
        fs.fit(X, y)
        raise AssertionError("Expected ValueError for negative values")
    except ValueError:
        pass


def test_random_forest():
    X, y = load_classification()
    fs = FeatureSelector(
        method="random_forest",
        params={"task": "classification", "threshold": "median"}
    )
    X_sel = fs.fit_transform(X, y)

    assert_true(0 < X_sel.shape[1] < X.shape[1], "RF selection failed")


def test_xgboost():
    X, y = load_classification()
    fs = FeatureSelector(
        method="xgboost",
        params={"task": "classification", "threshold": "mean"}
    )
    X_sel = fs.fit_transform(X, y)

    assert_true(X_sel.shape[1] > 0, "XGBoost selected zero features")


def test_boruta():
    X, y = load_classification()
    fs = FeatureSelector(
        method="boruta",
        params={
            "task": "classification",
            "max_iter": 10,
            "perc": 90
        }
    )
    X_sel = fs.fit_transform(X, y)

    assert_true(X_sel.shape[1] > 0, "Boruta selected zero features")


def test_permutation():
    X, y = load_classification()
    fs = FeatureSelector(
        method="permutation",
        params={
            "task": "classification",
            "n_repeats": 3,
            "threshold": 0.0
        }
    )
    X_sel = fs.fit_transform(X, y)

    assert_true(X_sel.shape[1] > 0, "Permutation importance selected zero features")


def test_all():
    X, y = load_classification()
    fs = FeatureSelector(method="all")
    X_sel = fs.fit_transform(X, y)

    assert_true(X_sel.shape == X.shape, "'all' should return all features")
    assert_true(list(X_sel.columns) == list(X.columns), "Column mismatch")


def test_transform_without_fit():
    X, _ = load_classification()
    fs = FeatureSelector(method="variance")

    try:
        fs.transform(X)
        raise AssertionError("Expected RuntimeError before fit")
    except RuntimeError:
        pass


def test_missing_y_error():
    X, _ = load_classification()

    methods = [
        "mutual_info",
        "chi2",
        "random_forest",
        "xgboost",
        "boruta",
        "permutation"
    ]

    for method in methods:
        fs = FeatureSelector(method=method)
        try:
            fs.fit(X)
            raise AssertionError(f"{method} should require y")
        except ValueError:
            pass


# ===============================
# Runner
# ===============================

if __name__ == "__main__":
    print("\n🚀 FeatureSelector – manual test suite\n")

    tests = [
        test_variance,
        test_mutual_info_classification,
        test_mutual_info_regression,
        test_chi2,
        test_chi2_negative_error,
        test_random_forest,
        test_xgboost,
        test_boruta,
        test_permutation,
        test_all,
        test_transform_without_fit,
        test_missing_y_error,
    ]

    for test in tests:
        run_test(test.__name__, test)

    print("\n✅ ALL TESTS FINISHED\n")
