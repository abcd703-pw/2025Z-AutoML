"""Main script to run MiniAutoML for binary classification tasks."""

import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score
from utils import load_data, load_models_config
from automl import MiniAutoML
from config import SUBSET_FRACTION

if __name__ == '__main__':

    X_train, X_test, y_train, y_test = load_data()
    print(f"Data loaded. Training set shape: {X_train.shape}, Test set shape: {X_test.shape}\n")

    models_config = load_models_config()
    
    print("--- Running default AutoML ---")
    automl = MiniAutoML(models_config=models_config, time_limit=600, subset_fraction=SUBSET_FRACTION)
    automl.fit(X_train, y_train)
    predictions = automl.predict(X_test)
    probabilities = automl.predict_proba(X_test)
    
    results_df = pd.DataFrame({
        'y_true': y_test,
        'y_pred': predictions,
        'y_proba': probabilities
    })
    
    results_df.to_csv('results/predictions.csv', index=False)
    print("Predictions saved to 'results/predictions.csv'")
    
    print("\n--- Evaluation ---")
    print(f"Best model: {automl.top_models[0]['name']}")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, predictions):.4f}")
    #print(f"ROC AUC Score: {roc_auc_score(y_test, probabilities):.4f}")
    print("--------------------------------------\n")