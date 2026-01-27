import os
import time
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from automl import MiniAutoML

def run_auto_ml(data_dir, json_path='models_final.json', total_time_limit=5, max_rows_limit=10000, top_k=5, random_state=42):
    
    print(f"\n{' MINI AUTOML ':^80}")
    print("="*80)

    x_path = os.path.join(data_dir, 'X.csv')
    y_path = os.path.join(data_dir, 'y.csv')
    
    try:
        print(f"[DATA] Loading data from: '{data_dir}'")
        print(f"[CONF] Configuration:     '{json_path}'")
        
        X = pd.read_csv(x_path)
        y = pd.read_csv(y_path)
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]
            
        n_samples, n_features = X.shape
        print(f"[INFO] Dataset loaded:    {n_samples} samples, {n_features} features")
        
        balance = y.value_counts(normalize=True)
        balance_str = " | ".join([f"Class {k}: {v:.1%}" for k, v in balance.items()])
        print(f"[INFO] Class Balance:     {balance_str}")

    except Exception as e:
        print(f"\n[CRITICAL ERROR] Error loading data: {e}")
        return None, None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    print("-" * 80)
    print(f"[TRAIN] Starting search (Time Budget: {total_time_limit} min, Top-K: {top_k})...")
    
    start_time = time.time()
    
    try:
        automl = MiniAutoML(json_path)
        automl.fit(
            X_train, y_train, top_k=top_k, 
            total_time_limit=total_time_limit, 
            max_rows_limit=max_rows_limit,
        )
    except Exception as e:
        print(f"\n[ERROR] AutoML Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

    duration = time.time() - start_time
    print(f"[DONE] Process completed in: {duration:.2f} s")

    if hasattr(automl, 'leaderboard') and automl.leaderboard is not None:
        print("\n" + "="*80)
        print(f"{' INTERNAL LEADERBOARD (TOP 10) ':^80}")
        print("="*80)
        
        cols_to_show = ['name', 'family', 'score', 'threshold']
        actual_cols = [c for c in cols_to_show if c in automl.leaderboard.columns]
        
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        pd.set_option('display.precision', 4)
        print(automl.leaderboard[actual_cols].head(10).to_string(index=False, justify='center'))
        print("-" * 80)

    try:
        y_pred = automl.predict(X_test) 
        y_proba = automl.predict_proba(X_test)

        if y_proba.ndim == 2:
            y_proba = y_proba[:, 1]

        bal_acc = balanced_accuracy_score(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        try:
            auc = roc_auc_score(y_test, y_proba)
        except:
            auc = 0.5 

        print("\n" + "#"*60)
        print(f"{' FINAL TEST SET RESULTS ':^60}")
        print("#"*60)
        print(f"{'Metric':<30} | {'Value':>20}")
        print("-" * 60)
        print(f"{'Balanced Accuracy':<30} | {bal_acc:>20.4f}")
        print(f"{'ROC AUC Score':<30} | {auc:>20.4f}")
        print(f"{'Accuracy':<30} | {acc:>20.4f}")
        print("-" * 60)
        
        model_name = type(automl.final_model).__name__
        print(f"\n[WINNING MODEL DETAILS]")
        print(f" -> Type: {model_name}")
        
        if isinstance(automl.final_model, VotingClassifier):
             n_estimators = len(automl.final_model.estimators)
             print(f" -> Structure: Ensemble of {n_estimators} models")
        
        print(f" -> Cutoff Threshold: {automl.final_threshold:.4f}")
        print("#"*60 + "\n")

        metrics = {
            "balanced_accuracy": bal_acc,
            "accuracy": acc,
            "roc_auc": auc,
            "duration": duration
        }
        return automl, metrics

    except Exception as e:
        print(f"[ERROR] Evaluation failed: {e}")
        return automl, None