import json
import importlib
import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import random
from yaml import warnings
os.environ['PYTHONWARNINGS'] = 'ignore'


class MiniAutoML:
    def __init__(self, models_config, time_limit=None):
        """
        models_config: lista konfiguracji z JSON.
        time_limit: maksymalny czas w minutach (opcjonalnie).
        """
        self.models_config = models_config
        self.time_limit = time_limit 
        self.fitted_ensemble = []
        self.model_weights = []
        self.preprocessor = None

    def _get_model_instance(self, config):
        # mądre importowanie bibliotek dla modeli z models.json
        try:
            module_name, class_name = config['class'].rsplit('.', 1)
            module = importlib.import_module(module_name)
            return getattr(module, class_name)(**config['params'])
        except: return None

    def _create_preprocessor(self, X):
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object', 'category']).columns
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler())
        ])
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        return ColumnTransformer(transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    def fit(self, X_train, y_train):
            start_time = time.time()
            limit_seconds = self.time_limit * 60 if self.time_limit else float('inf')
            buffer = 90  # Czas na trening finałowy

            # 1. Preprocessing
            self.preprocessor = self._create_preprocessor(X_train)
            X_processed = self.preprocessor.fit_transform(X_train)
            
            # 2. Strategia CV
            cv_strategy = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
            shuffled_configs = list(self.models_config)
            random.seed(42)
            random.shuffle(shuffled_configs)
            
            results = []
            print(f"Rozpoczynam Smart Ranking. Limit: {self.time_limit} min.")
            
            for i, config in enumerate(shuffled_configs):
                # Bezpiecznik czasowy
                elapsed = time.time() - start_time
                if elapsed > (limit_seconds - buffer):
                    print(f"\n[!] Osiągnięto limit czasu ({elapsed/60:.2f} min). Kończę ranking.")
                    break

                family = config.get('model_name', '').split('_')[0]
                
                # --- SMART RANKING: Przycinanie parametrów do szybkich testów ---
                test_params = config['params'].copy()
                if 'iterations' in test_params: test_params['iterations'] = min(test_params['iterations'], 200)
                if 'n_estimators' in test_params: test_params['n_estimators'] = min(test_params['n_estimators'], 250)
                if 'max_iter' in test_params: test_params['max_iter'] = min(test_params['max_iter'], 200)
                
                try:
                    module_name, class_name = config['class'].rsplit('.', 1)
                    module = importlib.import_module(module_name)
                    model_to_test = getattr(module, class_name)(**test_params)
                    
                    scores = cross_val_score(model_to_test, X_processed, y_train, 
                                            cv=cv_strategy, scoring='balanced_accuracy', n_jobs=-1)
                    
                    mean_s = np.mean(scores)
                    results.append({'score': mean_s, 'config': config, 'family': family})
                    
                    print(f"[{i+1}/{len(self.models_config)}] {config['model_name']}: {mean_s:.4f} (Elapsed: {elapsed/60:.1f}m)")
                except Exception:
                    continue

            if not results:
                raise RuntimeError("Brak wyników - sprawdź limity czasowe lub konfigurację!")

            # Selekcja różnorodna
            results_df = pd.DataFrame(results).sort_values(by='score', ascending=False)
            selected_models_configs = []
            families_seen = set()
            
            # Wybieramy najlepszego przedstawiciela z każdej rodziny
            for _, row in results_df.iterrows():
                if row['family'] not in families_seen:
                    selected_models_configs.append(row)
                    families_seen.add(row['family'])
                if len(selected_models_configs) >= 3: break
                
            # Dopełnienie do 5 najlepszymi modelami ogółem, jeśli rodzin było mało
            if len(selected_models_configs) < 3:
                for _, row in results_df.iterrows():
                    if row['config']['model_name'] not in [s['config']['model_name'] for s in selected_models_configs]:
                        selected_models_configs.append(row)
                    if len(selected_models_configs) >= 3: break

            # --- TRENING FINAŁOWY ---
            print("\nTrening finałowego Ensemble (Top 5 na pełnych parametrach):")
            self.fitted_ensemble = []
            final_scores = []
            
            for row in selected_models_configs:
                print(f"- Trenuję: {row['config']['model_name']}...")
                model = self._get_model_instance(row['config'])
                model.fit(X_processed, y_train)
                self.fitted_ensemble.append(model)
                final_scores.append(row['score'])

            # Wagi softmax-like na podstawie wyników CV
            exp_scores = np.exp(np.array(final_scores) * 10)
            self.model_weights = exp_scores / np.sum(exp_scores)
            
            print(f"\nSukces! Proces zakończony w {(time.time() - start_time)/60:.2f} min.")
            return self

    def predict_proba(self, X_test):
        X_processed = self.preprocessor.transform(X_test)
        weighted_probs = np.zeros(X_processed.shape[0])
        for model, weight in zip(self.fitted_ensemble, self.model_weights):
            weighted_probs += model.predict_proba(X_processed)[:, 1] * weight
        return weighted_probs

    def predict(self, X_test):
        return (self.predict_proba(X_test) >= 0.5).astype(int)
    
if __name__ == "__main__":
    with open('models.json', 'r') as f:
         portfolio = json.load(f)
    automl = MiniAutoML(portfolio, time_limit=8)
    
    X = pd.read_csv('X_projekt.csv') 
    y = pd.read_csv('y_projekt.csv').values.ravel()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    automl.fit(X_train, y_train)

    # Wynik
    preds = automl.predict(X_test)
    score = balanced_accuracy_score(y_test, preds)
    print(f"\n--- WYNIK KOŃCOWY ---")
    print(f"Balanced Accuracy: {score:.4f}")