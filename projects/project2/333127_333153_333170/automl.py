import json
import importlib
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer, OneHotEncoder
from sklearn.metrics import balanced_accuracy_score
from sklearn.ensemble import VotingClassifier

class MiniAutoML:
    def __init__(self, models_config_path):
        with open(models_config_path, 'r') as f:
            self.models_config = json.load(f)
            
        self.final_model = None
        self.final_threshold = 0.5
        self.preprocessor = None
        self.leaderboard = None
        
    def _get_class(self, class_path):
        module_path, class_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    def _build_pipeline(self, X):
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object', 'category', 'bool']).columns
        
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', PowerTransformer(method='yeo-johnson'))
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            verbose_feature_names_out=False
        )
        return self.preprocessor

    def _find_best_threshold(self, model, X_val, y_val):
        try:
            y_proba = model.predict_proba(X_val)[:, 1]
        except:
            return 0.5, balanced_accuracy_score(y_val, model.predict(X_val))

        best_th = 0.5
        best_score = 0
        thresholds = np.arange(0.2, 0.81, 0.05)
        
        for th in thresholds:
            y_pred = (y_proba >= th).astype(int)
            score = balanced_accuracy_score(y_val, y_pred)
            if score > best_score:
                best_score = score
                best_th = th
                
        return best_th, best_score

    def _get_model_family(self, class_path):
        class_path = class_path.lower()
        if 'xgboost' in class_path: return 'xgboost'
        if 'catboost' in class_path: return 'catboost'
        if 'randomforest' in class_path: return 'randomforest'
        if 'gradientboosting' in class_path: return 'gbm_sklearn'
        if 'linear' in class_path or 'logistic' in class_path: return 'linear'
        if 'neighbors' in class_path or 'knn' in class_path: return 'knn'
        if 'svm' in class_path or 'svc' in class_path: return 'svm'
        if 'naive' in class_path: return 'bayes'
        return 'other'

    def fit(self, X_train, y_train, top_k=5, total_time_limit=20, max_rows_limit=20000):
        np.random.seed(42)
        global_start = time.time()
        
        if total_time_limit:
            reserve_time_min = max(1.0, total_time_limit * 0.25)
            if total_time_limit < 5: reserve_time_min = 0.5
            search_budget_sec = (total_time_limit - reserve_time_min) * 60
            print(f"--- TIME BUDGET: {total_time_limit} min (Search: {search_budget_sec/60:.1f} min) ---")
        else:
            search_budget_sec = float('inf')

        print("Preprocessing data...")
        self._build_pipeline(X_train)
        X_processed = self.preprocessor.fit_transform(X_train)

        search_rows = max_rows_limit
        if X_processed.shape[0] > search_rows:
            print(f" -> Subsampling to {search_rows} rows for model selection...")
            indices = np.random.choice(X_processed.shape[0], search_rows, replace=False)
            if hasattr(X_processed, "toarray"):
                X_selection = X_processed[indices]
            else:
                X_selection = X_processed[indices, :]
            
            if isinstance(y_train, (pd.Series, pd.DataFrame)):
                y_selection = y_train.iloc[indices]
            else:
                y_selection = y_train[indices]
        else:
            X_selection = X_processed
            y_selection = y_train

        X_t, X_v, y_t, y_v = train_test_split(
            X_selection, y_selection, test_size=0.2, random_state=42, stratify=y_selection
        )
        
        print(f"Starting model tournament...")
        results = []
        
        for i, config in enumerate(self.models_config):
            if (time.time() - global_start) > search_budget_sec:
                print(f"!!! TIME LIMIT REACHED. Stopping search.")
                break

            try:
                ModelClass = self._get_class(config['class'])
                model = ModelClass(**config['params'])

                model.fit(X_t, y_t)
                best_thresh, score = self._find_best_threshold(model, X_v, y_v)
                family = self._get_model_family(config['class'])

                results.append({
                    'name': config['name'], 'family': family, 'config': config,
                    'model_obj': model, 'score': score, 'threshold': best_thresh
                })
                print(f"Model {i+1}/{len(self.models_config)}: {config['name']} -> BalAcc: {score:.4f}")

            except Exception as e:
                print(f"Model {i+1}/{len(self.models_config)}: {config['name']} -> ERROR: {e}")

        if not results:
            raise ValueError("No models trained successfully within the time limit.")

        results.sort(key=lambda x: x['score'], reverse=True)
        self.leaderboard = pd.DataFrame([{k: v for k, v in r.items() if k not in ['model_obj', 'config']} for r in results])
        
        best_single = results[0]
        print(f"\nBest Single Model: {best_single['name']} (BalAcc: {best_single['score']:.4f})")
        
        ens_score = -1
        if len(results) > 1:
            top_candidates = []
            used_families = set()
            for r in results:
                if len(top_candidates) >= top_k: break
                if r['family'] not in used_families:
                    top_candidates.append(r)
                    used_families.add(r['family'])
            if not top_candidates: top_candidates = results[:top_k]
            
            if len(top_candidates) > 1:
                est = [(c['name'], c['model_obj']) for c in top_candidates]
                ens = VotingClassifier(est, voting='soft')
                ens.fit(X_t, y_t)
                ens_thresh, ens_score = self._find_best_threshold(ens, X_v, y_v)
                print(f"Ensemble score: {ens_score:.4f}")

        if ens_score > best_single['score']:
            print(">>> Selected: ENSEMBLE")
            self.final_threshold = ens_thresh
            
            final_estimators = []
            for cand in top_candidates:
                cfg = cand['config']
                ModelClass = self._get_class(cfg['class'])
                m = ModelClass(**cfg['params'])
                final_estimators.append((cfg['name'], m))
            
            self.final_model = VotingClassifier(final_estimators, voting='soft', n_jobs=1)
        else:
            print(f">>> Selected: SINGLE ({best_single['name']})")
            self.final_threshold = best_single['threshold']
            
            cfg = best_single['config']
            ModelClass = self._get_class(cfg['class'])
            self.final_model = ModelClass(**cfg['params'])
            
        print("Retraining winning model on full data...")
        FINAL_SAFE_LIMIT = 100000
        if total_time_limit and X_processed.shape[0] > FINAL_SAFE_LIMIT:
            idx = np.random.choice(X_processed.shape[0], FINAL_SAFE_LIMIT, replace=False)
            if hasattr(X_processed, "toarray"):
                X_final = X_processed[idx]
            else:
                X_final = X_processed[idx, :]
                
            y_final = y_train.iloc[idx] if isinstance(y_train, (pd.Series, pd.DataFrame)) else y_train[indices]
            self.final_model.fit(X_final, y_final)
        else:
            self.final_model.fit(X_processed, y_train)

        total_time = (time.time() - global_start) / 60
        print(f"Done! Total time: {total_time:.2f} min.")
        return self

    def predict(self, X_test):
        X_test_processed = self.preprocessor.transform(X_test)
        if hasattr(self.final_model, "predict_proba"):
            probas = self.final_model.predict_proba(X_test_processed)[:, 1]
            predictions = (probas >= self.final_threshold).astype(int)
        else:
            predictions = self.final_model.predict(X_test_processed)
        return predictions

    def predict_proba(self, X_test):
        X_test_processed = self.preprocessor.transform(X_test)
        if hasattr(self.final_model, "predict_proba"):
            return self.final_model.predict_proba(X_test_processed)
        else:
            preds = self.final_model.predict(X_test_processed)
            return np.column_stack((1 - preds, preds))