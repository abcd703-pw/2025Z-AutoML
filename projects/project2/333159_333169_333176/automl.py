import json
import importlib
import time
import numpy as np
import pandas as pd
from preprocessor import Preprocessor
import os
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold

class MiniAutoML:
    def __init__(self, time_limit=300, random_state=42):
        self.time_limit = time_limit
        self.random_state = random_state
        self.model_portfolio = self._load_model_portfolio()
        self.best_model = None
        self.best_model_name = None
        self.best_score = None
        self.preprocessor = None  
        self.is_fitted = False
        

    # Loading model portfolio
    def _load_model_portfolio(self):
        base_path = os.path.dirname(__file__)
        # Json file is expected to be in the same directory as this script
        file_path = os.path.join(base_path, "models.json")
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading model portfolio from {file_path}: {e}")
            return []
        

    # Instantiating model
    def _instantiate_model(self, model_config):
        try:
            module_path, class_name = model_config["class"].rsplit(".", 1)
            module = importlib.import_module(module_path)
            model_class = getattr(module, class_name)
            params = model_config.get("params", {}).copy()
            model_class_str = model_config["class"].lower()

            # Disable output/warnings LightGBM
            if 'lightgbm' in model_class_str:
                params['verbosity'] = -1
                params['importance_type'] = 'split'

            # Disable output/warnings XGBoost
            if 'xgboost' in model_class_str:
                params['verbosity'] = 0
            
            # Random state if the model does not have it yet
            if "random_state" in model_class.__init__.__code__.co_varnames:
                params["random_state"] = self.random_state
            
            return model_class(**params)
        except Exception:
            return None

    def fit(self, X, y):

        start_time = time.time()
        
        print("---- Preprocessing ---")
        self.preprocessor = Preprocessor()
        X_processed = self.preprocessor.fit_transform(X)

    
        n_samples = len(X_processed)
        # Subsampling if the dataset has more then 10000 rows to speed the screening process
        if n_samples > 10000:
            indices = np.random.RandomState(self.random_state).choice(n_samples, 10000, replace=False)
            X_eval = X_processed.iloc[indices]
            y_eval = y.iloc[indices]
        else:
            # If there are less then 10000 rows nothing happens
            X_eval, y_eval = X_processed, y

        # Screening using cross validation with metric balanced_accuracy
        print(f"--- Screening {len(self.model_portfolio)} models ---")
        screening_cv = KFold(n_splits=3, shuffle=True, random_state=self.random_state)
        screening_results = []
        for config in self.model_portfolio:
            if time.time() - start_time > self.time_limit:
                print("Time limit reached during screening.")
                break
            model = self._instantiate_model(config)
            if model is None: continue
            try:
                scores = cross_val_score(model, X_eval, y_eval, cv=screening_cv, scoring='balanced_accuracy')
                screening_results.append({'config': config, 'score': np.mean(scores)})
            except: continue

        if not screening_results:
            raise RuntimeError("No models were trained within the time limit.")

        screening_results.sort(key=lambda x: x['score'], reverse=True)

        # Choosing the best models
        diverse_results = []
        seen_classes = set()

        for res in screening_results:
            model_class = res['config']['class']
            if model_class not in seen_classes:
                diverse_results.append(res)
                seen_classes.add(model_class)
            if len(diverse_results) >= 3:
                break

        for res in screening_results:
            if res not in diverse_results:
                diverse_results.append(res)
            if len(diverse_results) >= 5:
                break

        top_configs = diverse_results
        print(top_configs)

        candidates = {}

        stack_cv = KFold(n_splits=3, shuffle=True, random_state=self.random_state)

        # Best single
        best_cfg = top_configs[0]['config']
        candidates[best_cfg['name']] = self._instantiate_model(best_cfg)

        def get_remaining_time():
            return self.time_limit - (time.time() - start_time)
        
        best_3 = [(f"m_{i}", self._instantiate_model(top_configs[i]['config'])) for i in range(min(3, len(top_configs)))]
        best_5 = [(f"m_{i}", self._instantiate_model(top_configs[i]['config'])) for i in range(len(top_configs))]

        if get_remaining_time() > (0.3 * self.time_limit):
            print("Time remains: adding Voting...")
            # Ensemble
            # Voting Top 3
            candidates['Voting_Top3'] = VotingClassifier(estimators=best_3, voting='soft', n_jobs=1)

            # Voting Top 5
            candidates['Voting_Top5'] = VotingClassifier(estimators=best_5, voting='soft', n_jobs=1)

        if get_remaining_time() > (0.3* self.time_limit):
            print("Time remains: adding Stacking...")
            # Stacking
            candidates['Stacking_Top3'] = StackingClassifier(
                estimators=best_3,
                final_estimator=LogisticRegression(random_state=self.random_state),
                cv=stack_cv,
                n_jobs=1
            )

            candidates['Stacking_Top3_Passthrough'] = StackingClassifier(
            estimators=best_3,
            final_estimator=LogisticRegression(random_state=self.random_state),
            passthrough=True,
            cv=stack_cv,
            n_jobs=1
            )

            candidates['Stacking_RF'] = StackingClassifier(
            estimators=best_3,
            final_estimator=RandomForestClassifier(n_estimators=100, max_depth=3, random_state=self.random_state),
            cv=stack_cv, 
            n_jobs=1
            )

        # Final evaluation using cross validation
        best_final_score = -1
        print("\n--- Candidates ranking (Balanced Accuracy) ---")
        final_cv = KFold(n_splits=5, shuffle=True, random_state=self.random_state)

        for name, model in candidates.items():
            try:
                score = np.mean(cross_val_score(model, X_eval, y_eval, cv=final_cv, scoring='balanced_accuracy', n_jobs=1))
                print(f"{name}: {score:.4f}")
                
                if score > best_final_score:
                    best_final_score = score
                    self.best_model = model
                    self.best_model_name = name
            except:
                continue

        # Training final best model - full dataset
        print(f"\n>>> Winner: {self.best_model_name}, score (balanced_accuracy): {best_final_score:.4f}")
        self.best_model.fit(X_processed, y)
        self.best_score = best_final_score
        self.is_fitted = True
        
        return self

    def predict(self, X):
        if not self.is_fitted: raise ValueError("Model is not trained!")
        X_p = self.preprocessor.transform(X)
        return self.best_model.predict(X_p)

    def predict_proba(self, X):
        if not self.is_fitted: raise ValueError("Model is not trained!")
        X_p = self.preprocessor.transform(X)
        return self.best_model.predict_proba(X_p)[:, 1]
    