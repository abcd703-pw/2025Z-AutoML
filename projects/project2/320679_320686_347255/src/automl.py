"""MiniAutoML: A simplified automated machine learning system for binary classification tasks."""

import numpy as np
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from utils import create_model_instance
from config import MAX_ENSEMBLE_SIZE, RANDOM_STATE
import time

class MiniAutoML:
    def __init__(self, models_config,
                 selection_strategy='stability', cv=3, scoring='balanced_accuracy',
                 n_ensemble=2, ensemble_strategy='hard_voting',
                 time_limit=None, stability_threshold=0.28093450968738076, subset_fraction=0.5):
        """Initializes the MiniAutoML system.
        
        Parameters:
        -----------
        models_config : list
            List of model configurations
        selection_strategy : str
            Model selection strategy: 'cross_val', 'stability', 'heuristic'
        cv : int
            Number of cross-validation folds
        scoring : str
            Scoring metric for model selection
        n_ensemble : int
            Number of models to include in ensemble (1-5)
        ensemble_strategy : str
            Ensemble method: 'soft_voting', 'hard_voting', 'stacking'
        time_limit : int or None
            Time limit in seconds for the fitting process
        stability_threshold : float
            Threshold for stability-based selection
        subset_fraction : float
            Fraction of the training data to use for the first stage of model ranking
        """
        self.models_config = models_config

        valid_selection_strategies = ['cross_val', 'stability', 'heuristic']
        if selection_strategy not in valid_selection_strategies:
            raise ValueError(f"selection_strategy must be one of {valid_selection_strategies}")
        self.selection_strategy = selection_strategy

        self.cv = cv
        self.scoring = scoring
        self.stability_threshold = stability_threshold
        self.subsest_fraction = subset_fraction

        if n_ensemble > len(models_config):
            raise ValueError("n_ensemble cannot be greater than available models number.") 
        if not (1 <= n_ensemble <= MAX_ENSEMBLE_SIZE):
            raise ValueError(f"n_ensemble cannot be lower than 1 or greater than {MAX_ENSEMBLE_SIZE}.")  
        self.n_ensemble = n_ensemble

        valid_ensemble_strategies = ['soft_voting', 'hard_voting', 'stacking']
        if ensemble_strategy not in valid_ensemble_strategies:
            raise ValueError(f"ensemble_strategy must be one of {valid_ensemble_strategies}")
        self.ensemble_strategy = ensemble_strategy
        
        self.time_limit = time_limit
        self.final_model = None
        self.preprocessor = None
        self.models_ranking = None
        self.top_models = None

    def fit(self, X_train, y_train):
        """Fits the MiniAutoML model on the training data.
        
        If time_limit is set, the model ranking phase will stop early when the limit
        is reached, using only the models evaluated so far.
        """
        start_time = time.time()
        
        self.preprocessor = self._create_preprocessor(X_train)
        self.models_ranking = self._rank_models(X_train, y_train, start_time)

        # Check if we have enough models for the requested ensemble size
        if len(self.models_ranking) < self.n_ensemble:
            print(f"Warning: Only {len(self.models_ranking)} models evaluated due to time limit.")
            print(f"Reducing ensemble size from {self.n_ensemble} to {len(self.models_ranking)}.")
            actual_n_ensemble = len(self.models_ranking)
        else:
            actual_n_ensemble = self.n_ensemble

        # If no models were evaluated (e.g., due to a very strict time limit),
        # we cannot proceed to build an ensemble or single model.
        if actual_n_ensemble == 0:
            raise RuntimeError(
                "No models were evaluated before the time limit was reached; "
                "MiniAutoML cannot be fitted. Consider increasing 'time_limit' "
                "or reducing the number/complexity of candidate models."
            )

        self.top_models = self.models_ranking[:actual_n_ensemble]
        
        if actual_n_ensemble > 1:
            ensemble = self._create_ensemble(self.top_models)
            self.final_model = self._create_model_pipeline(self.preprocessor, ensemble)
        else:
            self.final_model = self._create_model_pipeline(
                self.preprocessor,
                create_model_instance(self.top_models[0]['config'])
            )

        self.final_model.fit(X_train, y_train)
        
        return self.final_model

    def predict(self, X_test):
        """Make predictions using the fitted MiniAutoML model."""
        if self.final_model is None:
            raise RuntimeError("The model has not been fitted yet. Please call 'fit' before 'predict'.")
        return self.final_model.predict(X_test)

    def predict_proba(self, X_test):
        """Make probability predictions for the positive class (class 1) using the fitted MiniAutoML model."""
        if self.final_model is None:
            raise RuntimeError("The model has not been fitted yet. Please call 'fit' before 'predict_proba'.")

        if not hasattr(self.final_model, 'predict_proba'):
            return None
        
        try:
            return self._predict_proba(X_test)[:, 1]
        except AttributeError:
            return None

    def _predict_proba(self, X_test):
        """Make probability predictions using the fitted MiniAutoML model."""
        if self.final_model is None:
            raise RuntimeError("The model has not been fitted yet. Please call 'fit' before 'predict_proba'.")
        return self.final_model.predict_proba(X_test)

    def _create_preprocessor(self, X):
        """Creates a preprocessing pipeline for the data."""
        # TODO: Maybe use AutoGluon: https://auto.gluon.ai/0.4.1/tutorials/tabular_prediction/tabular-feature-engineering.html#feature-engineering-example

        categorical_features = X.select_dtypes(include=['object', 'category']).columns
        numerical_features = X.select_dtypes(include=np.number).columns

        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='passthrough',
            verbose_feature_names_out=False
        ).set_output(transform="pandas")

        return preprocessor

    def _create_model_pipeline(self, preprocessor, model):
        """Creates a full pipeline with preprocessing and model."""
        return Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])

    def _rank_models(self, X_train, y_train, start_time):
        """Ranks models based on the selected selection method.
        Stage 1: evaluate all candidates on a subset of the training data to get a fast
        approximation. Stage 2: re-rank the best 2*n_ensemble models on the full data.
        """

        try:
            X_subset, _, y_subset, _ = train_test_split(
                X_train,
                y_train,
                train_size=self.subsest_fraction,
                stratify=y_train,
                random_state=RANDOM_STATE
            )
        except ValueError:
            X_subset, _, y_subset, _ = train_test_split(
                X_train,
                y_train,
                train_size=self.subsest_fraction,
                random_state=RANDOM_STATE
            )

        if self.selection_strategy == 'cross_val':
            first_stage = self._rank_models_cv(X_subset, y_subset, start_time)
        elif self.selection_strategy == 'stability':
            first_stage = self._rank_models_stability(X_subset, y_subset, start_time)
        elif self.selection_strategy == 'heuristic':
            first_stage = self._rank_models_heuristic(X_subset, y_subset, start_time)
        else:
            raise ValueError("Selection method not recognized.")
    
        if not first_stage:
            return first_stage

        # Second stage: refine top candidates on full training data
        refine_k = min(len(first_stage), max(2 * self.n_ensemble, 1))
        top_candidates = [m['config'] for m in first_stage[:refine_k]]

        if self.selection_strategy == 'cross_val':
            second_stage = self._rank_models_cv(X_train, y_train, start_time, candidate_models=top_candidates)
        elif self.selection_strategy == 'stability':
            second_stage = self._rank_models_stability(X_train, y_train, start_time, candidate_models=top_candidates)
        elif self.selection_strategy == 'heuristic':
            second_stage = self._rank_models_heuristic(X_train, y_train, start_time, candidate_models=top_candidates)
        else:
            raise ValueError("Selection method not recognized.")

        if not second_stage:
            return first_stage

        # Merge: start from first-stage ranking, update re-evaluated candidates with new scores
        first_stage_map = {m['name']: dict(m) for m in first_stage}
        for updated in second_stage:
            first_stage_map[updated['name']] = dict(updated)

        merged = list(first_stage_map.values())
        merged.sort(key=lambda x: x['mean_score'], reverse=True)
        return merged

    def _rank_models_cv(self, X_train, y_train, start_time, candidate_models=None):
        """Ranks models based on cross-validation scores."""
        models_ranking = []

        models_to_evaluate = candidate_models if candidate_models is not None else self.models_config

        for i, model_info in enumerate(models_to_evaluate):
            if self.time_limit:
                elapsed = time.time() - start_time
                if elapsed > self.time_limit:
                    print(f"Time limit reached after evaluating {i}/{len(models_to_evaluate)} models.")
                    print(f"Stopping model ranking early. Using {len(models_ranking)} evaluated models.")
                    break

            model = create_model_instance(model_info)
            pipeline = self._create_model_pipeline(self.preprocessor, model)

            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=self.cv, scoring=self.scoring)
            mean_score = np.mean(cv_scores)

            models_ranking.append({
                'name': model_info['name'],
                'config': model_info,
                'mean_score': mean_score
            })

        models_ranking.sort(key=lambda x: x['mean_score'], reverse=True)
        return models_ranking
  
    def _create_ensemble(self, top_models):
        """Creates an ensemble model from the top models."""
        estimators = [(model_info['name'], create_model_instance(model_info['config'])) for model_info in top_models]
        
        if self.ensemble_strategy == 'hard_voting':
            return VotingClassifier(estimators=estimators, voting='hard', weights=[m['mean_score'] for m in top_models])
        elif self.ensemble_strategy == 'soft_voting':
            return VotingClassifier(estimators=estimators, voting='soft', weights=[m['mean_score'] for m in top_models])
        elif self.ensemble_strategy == 'stacking':
            # logistic regression as meta-learner for stacking
            return StackingClassifier(
                estimators=estimators,
                final_estimator=LogisticRegression(max_iter=1000),
                cv=self.cv
            )
        else:
            raise ValueError("Ensemble method not recognized.")

    def _rank_models_stability(self, X_train, y_train, start_time, candidate_models=None):
        """Ranks models based on stability (variance) of cross-validation scores.
        
        Models with higher mean scores and lower variance are preferred.
        """
        models_ranking = []

        models_to_evaluate = candidate_models if candidate_models is not None else self.models_config

        for i, model_info in enumerate(models_to_evaluate):
            if self.time_limit:
                elapsed = time.time() - start_time
                if elapsed > self.time_limit:
                    print(f"Time limit reached after evaluating {i}/{len(models_to_evaluate)} models.")
                    print(f"Stopping model ranking early. Using {len(models_ranking)} evaluated models.")
                    break

            model = create_model_instance(model_info)
            pipeline = self._create_model_pipeline(self.preprocessor, model)

            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=self.cv, scoring=self.scoring)
            mean_score = np.mean(cv_scores)
            std_score = np.std(cv_scores)
            
            # Combined score: high mean and low variance
            # Penalize models with high variance
            stability_score = mean_score - (std_score * self.stability_threshold)

            models_ranking.append({
                'name': model_info['name'],
                'config': model_info,
                'mean_score': stability_score,
                'cv_mean': mean_score,
                'cv_std': std_score
            })

        models_ranking.sort(key=lambda x: x['mean_score'], reverse=True)
        return models_ranking

    def _rank_models_heuristic(self, X_train, y_train, start_time, candidate_models=None):
        """Ranks models based on dataset characteristics heuristics.
        
        Uses simple rules based on dataset size, dimensionality, and class balance
        to assign initial scores, then validates top candidates with CV.
        """
        n_samples, n_features = X_train.shape
        class_balance = np.bincount(y_train)[1] / len(y_train)  # Proportion of positive class
        
        models_ranking = []

        models_to_evaluate = candidate_models if candidate_models is not None else self.models_config

        for model_info in models_to_evaluate:
            # Initialize heuristic score
            heuristic_score = 0.5
            model_name = model_info['name'].lower()
            
            # Simple heuristics based on dataset characteristics
            # For small datasets (< 1000 samples), prefer simpler models
            if n_samples < 1000:
                if 'logreg' in model_name or 'knn' in model_name:
                    heuristic_score += 0.2
                elif 'gbm' in model_name or 'xgb' in model_name:
                    heuristic_score -= 0.1
            # For large datasets, prefer tree-based ensembles
            elif n_samples > 10000:
                if 'rf' in model_name or 'gbm' in model_name or 'xgb' in model_name or 'lightgbm' in model_name:
                    heuristic_score += 0.2
            
            # For high-dimensional data, prefer regularized models
            if n_features > 100:
                if 'logreg' in model_name or 'ridge' in model_name or 'lasso' in model_name:
                    heuristic_score += 0.15
            
            # For imbalanced datasets
            if class_balance < 0.3 or class_balance > 0.7:
                if 'tree' in model_name or 'forest' in model_name or 'gbm' in model_name:
                    heuristic_score += 0.1
            
            models_ranking.append({
                'name': model_info['name'],
                'config': model_info,
                'heuristic_score': heuristic_score
            })
        
        # Sort by heuristic score and evaluate top 10 with CV
        models_ranking.sort(key=lambda x: x['heuristic_score'], reverse=True)
        top_candidates = models_ranking[:min(10, len(models_ranking))]
        
        # Validate top candidates with cross-validation
        validated_ranking = []
        for i, model_info in enumerate(top_candidates):
            # Check time limit during CV validation
            if self.time_limit:
                elapsed = time.time() - start_time
                if elapsed > self.time_limit:
                    print(f"Time limit reached after validating {i}/{len(top_candidates)} candidates.")
                    # Use heuristic scores for remaining models
                    for remaining_model in top_candidates[i:]:
                        validated_ranking.append({
                            'name': remaining_model['name'],
                            'config': remaining_model['config'],
                            'mean_score': remaining_model['heuristic_score'],
                            'heuristic_score': remaining_model['heuristic_score']
                        })
                    break
            
            model = create_model_instance(model_info['config'])
            pipeline = self._create_model_pipeline(self.preprocessor, model)
            
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=self.cv, scoring=self.scoring)
            mean_score = np.mean(cv_scores)
            
            validated_ranking.append({
                'name': model_info['name'],
                'config': model_info['config'],
                'mean_score': mean_score,
                'heuristic_score': model_info['heuristic_score']
            })
        
        # Add remaining models with lower heuristic scores but no CV
        for model_info in models_ranking[min(10, len(models_ranking)):]:
            validated_ranking.append({
                'name': model_info['name'],
                'config': model_info['config'],
                'mean_score': model_info['heuristic_score'],  # Use heuristic as proxy
                'heuristic_score': model_info['heuristic_score']
            })
        
        validated_ranking.sort(key=lambda x: x['mean_score'], reverse=True)
        return validated_ranking