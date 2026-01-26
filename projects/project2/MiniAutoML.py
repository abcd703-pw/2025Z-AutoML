import pandas as pd
import numpy as np
import warnings
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from wrappers.wrapper_model import ModelWrapper
from Preprocessing.AutoMLPreprocessor import AutoMLPreprocessor
from scipy.stats import rankdata


class HeuristicEnsemble:
    def __init__(self, base_models, preprocessor, mode="mean", weights=None):
        self.base_models = base_models
        self.preprocessor = preprocessor
        self.mode = mode
        self.weights = weights
        self.model = self

    def predict_proba(self, X_raw):
        X_trans = self.preprocessor.transform(X_raw.copy(), None)

        preds = []
        cat_cols = self.preprocessor.get_categorical_cols(X_trans)
        for wrapper in self.base_models:
            X_model = X_trans.copy()
            model_name = wrapper.model.__class__.__name__

            if ("XGBClassifier" in model_name or "LGBMClassifier" in model_name) and cat_cols:
                for col in cat_cols:
                    X_model[col] = X_model[col].astype("category")

            preds.append(wrapper.model.predict_proba(X_model)[:, 1])

        preds = np.column_stack(preds)

        # 3. Logika łączenia (bez ponownego preprocessingu)
        if self.mode == "mean":
            return np.mean(preds, axis=1)

        elif self.mode == "weighted":
            return np.average(preds, axis=1, weights=self.weights)

        elif self.mode == "rank":
            ranked_preds = np.apply_along_axis(lambda x: rankdata(x) / len(x), 0, preds)
            return np.mean(ranked_preds, axis=1)

        return preds[:, 0]

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

    def get_params(self, deep=True):
        return {"mode": self.mode}
    

class StackingEnsemble:
    def __init__(self, base_models, preprocessor, meta_model=None, threshold=0.5, refit_meta=True):
        """
        :param base_models: Lista słowników {'wrapper': ..., 'preprocessor': ...}
        :param refit_meta: Jeśli False, zakłada, że meta_model jest już wytrenowany
                           i pomija kosztowne generowanie OOF w metodzie fit().
        """
        self.base_models = base_models
        self.preprocessor = preprocessor
        self.meta_model = meta_model if meta_model else LogisticRegression()
        self.threshold = threshold
        self.refit_meta = refit_meta
        self.fitted_ = False

    def fit(self, X, y):
        """
        Metoda przyjmuje dane już przetworzone.
        Nie wykonuje transform(), jedynie dostosowuje typy kolumn dla XGB/LGBM.
        """
        print("  -> Fitting base models on full dataset...")
        cat_cols = self.preprocessor.get_categorical_cols(X)
        
        for wrapper in self.base_models:
            # Tworzymy kopię X dla danego modelu, żeby specyficzne rzutowanie nie psuło innych
            X_model = X.copy()
            model_name = wrapper.model.__class__.__name__
            
            # --- Specyficzna obsługa KATEGORII ---
            if "CatBoost" in model_name:
                 wrapper.model.set_params(cat_features=cat_cols)
            
            elif ("XGBClassifier" in model_name or "LGBMClassifier" in model_name) and cat_cols:
                # XGB/LGBM wymagają fizycznego typu 'category'
                for col in cat_cols:
                    X_model[col] = X_model[col].astype("category")
                
                if "XGBClassifier" in model_name:
                    wrapper.model.set_params(enable_categorical=True, tree_method="hist")

            wrapper.model.fit(X_model, y)

        # 2. Meta-model: Trenujemy TYLKO jeśli refit_meta=True
        if self.refit_meta:
            print("  -> Generating OOF predictions for Meta-Learner (Slow)...")
            meta_features = []
            
            for wrapper in self.base_models:
                X_oof = X.copy()
                model_name = wrapper.model.__class__.__name__
                
                # Powtórka logiki kategorii dla OOF
                if "CatBoost" in model_name:
                     wrapper.model.set_params(cat_features=cat_cols)
                elif ("XGBClassifier" in model_name or "LGBMClassifier" in model_name) and cat_cols:
                    for col in cat_cols:
                        X_oof[col] = X_oof[col].astype("category")

                try:
                    oof_pred = cross_val_predict(wrapper.model, X_oof, y, cv=5, method="predict_proba", n_jobs=-1)[:, 1]
                except:
                    oof_pred = cross_val_predict(wrapper.model, X_oof, y, cv=5, method="predict", n_jobs=-1)
                
                meta_features.append(oof_pred)

            X_meta = np.column_stack(meta_features)
            self.meta_model.fit(X_meta, y)
        else:
            print("  -> Using pre-trained Meta-Learner params.")
            pass
            
        self.fitted_ = True
        return self

    def predict_proba(self, X_raw):
        """
        Tutaj wchodzą SUROWE dane, więc musimy je przetworzyć raz globalnie.
        """
        if not self.fitted_: raise ValueError("Ensemble not fitted")
        
        # 1. Globalna transformacja
        X_trans = self.preprocessor.transform(X_raw.copy(), None)
        # 2. Generowanie cech dla meta-modelu
        meta_features = self._get_meta_features(X_trans)
        
        # 3. Predykcja meta-modelu
        return self.meta_model.predict_proba(meta_features)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)

    def _get_meta_features(self, X_processed):
        """Generuje predykcje modeli bazowych na PRZETWORZONYCH danych."""
        preds = []
        cat_cols = self.preprocessor.get_categorical_cols(X_processed)
        
        for wrapper in self.base_models:
            X_model = X_processed.copy()
            model_name = wrapper.model.__class__.__name__
            
            if ("XGBClassifier" in model_name or "LGBMClassifier" in model_name) and cat_cols:
                for col in cat_cols:
                    X_model[col] = X_model[col].astype("category")
            
            preds.append(wrapper.model.predict_proba(X_model)[:, 1])
            
        return np.column_stack(preds)


class MiniAutoML:

    def __init__(self, models_config, metric="balanced_accuracy"):
        self.models_config = models_config
        self.metric = metric
        self.leaderboard = None
        self.preprocessor = AutoMLPreprocessor( # tunowane parametry, lepiej nie zmieniać
                 add_kmeans_features=True,
                 feature_selection= True,
                 add_poly_features=True, 
                 remove_outliers=False,
                 remove_multicollinearity=True, 
                 multicollinearity_threshold=0.95, 
                 id_threshold=0.95,
                 random_state=42)
        self.best_model = None

        warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

    def fit(self, X_train, y_train, cv=5):
        warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
        cv_stratedy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        scores = []
        n_samples, n_features = X_train.shape

        print("Begin preprocessing...")
        X_train_proc, y_train = self.preprocessor.fit_transform(X_train, y_train)
        cat_cols = self.preprocessor.get_categorical_cols(X_train_proc)

        X_train_cat = X_train_proc.copy()
        if cat_cols:
            for col in cat_cols:
                X_train_cat[col] = X_train_cat[col].astype("category")
        print("Preprocessing done.")

        # ======================================================================
        # STAGE 1: SCREENING
        # ======================================================================
        print(f"--- Stage 1: Screening {len(self.models_config)} models ---")

        for model_config in self.models_config:
            constraints = model_config.get("constraints", {})
            if n_samples > constraints.get("max_samples", float("inf")):
                continue
            if n_features > constraints.get("max_features", float("inf")):
                continue

            wrapper = ModelWrapper(model_config)
            if "random_state" in wrapper.model.get_params():
                wrapper.model.set_params(random_state=42)
            X_current = X_train_proc

            if "CatBoost" in model_config["class"] and cat_cols:
                wrapper.model.set_params(cat_features=cat_cols)

            try:
                cv_scores = cross_val_score(
                    wrapper.model, X_current, y_train, cv=cv_stratedy, scoring=self.metric, n_jobs=-1
                )
                mean_score = np.mean(cv_scores)
            except Exception as e:
                print(f"Error in {model_config['name']}: {e}")
                continue

            scores.append({
                "Model Name": model_config["name"],
                "Model Class": model_config["class"],
                "Metric Score": mean_score,
                "Wrapper": wrapper,
                "Config": model_config,
                "Params": wrapper.model.get_params()
            })
            print(f"{model_config['name']} → BA = {mean_score:.4f}")

        leaderboard = pd.DataFrame(scores).sort_values(
            by="Metric Score", ascending=False
        ).reset_index(drop=True)

        # ======================================================================
        # STAGE 2: STACKING (BEST + UNIQUE)
        # ======================================================================
        print("\n--- Stage 2: Stacking Ensembles ---")

        oof_cache = {}

        # --- Funkcja pomocnicza do pobierania/liczenia OOF ---
        def get_or_calc_oof(row_data):
            m_name = row_data["Model Name"]
            if m_name in oof_cache:
                return oof_cache[m_name]


            wrapper = row_data["Wrapper"]
            X_curr = X_train_proc
            if ("XGBClassifier" in row_data["Config"]["class"] or "LGBMClassifier" in row_data["Config"][
                "class"]) and cat_cols:
                X_curr = X_train_cat

                if "XGBClassifier" in row_data["Config"]["class"]:
                    wrapper.model.set_params(enable_categorical=True, tree_method="hist")

            try:
                oof = cross_val_predict(wrapper.model, X_curr, y_train, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
            except:
                oof = cross_val_predict(wrapper.model, X_curr, y_train, cv=cv, method="predict", n_jobs=-1)

            oof_cache[m_name] = oof
            return oof

        # --- A. STACKING TOP 3 BEST  ---
        selected_best = leaderboard.head(3).to_dict("records")

        if len(selected_best) >= 2:
            meta_feats_best = []
            wrappers_best = []

            for row in selected_best:
                oof = get_or_calc_oof(row)
                meta_feats_best.append(oof)

                nw = ModelWrapper(row["Config"])
                nw.model.set_params(**row["Wrapper"].model.get_params())
                wrappers_best.append(nw)

            X_meta_best = np.column_stack(meta_feats_best)


            meta_best = LogisticRegression(class_weight="balanced")
            meta_best.fit(X_meta_best, y_train)
            score_best = balanced_accuracy_score(y_train, meta_best.predict(X_meta_best))

            scores.append({
                "Model Name": "Ensemble: Stacking Top 3 Best",
                "Model Class": "Ensemble",
                "Metric Score": score_best,
                "Wrapper": StackingEnsemble(wrappers_best, self.preprocessor, meta_best, refit_meta=False),
                "Config": {}, "Params": "Stacking Top 3 Best"
            })
            print(f"Stacking Top 3 Best → {score_best:.4f}")

        # --- B. STACKING TOP 3 UNIQUE  ---
        unique_candidates = []
        seen_classes = set()

        for row in leaderboard.to_dict("records"):
            m_class = row["Config"]["class"]
            if m_class not in seen_classes:
                unique_candidates.append(row)
                seen_classes.add(m_class)

            if len(unique_candidates) >= 3:
                break

        if len(unique_candidates) >= 2:
            meta_feats_unique = []
            wrappers_unique = []
            names_unique = []

            for row in unique_candidates:
                names_unique.append(row["Model Name"])
                oof = get_or_calc_oof(row)
                meta_feats_unique.append(oof)

                nw = ModelWrapper(row["Config"])
                nw.model.set_params(**row["Wrapper"].model.get_params())
                wrappers_unique.append(nw)

            X_meta_unique = np.column_stack(meta_feats_unique)

            meta_unique = LogisticRegression(class_weight="balanced")
            meta_unique.fit(X_meta_unique, y_train)
            score_unique = balanced_accuracy_score(y_train, meta_unique.predict(X_meta_unique))

            scores.append({
                "Model Name": "Ensemble: Stacking Top 3 Unique",
                "Model Class": "Ensemble",
                "Metric Score": score_unique,
                "Wrapper": StackingEnsemble(wrappers_unique, self.preprocessor, meta_unique, refit_meta=False),
                "Config": {}, "Params": f"Unique Models: {names_unique}"
            })
            print(f"Stacking Top 3 Unique → {score_unique:.4f} (Models: {len(unique_candidates)})")

        # ======================================================================
        # STAGE 3: HEURISTIC ENSEMBLES (Mean, Rank, Champ+Rebel)
        # ======================================================================
        print("\n--- Stage 3: Fast Heuristics (Top 5) ---")

        top_n = min(5, len(leaderboard))
        selected_heuristics = leaderboard.head(top_n).to_dict("records")

        heuristic_wrappers = []
        collected_oofs = []

        for row in selected_heuristics:
            oof = get_or_calc_oof(row)
            collected_oofs.append(oof)

            nw = ModelWrapper(row["Config"])
            nw.model.set_params(**row["Wrapper"].model.get_params())
            heuristic_wrappers.append(nw)

        X_heuristics = np.column_stack(collected_oofs)

        # 1. Simple Mean
        mean_preds = np.mean(X_heuristics, axis=1)
        score_mean = balanced_accuracy_score(y_train, (mean_preds >= 0.5).astype(int))
        scores.append({
            "Model Name": f"Ensemble: Mean Top {top_n}",
            "Model Class": "Heuristic",
            "Metric Score": score_mean,
            "Wrapper": HeuristicEnsemble(heuristic_wrappers, self.preprocessor, mode="mean"),
            "Config": {}, "Params": "Simple Mean"
        })
        print(f"Ensemble: Mean Top {top_n} → {score_mean:.4f}")

        # 2. Rank Avg
        ranked_preds = np.apply_along_axis(lambda x: rankdata(x) / len(x), 0, X_heuristics)
        mean_rank = np.mean(ranked_preds, axis=1)
        score_rank = balanced_accuracy_score(y_train, (mean_rank >= 0.5).astype(int))
        scores.append({
            "Model Name": f"Ensemble: Rank Avg Top {top_n}",
            "Model Class": "Heuristic",
            "Metric Score": score_rank,
            "Wrapper": HeuristicEnsemble(heuristic_wrappers, self.preprocessor, mode="rank"),
            "Config": {}, "Params": "Rank Averaging"
        })
        print(f"Ensemble: Rank Avg Top {top_n} → {score_rank:.4f}")

        # 3. Champ + Rebel
        if top_n >= 2:
            champ_oof = collected_oofs[0]  # #1 model
            best_rebel_idx = -1
            min_corr = 1.0

            for i in range(1, top_n):
                corr = np.corrcoef(champ_oof, collected_oofs[i])[0, 1]
                if corr < min_corr:
                    min_corr = corr
                    best_rebel_idx = i

            if best_rebel_idx != -1:
                rebel_oof = collected_oofs[best_rebel_idx]
                blend_oof = 0.7 * champ_oof + 0.3 * rebel_oof
                score_blend = balanced_accuracy_score(y_train, (blend_oof >= 0.5).astype(int))

                blend_wrapper = HeuristicEnsemble(
                    [heuristic_wrappers[0], heuristic_wrappers[best_rebel_idx]],
                    self.preprocessor, mode="weighted", weights=[0.7, 0.3]
                )

                scores.append({
                    "Model Name": "Ensemble: Champ+Rebel",
                    "Model Class": "Heuristic",
                    "Metric Score": score_blend,
                    "Wrapper": blend_wrapper,
                    "Config": {}, "Params": f"Champ & #{best_rebel_idx + 1} (Corr {min_corr:.2f})"
                })
                print(f"Ensemble: Champ+Rebel → {score_blend:.4f}")

        # ======================================================================
        # FINAL SELECTION
        # ======================================================================
        leaderboard = pd.DataFrame(scores).sort_values(
            by="Metric Score", ascending=False
        ).reset_index(drop=True)

        best = leaderboard.iloc[0]
        self.leaderboard = leaderboard
        self.best_model = best["Wrapper"]

        print("\n==============================")
        print(f"WINNER: {best['Model Name']}")
        print(f"Balanced Accuracy: {best['Metric Score']:.4f}")
        print("==============================")

        print(f"Final fitting of {best['Model Name']}...")

        # Fit logic
        if isinstance(self.best_model, StackingEnsemble):
            self.best_model.fit(X_train_proc, y_train)
        elif isinstance(self.best_model, HeuristicEnsemble):
            for w in self.best_model.base_models:
                m_class = w.model.__class__.__name__
                if ("XGBClassifier" in m_class or "LGBMClassifier" in m_class) and cat_cols:
                    w.model.fit(X_train_cat, y_train)
                elif "CatBoost" in m_class and cat_cols:
                    w.model.set_params(cat_features=cat_cols)
                    w.model.fit(X_train_proc, y_train)
                else:
                    w.model.fit(X_train_proc, y_train)
        else:
            model_class = best["Config"]["class"]
            if ("XGBClassifier" in model_class or "LGBMClassifier" in model_class) and cat_cols:
                self.best_model.fit(X_train_cat, y_train)
            elif "CatBoost" in model_class and cat_cols:
                self.best_model.model.set_params(cat_features=cat_cols)
                self.best_model.fit(X_train_proc, y_train)
            else:
                self.best_model.fit(X_train_proc, y_train)

        return self.best_model

    def predict(self, X_test):
        if not self.best_model: raise ValueError("Call fit() first.")

        if isinstance(self.best_model, (StackingEnsemble, HeuristicEnsemble)):
            return self.best_model.predict(X_test)  

        X_test_proc = self.preprocessor.transform(X_test, None)

        cat_cols = self.preprocessor.get_categorical_cols(X_test_proc)
        model_name = self.best_model.model.__class__.__name__

        if ("XGBClassifier" in model_name or 
            "LGBMClassifier" in model_name) and cat_cols:
             for col in cat_cols:
                X_test_proc[col] = X_test_proc[col].astype("category")
                
        return self.best_model.model.predict(X_test_proc)

    def predict_proba(self, X_test):
        if not self.best_model:
            raise ValueError("Call fit() first.")

        if isinstance(self.best_model, StackingEnsemble) :
            return self.best_model.predict_proba(X_test)[:, 1]

        if isinstance(self.best_model, HeuristicEnsemble):
            return self.best_model.predict_proba(X_test)

        X_test_proc = self.preprocessor.transform(X_test, None)

        cat_cols = self.preprocessor.get_categorical_cols(X_test_proc)
        model_name = self.best_model.model.__class__.__name__
        
        if ("XGBClassifier" in model_name or "LGBMClassifier" in model_name) and cat_cols:
             for col in cat_cols:
                X_test_proc[col] = X_test_proc[col].astype("category")

        return self.best_model.model.predict_proba(X_test_proc)[:, 1]

    def display_leaderboard(self, mode="short"):
        if self.leaderboard is None: raise ValueError("No leaderboard.")
        print("================ Leaderboard ================")
        if mode == "short":
             cols = ["Model Name", "Metric Score"]
             return self.leaderboard[cols]
        else:
             return self.leaderboard