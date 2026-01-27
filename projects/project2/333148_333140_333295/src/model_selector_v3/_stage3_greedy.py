import numpy as np
import time
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.base import clone
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.utils.class_weight import compute_sample_weight
from ._utils import DataMapper

class Stage3Greedy:
    """
    Realizuje trzeci etap: budowę zespołu modeli (Ensemble).
    Używa metody Greedy Forward Selection optymalizowanej miarą łączącą
    skuteczność modelu z dywersyfikacją (korelacja predykcji).
    """

    def __init__(self, config, task, random_state, verbose=0):
        """
        Inicjalizuje etap ensemble.

        Args:
            config (dict): Konfiguracja.
            task (str): Typ zadania.
            random_state (int): Ziarno losowości.
            verbose (int): Poziom logowania.
        """
        self.top_k = config.get('greedy_top_k', 5)
        self.task = task
        self.random_state = random_state
        self.verbose = verbose
        self.mapper = DataMapper(config)
        self.diversity_w = config.get('st3_diversity_weight', 0.3)

    def _calculate_ensemble_score(self, oof_subset, y_true):
        """
        Oblicza wynik dla zestawu predykcji OOF uwzględniając dywersyfikację.
        Dla klasyfikacji używa Weighted Log Loss.
        """
        if oof_subset.ndim == 1:
            avg_preds = oof_subset
            diversity = 0
        else:
            avg_preds = np.mean(oof_subset, axis=1)
            corr_matrix = np.corrcoef(oof_subset, rowvar=False)
            # Średnia korelacja pozadiagonalna
            diversity = 1 - (np.sum(corr_matrix) - oof_subset.shape[1]) / (oof_subset.shape[1]**2 - oof_subset.shape[1] + 1e-6)

        if self.task == 'classification':
            sw = compute_sample_weight('balanced', y_true)
            performance = -log_loss(y_true, avg_preds, labels=[0, 1], sample_weight=sw)
        else:
            performance = -mean_squared_error(y_true, avg_preds)

        return (1 - self.diversity_w) * performance + self.diversity_w * diversity

    def _generate_oof(self, model, X, y, cv, method):
        """
        Generuje predykcje Out-Of-Fold dla pojedynczego modelu.
        """
        y_np = y.values if hasattr(y, 'values') else y
        oof = np.zeros(len(y_np))

        fit_params = {}
        model_name = model.__class__.__name__.lower()
        if 'catboost' in model_name and hasattr(X, 'columns'):
            cat_cols = [c for c in X.columns if X[c].dtype.name in ['object', 'category']]
            if cat_cols:
                fit_params['cat_features'] = cat_cols

        for train_idx, val_idx in cv.split(X, y):
            X_tr = X.iloc[train_idx] if hasattr(X, 'iloc') else X[train_idx]
            X_val = X.iloc[val_idx] if hasattr(X, 'iloc') else X[val_idx]
            m_clone = clone(model)
            m_clone.fit(X_tr, y_np[train_idx], **fit_params)

            if method == 'predict_proba':
                p = m_clone.predict_proba(X_val)
                oof[val_idx] = p[:, 1] if p.shape[1] == 2 else p[:, 0]
            else:
                oof[val_idx] = m_clone.predict(X_val)
        return oof

    def run(self, candidates, X_dict, y, time_limit=60):
        """
        Główna pętla etapu Ensemble korzystająca z Greedy Selection i Diversity Measure.
        """
        start_time = time.time()
        if self.verbose > 0:
            print(f"[Etap 3: Greedy] Start. Kandydaci: {len(candidates)}. Limit: {time_limit:.2f}s")

        if time_limit <= 5:
            time_limit = 15

        if self.task == 'classification':
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            method = 'predict_proba'
        else:
            cv = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
            method = 'predict'

        oof_preds, valid_models = [], []
        for i, model in enumerate(candidates):
            if (time.time() - start_time) > time_limit:
                break
            try:
                X = self.mapper.get_X_for_model(model.__class__.__name__, X_dict)
                oof_preds.append(self._generate_oof(model, X, y, cv, method))
                valid_models.append(model)
            except Exception:
                continue

        if not oof_preds: return [], None
        oof_matrix = np.column_stack(oof_preds)
        n_valid = len(valid_models)

        selected_indices = []
        best_score = -np.inf

        for i in range(n_valid):
            score = self._calculate_ensemble_score(oof_matrix[:, i], y)
            if score > best_score:
                best_score, selected_indices = score, [i]

        while len(selected_indices) < self.top_k and len(selected_indices) < n_valid:
            if (time.time() - start_time) > (time_limit - 2):
                break

            best_iter_score, best_iter_idx = -np.inf, -1
            for idx in range(n_valid):
                if idx in selected_indices: continue
                score = self._calculate_ensemble_score(oof_matrix[:, selected_indices + [idx]], y)
                if score > best_iter_score:
                    best_iter_score, best_iter_idx = score, idx

            if best_iter_idx != -1:
                selected_indices.append(best_iter_idx)
                best_score = best_iter_score
            else:
                break

        final_models = [valid_models[i] for i in selected_indices]
        final_oof = oof_matrix[:, selected_indices]

        if self.verbose > 0:
            print(f"[Etap 3] Koniec. Wybrano {len(final_models)} modeli.")

        return final_models, final_oof