import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from scipy.stats import median_abs_deviation
from ._factory import ModelFactory
from ._utils import DataMapper


class Stage2CV:
    """
    Realizuje drugi etap selekcji: pełną walidację krzyżową (CV) dla kandydatów
    wyłonionych w etapie Hyperband. Oblicza wynik odporny (robust score) uwzględniający
    odchylenie standardowe wyników, aby promować modele stabilne.
    """

    def __init__(self, config, task, random_state, verbose=0):
        """
        Inicjalizuje etap walidacji krzyżowej.

        Args:
            config (dict): Konfiguracja.
            task (str): Typ zadania.
            random_state (int): Ziarno losowości.
            verbose (int): Poziom logowania.
        """
        self.config = config
        self.task = task
        self.random_state = random_state
        self.verbose = verbose

        self.mapper = DataMapper(config)
        self.rules = config.get('stage2_rules', [])
        self.cv_folds = config.get('cv_folds', 5)

    def _resolve_rule(self, model_type):
        """
        Przypisuje typ modelu do reguły limitów zdefiniowanej w konfiguracji Stage 2.
        """
        family = self.mapper.get_family(model_type)
        if not family: return None
        return next((r for r in self.rules if family in r['families']), None)

    def run(self, candidates, X_dict, y, time_budget):
        """
        Przeprowadza walidację krzyżową dla każdego kandydata.
        Modele są oceniane, a następnie filtrowane i limitowane zgodnie z regułami
        grupowymi (np. max 2 modele typu boosting).

        Args:
            candidates (list): Lista konfiguracji modeli z Etapu 1.
            X_dict (dict): Dane wejściowe.
            y (array-like): Zmienna celowa.
            time_budget (float): (Obecnie nieużywany w pętli synchronicznej CV, ale dostępny).

        Returns:
            list: Lista zainicjalizowanych obiektów modeli (nie wytrenowanych na pełnym zbiorze),
                  które przeszły selekcję.
        """
        if self.verbose > 0:
            print(f"[Etap 2: Full CV] Weryfikacja {len(candidates)} modeli.")

        results = []
        if self.task == 'classification':
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            scoring = 'f1_weighted'
        else:
            cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            scoring = 'neg_mean_squared_error'

        for c_name, c_conf in candidates:
            X = self.mapper.get_X_for_model(c_conf['type'], X_dict)
            fit_params = {}
            if 'catboost' in c_conf['type'].lower() and hasattr(X, 'columns'):
                cat_cols = [c for c in X.columns if X[c].dtype.name in ['object', 'category']]
                if cat_cols:
                    fit_params['cat_features'] = cat_cols

            try:
                model = ModelFactory.create_model(c_conf, self.task, self.random_state, n_jobs=-1)
                scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=1, params=fit_params)

                k = 1.0
                med = np.median(scores)
                mad = median_abs_deviation(scores, scale='normal')
                robust_score = med - (k * mad)
                results.append((c_name, c_conf, robust_score))

            except Exception as e:
                if self.verbose >= 2:
                    print(f"    Błąd CV dla {c_name}: {e}")

        results.sort(key=lambda x: x[2], reverse=True)

        final_models = []
        group_counts = {r['name']: 0 for r in self.rules}

        if self.verbose > 0:
            print("[Etap 2] Selekcja finałowa:")

        for c_name, c_conf, score in results:
            rule = self._resolve_rule(c_conf['type'])
            if not rule: continue

            if group_counts[rule['name']] < rule['limit']:
                if self.verbose >= 2:
                    print(f" Akceptacja: {c_name} (Grupa: {rule['name']}) -> Score: {score:.4f}")
                model = ModelFactory.create_model(c_conf, self.task, self.random_state, n_jobs=-1)
                final_models.append(model)
                group_counts[rule['name']] += 1

        return final_models