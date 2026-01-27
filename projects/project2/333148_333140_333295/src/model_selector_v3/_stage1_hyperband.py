import optuna
import numpy as np
import time
import warnings
from copy import deepcopy
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.base import clone
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.utils.class_weight import compute_sample_weight
from optuna.integration import LightGBMPruningCallback
from ._factory import ModelFactory
from ._utils import DataMapper


class Stage1Hyperband:
    """
    Realizuje pierwszy etap selekcji modeli wykorzystujący algorytm Hyperband (poprzez Optuna).
    Celem jest szybkie przetestowanie dużej liczby konfiguracji na ograniczonych zasobach
    (mała liczba foldów lub iteracji) i odrzucenie nierokujących modeli.
    """

    def __init__(self, config, portfolio, task, random_state, verbose=0):
        """
        Inicjalizuje etap Hyperband.

        Args:
            config (dict): Główna konfiguracja.
            portfolio (dict): Przestrzeń dostępnych modeli i ich parametrów.
            task (str): Typ zadania ('classification'/'regression').
            random_state (int): Ziarno losowości.
            verbose (int): Poziom logowania.
        """
        self.config = config
        self.portfolio = portfolio
        self.task = task
        self.random_state = random_state
        self.verbose = verbose

        self.mapper = DataMapper(config)
        self.rules = config.get('stage1_rules', [])
        self.cv_folds = config.get('cv_folds', 5)
        self.penalty_k = config.get('st1_penalty_k', 0.1)

    def _get_metric_func(self):
        """
        Wybiera odpowiednią funkcję metryki. Dla klasyfikacji używa Weighted Log Loss.
        Dla regresji używa ujemnego błędu średniokwadratowego (MSE).
        """
        if self.task != 'classification':
            return (lambda y_t, y_p: -mean_squared_error(y_t, y_p)), False

        def weighted_log_loss(y_t, y_p):
            try:
                sw = compute_sample_weight('balanced', y_t)
                return -log_loss(y_t, y_p, labels=[0, 1], sample_weight=sw)
            except Exception:
                return -100.0

        return weighted_log_loss, True

    def _resolve_rule(self, model_type):
        """
        Znajduje regułę limitów (Stage 1 rules) pasującą do danej rodziny modeli.
        """
        family = self.mapper.get_family(model_type)
        if not family: return None
        return next((r for r in self.rules if family in r['families']), None)

    def objective(self, trial, X_dict, y, active_keys, deadline):
        """
        Funkcja celu dla Optuny. Trenuje wybrany model w pętli walidacji krzyżowej,
        raportując wyniki pośrednie. Wynik końcowy uwzględnia karę za brak stabilności (SD).
        """
        if time.time() > deadline:
            raise optuna.exceptions.TrialPruned()

        cfg_name = trial.suggest_categorical('config_name', active_keys)
        model_conf = deepcopy(self.portfolio[cfg_name])
        X = self.mapper.get_X_for_model(model_conf['type'], X_dict)

        if 'params' in model_conf:
            if 'n_estimators' in model_conf['params']:
                model_conf['params']['n_estimators'] = min(model_conf['params']['n_estimators'], 100)
            if 'iterations' in model_conf['params']:
                model_conf['params']['iterations'] = min(model_conf['params']['iterations'], 100)

        try:
            model = ModelFactory.create_model(model_conf, self.task, self.random_state, n_jobs=1)
        except Exception:
            return -np.inf

        kf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True,
                             random_state=self.random_state) if self.task == 'classification' else KFold(
            n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)

        metric_func, use_proba = self._get_metric_func()
        scores = []
        m_type = model_conf['type'].lower()

        for i, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            if time.time() > deadline:
                raise optuna.exceptions.TrialPruned()

            X_tr, X_val = (X.iloc[train_idx], X.iloc[val_idx]) if hasattr(X, 'iloc') else (X[train_idx], X[val_idx])
            y_tr, y_val = (y.iloc[train_idx], y.iloc[val_idx]) if hasattr(y, 'iloc') else (y[train_idx], y[val_idx])

            fit_params = {}
            if 'catboost' in m_type and hasattr(X_tr, 'columns'):
                cat_features = [c for c in X_tr.columns if X_tr[c].dtype.name in ['object', 'category']]
                if cat_features:
                    fit_params['cat_features'] = cat_features

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if 'lgbm' in m_type:
                    fit_params['callbacks'] = [
                        LightGBMPruningCallback(trial, "binary_logloss" if self.task == 'classification' else "l2")
                    ]
                    fit_params['verbose'] = -1

                model_clone = clone(model)
                try:
                    model_clone.fit(X_tr, y_tr, **fit_params)
                    preds = model_clone.predict_proba(X_val) if use_proba else model_clone.predict(X_val)
                    if use_proba and preds.shape[1] == 2: preds = preds[:, 1]
                    scores.append(metric_func(y_val, preds))
                except optuna.exceptions.TrialPruned:
                    raise
                except Exception:
                    return -np.inf

            trial.report(np.mean(scores), step=i)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return np.mean(scores) - self.penalty_k * np.std(scores)

    def run(self, X_dict, y, time_budget):
        """
        Uruchamia badanie (study) Optuny. Zarządza budżetem czasowym,
        konfiguruje sampler i pruner, a następnie wybiera najlepsze konfiguracje.
        """
        if self.verbose > 0:
            print(f"[Etap 1: Hyperband] Start. Budżet: {time_budget}s")

        deadline = time.time() + time_budget
        active_keys = []
        skipped_results = []

        for name, conf in self.portfolio.items():
            rule = self._resolve_rule(conf['type'])
            if not rule: continue
            if rule.get('skip_hyperband', False):
                skipped_results.append((name, conf))
            else:
                active_keys.append(name)

        candidates = []
        if active_keys:
            optuna.logging.set_verbosity(optuna.logging.WARNING if self.verbose == 0 else optuna.logging.INFO)

            study = optuna.create_study(
                direction='maximize',
                sampler=optuna.samplers.RandomSampler(seed=self.random_state),
                pruner=optuna.pruners.HyperbandPruner(
                    min_resource=self.config.get('hb_min_resource', 1),
                    max_resource=self.cv_folds,
                    reduction_factor=self.config.get('hb_reduction_factor', 2)
                )
            )

            try:
                study.optimize(
                    lambda t: self.objective(t, X_dict, y, active_keys, deadline),
                    timeout=time_budget,
                    n_jobs=-1
                )
            except Exception as e:
                if self.verbose > 0:
                    print(f"Błąd w Optuna: {e}")

            trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            trials.sort(key=lambda t: t.value, reverse=True)

            group_counts = {r['name']: 0 for r in self.rules}
            seen = set()

            for t in trials:
                c_name = t.params['config_name']
                if c_name in seen: continue
                conf = self.portfolio[c_name]
                rule = self._resolve_rule(conf['type'])

                if rule and group_counts[rule['name']] < rule['limit']:
                    candidates.append((c_name, conf))
                    group_counts[rule['name']] += 1
                    seen.add(c_name)

        return candidates + skipped_results