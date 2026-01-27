import json
import os
import time
import pandas as pd
import numpy as np
from ._stage1_hyperband import Stage1Hyperband
from ._stage2_cv import Stage2CV
from ._stage3_greedy import Stage3Greedy

class ModelSelectorV3:
    """
    Główna klasa orkiestrująca proces automatycznego doboru modeli (AutoML).
    Zarządza konfiguracją, wczytywaniem portfolio modeli oraz przepływem danych
    przez trzy etapy selekcji: Hyperband, Walidację Krzyżową (CV) oraz Greedy Ensemble.
    """

    def __init__(self, config_path=None, verbose=2):
        """
        Inicjalizuje selektor, ładuje konfigurację i portfolio modeli.

        Args:
            config_path (str, optional): Ścieżka do pliku konfiguracyjnego JSON.
                Jeśli None, szuka 'config.json' w katalogu pakietu.
            verbose (int): Poziom szczegółowości logowania.
        """
        self.verbose = verbose

        if config_path is None:
            base_path = os.path.dirname(__file__)
            config_path = os.path.join(base_path, 'config.json')

        with open(config_path, 'r') as f:
            self.config = json.load(f)

        portfolio_path = os.path.join(os.path.dirname(config_path), self.config['portfolio_path'])
        with open(portfolio_path, 'r') as f:
            self.portfolio = json.load(f)

        self.time_budget = self.config.get('time_budget', 300)
        self.random_state = self.config.get('random_state', 42)

    def _sanitize_catboost_data(self, X_dict):
        """
        Przygotowuje dane wejściowe specyficznie dla modeli CatBoost.
        Konwertuje kolumny numeryczne na float oraz kategoryczne na stringi z oznaczeniem 'MISSING'.
        """
        cb_key = self.config.get('data_mapping', {}).get('catboost')

        if cb_key and cb_key in X_dict:
            if self.verbose >= 2:
                print(f"[Sanitizer] Czyszczenie danych dla CatBoosta (klucz: '{cb_key}')...")

            df = X_dict[cb_key].copy()

            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].astype(float)
                else:
                    if hasattr(df[col], 'cat'):
                        if df[col].isnull().any():
                            df[col] = df[col].cat.add_categories('MISSING').fillna('MISSING')
                    else:
                        df[col] = df[col].fillna('MISSING').astype(str)

            X_dict[cb_key] = df

    def fit(self, X_dict, y, task='classification'):
        """
        Uruchamia pełny proces selekcji modeli.

        Proces składa się z:
        1. Sanityzacji danych dla CatBoosta.
        2. Etapu 1 (Hyperband): Szybka eliminacja z użyciem Weighted Log Loss i kary za SD.
        3. Etapu 2 (Full CV): Dokładna walidacja z nowym grupowaniem modeli prostych.
        4. Etapu 3 (Greedy Ensemble): Budowa zespołu z miarą dywersyfikacji.
        """
        start_global = time.time()

        self._sanitize_catboost_data(X_dict)

        stage1 = Stage1Hyperband(self.config, self.portfolio, task, self.random_state, verbose=self.verbose)
        t1_budget = self.time_budget * 0.4
        candidates_stage1 = stage1.run(X_dict, y, t1_budget)

        if not candidates_stage1:
            if self.verbose > 0:
                print("Brak kandydatów po Etapie 1!")
            return [], None

        stage2 = Stage2CV(self.config, task, self.random_state, verbose=self.verbose)
        time_left = self.time_budget - (time.time() - start_global)
        if time_left < 10: time_left = 10
        t2_budget = time_left * 0.8

        candidates_stage2 = stage2.run(candidates_stage1, X_dict, y, t2_budget)

        if not candidates_stage2:
            if self.verbose > 0:
                print("Brak kandydatów po Etapie 2!")
            return [], None

        stage3 = Stage3Greedy(self.config, task, self.random_state, verbose=self.verbose)
        time_left_final = self.time_budget - (time.time() - start_global)

        final_models, oof_preds = stage3.run(candidates_stage2, X_dict, y, time_left_final)

        if self.verbose > 0:
            print(f"[ModelSelector] Koniec procesu. Wybrano {len(final_models)} modeli.")

        return final_models, oof_preds