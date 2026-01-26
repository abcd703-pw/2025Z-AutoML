import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import KBinsDiscretizer, PolynomialFeatures, StandardScaler, PowerTransformer, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.cluster import MiniBatchKMeans

from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor


import warnings

class AutoMLPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, target_col=None, 
                 add_kmeans_features=True,
                 feature_selection=True, # opcje to True, False
                 add_poly_features=True, 
                 remove_outliers=False, # bardzo zle przy niezbalansowanych, lepiej False
                 remove_multicollinearity=True, # istotne do selekcji cech
                 multicollinearity_threshold=0.95,
                 id_threshold=0.95,
                 random_state=None):
        
        """
        Kompleksowy preprocesor AutoML, integrujący czyszczenie danych, inżynierię cech
        oraz redukcję wymiarowości.

        Pipeline przetwarzania:
        1. Detekcja typów (liczby, kategorie, daty).
        2. Przetwarzanie dat (cykliczne sin/cos).
        3. Imputacja braków (mediana/moda).
        4. Transformacja Yeo-Johnson i skalowanie.
        5. Generowanie cech (KMeans, Interakcje).
        6. Usuwanie outlierów.
        7. Selekcja cech (PCA/SFS).
        
        Przykładowe użycie:
        
        Processor = AutoMLPreprocessor(feature_selection_method='sfs', add_poly_features=True)
        X_train, X_test, y_train, y_test = Processor.process(X, y)


        Parametry:
        ----------
        target_col : str, opcjonalnie (domyślnie=None)
            Nazwa kolumny docelowej. Jeśli None, klasa spróbuje ją wywnioskować z przekazanego y.

        add_kmeans_features : bool (domyślnie=True)
            Czy generować cechy oparte na klastrowaniu (MiniBatchKMeans).
            Dodaje kolumny z dystansem do centroidów oraz ID klastra.
            Pomaga modelom wykrywać nieliniowe grupy w danych.

        feature_selection : {True, False} (domyślnie=True)
            Redukcja liczby cech
            
        n_features : float lub int (domyślnie=0.25)
            Ile cech zachować po selekcji:
            - float (0.0 - 1.0): Procent początkowych kolumn (np. 0.25 to 25%).
            - int (> 1): Dokładna liczba kolumn do pozostawienia.

        add_poly_features : bool (domyślnie=False)
            Czy tworzyć interakcje między zmiennymi numerycznymi (A*B).
            Może znacznie poprawić wynik, ale generuje dużo nowych kolumn,
            więc zaleca się używanie tego łącznie z `select_features=True`.

        remove_outliers : bool (domyślnie=True)
            Czy usuwać obserwacje odstające (outliery) ze zbioru treningowego
            przy użyciu algorytmu IsolationForest.

        random_state : int, opcjonalnie (domyślnie=None)
            Ziarno losowości dla zapewnienia powtarzalności wyników 
            (dla KMeans, PCA, SFS, IsolationForest).
        """

        self.random_state = random_state
        self.is_fitted = False
        
        self.target_col = target_col
        self.remove_outliers = remove_outliers
        
        # Column types
        self.original_num_cols = [] # Oryginalne kolumny numeryczne na samym początku
        self.num_cols = [] # Lista kolumn numerycznych
        self.cat_cols = [] # Lista kolumn kategorycznych
        self.date_cols = [] # Lista kolumn datowych
        
        # Usuwanie IDków
        self.id_threshold = id_threshold
        self.cols_to_drop_early = []

        # Imputers and transformers
        # IterativeImputer - MICE (Multivariate Imputation by Chained Equations)
        # Używamy ExtraTreesRegressor dla szybkości i nieliniowości.
        self.imputer_num = IterativeImputer(
            estimator=ExtraTreesRegressor(n_jobs=-1, min_samples_leaf=10, n_estimators=20, random_state=random_state),
            max_iter=5,            # 5 iteracji wystarczy dla dobrego przybliżenia (szybkość)
            random_state=random_state,
            initial_strategy='median' # Na start wstawia medianę, potem ją poprawia drzewami
        )
        self.cols_to_impute_num = [] 

        # Dla kategorii SimpleImputer, dla szybkości.
        # Imputacja MICE dla kategorii jest bardzo kosztowna obliczeniowo.
        self.imputer_cat = SimpleImputer(strategy='most_frequent')
        self.cols_to_impute_cat = [] 
        
        self.power_transformer = PowerTransformer(method='yeo-johnson', standardize=True)
        # unknown_value=-1 jest bezpieczne, bo rzutujemy na int -> -1 (int)
        self.cat_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

        # Frequency Encoding
        self.freq_enc_map = {} # Słownik do trzymania mapowań: { 'Kolumna': { 'Wartosc': 0.5, ... } }
        self.freq_cols = []    # Lista kolumn, które kodujemy

        #-- Nowe pola dla Binning ---
        self.binning_transformer = None
        self.binning_cols = []       # Kolumny źródłowe
        self.binning_new_names = []  # Nazwy nowych kolumn BIN_...

        # --- Nowe pola dla KMeans ---
        self.add_kmeans_features = add_kmeans_features
        self.kmeans_model = None
        self.kmeans_num_cols = [] # Lista nazw num w kmeans
        self.kmeans_cat_cols = [] # Lista nazw cat w kmeans

        # --- Nowe pola dla interakcji ---
        self.add_poly_features = add_poly_features
        self.cols_to_poly = [] # Lista kolumn do interakcji
        self.added_poly_cols = [] # Nazwy nowych cech z interakcji

        # Target processing
        self.imputer_y = SimpleImputer(strategy='most_frequent')
        self.encoder_y = LabelEncoder()

        # Feature Selection
        # Feature Selection Config
        self.feature_selection = feature_selection
        # Stan selekcji
        self.selection_mode = None          # 'pca', 'sfs', 'filter_only' lub None
        self.final_selected_cols = []       # Dla SFS/Filtra
        self.pca_model = None               # Dla PCA
        self.selector_model = None          # Dla SFS

        # --- Współliniowość ---
        self.remove_multicollinearity = remove_multicollinearity
        self.multicollinearity_threshold = multicollinearity_threshold
        self.collinear_drop_cols = [] # Lista kolumn do usunięcia

    def process(self, X, y=None):
        # 1. Scenariusz: Użytkownik podaje tylko X (dataframe z targetem), y=None
        if y is None:
            if self.target_col is not None and self.target_col in X.columns:
                y = X[self.target_col]
            elif self.target_col is not None:
                raise Exception("Nie podano y ani nie znaleziono target_col w X!")
            else:
                raise Exception("Nie podano y ani target_col! Nie można wywnioskować targetu.")
            # Jeśli target_col jest None i y jest None -> Nie mamy targetu. 

        # 2. Scenariusz: Użytkownik podał y, ale nie podał nazwy target_col w __init__
        if self.target_col is None and y is not None:
            if hasattr(y, 'name'): 
                self.target_col = y.name
            elif hasattr(y, 'columns') and len(y.columns) > 0: 
                self.target_col = y.columns[0]
            else:
                self.target_col = 'target'
        
        # 3. Usuwamy target z X (jeśli tam jest), żeby nie było wycieku danych (Data Leakage)
        if self.target_col and self.target_col in X.columns:
            X = X.drop(columns=[self.target_col], errors='ignore')

        # 4. Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )
        
        X_train, y_train = self.fit_transform(X_train, y_train)
        X_test, y_test = self.transform(X_test, y_test)
        
        return X_train, X_test, y_train, y_test


    def _detect_columns(self, X):
        """
        Automatyczne wykrywanie typów kolumn z zaawansowaną heurystyką.
        Kolejność:
        1. Daty (po próbce danych).
        2. Niska liczność (<10 unikalnych) -> KATEGORIA (nawet jeśli to liczby/bool).
        3. Typy numeryczne -> NUMERYCZNE.
        4. Object, który da się rzutować na liczbę (np. "3.14") -> NUMERYCZNE.
        5. Reszta -> KATEGORIA.
        """
        self.date_cols = []
        self.cat_cols = []
        self.num_cols = []
        self.original_num_cols = []

        temp_X = X.copy()
        
        # 1. Wykrywanie DAT (Twoja logika z optymalizacją)
        # Iterujemy tylko po object i datetime, żeby nie sprawdzać floatów
        potential_dates = temp_X.select_dtypes(include=['object', 'datetime']).columns
        
        for col in potential_dates:
            try:
                # Jeśli to tekst wyglądający na liczbę, pomijamy sprawdzanie daty
                # (chyba że to timestamp, ale to rzadkość w csv bez formatowania)
                if temp_X[col].dtype == 'object':
                    # Szybki test czy to nie same cyfry (unika mielenia IDków przez to_datetime)
                    # Używamy próbki dla szybkości
                    sample_str = temp_X[col].dropna().astype(str).iloc[:100]
                    if sample_str.str.replace('.', '', 1).str.isnumeric().all(): 
                        continue

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    sample = temp_X[col].dropna().iloc[:100]
                    if len(sample) > 0:
                        pd.to_datetime(sample, errors='raise')
                        self.date_cols.append(col)
            except (ValueError, TypeError):
                continue
        
        
        remaining_cols = [c for c in temp_X.columns if c not in self.date_cols]
        n_rows = len(temp_X)

        for col in remaining_cols:
            series = temp_X[col]
            
            # Pomijamy puste kolumny
            if series.dropna().empty:
                continue

            # A. Sprawdzenie typu obiektowego / bool
            # Jeśli to stringi lub boole -> KATEGORIA
            if pd.api.types.is_object_dtype(series) or pd.api.types.is_bool_dtype(series) or pd.api.types.is_categorical_dtype(series):
                self.cat_cols.append(col)
                continue

            # B. Sprawdzenie NUMERYCZNE (int i float)
            if pd.api.types.is_numeric_dtype(series):
                
                # 1. Test na "Prawdziwy Float"
                # Jeśli kolumna jest floatem i ma wartości po przecinku (np. 1.23), to NA PEWNO liczba.
                # (np. stopy procentowe, waga, cena)
                if pd.api.types.is_float_dtype(series):
                    # Sprawdzamy czy ma część ułamkową inną niż 0
                    # dropna() ważne, bo modulo z NaN daje NaN
                    if (series.dropna() % 1 != 0).any():
                        self.num_cols.append(col)
                        continue

                # 2. Analiza Liczności (Cardinality) dla liczb całkowitych (i floatów wyglądających jak int)
                n_unique = series.nunique(dropna=True)
                
                # Próg absolutny (dla małych zbiorów wartości, np. płeć 0/1, status 1/2/3)
                ABS_THRESHOLD = 20 
                
                # Próg relatywny (dla dużych danych)
                # Jeśli unikalnych wartości jest mniej niż 5% liczby wierszy -> KATEGORIA
                # (np. kod pocztowy w bazie milionowej - jest liczbą, ale zachowuje się jak kategoria)
                RATIO_THRESHOLD = 0.05 
                
                # Wyjątek: Jeśli unikalnych wartości jest bardzo dużo (np. > 1000), 
                # to nawet przy małym ratio wolimy to jako liczbę (chyba że to ID, ale ID usuwamy gdzie indziej).
                MAX_CAT_LIMIT = 500

                if n_unique <= ABS_THRESHOLD:
                    # Bardzo mało wartości (np. 0-20) -> Traktujemy jako KATEGORIĘ
                    # (Dla CatBoost/LGBM to często lepsze niż liczba)
                    self.cat_cols.append(col)
                
                elif n_unique < n_rows * RATIO_THRESHOLD and n_unique <= MAX_CAT_LIMIT:
                    # Mało wartości w stosunku do wielkości danych -> KATEGORIA
                    self.cat_cols.append(col)
                
                else:
                    # W przeciwnym razie -> LICZBA
                    self.num_cols.append(col)

            else:
                # Fallback (rzadkie przypadki) -> Kategoria
                self.cat_cols.append(col)

    def _process_dates_cyclical(self, df):
        df_out = df.copy()
        for col in self.date_cols:
            if col in df_out.columns:
                dates = pd.to_datetime(df_out[col], errors='coerce')
                cycles = {
                    'month': (dates.dt.month, 12),
                    'day': (dates.dt.day, 31),
                    'dayofweek': (dates.dt.dayofweek, 7),
                    'dayofyear': (dates.dt.dayofyear, 365)
                }
                for part, (values, period) in cycles.items():
                    df_out[f'{col}_{part}_sin'] = np.sin(2 * np.pi * values / period)
                    df_out[f'{col}_{part}_cos'] = np.cos(2 * np.pi * values / period)
                
                df_out[f'{col}_year'] = dates.dt.year
                df_out[f'{col}_is_weekend'] = (dates.dt.dayofweek >= 5).astype(int)
                df_out.drop(columns=[col], inplace=True)
        
        return df_out

    def _process_target(self, y, fit=False):
        """
        Jeden pipeline dla targetu: Standaryzacja typu -> Imputacja -> Encoding.
        Obsługuje: bool, int, str, object oraz braki danych.
        """
        if y is None:
            return None
        
        # 1. Konwersja na Pandas Series (ułatwia obsługę typów i NaN)
        y = np.ravel(y)
        y_s = pd.Series(y)
        
        # 2. Naprawa błędu "dtype bool":
        # Jeśli dane są boolami (lub zawierają boole i NaNy), zamieniamy je na spójny format.
        # Najbezpieczniej zamienić wszystko na Stringi, ale musimy uważać, 
        # bo astype(str) zamienia np.nan na napis "nan".
        
        # Krok A: Zamiana na stringi (True -> "True", 1 -> "1", NaN -> "nan")
        y_s = y_s.astype(str)
        
        # Krok B: Przywrócenie prawdziwych NaN-ów (żeby Imputer wiedział co uzupełniać)
        # Traktujemy napisy 'nan', 'None', '<NA>' jako braki danych.
        y_s = y_s.replace({'nan': np.nan, 'None': np.nan, '<NA>': np.nan})
        
        # 3. Konwersja do numpy array 2D (wymóg SimpleImputera)
        y_vals = y_s.values.reshape(-1, 1)
        
        # 4. Pipeline Imputacja -> Encoding
        if fit:
            # Uczymy się najczęstszej wartości (imputer) i mapowania klas (encoder)
            y_filled = self.imputer_y.fit_transform(y_vals).ravel()
            y_encoded = self.encoder_y.fit_transform(y_filled)
        else:
            # Tylko transformujemy
            y_filled = self.imputer_y.transform(y_vals).ravel()
            try:
                y_encoded = self.encoder_y.transform(y_filled)
            except ValueError:
                # Fallback: jeśli w nowych danych pojawi się klasa, której nie było w treningu
                # (np. "Maybe" zamiast "True"/"False"), zwracamy wypełnione wartości surowe 
                # lub (lepiej) domyślną klasę 0.
                print("Uwaga: Nieznana klasa w targecie. Mapuję na 0.")
                # Opcja bezpieczna: zwracamy zera (zakładając binarną klasyfikację)
                return np.zeros(len(y_filled), dtype=int)
                
        return y_encoded
    
    def _fit_frequency_features(self, X):
        """Uczy się częstości występowania kategorii na zbiorze treningowym."""
        if not self.cat_cols:
            return

        # Wybieramy kandydatów (np. > 5 unikalnych wartości)
        cols_to_process = [c for c in self.cat_cols if c in X.columns]
        unique_counts = {c: X[c].nunique() for c in cols_to_process}
        
        # Bierzemy top 10 kolumn o największej liczności
        candidates = [c for c, count in unique_counts.items() if count > 5]
        self.freq_cols = sorted(candidates, key=lambda c: unique_counts[c], reverse=True)[:10]

        self.freq_enc_map = {}
        for col in self.freq_cols:
            # Obliczamy częstość (procentowy udział)
            freq_map = X[col].value_counts(normalize=True).to_dict()
            self.freq_enc_map[col] = freq_map

    def _transform_frequency_features(self, X):
        """Aplikuje zapamiętane częstości do danych."""
        if not self.freq_cols or not self.freq_enc_map:
            return X
        
        new_feats = {}
        for col in self.freq_cols:
            if col in X.columns:
                mapping = self.freq_enc_map[col]
                # Mapujemy. fillna(0) jest kluczowe!
                # Jeśli w teście pojawi się nowa kategoria (nieznana w treningu), dostanie 0.
                new_feats[f'FREQ_{col}'] = X[col].map(mapping).fillna(0)
        
        if new_feats:
            X_freq = pd.DataFrame(new_feats, index=X.index)
            X = pd.concat([X, X_freq], axis=1)
            
            # W fit() dodajemy te kolumny do num_cols, tutaj tylko zwracamy X
            # (aktualizację listy num_cols robimy tylko w głównym fit)
            
        return X    


    def _fit_binning_features(self, X):
        """Uczy się przedziałów (kwantyli) na zbiorze treningowym."""
        # Logika wyboru kolumn (np. te same co do interakcji lub top numeryczne)
        if self.cols_to_poly:
            self.binning_cols = [c for c in self.cols_to_poly if c in X.columns]
        else:
            # Fallback: pierwsze 10 numerycznych
            self.binning_cols = [c for c in self.num_cols if c in X.columns][:10]
            
        if not self.binning_cols:
            return

        # Inicjalizacja i nauka
        self.binning_transformer = KBinsDiscretizer(
            n_bins=5, encode='ordinal', strategy='quantile', subsample=200_000, random_state=self.random_state
        )
        
        try:
            self.binning_transformer.fit(X[self.binning_cols])
            self.binning_new_names = [f'BIN_{c}' for c in self.binning_cols]
        except Exception as e:
            print(f"   Błąd fitowania Binningu: {e}")
            self.binning_transformer = None
            self.binning_cols = []

    def _transform_binning_features(self, X):
        """Aplikuje wyuczone przedziały."""
        if self.binning_transformer is None or not self.binning_cols:
            return X
            
        # Sprawdzamy obecność kolumn
        valid_cols = [c for c in self.binning_cols if c in X.columns]
        if len(valid_cols) != len(self.binning_cols):
            # Jeśli brakuje kolumn (rzadkie), pomijamy transformację dla bezpieczeństwa
            print("   Uwaga: Brak kolumn do Binningu w danych. Pomijam transformację Binningu.")
            return X

        try:
            X_binned = self.binning_transformer.transform(X[self.binning_cols]).astype(int)
            
            X_bin_df = pd.DataFrame(X_binned, columns=self.binning_new_names, index=X.index)
            X = pd.concat([X, X_bin_df], axis=1)
            
        except Exception as e:
            print(f"   Błąd transformacji Binningu: {e}")
            # W rzadkich przypadkach (np. zbyt mała wariancja w kolumnie) może rzucić błąd
            
        return X


    def _fit_kmeans(self, X):
        """Logika uczenia KMeans inspirowana biblioteką MLJAR."""
        # Wybór liczby klastrów (heurystyka MLJAR)
        n_clusters = int(np.log10(X.shape[0]) * 8)
        n_clusters = max(2, n_clusters)      # Minimum 2 klastry
        n_clusters = min(n_clusters, 15)     # Ograniczamy max (żeby nie zrobiło 100 kolumn)
        # MLJAR robi min(n, X.shape[1]), ale tutaj bezpieczniej dać sztywny limit górny dla wydajności
        
        self.kmeans_model = MiniBatchKMeans(
            n_clusters=n_clusters, 
            init="k-means++", 
            batch_size=256,
            random_state=self.random_state,
            n_init='auto'
        )
        self.kmeans_model.fit(X)
        
        # Zapamiętanie nazw nowych cech
        self.kmeans_num_cols = [f"Dist_Cluster_{i}" for i in range(n_clusters)]
        self.kmeans_cat_cols = ["Cluster"]
        
        self.num_cols.extend(self.kmeans_num_cols)
        self.cat_cols.extend(self.kmeans_cat_cols)
        print(f"--- KMeans: Wytrenowano {n_clusters} klastrów ---")

    def _transform_with_kmeans(self, X):
        """Aplikuje KMeans i dokleja nowe kolumny do X."""
        if not self.add_kmeans_features or self.kmeans_model is None:
            return X
        
        X_out = X.copy()
        
        distances = self.kmeans_model.transform(X_out)
        clusters = self.kmeans_model.predict(X_out)
        
        # Dodawanie kolumn do DataFrame
        # 1. Dystanse do centroidów
        dist_cols = self.kmeans_num_cols
        X_out[dist_cols] = distances
        
        # 2. ID Klastra (jako kategoria/int)
        X_out[self.kmeans_cat_cols[0]] = clusters
        
        return X_out

    def _add_poly_features(self, X, y):
        """
        Generuje zaawansowane cechy inżynierskie:
        1. Unarne: log(x), x^2
        2. Binarne (pary najważniejszych cech): A+B, A-B, A*B, A/B
        """
        if not self.add_poly_features:
            return X
        
        # Limit cech bazowych, z których robimy kombinacje.
        # Jeśli dasz 5, to par będzie 5 po 2 = 10 par. Każda para ma 4 operacje (+,-,*,/) = 40 nowych cech.
        NUM_OF_FEATURES_LIMIT = 6

        # Bierzemy tylko numeryczne, żeby nie mnożyć kategorii
        valid_cols = [c for c in self.num_cols if c in X.columns]
        
        if not valid_cols:
            return X

        if len(valid_cols) > NUM_OF_FEATURES_LIMIT:
            if y is not None:
                print(f"   -> Wybieranie top {NUM_OF_FEATURES_LIMIT} cech do interakcji (ExtraTrees)...")
                try:
                    # Lekki model do szybkiego rankingu
                    et = ExtraTreesClassifier(
                        n_estimators=50, 
                        max_depth=5,     # Płytkie drzewa wystarczą do oceny ważności
                        n_jobs=-1, 
                        random_state=self.random_state,
                        class_weight='balanced'
                    )
                    et.fit(X[valid_cols], y)
                    
                    # Pobieramy ważności i sortujemy
                    importances = et.feature_importances_
                    indices = np.argsort(importances)[::-1] # Indeksy malejąco
                    top_indices = indices[:NUM_OF_FEATURES_LIMIT]
                    
                    self.cols_to_poly = [valid_cols[i] for i in top_indices]
                    print(f"   Wybrane kolumny do interakcji: {self.cols_to_poly}")
                except Exception as e:
                    print(f"   Błąd selekcji do interakcji ({e}). Biorę pierwsze {NUM_OF_FEATURES_LIMIT}.")
                    self.cols_to_poly = valid_cols[:NUM_OF_FEATURES_LIMIT]
            else:
                # Fallback jeśli brak y
                self.cols_to_poly = valid_cols[:NUM_OF_FEATURES_LIMIT]
        else:
            # Jeśli mało kolumn, bierzemy wszystkie
            self.cols_to_poly = valid_cols
            
        # --- GENEROWANIE CECH ---
        # Tutaj wywołujemy logikę transformacji od razu, żeby zaktualizować X
        # Nie potrzebujemy 'fit' dla matematyki (dodawanie to dodawanie), 
        # musimy tylko pamiętać 'cols_to_poly' (co zrobiliśmy wyżej).
        
        return self._transform_poly_features(X)


    def _transform_poly_features(self, X):
        """Aplikuje operacje matematyczne na wybranych kolumnach."""
        if not self.add_poly_features or not hasattr(self, 'cols_to_poly') or not self.cols_to_poly:
            return X
        
        # Sprawdzamy czy wybrane kolumny są w X
        valid_cols = [c for c in self.cols_to_poly if c in X.columns]
        if len(valid_cols) < 1:
            return X
        
        new_features = {}
        epsilon = 1e-6 # Zabezpieczenie przed dzieleniem przez zero
        
        # A. Transformacje UNARNE (pojedyncze kolumny)
        for col in valid_cols:
            # 1. Logarytm (bezpieczny: log(abs(x) + 1))
            # Obsługuje ujemne wartości zamieniając je na log z modułu
            new_features[f'LOG_{col}'] = np.log1p(np.abs(X[col]))
            
            # 2. Kwadrat (zastępuje PolynomialFeatures degree=2 dla samego siebie)
            new_features[f'SQR_{col}'] = X[col] ** 2
            
            # Opcjonalnie: Pierwiastek (tylko z modułu)
            # new_features[f'SQRT_{col}'] = np.sqrt(np.abs(X[col]))

        # B. Transformacje BINARNE (pary kolumn)
        import itertools
        # Tworzymy pary z wybranych kolumn (np. V1 i V4)
        for col_a, col_b in itertools.combinations(valid_cols, 2):
            val_a = X[col_a]
            val_b = X[col_b]
            
            # 3. Mnożenie (Interakcja)
            new_features[f'MUL_{col_a}_x_{col_b}'] = val_a * val_b
            
            # 4. A / B
            # Logika: Jeśli val_b == 0, użyj epsilon. W przeciwnym razie użyj val_b.
            # Nie dodajemy epsilona do liczb, które nie są zerem, żeby nie zniekształcać danych.
            denom_b = np.where(val_b == 0, epsilon, val_b)
            new_features[f'DIV_{col_a}_by_{col_b}'] = val_a / denom_b
            
            # W drugą stronę: B / A
            denom_a = np.where(val_a == 0, epsilon, val_a)
            new_features[f'DIV_{col_b}_by_{col_a}'] = val_b / denom_a
            
            # 5. Różnica (Difference)
            new_features[f'SUB_{col_a}_{col_b}'] = val_a - val_b
            
            # 6. Suma (często mniej ważna dla drzew, ale może się przydać)
            new_features[f'ADD_{col_a}_{col_b}'] = val_a + val_b

        # Tworzenie DataFrame z nowych cech
        X_new_feats = pd.DataFrame(new_features, index=X.index)
        
        # Filtrowanie duplikatów (jeśli już są w X - rzadkie, ale możliwe)
        self.added_poly_cols = [c for c in X_new_feats.columns if c not in X.columns]
        
        if self.added_poly_cols:
            X = pd.concat([X, X_new_feats[self.added_poly_cols]], axis=1)
            # Aktualizacja listy numerycznych (tylko w fit, ale tu robimy trick sprawdzając typ)
            # Jeśli wywołujemy to z transform(), to self.num_cols nie powinno się zmieniać trwale 
            # w sposób, który zepsułby pipeline, ale dla spójności warto wiedzieć, że to liczby.
            # (W metodzie fit() w klasie głównej i tak nadpisujesz num_cols na końcu, więc jest OK).
        
        return X

    def _remove_collinear(self, X):
        """Usuwa kolumny silnie skorelowane ze sobą."""
        if not self.remove_multicollinearity:
            return X
            
        # Obliczamy macierz korelacji (wartość bezwzględna, bo -0.99 to też silna korelacja)
        # Robimy to tylko dla kolumn numerycznych
        
        if not self.num_cols:
            return X

        corr_matrix = X[self.num_cols].corr().abs()

        # Wybieramy górny trójkąt macierzy (żeby nie sprawdzać A z B i B z A)
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Znajdujemy kolumny, które mają korelację większą niż próg
        to_drop = [column for column in upper.columns if any(upper[column] > self.multicollinearity_threshold)]
        
        # Zapisujemy do self.collinear_drop_cols, żeby w transform usunąć te same
        self.collinear_drop_cols = to_drop
        
        if to_drop:
            n_total = X.shape[1]
            print(f"--- Współliniowość: Usunięto {len(to_drop)} z {n_total} kolumn (korelacja > {self.multicollinearity_threshold}) ---")
            # print(f"-> Usunięte: {to_drop}") # Opcjonalnie wypisz nazwy
        
        self.num_cols = [c for c in self.num_cols if c not in self.collinear_drop_cols]
        return X.drop(columns=self.collinear_drop_cols)
    
    def _detect_useless_features(self, X):
        """
        Wykrywa kolumny stałe (0 wariancji) oraz kolumny ID (zbyt wysoka unikalność).
        ZABEZPIECZENIE: Nie usuwa kolumn, które wyglądają jak daty.
        """
        self.cols_to_drop_early = []
        
        n_rows = len(X)
        if n_rows == 0: return

        for col in X.columns:
            # 1. Kolumny STAŁE (tylko 1 wartość)
            if X[col].nunique(dropna=False) <= 1:
                self.cols_to_drop_early.append(col)
                continue
            
            # 2. Kolumny ID (High Cardinality)
            # Sprawdzamy to TYLKO dla obiektów/kategorii. 
            if X[col].dtype == 'object' or hasattr(X[col], 'cat'):
                n_unique = X[col].nunique()
                
                if n_unique / n_rows > self.id_threshold:
                    
                    # Zanim usuniemy, sprawdzamy czy to nie jest data (np. timestamp)
                    # Timestampy są unikalne, ale niosą cenną informację!
                    is_date_candidate = False
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            # Bierzemy próbkę niepustych wartości
                            sample = X[col].dropna().iloc[:100]
                            if len(sample) > 0:
                                # Sprawdzamy czy to nie są same cyfry (bo ID=12345 to nie data)
                                # Jeśli to tekstowe liczby, to na pewno ID, więc pozwalamy usunąć.
                                if sample.astype(str).str.isnumeric().all():
                                    is_date_candidate = False
                                else:
                                    # Próba konwersji na datę
                                    pd.to_datetime(sample, errors='raise')
                                    is_date_candidate = True
                    except (ValueError, TypeError):
                        is_date_candidate = False
                    
                    if is_date_candidate:
                        # To wygląda na datę, więc JEJ NIE USUWAMY mimo wysokiej unikalności
                        continue
                    
                    # Jeśli dotarliśmy tutaj, to nie jest data, a ma dużą unikalność -> ID -> Usuwamy
                    self.cols_to_drop_early.append(col)

        if self.cols_to_drop_early:
            print(f"--- Wstępne czyszczenie: Usunięto {len(self.cols_to_drop_early)} kolumn (stałe lub ID) ---")
            # print(f"-> Usunięte: {self.cols_to_drop_early}")

    def _fit_feature_selection(self, X, y):
        """
        Inteligentna selekcja cech:
        1. ENSEMBLE SCREENING (L2 + ExtraTrees):
        - Usuwa zmienne, które są słabe ZARÓWNO liniowo, jak i nieliniowo.
        - To jest 'miękkie' wstępne czyszczenie.
        2. HYBRYDOWA REDUKCJA:
        - Jeśli cech zostało mało (<60) -> SFS Backward (Maksymalna precyzja).
        - Jeśli cech zostało dużo (>60) -> PCA (Kompresja informacji).
        """
        if not self.feature_selection:
            self.selection_mode = None
            return

        print(f"\n--- Selekcja Cech (Hybrid) ---")
        X_curr = X.copy()
        initial_cols = X_curr.columns.tolist()
        n_start = len(initial_cols)
        
        # Próg przełączenia między SFS a PCA
        SFS_CAP_THRESHOLD = 150

        # ==========================================
        # ETAP 1: ENSEMBLE SCREENING (L1 + ExtraTrees)
        # ==========================================
        print(f"-> Etap 1: Wstępne usuwanie szumu (L1 + ExtraTrees)...")
        
        try:
            # A. Ocena Liniowa (L1 / Lasso)
            # C=0.5 to umiarkowana regularyzacja. threshold='0.1*mean' jest bardzo łagodny.
            # Chcemy wyrzucić tylko totalne zera.
            print("   -> Ocena Liniowa (L1 Logistic Regression)...")
            LR_threshold = "0.1*mean"  # próg
            l1_model = LogisticRegression(
                class_weight = 'balanced', 
                penalty = 'l1',
                solver='liblinear',
                max_iter = 1000,
                C = 0.5,
                tol = 1e-3,
                random_state = self.random_state
            )
            l1_selector = SelectFromModel(estimator=l1_model, threshold=LR_threshold)
            l1_selector.fit(X_curr, y)
            mask_l1 = l1_selector.get_support()
            
            # B. Ocena Nieliniowa (ExtraTrees)
            # Drzewa widzą interakcje.
            print("   -> Ocena Nieliniowa (ExtraTreesClassifier)...")
            ETC_threshold = "0.34*mean"  # próg
            et_model = ExtraTreesClassifier(
                n_estimators=100, 
                min_samples_leaf=10, 
                class_weight='balanced',
                random_state=self.random_state, 
                n_jobs=-1)
            et_selector = SelectFromModel(estimator=et_model, threshold=ETC_threshold)
            et_selector.fit(X_curr, y)
            mask_et = et_selector.get_support()
            
            # C. Suma Zbiorów (Union)
            # Zostawiamy cechę, jeśli L1 LUB Drzewa uważają ją za ważną.
            final_mask = mask_l1 | mask_et # Bitwise OR
            
            # Aktualizacja danych
            filtered_cols = np.array(initial_cols)[final_mask].tolist()
            
            # Zabezpieczenie: jeśli algorytm chce usunąć wszystko (mało prawdopodobne), bierzemy wynik ExtraTrees
            if len(filtered_cols) < 2:
                print("   Uwaga: Ensemble chciał usunąć prawie wszystko. Cofam do wyniku samych drzew.")
                filtered_cols = np.array(initial_cols)[mask_et].tolist()

            X_curr = X_curr[filtered_cols]
            n_after_screen = len(filtered_cols)
            
            n_dropped = n_start - n_after_screen
            print(f"   L1 wybrało: {sum(mask_l1)}, Trees wybrało: {sum(mask_et)}")
            print(f"   Wspólna decyzja: Zachowano {n_after_screen} z {n_start} cech (usunięto {n_dropped} najsłabszych).")

        except Exception as e:
            print(f"   Błąd w Screeningu ({e}). Pomijam ten etap.")
            filtered_cols = initial_cols
            n_after_screen = n_start

        # Zapamiętujemy stan po screeningu na wypadek błędu w etapie 2
        self.final_selected_cols = filtered_cols

        # ==========================================
        # ETAP 2: DECYZJA (PCA vs SFS)
        # ==========================================
        # LOGIKA HYBRYDOWA
        if n_after_screen <= SFS_CAP_THRESHOLD:
            # --- ŚCIEŻKA A: Mało cech -> SFS (Precyzja) ---
            print(f"-> Etap 2: Liczba cech ({n_after_screen}) <= {SFS_CAP_THRESHOLD}. Uruchamiam SFS Backward.")
            
            self.selection_mode = 'sfs'
            
            # est = LogisticRegression( # średnio wolny
            #     class_weight='balanced', 
            #     solver='liblinear',
            #     max_iter= 1000,
            #     C=0.1, # może być tunowane
            #     #tol=1e-3,
            #     random_state=self.random_state
            # )

            # est = ExtraTreesClassifier( # wolny
            #     n_estimators=20, 
            #     max_depth=7,
            #     n_jobs=-1, 
            #     class_weight='balanced',
            #     random_state=self.random_state
            # )

            est = KNeighborsClassifier(n_neighbors=4, n_jobs=-1) #szybki i całkiem dobry

            
            # from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            # est = LinearDiscriminantAnalysis()
            
            self.selector_model = SFS(
                est,
                k_features='best', # best lub parsimonious
                forward=False, 
                floating=False, 
                verbose=0,  # (2 = szczegółowa)
                scoring='roc_auc',
                cv=3,
                n_jobs=-1
            )
            
            try:
                self.selector_model.fit(X_curr, y)
                self.final_selected_cols = self.final_selected_cols = list(self.selector_model.k_feature_names_)
                print(f"-> SFS zakończony. Wybrano {len(self.final_selected_cols)} najlepszych cech.")
            except Exception as e:
                print(f"   Błąd SFS ({e}). Zostawiam wynik po screeningu.")
                self.selection_mode = 'filter_only'
                # final_selected_cols już jest ustawione na filtered_cols

        else:
            # --- ŚCIEŻKA B: Dużo cech -> PCA (Kompresja) ---
            print(f"-> Etap 2: Liczba cech ({n_after_screen}) > {SFS_CAP_THRESHOLD}. Uruchamiam PCA dla wydajności.")
            
            self.selection_mode = 'pca'
            
            # AUTOMATYCZNY DOBÓR:
            # To eliminuje współliniowość i szum, ale zostawia prawie cały sygnał.
            target_variance = 0.999
            
            # Bezpiecznik: PCA nie może stworzyć więcej komponentów niż mamy próbek
            # (choć sklearn z floatem i tak by to obsłużył, svd_solver='full' jest precyzyjny)
            self.pca_model = PCA(n_components=target_variance, random_state=self.random_state, svd_solver='full')
            
            # Uczymy PCA na kolumnach, które przeszły screening
            self.final_selected_cols = filtered_cols 
            
            # Fitujemy
            self.pca_model.fit(X_curr)
            
            n_comps = self.pca_model.n_components_
            var_explained = np.sum(self.pca_model.explained_variance_ratio_)
            
            print(f"-> PCA zakończone. Skompresowano {n_after_screen} cech do {n_comps} komponentów.")
            print(f"   Wyjaśniona wariancja: {var_explained:.4f} (Cel: {target_variance})")
        
        # ==========================================
        # ETAP 3: AKTUALIZACJA LISTY KOLUMN (KLUCZOWE!)
        # ==========================================
        if self.selection_mode == 'pca':
            # PCA tworzy zupełnie nowe kolumny numeryczne (PC1, PC2...)
            # Tracimy oryginalne kategorie i liczby.
            n_comps = self.pca_model.n_components_
            new_pca_cols = [f"PC{i+1}" for i in range(n_comps)]
            
            self.num_cols = new_pca_cols
            self.cat_cols = [] # Po PCA nie ma już kategorii
            
        else:
            # SFS lub Filter - tylko filtrujemy istniejące listy
            # Musimy zostawić tylko te, które są w final_selected_cols
            final_set = set(self.final_selected_cols)
            
            self.num_cols = [c for c in final_set if c in self.num_cols]
            self.cat_cols = [c for c in final_set if c in self.cat_cols]
        
        print(f"--- Selekcja Cech Zakończona: Tryb = {self.selection_mode} ---\n")
        print(f"Ostateczne kolumny:{self.num_cols + self.cat_cols}\n")

    def _transform_feature_selection(self, X):
        """Aplikuje selekcję/transformację zgodnie z ustalonym trybem."""
        if self.selection_mode is None:
            return X
            
        # Krok 1: Wybieramy kolumny po screeningu (wspólne dla wszystkich trybów)
        # Musimy sprawdzić, czy cols są w X (bo usuwanie kolinearności mogło namieszać, choć nie powinno)
        if hasattr(self, 'final_selected_cols') and self.final_selected_cols:
            valid_cols = [c for c in self.final_selected_cols if c in X.columns]
            
            # Jeśli tryb to PCA, to final_selected_cols to kolumny WEJŚCIOWE do PCA
            X_subset = X[valid_cols]
            
            if self.selection_mode == 'pca' and self.pca_model:
                X_pca = self.pca_model.transform(X_subset)
                cols = [f"PC{i+1}" for i in range(X_pca.shape[1])]
                return pd.DataFrame(X_pca, columns=cols, index=X.index)
            
            # Jeśli tryb to 'sfs' lub 'filter_only', to final_selected_cols to już wynik końcowy
            return X_subset

        return X



    def get_categorical_cols(self, X_processed):
        """
        Zwraca listę kolumn kategorycznych, które znajdują się w przetworzonym zbiorze danych.
        Przydatne dla CatBoost/XGBoost po selekcji cech (SFS).
        """
        if self.selection_mode == 'pca':
            # PCA zamienia wszystko na liczby (PC1, PC2...), więc nie ma już kategorii.
            return []
        
        # Sprawdzamy, które z oryginalnie wykrytych kategorii przetrwały usuwanie i selekcję
        # X_processed to DataFrame zwrócony przez transform()
        valid_cats = [col for col in self.cat_cols if col in X_processed.columns]
        return valid_cats

    def fit(self, X, y):
        # Zabezpieczenie na wypadek gdyby fit było wołane ręcznie bez process()
        if self.target_col is None and y is not None:
             if hasattr(y, 'name'): self.target_col = y.name
             else: self.target_col = 'target'

        X = X.copy()
        y_proc = self._process_target(y, fit=True)
        
        # Wczesne usuwanie bezużytecznych cech
        self._detect_useless_features(X)
        if self.cols_to_drop_early:
            X = X.drop(columns=self.cols_to_drop_early)
        
        # 1. Wykrywanie typów i daty
        self._detect_columns(X)
        X = self._process_dates_cyclical(X)
        
        current_cols = X.columns
        self.num_cols = [c for c in current_cols if c not in self.cat_cols]

        # braki w danych - imputacja
        if self.num_cols:
            self.imputer_num.fit(X[self.num_cols])
            self.cols_to_impute_num = self.num_cols.copy()
            X[self.num_cols] = self.imputer_num.transform(X[self.num_cols])

        if self.cat_cols:
            self.imputer_cat.fit(X[self.cat_cols])
            self.cols_to_impute_cat = self.cat_cols.copy()
            X[self.cat_cols] = self.imputer_cat.transform(X[self.cat_cols])

        # Skalowanie i transformacja Yeo-Johnson
        if self.num_cols:
            self.power_transformer.fit(X[self.num_cols])
            X[self.num_cols] = self.power_transformer.transform(X[self.num_cols])
        # Encoding kategorii
        if self.cat_cols:
            X[self.cat_cols] = X[self.cat_cols].astype(str)
            self.cat_encoder.fit(X[self.cat_cols])
            X[self.cat_cols] = self.cat_encoder.transform(X[self.cat_cols]).astype(int)
        
        # --- 2. Generowanie Cech (Feature Engineering) ---
        
        # A. Frequency Encoding (Uczymy się mapy i transformujemy X)
        self._fit_frequency_features(X)
        X = self._transform_frequency_features(X)
        # Aktualizacja listy numerycznej o nowe kolumny FREQ_
        new_freq_cols = [c for c in X.columns if c.startswith('FREQ_')]
        self.num_cols.extend(new_freq_cols)

        # B. KMeans
        if self.add_kmeans_features:
            self._fit_kmeans(X)
            X = self._transform_with_kmeans(X)

        # C. Interakcje (na podstawie num_cols + kmeans_cols)
        if self.add_poly_features:
            X = self._add_poly_features(X, y_proc)
            self.num_cols.extend(self.added_poly_cols)

        # D. Binning (Uczymy się przedziałów i transformujemy X)
        self._fit_binning_features(X)
        X = self._transform_binning_features(X)
        # Aktualizacja listy numerycznej o nowe kolumny BIN_
        # (Dla modeli drzewiastych traktujemy je jako numeryczne/ordinal)
        if self.binning_new_names:
            self.cat_cols.extend(self.binning_new_names)

        # --- 3. Czyszczenie (Współliniowość) ---
        # Robimy to PO wygenerowaniu wszystkiego, żeby usunąć np. interakcje skorelowane z oryginałem
        if self.remove_multicollinearity:
            X = self._remove_collinear(X)

        # --- 4. Selekcja Cech ---
        if self.feature_selection:
            self._fit_feature_selection(X, y_proc)
            X = self._transform_feature_selection(X)
        self.is_fitted = True
        
        return self

    def transform(self, X, y=None):
        if not self.is_fitted:
            raise Exception("Najpierw uruchom fit()!")
        
        X = X.copy()
        y_transformed = self._process_target(y, fit=False)
        
        if self.cols_to_drop_early:
            # errors='ignore' na wypadek gdyby w X (np. testowym) tych kolumn w ogóle nie było
            X = X.drop(columns=self.cols_to_drop_early, errors='ignore')

        X = self._process_dates_cyclical(X)
        
        # --- Imputacja, Skalowanie, Encoding ---
        valid_impute = [c for c in self.cols_to_impute_num if c in X.columns]
        if len(valid_impute) > 0:
            X[valid_impute] = self.imputer_num.transform(X[valid_impute])
            X[valid_impute] = self.power_transformer.transform(X[valid_impute])

        valid_impute = [c for c in self.cols_to_impute_cat if c in X.columns]
        if len(valid_impute) > 0:
            X[valid_impute] = self.imputer_cat.transform(X[valid_impute])
            X[valid_impute] = X[valid_impute].astype(str)
            X[valid_impute] = self.cat_encoder.transform(X[valid_impute]).astype(int)

        # --- Generowanie Cech (Kolejność jak w fit!) ---
        
        X = self._transform_frequency_features(X)

        if self.add_kmeans_features:
            X = self._transform_with_kmeans(X)

        if self.add_poly_features:
            X = self._transform_poly_features(X)

        X = self._transform_binning_features(X)

        # --- Czyszczenie (Współliniowość) ---
        if self.remove_multicollinearity and self.collinear_drop_cols:
            # Usuwamy te same kolumny co w fit
            cols_to_drop = [c for c in self.collinear_drop_cols if c in X.columns]
            X = X.drop(columns=cols_to_drop)

        # --- Selekcja ---
        if self.feature_selection:
            X = self._transform_feature_selection(X)
        
        if y is None:
            return X
        return X, y_transformed
    
    def fit_transform(self, X, y):
        """Uczy się, usuwa outliery (opcjonalnie) i zwraca gotowe dane."""
        # 1. Nauka parametrów (średnie, odchylenia, wagi SFS itd.)
        self.fit(X, y)
        
        # 2. Przygotowanie kopii roboczych
        X_proc = X.copy()
        # Zabezpieczenie: y może być None (np. w unsupervised), choć tu rzadko
        y_proc = y.copy() if y is not None else None
        
        # 3. Wykrywanie Outlierów (OPCJONALNE)
        if self.remove_outliers and self.original_num_cols:
            # Tworzymy TYMCZASOWĄ wersję danych tylko dla algorytmu IsolationForest.
            # Musimy zamienić daty i braki na liczby, żeby algorytm zadziałał.
            temp_X = X_proc.copy()
            if self.cols_to_drop_early:
                X_proc = X_proc.drop(columns=self.cols_to_drop_early)

            temp_X = self._process_dates_cyclical(temp_X)
            
            valid_num = [c for c in self.original_num_cols if c in temp_X.columns]
            if valid_num:
                temp_X[valid_num] = self.imputer_num.transform(temp_X[valid_num])
                temp_X[valid_num] = self.power_transformer.transform(temp_X[valid_num])
            
            # Wykrywamy outliery
            iso = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
            preds = iso.fit_predict(temp_X[valid_num])
            
            # Filtrujemy SUROWE dane (X_proc)
            mask = preds != -1
            print(f"Usunięto {sum(preds == -1)} wierszy (outliery).")
            
            X_proc = X_proc[mask]
            if y_proc is not None:
                # Uwaga: y musi być typu pandas Series/DataFrame lub numpy array, żeby obsłużyć maskowanie
                # Jeśli y to lista, trzeba zamienić: np.array(y_proc)[mask]
                if isinstance(y_proc, list):
                    y_proc = np.array(y_proc)[mask]
                else:
                    y_proc = y_proc[mask]

        # 4. FINALNE PRZETWARZANIE
        # Tutaj dzieje się magia: transform() bierze X_proc (który ma wciąż surowe daty),
        # przetwarza daty, imputuje braki, skaluje i selekcjonuje cechy.
        # Działa to niezależnie od tego, czy outliery były usuwane, czy nie.
        return self.transform(X_proc, y_proc)
    
