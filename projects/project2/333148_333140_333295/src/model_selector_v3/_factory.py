from copy import deepcopy
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

# Próba importu bibliotek zewnętrznych
try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:
    XGBClassifier, XGBRegressor = None, None

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except ImportError:
    LGBMClassifier, LGBMRegressor = None, None

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
except ImportError:
    CatBoostClassifier, CatBoostRegressor = None, None


class ModelFactory:
    @staticmethod
    def create_model(config, task, random_state=42, n_jobs=-1):
        model_type = config['type'].lower()
        params = deepcopy(config.get('params', {}))

        # Konwersja stringów "True"/"False" na boole (częsty problem z JSON)
        for k, v in params.items():
            if v == "True": params[k] = True
            if v == "False": params[k] = False

        # --- XGBOOST ---
        if 'xgboost' in model_type:
            if XGBClassifier is None:
                raise ImportError("Biblioteka XGBoost nie jest zainstalowana!")

            Cls = XGBClassifier if task == 'classification' else XGBRegressor
            return Cls(**params, random_state=random_state, n_jobs=n_jobs, verbosity=0)

        # --- LIGHTGBM ---
        elif 'lgbm' in model_type or 'lightgbm' in model_type:
            if LGBMClassifier is None:
                raise ImportError("Biblioteka LightGBM nie jest zainstalowana!")

            Cls = LGBMClassifier if task == 'classification' else LGBMRegressor
            # LightGBM używa verbose=-1 żeby nie spamować
            return Cls(**params, random_state=random_state, n_jobs=n_jobs, verbose=-1)

        # --- CATBOOST ---
        elif 'catboost' in model_type:
            if CatBoostClassifier is None:
                raise ImportError("Biblioteka CatBoost nie jest zainstalowana!")

            Cls = CatBoostClassifier if task == 'classification' else CatBoostRegressor
            # CatBoost ma thread_count zamiast n_jobs i allow_writing_files=False
            return Cls(**params, random_state=random_state, thread_count=n_jobs, verbose=0, allow_writing_files=False)

        # --- RANDOM FOREST ---
        elif 'rf' in model_type or 'randomforest' in model_type or 'extratrees' in model_type:
            Cls = RandomForestClassifier if task == 'classification' else RandomForestRegressor
            return Cls(**params, random_state=random_state, n_jobs=n_jobs)

        # --- LINEAR / LOGISTIC ---
        elif 'linear' in model_type or 'logistic' in model_type or 'regression' in model_type:
            if task == 'classification':
                return LogisticRegression(**params, random_state=random_state, n_jobs=n_jobs, solver='lbfgs')
            else:
                return LinearRegression(**params, n_jobs=n_jobs)

        # --- KNN ---
        elif 'knn' in model_type or 'neighbor' in model_type:
            Cls = KNeighborsClassifier if task == 'classification' else KNeighborsRegressor
            return Cls(**params, n_jobs=n_jobs)

        raise ValueError(f"Unknown model type: {model_type}")