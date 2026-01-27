import importlib
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import train_test_split

import torch
import functools

# --- FIX DLA TABPFN NA PYTORCH 2.6+ ---
# Ten kod automatycznie wymusza weights_only=False,
# naprawiając błąd bez ingerencji w pliki biblioteki.
if hasattr(torch, 'load'):
    _original_torch_load = torch.load


    @functools.wraps(_original_torch_load)
    def _patched_load(*args, **kwargs):
        # Sprawdzamy, czy funkcja obsługuje argument 'weights_only'
        # i jeśli nie został podany, ustawiamy go na False
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        return _original_torch_load(*args, **kwargs)


    torch.load = _patched_load
class ModelWrapper:
    def __init__(self, model_config):
        self.model_config = model_config
        self.model = self._initialize_model()
        self.params = self.model_config.get("params", {}) or {}

    def _initialize_model(self):
        fqcn = self.model_config.get("class")
        if not fqcn:
            raise ValueError(f"Missing 'class' field in model configuration: {self.model_config}")

        module_name, cls_name = fqcn.rsplit(".", 1)
        module = importlib.import_module(module_name)
        Cls = getattr(module, cls_name)

        params = self.model_config.get("params", {}) or {}
        try:
            return Cls(**params)
        except Exception:
            raise ValueError(f"Failed to initialize model {fqcn} with parameters: {params}")

    def fit(self, X_train, y_train):
        early_stopping_rounds = self.model_config.get("params", {}).get("early_stopping_rounds", None)

        if early_stopping_rounds:
            X_train_split, X_val, y_train_split, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )
            self.model.fit(
                X_train_split, y_train_split,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=early_stopping_rounds,
                verbose=False
            )
        else:
            self.model.fit(X_train, y_train)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def evaluate(self, X_test, y_test):
        probs = self.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probs)
        brier = brier_score_loss(y_test, probs)
        return {"auc": auc, "brier": brier}

    def get_params(self, deep=True):
        """Return model parameters for compatibility with scikit-learn."""
        return {"class": self.model_config.get("class"), **self.params}

    def set_params(self, **params):
        """Set model parameters for compatibility with scikit-learn."""
        self.params.update(params)
        self.model = self._initialize_model()