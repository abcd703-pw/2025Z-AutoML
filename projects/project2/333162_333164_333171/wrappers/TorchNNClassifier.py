import numpy as np
from sklearn.discriminant_analysis import unique_labels
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split

class TorchNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, 
                 hidden_size=128, 
                 num_layers=2, 
                 dropout_prob=0.1, 
                 activation="relu", 
                 use_batchnorm=True, 
                 learning_rate=1e-3, 
                 weight_decay=1e-5,
                 epochs=20,          # Domyślna wartość, jeśli nie podano w JSON
                 batch_size=64,
                 random_state=None):
        
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.dropout_prob = dropout_prob
        self.activation = activation
        self.use_batchnorm = use_batchnorm
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        
        self.classes_ = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_model(self, input_dim):
        layers = []
        in_features = input_dim
        
        # Funkcja aktywacji
        act_fn = nn.ReLU() if self.activation.lower() == "relu" else nn.Tanh()
        
        for _ in range(self.num_layers):
            layers.append(nn.Linear(in_features, self.hidden_size))
            if self.use_batchnorm:
                layers.append(nn.BatchNorm1d(self.hidden_size))
            layers.append(act_fn)
            layers.append(nn.Dropout(self.dropout_prob))
            in_features = self.hidden_size
            
        # Warstwa wyjściowa (dla klasyfikacji binarnej: 2 wyjścia lub 1 + sigmoid)
        # Tutaj używamy 2 wyjść + CrossEntropyLoss dla stabilności
        layers.append(nn.Linear(self.hidden_size, 2))
        
        return nn.Sequential(*layers).to(self.device)

    def fit(self, X, y):

        self.classes_ = unique_labels(y)

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        # Konwersja danych
        X_tensor = torch.FloatTensor(X.values if hasattr(X, 'values') else X).to(self.device)
        y_tensor = torch.LongTensor(y.values if hasattr(y, 'values') else y).to(self.device)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Budowa modelu
        self.model = self._build_model(input_dim=X.shape[1])
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        criterion = nn.CrossEntropyLoss()
        
        self.model.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        return self

    def predict_proba(self, X):
        self.model.eval()
        X_tensor = torch.FloatTensor(X.values if hasattr(X, 'values') else X).to(self.device)
        
        with torch.no_grad():
            logits = self.model(X_tensor)
            probs = torch.softmax(logits, dim=1)
            
        return probs.cpu().numpy()

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)