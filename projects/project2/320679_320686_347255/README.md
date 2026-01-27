# MiniAutoML

A simplified automated machine learning (AutoML) system for binary classification tasks. MiniAutoML automatically selects and trains the best machine learning models from a configurable set of classifiers, and can create ensemble models for improved performance.

## Project Structure

```
MiniAutoML/
├── README.md
├── requirements.txt
├── data/
│   ├── data.csv          # Full dataset
│   ├── X.csv             # Feature data
│   └── y.csv             # Target labels
├── models/
│   └── models.json       # Model configurations
├── results/              # Predictions and experiments results
├── docs/
│   └── report.pdf        # Project report
└── src/
    ├── automl.py         # Core MiniAutoML implementation
    ├── config.py         # Configuration settings
    ├── generate_data.py  # Data generation utilities
    ├── main.py           # Main execution script
    └── utils.py          # Utility functions
```

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd MiniAutoML
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from automl import MiniAutoML
from utils import load_data, load_models_config

# Load data
X_train, X_test, y_train, y_test = load_data()

# Load model configurations
models_config = load_models_config()

# Create and fit AutoML with single best model
automl = MiniAutoML(models_config=models_config, n_ensemble=1)
automl.fit(X_train, y_train)

# Make predictions
predictions = automl.predict(X_test)
probabilities = automl.predict_proba(X_test)
```

### Model Configuration

Models are configured in `models/models.json`. Example:

```json
[
  {
    "name": "RandomForest",
    "class": "sklearn.ensemble.RandomForestClassifier",
    "params": {
      "n_estimators": 100,
      "random_state": 42
    }
  },
  {
    "name": "LogisticRegression",
    "class": "sklearn.linear_model.LogisticRegression",
    "params": {
      "max_iter": 1000,
      "random_state": 42
    }
  }
]
```
