"""Hyperparameter optimization for MiniAutoML using Optuna on multiple OpenML datasets."""

import os
import optuna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from automl import MiniAutoML
from utils import load_models_config
from config import RANDOM_STATE, MODELS_CONFIG

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

OPENML_DATASET_IDS = [
    37,    # diabetes
    1067,  # kc1
    44,    # spambase
    1461,  # bank-marketing
    1590,  # adult
]


def load_openml_dataset(dataset_id, test_size=0.25):
    """Load and prepare an OpenML dataset for binary classification."""
    
    print(f"Loading OpenML dataset {dataset_id}...")
    
    dataset = fetch_openml(data_id=dataset_id, as_frame=True, parser='auto')
    X = dataset.data
    y = dataset.target
    y = pd.factorize(y)[0]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"Dataset {dataset_id} loaded: {X_train.shape[0]} train samples, {X_test.shape[0]} test samples")
    
    return X_train, X_test, y_train, y_test


def evaluate_configuration(config, models_config, datasets):
    """Evaluate a MiniAutoML configuration on multiple datasets. 
    Returns mean balanced accuracy across all datasets."""
    scores = []
    
    for i, (X_train, X_test, y_train, y_test) in enumerate(datasets):
        try:
            automl = MiniAutoML(
                models_config=models_config,
                selection_strategy=config['selection_strategy'],
                cv=config['cv'],
                scoring='balanced_accuracy',
                n_ensemble=config['n_ensemble'],
                ensemble_strategy=config['ensemble_strategy'],
                time_limit=config.get('time_limit', None),
                stability_threshold=config['stability_threshold']
            )
            
            print(f"  Evaluating on dataset {i+1}...")
            automl.fit(X_train, y_train)
            print("  Training completed.")
            y_pred = automl.predict(X_test)
            
            score = balanced_accuracy_score(y_test, y_pred)
            scores.append(score)
            
            print(f"  Dataset {i+1}: Balanced Accuracy = {score:.4f}")
            
        except Exception as e:
            print(f"  Dataset {i+1}: Failed with error: {str(e)}")
            # Use a low score for failed evaluations
            scores.append(0.5)
    
    mean_score = np.mean(scores)
    return mean_score


def plot_optimization_results(study, save_dir='plots'):
    """Generate and save visualization plots for the optimization study."""
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Optimization History Plot
    plt.figure(figsize=(12, 6))
    trials_df = study.trials_dataframe()
    
    plt.plot(trials_df['number'], trials_df['value'], 'o-', alpha=0.6, label='Trial Score')
    plt.plot(trials_df['number'], trials_df['value'].cummax(), 'r-', linewidth=2, label='Best Score')
    plt.xlabel('Trial Number', fontsize=22)
    plt.ylabel('Balanced Accuracy', fontsize=22)
    #plt.title('Optimization History', fontsize=14, fontweight='bold')
    plt.legend(fontsize=18)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/optimization_history.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {save_dir}/optimization_history.png")
    plt.close()
    
    # Parameter Importance Plot
    importance = optuna.importance.get_param_importances(study)
    
    plt.figure(figsize=(12, 6))
    params = list(importance.keys())
    values = list(importance.values())
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(params)))
    
    plt.barh(params, values, color=colors, edgecolor='black')
    plt.xlabel('Importance', fontsize=22)
    plt.ylabel('Parameter', fontsize=22)
    plt.yticks(fontsize=18)
    plt.xticks(fontsize=16)
    #plt.title('Parameter Importance', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/parameter_importance.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {save_dir}/parameter_importance.png")
    plt.close()
    
    # Top 10 Configurations Plot
    top_trials = sorted(study.trials, key=lambda t: t.value if t.state == optuna.trial.TrialState.COMPLETE else -1, reverse=True)[:10]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    trial_numbers = [t.number for t in top_trials]
    trial_scores = [t.value for t in top_trials]
    
    colors = plt.cm.RdYlGn(np.linspace(0.5, 1, len(trial_numbers)))
    bars = ax.barh(range(len(trial_numbers)), trial_scores, color=colors, edgecolor='black')
    
    ax.set_yticks(range(len(trial_numbers)))
    ax.set_yticklabels([f'Trial {n}' for n in trial_numbers], fontsize=16)
    ax.set_xlabel('Balanced Accuracy', fontsize=22)
    #ax.set_title('Top 10 Configurations', fontsize=22, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, trial_scores)):
        ax.text(score + 0.002, bar.get_y() + bar.get_height()/2, 
                f'{score:.4f}', va='center', fontsize=16)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/top_configurations.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {save_dir}/top_configurations.png")
    plt.close()
    
    print(f"\nAll plots saved to '{save_dir}/' directory")


def save_study(study, filename='optuna_study.pkl'):
    """Save the complete Optuna study."""
    
    joblib.dump(study, filename)
    print(f"Study saved to '{filename}'")
    
    trials_df = study.trials_dataframe()
    csv_filename = filename.replace('.pkl', '_trials.csv')
    trials_df.to_csv(csv_filename, index=False)
    print(f"Trials dataframe saved to '{csv_filename}'")


def load_study(filename='optuna_study.pkl'):
    """Load a saved Optuna study."""
    
    study = joblib.load(filename)
    print(f"Study loaded from '{filename}'")
    print(f"Study contains {len(study.trials)} trials")
    print(f"Best value: {study.best_value:.4f}")
    
    return study


def objective(trial, models_config, datasets):
    """Optuna objective function to optimize MiniAutoML configuration."""

    selection_strategy = trial.suggest_categorical(
        'selection_strategy', 
        ['cross_val', 'stability', 'heuristic']
    )
    
    cv = trial.suggest_int('cv', 3, 5)
    n_ensemble = trial.suggest_int('n_ensemble', 1, 5)
    
    if n_ensemble > 1:
        ensemble_strategy = trial.suggest_categorical(
            'ensemble_strategy', 
            ['soft_voting', 'hard_voting', 'stacking']
        )
    else:
        ensemble_strategy = 'soft_voting'
    
    if selection_strategy == 'stability':
        stability_threshold = trial.suggest_float('stability_threshold', 0.0, 1.0)
    else:
        stability_threshold = 0.1
    
    config = {
        'selection_strategy': selection_strategy,
        'cv': cv,
        'n_ensemble': n_ensemble,
        'ensemble_strategy': ensemble_strategy,
        'stability_threshold': stability_threshold,
        'time_limit': 600  # 10 minutes per dataset
    }
    
    print(f"\nTrial {trial.number}:")
    print(f"  Config: {config}")
    
    mean_score = evaluate_configuration(config, models_config, datasets)
    
    print(f"  Mean Balanced Accuracy: {mean_score:.4f}")
    
    return mean_score


def optimize_automl_config(n_trials=50, n_datasets=5):
    """Run Optuna optimization to find best MiniAutoML configuration."""
    
    print("="*80)
    print("MiniAutoML Hyperparameter Optimization using Optuna")
    print("="*80)
    
    print("\nLoading models configuration...")
    models_config = load_models_config(MODELS_CONFIG)
    print(f"Loaded {len(models_config)} model configurations")
    
    print("\nLoading OpenML datasets...")
    datasets = []
    dataset_ids_to_use = OPENML_DATASET_IDS[:n_datasets]
    
    for dataset_id in dataset_ids_to_use:
        dataset = load_openml_dataset(dataset_id)
        datasets.append(dataset)
    
    print(f"\nSuccessfully loaded {len(datasets)} datasets")
    
    print(f"\nStarting Optuna optimization with {n_trials} trials...")
    study = optuna.create_study(
        direction='maximize',
        study_name='miniAutoML_optimization',
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
    )
    
    study.optimize(
        lambda trial: objective(trial, models_config, datasets),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    print("\n" + "="*80)
    print("Optimization Complete!")
    print("="*80)
    print(f"\nBest Trial: {study.best_trial.number}")
    print(f"Best Mean Balanced Accuracy: {study.best_value:.4f}")
    print("\nBest Configuration:")
    for param, value in study.best_params.items():
        print(f"  {param}: {value}")
    
    print("\n" + "-"*80)
    print("Parameter Importance:")
    try:
        importance = optuna.importance.get_param_importances(study)
        for param, imp in importance.items():
            print(f"  {param}: {imp:.4f}")
    except Exception as e:
        print(f"  Could not calculate parameter importance: {str(e)}")
    
    print("\n" + "-"*80)
    print("Generating visualization plots...")
    plot_optimization_results(study, save_dir='results/plots')
    
    print("\n" + "-"*80)
    print("Saving study object...")
    save_study(study, filename='results/optuna_study.pkl')
    
    return study.best_params


if __name__ == "__main__":
    best_config = optimize_automl_config(n_trials=30, n_datasets=2)
        
    with open('results/best_config.json', 'w') as f:
       json.dump(best_config, f, indent=2)
    
    print("\nBest configuration saved to 'results/best_config.json'")
    
    # study = load_study(filename='results/optuna_study.pkl')
    # plot_optimization_results(study, save_dir='results/plots')