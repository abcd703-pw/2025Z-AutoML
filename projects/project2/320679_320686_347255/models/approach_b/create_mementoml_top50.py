import pandas as pd
import json
from pathlib import Path
import numpy as np


def load_mementoml_benchmark(filepath):
    print(f"Loading MementoML benchmark data from {filepath}")
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} benchmark results")
    print(f"Models: {df['model'].unique().tolist()}")
    print(f"Datasets: {len(df['dataset'].unique())} unique datasets")
    print(f"Total configurations tested: {len(df[['model', 'param_index']].drop_duplicates())}")
    return df


def load_param_file(params_dir, model_name):
    param_files = {
        'xgboost': 'xgboost_params.csv',
        'catboost': 'catboost_params.csv',
        'ranger': 'ranger_params.csv',
        'randomForest': 'randomForest_params.csv',
        'kknn': 'kknn_params.csv',
        'gbm': 'gbm_params.csv',
        'glmnet': 'glmnet_params.csv',
        'svm': 'svm_params.csv'
    }
    
    if model_name not in param_files:
        return None
    
    filepath = Path(params_dir) / param_files[model_name]
    if filepath.exists():
        return pd.read_csv(filepath)
    return None


def aggregate_performance(benchmark_df, metric='auc'):
    agg_df = benchmark_df.groupby(['model', 'param_index']).agg({
        metric: ['mean', 'std', 'count', 'min', 'max'],
        'time': 'mean'
    }).reset_index()
    
    agg_df.columns = ['model', 'param_index', f'{metric}_mean', f'{metric}_std', 
                      'n_evaluations', f'{metric}_min', f'{metric}_max', 'time_mean']
    
    print("\nEvaluations per configuration:")
    for model in sorted(agg_df['model'].unique()):
        model_df = agg_df[agg_df['model'] == model]
        n_configs = len(model_df)
        avg_evals = model_df['n_evaluations'].mean()
        min_evals = model_df['n_evaluations'].min()
        max_evals = model_df['n_evaluations'].max()
        print(f"  {model}: {n_configs} configs, {avg_evals:.0f} avg evals/config (range: {min_evals:.0f}-{max_evals:.0f})")
    
    # Calculate coefficient of variation
    agg_df['cv'] = agg_df[f'{metric}_std'] / agg_df[f'{metric}_mean']
    
    # Calculate range (max - min)
    agg_df['range'] = agg_df[f'{metric}_max'] - agg_df[f'{metric}_min']
    
    # Create a composite score that balances mean performance and stability
    # Higher mean = better, Lower CV = better (more stable)
    # Normalize both to 0-1 range, then combine
    mean_normalized = (agg_df[f'{metric}_mean'] - agg_df[f'{metric}_mean'].min()) / \
                      (agg_df[f'{metric}_mean'].max() - agg_df[f'{metric}_mean'].min())
    
    cv_normalized = 1 - ((agg_df['cv'] - agg_df['cv'].min()) / 
                         (agg_df['cv'].max() - agg_df['cv'].min()))
    
    # Composite score: 70% mean performance, 30% stability
    agg_df['composite_score'] = 0.7 * mean_normalized + 0.3 * cv_normalized
    
    agg_df = agg_df.sort_values('composite_score', ascending=False)
    
    return agg_df


def select_top_configs(agg_df, max_total=50, max_per_model=None, metric='auc', exclude_models=None):
    if max_per_model is None:
        max_per_model = 10
    
    if isinstance(max_per_model, int):
        default_limit = max_per_model
        max_per_model = {}
    else:
        default_limit = 10
    
    if exclude_models:
        print(f"Excluding models: {', '.join(exclude_models)}")
        agg_df = agg_df[~agg_df['model'].isin(exclude_models)]
    
    selected_configs = []
    selected_param_indices = set()  # Track selected (model, param_index) to avoid duplicates
    model_counts = {}
    
    sorted_df = agg_df.sort_values('composite_score', ascending=False)
    
    for _, row in sorted_df.iterrows():
        model = row['model']
        param_index = int(row['param_index'])
        
        # Treat ranger and randomForest as the same model type
        model_type = 'RandomForest' if model in ['ranger', 'randomForest'] else model
        
        # Check for duplicates
        config_key = (model, param_index)
        if config_key in selected_param_indices:
            continue
        
        if len(selected_configs) >= max_total:
            break
        
        limit = max_per_model.get(model, max_per_model.get(model_type, default_limit))
        
        if model_counts.get(model_type, 0) >= limit:
            continue
        
        selected_configs.append({
            'model': model,
            'param_index': param_index,
            f'{metric}_mean': float(row[f'{metric}_mean']),
            f'{metric}_std': float(row[f'{metric}_std']),
            f'{metric}_min': float(row[f'{metric}_min']),
            f'{metric}_max': float(row[f'{metric}_max']),
            'cv': float(row['cv']),
            'composite_score': float(row['composite_score']),
            'n_evaluations': int(row['n_evaluations']),
            'time_mean': float(row['time_mean'])
        })
        
        selected_param_indices.add(config_key)
        model_counts[model_type] = model_counts.get(model_type, 0) + 1
    
    print(f"\nSelected {len(selected_configs)} configurations:")
    for model, count in sorted(model_counts.items()):
        print(f"  {model}: {count}")
    
    return selected_configs


def map_xgboost_params(row):
    params = {
        'n_estimators': int(row['nrounds']),
        'learning_rate': float(row['eta']),
        'max_depth': int(row['max_depth']),
        'subsample': float(row['subsample']),
        'colsample_bytree': float(row['colsample_bytree']),
        'colsample_bylevel': float(row['colsample_bylevel']),
        'min_child_weight': float(row['min_child_weight']),
        'booster': str(row['booster']),
        'random_state': 42
    }
    
    if params['booster'] == 'gbtree':
        params['tree_method'] = 'hist'
    
    return params


def map_catboost_params(row):
    return {
        'iterations': int(row['iterations']),
        'depth': int(row['depth']),
        'l2_leaf_reg': float(row['l2_leaf_reg']),
        'bagging_temperature': float(row['bagging_temperature']),
        'learning_rate': float(row['learning_rate']),
        'random_state': 42,
        'verbose': 0
    }


def map_ranger_params(row):
    params = {
        'n_estimators': int(row['num.trees']),
        'min_samples_leaf': int(row['min.node.size']),
        'bootstrap': bool(row['replace']),
        'random_state': 42
    }
    
    if row['splitrule'] == 'gini':
        params['criterion'] = 'gini'
    elif row['splitrule'] == 'extratrees':
        params['criterion'] = 'gini'
        params['max_features'] = 'sqrt'
    
    return params


def map_randomforest_params(row):
    params = {
        'n_estimators': int(row['ntree']),
        'bootstrap': bool(row['replace']),
        'random_state': 42
    }
    
    if 'nodesize' in row and not pd.isna(row['nodesize']):
        params['min_samples_leaf'] = int(row['nodesize'])
    
    return params


def map_kknn_params(row):
    params = {
        'n_neighbors': int(row['k']),
        'weights': 'distance'
    }
    
    if 'distance' in row:
        distance = int(row['distance'])
        if distance == 1:
            params['metric'] = 'manhattan'
        elif distance == 2:
            params['metric'] = 'euclidean'
        else:
            params['metric'] = 'minkowski'
            params['p'] = distance
    
    return params


def map_gbm_params(row):
    return {
        'n_estimators': int(row['n.trees']),
        'learning_rate': float(row['shrinkage']),
        'max_depth': int(row['interaction.depth']),
        'min_samples_leaf': int(row['n.minobsinnode']),
        'subsample': float(row['bag.fraction']),
        'random_state': 42
    }


def map_glmnet_params(row):
    alpha = float(row['alpha'])
    
    params = {
        'max_iter': 1000,
        'random_state': 42
    }
    
    if alpha == 0:
        params['penalty'] = 'l2'
        params['solver'] = 'lbfgs'
    elif alpha == 1:
        params['penalty'] = 'l1'
        params['solver'] = 'saga'
    else:
        params['penalty'] = 'elasticnet'
        params['solver'] = 'saga'
        params['l1_ratio'] = alpha
    
    if 'lambda' in row and not pd.isna(row['lambda']) and row['lambda'] > 0:
        params['C'] = 1.0 / float(row['lambda'])
    else:
        params['C'] = 1.0
    
    return params


def map_svm_params(row):
    params = {
        'probability': True,
        'random_state': 42
    }
    
    if 'kernel' in row and not pd.isna(row['kernel']):
        params['kernel'] = str(row['kernel'])
    
    if 'C' in row and not pd.isna(row['C']):
        params['C'] = float(row['C'])
    
    if 'gamma' in row and not pd.isna(row['gamma']):
        gamma_val = row['gamma']
        if isinstance(gamma_val, str) and gamma_val in ['scale', 'auto']:
            params['gamma'] = gamma_val
        else:
            params['gamma'] = float(gamma_val)
    
    if 'degree' in row and not pd.isna(row['degree']):
        params['degree'] = int(row['degree'])
    
    return params


def create_model_configs(selected_configs, params_dir, metric='auc', max_configs=50):
    model_mappers = {
        'xgboost': (map_xgboost_params, 'xgboost.XGBClassifier'),
        'catboost': (map_catboost_params, 'catboost.CatBoostClassifier'),
        'ranger': (map_ranger_params, 'sklearn.ensemble.RandomForestClassifier'),
        'randomForest': (map_randomforest_params, 'sklearn.ensemble.RandomForestClassifier'),
        'kknn': (map_kknn_params, 'sklearn.neighbors.KNeighborsClassifier'),
        'gbm': (map_gbm_params, 'sklearn.ensemble.GradientBoostingClassifier'),
        'glmnet': (map_glmnet_params, 'sklearn.linear_model.LogisticRegression'),
        'svm': (map_svm_params, 'sklearn.svm.SVC')
    }
    
    configs = []
    seen_params = {}  # Track parameter combinations per model type to avoid duplicates
    
    config_index = 0
    rank = 1
    
    while len(configs) < max_configs and config_index < len(selected_configs):
        selected = selected_configs[config_index]
        config_index += 1
        
        model_name = selected['model']
        param_index = selected['param_index']
        
        if model_name not in model_mappers:
            print(f"  Skipping {model_name} - no mapper defined")
            continue
        
        params_df = load_param_file(params_dir, model_name)
        if params_df is None:
            print(f"  Skipping {model_name} - parameter file not found")
            continue
        
        param_row = params_df[params_df['param_index'] == param_index]
        if param_row.empty:
            print(f"  Skipping: {model_name} param_index {param_index} not found")
            continue
        
        param_row = param_row.iloc[0]
        
        mapper_func, sklearn_class = model_mappers[model_name]
        try:
            params = mapper_func(param_row)
            
            # Create a hashable representation of parameters to check for duplicates
            # Convert params dict to a sorted tuple of items
            param_signature = tuple(sorted(params.items()))
            
            # Check if we've already added a config with these exact parameters
            model_type = sklearn_class.split('.')[-1]
            if model_type not in seen_params:
                seen_params[model_type] = set()
            
            if param_signature in seen_params[model_type]:
                print(f"  Skipping: {model_name} param_index {param_index} (duplicate parameters)")
                continue
            
            config = {
                'name': f"{model_name}_top{rank}_param{param_index}",
                'class': sklearn_class,
                'params': params,
                'metadata': {
                    f'{metric}_mean': selected[f'{metric}_mean'],
                    f'{metric}_std': selected[f'{metric}_std'],
                    f'{metric}_min': selected[f'{metric}_min'],
                    f'{metric}_max': selected[f'{metric}_max'],
                    'cv': selected['cv'],
                    'composite_score': selected['composite_score'],
                    'n_evaluations': selected['n_evaluations'],
                    'param_index': param_index,
                    'rank': rank
                }
            }
            
            configs.append(config)
            seen_params[model_type].add(param_signature)
            print(f"[{rank}] {config['name']} (auc={selected[f'{metric}_mean']:.4f}±{selected[f'{metric}_std']:.4f}, cv={selected['cv']:.4f}, score={selected['composite_score']:.4f})")
            rank += 1
            
        except Exception as e:
            print(f"Error creating config for {model_name} param {param_index}: {e}")
            continue
    
    return configs


def main():
    project_root = Path(__file__).parent.parent.parent
    benchmark_file = project_root / 'parameters' / 'MementoML.csv'
    params_dir = project_root / 'parameters'
    output_file = project_root / 'models' / 'approach_b' / 'models_mementoml_top50.json'
    
    benchmark_df = load_mementoml_benchmark(benchmark_file)
    
    agg_df = aggregate_performance(benchmark_df, metric='auc')
    print(f"   Total unique configurations: {len(agg_df)}")
    print(f"   Best AUC (mean): {agg_df['auc_mean'].max():.4f}")
    print(f"   Best composite score: {agg_df['composite_score'].max():.4f}")
    print(f"   Most stable (lowest CV): {agg_df['cv'].min():.4f}")
    
    # Exclude models that have installation issues (catboost) or no sklearn equivalents or missing param files
    exclude_models = ['catboost', 'agtboost', 'bartMachine', 'svm']
    
    max_per_model = {
        'xgboost': 15,
        'kknn': 10,  # resulting is 5
        'randomForest': 11, # resulting is 10
        'gbm': 11,  # resulting is 10
    }
    
    selected_configs = select_top_configs(
        agg_df, 
        max_total=70,  # Select extra to ensure we get 50 valid ones after duplicates/missing
        max_per_model=max_per_model, 
        metric='auc',
        exclude_models=exclude_models
    )
    
    model_configs = create_model_configs(selected_configs, params_dir, metric='auc', max_configs=50)
    print(f"\nSaving configurations to {output_file}")
    
    clean_configs = []
    for config in model_configs:
        clean_config = {
            'name': config['name'],
            'class': config['class'],
            'params': config['params']
        }
        clean_configs.append(clean_config)
    
    with open(output_file, 'w') as f:
        json.dump(clean_configs, f, indent=2)
    
    metadata_file = project_root / 'models' / 'approach_b' / 'models_mementoml_top50_with_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(model_configs, f, indent=2)
    print(f"Saved detailed version with metadata to {metadata_file.name}")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total configurations: {len(model_configs)}")
    
    model_counts = {}
    for config in model_configs:
        model_class = config['class'].split('.')[-1]
        model_counts[model_class] = model_counts.get(model_class, 0) + 1
    
    print("\nBreakdown by model type:")
    for model_type, count in sorted(model_counts.items()):
        print(f"  {model_type}: {count}")
    
    print(f"\nTop 5 configurations (ranked by composite score):")
    for i, config in enumerate(model_configs[:5], 1):
        meta = config['metadata']
        auc_mean = meta.get('auc_mean', 0)
        auc_std = meta.get('auc_std', 0)
        cv = meta.get('cv', 0)
        score = meta.get('composite_score', 0)
        print(f"  {i}. {config['name']}: AUC={auc_mean:.4f}±{auc_std:.4f}, CV={cv:.4f}, Score={score:.4f}")
    
if __name__ == '__main__':
    main()
