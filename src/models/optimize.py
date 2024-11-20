# src/models/optimize.py
import optuna
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from typing import Dict, List, Tuple

def prepare_data(train_features: pd.DataFrame, train_labels: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare data for modeling"""
    # Merge features with labels
    train_data = train_features.merge(train_labels, on='uid', how='inner')
    
    # Drop non-feature columns
    features_to_drop = ['uid', 'year']
    X = train_data.drop(features_to_drop + ['composite_score'], axis=1)
    y = train_data['composite_score']
    
    return X, y

def preprocess_data(df: pd.DataFrame, is_training: bool = True, encoders: Dict = None) -> Tuple[pd.DataFrame, Dict]:
    """Preprocess features"""
    df = df.copy()
    
    # Handle missing values
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # Fill numeric missing values with median
    for col in numeric_cols:
        if is_training:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
        else:
            df[col] = df[col].fillna(encoders[f'{col}_median'])
    
    # Fill categorical missing values with mode
    for col in categorical_cols:
        if is_training:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
        else:
            df[col] = df[col].fillna(encoders[f'{col}_mode'])
    
    # Label encode categorical columns
    if is_training:
        encoders = {}
        for col in categorical_cols:
            df[col] = pd.Categorical(df[col]).codes
            encoders[f'{col}_categories'] = pd.Categorical(df[col]).categories
        
        # Store medians and modes
        for col in numeric_cols:
            encoders[f'{col}_median'] = df[col].median()
        for col in categorical_cols:
            encoders[f'{col}_mode'] = df[col].mode()[0]
    else:
        for col in categorical_cols:
            df[col] = pd.Categorical(df[col], categories=encoders[f'{col}_categories']).codes
    
    return df, encoders

def objective(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> float:
    """Optuna objective function for hyperparameter optimization"""
    
    # Define hyperparameter search space
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'verbosity': -1
    }
    
    # K-fold cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Train model
        model = lgb.train(
            params,
            train_data,
            valid_sets=[train_data, val_data],
            num_boost_round=10000,
            callbacks=[
                lgb.early_stopping(100),
                lgb.log_evaluation(0)
            ]
        )
        
        # Predict and evaluate
        val_preds = model.predict(X_val)
        score = np.sqrt(mean_squared_error(y_val, val_preds))
        scores.append(score)
    
    # Return mean CV score
    return np.mean(scores)

def optimize_hyperparameters(X: pd.DataFrame, y: pd.Series, n_trials: int = 100) -> Dict:
    """Run hyperparameter optimization"""
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X, y), n_trials=n_trials)
    
    print('\nBest trial:')
    trial = study.best_trial
    print(f'  RMSE: {trial.value:.4f}')
    print('\nBest hyperparameters:')
    for key, value in trial.params.items():
        print(f'  {key}: {value}')
    
    return study.best_params

def main():
    """Main optimization function"""
    print("Loading data...")
    data_dir = Path('data/raw')
    train_features = pd.read_csv(data_dir / 'train_features.csv')
    train_labels = pd.read_csv(data_dir / 'train_labels.csv')
    
    print("Preparing data...")
    X, y = prepare_data(train_features, train_labels)
    
    print("Preprocessing data...")
    X_processed, encoders = preprocess_data(X)
    
    print("\nStarting hyperparameter optimization...")
    best_params = optimize_hyperparameters(X_processed, y, n_trials=100)
    
    # Save best parameters
    pd.DataFrame([best_params]).to_csv('models/best_params.csv', index=False)
    print("\nBest parameters saved to 'models/best_params.csv'")

if __name__ == "__main__":
    main()