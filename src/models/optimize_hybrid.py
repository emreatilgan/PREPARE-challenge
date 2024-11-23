import optuna
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from typing import Dict, List, Tuple
from src.features.build_features import FeatureEngineerHybrid

def encode_categorical_features(df: pd.DataFrame, encoders: Dict = None, is_training: bool = True) -> Tuple[pd.DataFrame, Dict]:
    """Encode categorical features"""
    df = df.copy()
    if encoders is None:
        encoders = {}
    
    # Handle categorical features
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        if is_training:
            encoders[col] = LabelEncoder()
            df[col] = encoders[col].fit_transform(df[col].astype(str))
        else:
            # Handle unseen categories for test data
            df[col] = df[col].astype(str)
            df[col] = df[col].map(lambda x: x if x in encoders[col].classes_ else encoders[col].classes_[0])
            df[col] = encoders[col].transform(df[col])
    
    return df, encoders

def prepare_data(train_features: pd.DataFrame, train_labels: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare data for modeling"""
    # Merge features with labels
    train_data = train_features.merge(
        train_labels,
        on='uid',
        how='inner'
    )
    
    # Drop non-feature columns
    features_to_drop = ['uid', 'year', 'composite_score']
    X = train_data.drop([col for col in features_to_drop if col in train_data.columns], axis=1)
    y = train_data['composite_score']
    
    # Encode categorical features
    X, _ = encode_categorical_features(X, is_training=True)
    
    return X, y

def suggest_hyperparameters(trial: optuna.Trial) -> Dict:
    """Suggest hyperparameters for LightGBM"""
    return {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        
        # Tree-related parameters
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
        
        # Learning parameters
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        
        # Sampling parameters
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        
        # Regularization parameters
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.001, 0.1),
        
        # Additional parameters
        'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 10.0, log=True)
    }

def objective(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> float:
    """Optuna objective function for hyperparameter optimization"""
    params = suggest_hyperparameters(trial)
    
    # K-fold cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Train model
        model = lgb.train(
            params,
            train_data,
            valid_sets=[train_data, val_data],
            callbacks=[
                lgb.early_stopping(100),
                lgb.log_evaluation(0)
            ]
        )
        
        # Predict and evaluate
        val_preds = model.predict(X_val)
        score = np.sqrt(mean_squared_error(y_val, val_preds))
        scores.append(score)
        
        # Report intermediate values
        trial.report(score, fold)
        
        # Handle pruning based on the intermediate value
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return np.mean(scores)

def optimize_hyperparameters(X: pd.DataFrame, y: pd.Series, n_trials: int = 100) -> Dict:
    """Run hyperparameter optimization"""
    print(f"\nStarting optimization with {len(X.columns)} features")
    print("\nFeature dtypes:")
    print(X.dtypes.value_counts())
    
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=20,
            interval_steps=10
        )
    )
    
    study.optimize(
        lambda trial: objective(trial, X, y),
        n_trials=n_trials,
        timeout=None,
        callbacks=[
            lambda study, trial: print(f"Trial {trial.number}: RMSE = {trial.value:.4f}")
        ]
    )
    
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
    
    # Feature engineering
    print("\nApplying feature engineering...")
    feature_engineer = FeatureEngineerHybrid()
    train_features_engineered = feature_engineer.fit_transform(train_features)
    
    print("\nPreparing and preprocessing data...")
    X, y = prepare_data(train_features_engineered, train_labels)
    
    best_params = optimize_hyperparameters(X, y, n_trials=50)
    
    # Save best parameters
    pd.DataFrame([best_params]).to_csv('models/best_params_hybrid.csv', index=False)
    print("\nBest parameters saved to 'models/best_params_hybrid.csv'")

if __name__ == "__main__":
    main()