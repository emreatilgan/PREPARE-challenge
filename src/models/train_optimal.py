# src/models/train_optimal.py
import pandas as pd
import numpy as np
from pathlib import Path
from src.models.lgb_model import LGBModel
from src.models.pipeline import ModelingPipeline

def process_parameters(params: dict) -> dict:
    """Convert parameters to appropriate types"""
    processed_params = {}
    
    # Define which parameters should be integers
    int_params = [
        'num_leaves', 'bagging_freq', 'min_child_samples',
        'max_depth', 'min_data_in_leaf'
    ]
    
    # Convert parameters to appropriate types
    for key, value in params.items():
        if key in int_params:
            processed_params[key] = int(value)
        else:
            processed_params[key] = float(value)
    
    return processed_params

def train_and_evaluate():
    """Train model with optimal parameters and create submission"""
    
    # Load data
    data_dir = Path('data/raw')
    train_features = pd.read_csv(data_dir / 'train_features.csv')
    train_labels = pd.read_csv(data_dir / 'train_labels.csv')
    test_features = pd.read_csv(data_dir / 'test_features.csv')
    submission_format = pd.read_csv(data_dir / 'submission_format.csv')
    
    # Load and process best parameters
    try:
        raw_params = pd.read_csv('models/best_params.csv').iloc[0].to_dict()
        best_params = process_parameters(raw_params)
        print("\nLoaded best parameters:")
        for key, value in best_params.items():
            print(f"  {key}: {value} ({type(value).__name__})")
    except FileNotFoundError:
        print("\nNo optimized parameters found. Running hyperparameter optimization first...")
        from src.models.optimize import main as optimize_main
        optimize_main()
        raw_params = pd.read_csv('models/best_params.csv').iloc[0].to_dict()
        best_params = process_parameters(raw_params)
    
    # Prepare training data
    print("\nPreparing training data...")
    train_data = train_features.merge(train_labels, on='uid', how='inner')
    
    # Initialize pipeline with optimal parameters
    pipeline = ModelingPipeline(
        model_class=LGBModel,
        model_params={
            'objective': 'regression',
            'metric': 'rmse',
            'verbosity': -1,
            **best_params
        }
    )
    
    # Prepare features
    features_to_drop = ['uid', 'year']
    X = train_data.drop(features_to_drop + ['composite_score'], axis=1)
    y = train_data['composite_score']
    
    print(f"\nTraining with {X.shape[1]} features on {len(X)} samples")
    
    # Train model
    cv_score, feature_importance = pipeline.train(
        X,
        y,
        n_splits=5,
        random_state=42
    )
    
    print(f"\nCV RMSE: {cv_score:.4f}")
    print("\nTop 20 Important Features:")
    print(feature_importance.head(20))
    
    # Make predictions
    print("\nMaking predictions...")
    test_predictions = pipeline.predict(test_features)
    
    # Round predictions to nearest integer and ensure they're within valid range
    test_predictions = np.round(test_predictions).clip(0, 384).astype(int)
    
    # Create a mapping from uid to prediction
    predictions_dict = dict(zip(test_features['uid'], test_predictions))
    
    # Create submission
    submission = submission_format.copy()
    submission['composite_score'] = submission['uid'].map(predictions_dict).astype(int)
    
    # Verify submission format
    print("\nVerifying submission format...")
    print("Submission shape:", submission.shape)
    print("Submission dtypes:")
    print(submission.dtypes)
    print("\nPrediction statistics:")
    print(submission['composite_score'].describe())
    
    # Save submission
    submission.to_csv('submissions/submission_optimal.csv', index=False)
    print(f"\nSubmission saved with {len(submission)} predictions")
    
    # Save model
    pipeline.save('models/model_optimal.pkl')
    print("\nModel saved")
    
    # Save feature importance analysis
    feature_importance_df = pd.DataFrame({
        'feature': feature_importance.index,
        'importance': feature_importance.values
    })
    feature_importance_df.to_csv('analysis/feature_importance_optimal.csv', index=False)
    print("\nFeature importance analysis saved")
    
    return cv_score, feature_importance

if __name__ == "__main__":
    train_and_evaluate()