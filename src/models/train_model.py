import pandas as pd
import numpy as np
from pathlib import Path
from src.models.lgb_model import LGBModel
from src.models.pipeline import ModelingPipeline

def prepare_data(train_features, train_labels):
    """Prepare data by properly merging features and labels"""
    # Merge features with labels
    train_data = train_features.merge(
        train_labels,
        on='uid',
        how='inner'
    )
    
    print(f"Number of unique individuals: {len(train_data['uid'].unique())}")
    print(f"Number of samples: {len(train_data)}")
    
    return train_data

def train_and_evaluate():
    """Train model and create submission"""
    
    # Load data
    data_dir = Path('data/raw')
    train_features = pd.read_csv(data_dir / 'train_features.csv')
    train_labels = pd.read_csv(data_dir / 'train_labels.csv')
    test_features = pd.read_csv(data_dir / 'test_features.csv')
    submission_format = pd.read_csv(data_dir / 'submission_format.csv')
    
    # Prepare training data
    print("Preparing training data...")
    train_data = prepare_data(train_features, train_labels)
    
    # Initialize pipeline with LightGBM model
    pipeline = ModelingPipeline(
        model_class=LGBModel,
        model_params={
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        }
    )
    
    # Drop columns not needed for training
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
    print("\nTop 10 Important Features:")
    print(feature_importance.head(10))
    
    # Make predictions
    print("\nMaking predictions...")
    
    # Create final submission dataframe
    submission = submission_format.copy()
    
    # Get predictions for test set
    test_predictions = pipeline.predict(test_features)
    
    # Round predictions to nearest integer and ensure they're within valid range
    test_predictions = np.round(test_predictions).clip(0, 384).astype(int)
    
    # Create a mapping from uid to prediction
    predictions_dict = dict(zip(test_features['uid'], test_predictions))
    
    # Map predictions to submission format
    submission['composite_score'] = submission['uid'].map(predictions_dict).astype(int)
    
    # Verify submission format
    print("\nVerifying submission format...")
    print("Submission shape:", submission.shape)
    print("Submission dtypes:")
    print(submission.dtypes)
    print("\nPrediction statistics:")
    print(submission['composite_score'].describe())
    
    # Save submission
    submission.to_csv('submissions/baseline_submission.csv', index=False)
    print(f"\nSubmission saved with {len(submission)} predictions")
    
    # Save model
    pipeline.save('models/baseline_model.pkl')
    print("\nModel saved")
    
    return cv_score, feature_importance

if __name__ == "__main__":
    train_and_evaluate()