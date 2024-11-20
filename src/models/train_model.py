import pandas as pd
from pathlib import Path
from src.models.lgb_model import LGBModel
from src.models.pipeline import ModelingPipeline

def train_and_evaluate():
    """Train model and create submission"""
    
    # Load data
    data_dir = Path('data/raw')
    train_features = pd.read_csv(data_dir / 'train_features.csv')
    train_labels = pd.read_csv(data_dir / 'train_labels.csv')
    test_features = pd.read_csv(data_dir / 'test_features.csv')
    submission_format = pd.read_csv(data_dir / 'submission_format.csv')
    
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
    
    # Train model
    cv_score, feature_importance = pipeline.train(
        train_features,
        train_labels,
        n_splits=5,
        random_state=42
    )
    
    print(f"\nCV RMSE: {cv_score:.4f}")
    print("\nTop 10 Important Features:")
    print(feature_importance.head(10))
    
    # Make predictions on test set
    predictions = pipeline.predict(test_features)
    
    # Create submission file
    submission = submission_format.copy()
    submission['composite_score'] = predictions
    submission.to_csv('../submissions/baseline_submission.csv', index=False)
    
    # Save model
    pipeline.save('../models/baseline_model.pkl')
    
    return cv_score, feature_importance

if __name__ == "__main__":
    train_and_evaluate()