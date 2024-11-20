import pandas as pd
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
    
    # Make predictions for each required year
    print("\nMaking predictions...")
    all_predictions = []
    
    for year in submission_format['year'].unique():
        year_predictions = pipeline.predict(test_features)
        submission_year = submission_format[submission_format['year'] == year].copy()
        submission_year['composite_score'] = year_predictions
        all_predictions.append(submission_year)
    
    # Combine predictions for all years
    final_submission = pd.concat(all_predictions)
    final_submission.to_csv('submissions/baseline_submission.csv', index=False)
    print(f"\nSubmission saved with {len(final_submission)} predictions")
    
    # Save model
    pipeline.save('models/baseline_model.pkl')
    print("\nModel saved")
    
    return cv_score, feature_importance

if __name__ == "__main__":
    train_and_evaluate()