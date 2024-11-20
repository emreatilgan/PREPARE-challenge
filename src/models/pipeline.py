import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import List, Dict, Tuple

class ModelingPipeline:
    """End-to-end modeling pipeline"""
    
    def __init__(self, model_class, model_params=None):
        self.model_class = model_class
        self.model_params = model_params
        self.label_encoders = {}
        self.model = None
    
    def preprocess_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """Preprocess features for modeling"""
        df = df.copy()
        
        # Handle missing values
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # Fill numeric missing values with median
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median() if is_training else self.numeric_medians[col])
        
        # Fill categorical missing values with mode
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0] if is_training else self.categorical_modes[col])
        
        # Label encode categorical columns
        for col in categorical_cols:
            if is_training:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col])
            else:
                # Handle unseen categories in test set
                unique_values = set(df[col].unique())
                missing_values = unique_values - set(self.label_encoders[col].classes_)
                if missing_values:
                    print(f"Warning: New categories found in {col}: {missing_values}")
                    df[col] = df[col].map(lambda x: x if x in self.label_encoders[col].classes_ else self.label_encoders[col].classes_[0])
                df[col] = self.label_encoders[col].transform(df[col])
        
        if is_training:
            self.numeric_medians = df[numeric_cols].median()
            self.categorical_modes = df[categorical_cols].mode().iloc[0]
        
        return df
    
    def train(self, train_features: pd.DataFrame, train_labels: pd.DataFrame,
              n_splits: int = 5, random_state: int = 42) -> Tuple[float, Dict]:
        """Train model and return CV score and feature importance"""
        
        # Preprocess features
        X = self.preprocess_features(train_features, is_training=True)
        y = train_labels['composite_score']
        
        # Initialize and train model
        self.model = self.model_class(self.model_params)
        self.model.fit(X, y, n_splits=n_splits, random_state=random_state)
        
        return self.model.cv_rmse, self.model.feature_importances_
    
    def predict(self, test_features: pd.DataFrame) -> np.ndarray:
        """Make predictions on test set"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        # Preprocess features
        X = self.preprocess_features(test_features, is_training=False)
        
        # Make predictions
        predictions = self.model.predict(X)
        return predictions
    
    def save(self, path: str):
        """Save pipeline to disk"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        self.model.save(path)
    
    @classmethod
    def load(cls, path: str, model_class):
        """Load pipeline from disk"""
        pipeline = cls(model_class)
        pipeline.model = model_class.load(path)
        return pipeline