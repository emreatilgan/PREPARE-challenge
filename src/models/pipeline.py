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
    
    def _convert_mixed_types(self, series):
        """Convert mixed-type series to strings"""
        return series.astype(str)
    
    def preprocess_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """Preprocess features for modeling"""
        df = df.copy()

        # Drop ID columns if present
        id_columns = ['uid', 'year']
        df = df.drop([col for col in id_columns if col in df.columns], axis=1)
        
        # Handle missing values
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        print(f"\nPreprocessing {'training' if is_training else 'test'} data:")
        print(f"Numeric columns: {len(numeric_cols)}")
        print(f"Categorical columns: {len(categorical_cols)}")
        
        # Fill numeric missing values with median
        for col in numeric_cols:
            if is_training:
                self.numeric_medians = df[numeric_cols].median()
            df[col] = df[col].fillna(self.numeric_medians[col])
        
        # Fill categorical missing values with mode
        for col in categorical_cols:
            if is_training:
                self.categorical_modes = df[categorical_cols].mode().iloc[0]
            df[col] = df[col].fillna(self.categorical_modes[col])
        
        # Label encode categorical columns
        for col in categorical_cols:
            # Convert mixed types to string
            df[col] = self._convert_mixed_types(df[col])
            
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
        
        return df
    
    def train(self, X: pd.DataFrame, y: pd.Series,
              n_splits: int = 5, random_state: int = 42) -> Tuple[float, Dict]:
        """Train model and return CV score and feature importance"""
        
        print("\nStarting training pipeline...")
        print(f"Input shape: {X.shape}")
        
        # Preprocess features
        X_processed = self.preprocess_features(X, is_training=True)
        print(f"Processed shape: {X_processed.shape}")
        
        # Initialize and train model
        self.model = self.model_class(self.model_params)
        self.model.fit(X_processed, y, n_splits=n_splits, random_state=random_state)
        
        return self.model.cv_rmse, self.model.feature_importances_
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on test set"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        print(f"\nMaking predictions on {len(X)} samples")
        
        # Preprocess features
        X_processed = self.preprocess_features(X, is_training=False)
        
        # Make predictions
        predictions = self.model.predict(X_processed)
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