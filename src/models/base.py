from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import pickle
from pathlib import Path

class BaseModel(ABC):
    """Abstract base class for all models"""
    
    def __init__(self, name: str, params: dict = None):
        self.name = name
        self.params = params or {}
        self.models = []  # Store models from each fold
        self.feature_importances_ = None
        self.cv_scores = []
        
    @abstractmethod
    def train_fold(self, X_train, y_train, X_val, y_val):
        """Train model on a single fold"""
        pass
    
    def fit(self, X, y, n_splits=5, random_state=42):
        """Train model using k-fold cross-validation"""
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        self.feature_names = X.columns.tolist()
        
        print(f"\nTraining {self.name}")
        print("-" * 50)
        
        fold_importance_df = pd.DataFrame()
        oof_predictions = np.zeros(len(X))
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            print(f"\nFold {fold + 1}")
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train model on fold
            model, importance = self.train_fold(X_train, y_train, X_val, y_val)
            self.models.append(model)
            
            # Make predictions
            oof_predictions[val_idx] = self.predict_with_model(model, X_val)
            
            # Calculate fold score
            fold_score = np.sqrt(mean_squared_error(y_val, oof_predictions[val_idx]))
            self.cv_scores.append(fold_score)
            print(f"Fold {fold + 1} RMSE: {fold_score:.4f}")
            
            # Store feature importance
            fold_importance_df = pd.concat([
                fold_importance_df,
                pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': importance,
                    'fold': fold + 1
                })
            ])
        
        # Calculate and store overall CV score
        self.cv_rmse = np.sqrt(mean_squared_error(y, oof_predictions))
        print(f"\nOverall CV RMSE: {self.cv_rmse:.4f}")
        
        # Calculate and store average feature importance
        self.feature_importances_ = fold_importance_df.groupby('feature')['importance'].mean()
        self.feature_importances_ = self.feature_importances_.sort_values(ascending=False)
        
        return self
    
    def predict(self, X):
        """Make predictions using all trained models"""
        predictions = np.zeros(len(X))
        for model in self.models:
            predictions += self.predict_with_model(model, X)
        return predictions / len(self.models)
    
    @abstractmethod
    def predict_with_model(self, model, X):
        """Make predictions using a single model"""
        pass
    
    def save(self, path):
        """Save model to disk"""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path):
        """Load model from disk"""
        with open(path, 'rb') as f:
            return pickle.load(f)