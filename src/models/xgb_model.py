import xgboost as xgb
from .base import BaseModel
import numpy as np

class XGBModel(BaseModel):
    """XGBoost implementation of BaseModel"""
    
    def __init__(self, params=None):
        default_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.9,
            'tree_method': 'hist',  # This is faster and works well on CPU
            'random_state': 42
        }
        if params:
            default_params.update(params)
        super().__init__('XGBoost', default_params)
    
    def train_fold(self, X_train, y_train, X_val, y_val):
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=10000,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=100,
            verbose_eval=100
        )
        
        importance = model.get_score(importance_type='gain')
        # Convert feature importance dict to array matching feature order
        importance_array = np.zeros(len(X_train.columns))
        for feat, imp in importance.items():
            try:
                idx = list(X_train.columns).index(feat)
                importance_array[idx] = imp
            except ValueError:
                continue
                
        return model, importance_array
    
    def predict_with_model(self, model, X):
        return model.predict(xgb.DMatrix(X))