import lightgbm as lgb
from .base import BaseModel

class LGBModel(BaseModel):
    """LightGBM implementation of BaseModel"""
    
    def __init__(self, params=None):
        default_params = {
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
        if params:
            default_params.update(params)
        super().__init__('LightGBM', default_params)
    
    def train_fold(self, X_train, y_train, X_val, y_val):
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val)
        
        model = lgb.train(
            self.params,
            train_data,
            num_boost_round=10000,
            valid_sets=[train_data, val_data],
            callbacks=[
                lgb.early_stopping(100),
                lgb.log_evaluation(100)
            ]
        )
        
        importance = model.feature_importance(importance_type='gain')
        return model, importance
    
    def predict_with_model(self, model, X):
        return model.predict(X)