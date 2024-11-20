from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np
import lightgbm as lgb

def train_model(X, y, params, n_folds=5):
    """Train model with k-fold cross validation."""
    models = []
    oof_predictions = np.zeros(len(X))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"Training fold {fold + 1}")
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='rmse',
            early_stopping_rounds=100,
            verbose=100
        )
        
        oof_predictions[val_idx] = model.predict(X_val)
        models.append(model)
        
    oof_rmse = np.sqrt(mean_squared_error(y, oof_predictions))
    print(f"Overall OOF RMSE: {oof_rmse}")
    
    return models, oof_rmse
