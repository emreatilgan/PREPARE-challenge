import numpy as np
from sklearn.metrics import mean_squared_error

def calculate_rmse(y_true, y_pred):
    """Calculate Root Mean Squared Error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def evaluate_predictions(y_true, y_pred, group_col=None, df=None):
    """Evaluate predictions, optionally by group."""
    if group_col is None:
        return calculate_rmse(y_true, y_pred)
    
    results = {}
    for group in df[group_col].unique():
        mask = df[group_col] == group
        group_rmse = calculate_rmse(y_true[mask], y_pred[mask])
        results[group] = group_rmse
    
    return results