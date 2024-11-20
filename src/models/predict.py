import numpy as np

def predict_test(models, X_test):
    """Make predictions on test data using trained models."""
    predictions = np.zeros(len(X_test))
    
    for model in models:
        predictions += model.predict(X_test)
    
    predictions /= len(models)
    return predictions