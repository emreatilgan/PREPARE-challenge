import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def handle_missing_values(df):
    """Handle missing values in the dataset."""
    # Numeric columns: fill with median
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # Categorical columns: fill with mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])
    
    return df

def encode_categorical_features(df, encoders=None):
    """Encode categorical features using LabelEncoder."""
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    if encoders is None:
        encoders = {}
        for col in categorical_cols:
            encoders[col] = LabelEncoder()
            df[col] = encoders[col].fit_transform(df[col])
    else:
        for col in categorical_cols:
            df[col] = encoders[col].transform(df[col])
    
    return df, encoders
