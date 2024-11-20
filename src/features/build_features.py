import pandas as pd
import numpy as np
from typing import List, Dict, Tuple

class FeatureEngineer:
    """Feature engineering pipeline for Alzheimer's prediction"""
    
    def __init__(self):
        self.temporal_suffixes = ['_03', '_12']
    
    def _convert_to_numeric(self, df: pd.DataFrame, col: str) -> pd.Series:
        """Convert a column to numeric, handling categorical variables"""
        if df[col].dtype == 'object':
            # For categorical health indicators, convert to numeric
            if 'glob_hlth' in col:
                # Map health categories to numeric scores
                health_map = {
                    'excellent': 5,
                    'very good': 4,
                    'good': 3,
                    'fair': 2,
                    'poor': 1
                }
                return df[col].map(health_map).fillna(df[col].map(health_map).median())
            else:
                # For other categorical variables, use label encoding
                return pd.Categorical(df[col]).codes
        return df[col]
    
    def create_temporal_changes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features representing changes between 2003 and 2012"""
        df = df.copy()
        
        # Find matching columns between years
        base_columns = []
        for col in df.columns:
            if col.endswith('_03'):
                base_name = col[:-3]
                if f"{base_name}_12" in df.columns:
                    base_columns.append(base_name)
        
        # Calculate changes for numeric columns
        for base_col in base_columns:
            col_03 = f"{base_col}_03"
            col_12 = f"{base_col}_12"
            
            # Convert both columns to numeric
            df[col_03] = pd.to_numeric(df[col_03], errors='coerce')
            df[col_12] = pd.to_numeric(df[col_12], errors='coerce')
            
            if pd.api.types.is_numeric_dtype(df[col_03]) and pd.api.types.is_numeric_dtype(df[col_12]):
                # Absolute change
                df[f"{base_col}_change"] = df[col_12] - df[col_03]
                
                # Percent change (handling division by zero)
                df[f"{base_col}_pct_change"] = np.where(
                    df[col_03] != 0,
                    (df[col_12] - df[col_03]) / df[col_03],
                    0
                )
                
                # Rate of change (per year)
                df[f"{base_col}_rate"] = df[f"{base_col}_change"] / 9  # 9 years between 2003 and 2012
        
        return df
    
    def create_health_composites(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create composite health indicators"""
        df = df.copy()
        
        # Overall health score for each year
        for suffix in self.temporal_suffixes:
            health_indicators = [
                f'glob_hlth{suffix}',
                f'n_adl{suffix}',
                f'n_iadl{suffix}',
                f'n_illnesses{suffix}'
            ]
            
            # Convert and normalize indicators
            normalized_indicators = []
            for col in health_indicators:
                if col in df.columns:
                    # Convert to numeric
                    numeric_col = self._convert_to_numeric(df, col)
                    # Normalize
                    if numeric_col.std() != 0:
                        df[f'{col}_norm'] = (numeric_col - numeric_col.mean()) / numeric_col.std()
                        normalized_indicators.append(f'{col}_norm')
            
            # Create composite score
            if normalized_indicators:
                df[f'health_score{suffix}'] = df[normalized_indicators].mean(axis=1)
        
        return df
    
    def create_cognitive_risk_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features related to cognitive risk factors"""
        df = df.copy()
        
        for suffix in self.temporal_suffixes:
            # Social engagement score
            social_cols = [
                f'married{suffix}',
                f'n_living_child{suffix}'
            ]
            if suffix == '_12':  # Additional columns only available for 2012
                social_cols.extend([
                    'attends_club_12',
                    'volunteer_12',
                    'games_12',
                    'table_games_12'
                ])
            
            # Convert all columns to numeric
            numeric_social_cols = []
            for col in social_cols:
                if col in df.columns:
                    df[f'{col}_numeric'] = self._convert_to_numeric(df, col)
                    numeric_social_cols.append(f'{col}_numeric')
            
            # Calculate social engagement score
            if numeric_social_cols:
                df[f'social_score{suffix}'] = df[numeric_social_cols].mean(axis=1)
            
            # Physical activity score
            activity_cols = [
                f'exer_3xwk{suffix}',
                f'n_adl{suffix}',
                f'n_iadl{suffix}'
            ]
            
            numeric_activity_cols = []
            for col in activity_cols:
                if col in df.columns:
                    df[f'{col}_numeric'] = self._convert_to_numeric(df, col)
                    numeric_activity_cols.append(f'{col}_numeric')
            
            if numeric_activity_cols:
                df[f'activity_score{suffix}'] = df[numeric_activity_cols].mean(axis=1)
            
            # Mental activity score (especially for 2012)
            if suffix == '_12':
                mental_cols = [
                    'reads_12',
                    'games_12',
                    'attends_class_12'
                ]
                numeric_mental_cols = []
                for col in mental_cols:
                    if col in df.columns:
                        df[f'{col}_numeric'] = self._convert_to_numeric(df, col)
                        numeric_mental_cols.append(f'{col}_numeric')
                
                if numeric_mental_cols:
                    df['mental_activity_score_12'] = df[numeric_mental_cols].mean(axis=1)
        
        return df
    
    def create_economic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create economic-related features"""
        df = df.copy()
        
        for suffix in self.temporal_suffixes:
            income_cols = [
                f'hincome{suffix}',
                f'hinc_business{suffix}',
                f'hinc_rent{suffix}',
                f'hinc_assets{suffix}',
                f'hinc_cap{suffix}',
                f'rinc_pension{suffix}'
            ]
            
            numeric_income_cols = []
            for col in income_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                    numeric_income_cols.append(col)
            
            if numeric_income_cols:
                # Total income
                df[f'total_income{suffix}'] = df[numeric_income_cols].sum(axis=1)
                
                # Income diversity (number of non-zero income sources)
                df[f'income_sources{suffix}'] = (df[numeric_income_cols] > 0).sum(axis=1)
                
                # Income stability (pension to total income ratio)
                pension_col = f'rinc_pension{suffix}'
                if pension_col in df.columns:
                    df[f'pension_ratio{suffix}'] = np.where(
                        df[f'total_income{suffix}'] > 0,
                        df[pension_col] / df[f'total_income{suffix}'],
                        0
                    )
        
        return df
    
    def create_healthcare_access_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create healthcare access related features"""
        df = df.copy()
        
        for suffix in self.temporal_suffixes:
            # Healthcare utilization score
            utilization_cols = [
                f'hosp{suffix}',
                f'visit_med{suffix}',
                f'out_proc{suffix}',
                f'visit_dental{suffix}'
            ]
            
            numeric_utilization_cols = []
            for col in utilization_cols:
                if col in df.columns:
                    df[f'{col}_numeric'] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                    numeric_utilization_cols.append(f'{col}_numeric')
            
            if numeric_utilization_cols:
                df[f'healthcare_utilization{suffix}'] = df[numeric_utilization_cols].mean(axis=1)
            
            # Insurance coverage score
            insurance_cols = [
                f'imss{suffix}',
                f'issste{suffix}',
                f'insur_private{suffix}',
                f'insur_other{suffix}'
            ]
            if suffix == '_12':
                insurance_cols.append('seg_pop_12')
            
            numeric_insurance_cols = []
            for col in insurance_cols:
                if col in df.columns:
                    df[f'{col}_numeric'] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                    numeric_insurance_cols.append(f'{col}_numeric')
            
            if numeric_insurance_cols:
                df[f'insurance_coverage{suffix}'] = df[numeric_insurance_cols].sum(axis=1)
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between important predictors"""
        df = df.copy()
        
        for suffix in self.temporal_suffixes:
            # Convert education to numeric if needed
            edu_col = f'edu_gru{suffix}'
            if edu_col in df.columns:
                df[f'{edu_col}_numeric'] = self._convert_to_numeric(df, edu_col)
                
                if suffix == '_12' and 'reads_12' in df.columns:
                    reads_numeric = self._convert_to_numeric(df, 'reads_12')
                    df[f'edu_reading_interaction{suffix}'] = df[f'{edu_col}_numeric'] * reads_numeric
                
                # Education and social interaction
                if f'social_score{suffix}' in df.columns:
                    df[f'edu_social_interaction{suffix}'] = df[f'{edu_col}_numeric'] * df[f'social_score{suffix}']
            
            # Age and health interaction
            age_col = f'age{suffix}'
            if age_col in df.columns:
                df[f'{age_col}_numeric'] = self._convert_to_numeric(df, age_col)
                if f'health_score{suffix}' in df.columns:
                    df[f'age_health_interaction{suffix}'] = df[f'{age_col}_numeric'] * df[f'health_score{suffix}']
        
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering steps"""
        print("Starting feature engineering pipeline...")
        print(f"Initial shape: {df.shape}")
        
        # Apply each feature engineering step
        df = self.create_temporal_changes(df)
        print("Created temporal changes")
        
        df = self.create_health_composites(df)
        print("Created health composites")
        
        df = self.create_cognitive_risk_factors(df)
        print("Created cognitive risk factors")
        
        df = self.create_economic_features(df)
        print("Created economic features")
        
        df = self.create_healthcare_access_features(df)
        print("Created healthcare access features")
        
        df = self.create_interaction_features(df)
        print("Created interaction features")
        
        print(f"Final shape: {df.shape}")
        return df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering to new data"""
        return self.fit_transform(df)