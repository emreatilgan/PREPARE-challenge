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
    
# src/features/build_features_v2.py
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple

class FeatureEngineerV2:
    """Focused feature engineering pipeline based on important predictors"""
    
    def __init__(self):
        self.temporal_suffixes = ['_03', '_12']
        
        # Define key feature groups based on importance analysis
        self.key_features = {
            'education': ['edu_gru', 'rameduc_m', 'rafeduc_m'],
            'demographics': ['age', 'n_living_child'],
            'economic': ['hincome', 'rjob_hrswk'],
            'social': ['rrfcntx_m', 'rsocact_m'],
            'health': ['n_depr', 'bmi'],
            'cognitive': ['reads', 'games']
        }
    
    def _convert_to_numeric(self, df: pd.DataFrame, col: str) -> pd.Series:
        """Convert column to numeric, handling categorical variables"""
        if df[col].dtype == 'object':
            if 'edu' in col:
                # Create ordinal encoding for education
                edu_order = {
                    'no education': 0,
                    'incomplete primary': 1,
                    'primary': 2,
                    'incomplete secondary': 3,
                    'secondary': 4,
                    'preparatory or higher': 5
                }
                return df[col].map(edu_order).fillna(df[col].map(edu_order).median())
            elif 'age' in col:
                # Convert age groups to numeric
                age_map = {
                    '50-54': 52, '55-59': 57, '60-64': 62,
                    '65-69': 67, '70-74': 72, '75-79': 77,
                    '80-84': 82, '85+': 87
                }
                return df[col].map(age_map).fillna(df[col].map(age_map).median())
            else:
                # For other categorical variables, use label encoding
                return pd.Categorical(df[col]).codes
        return df[col]
    
    def create_temporal_changes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal changes for key features"""
        df = df.copy()
        
        for feature_group in self.key_features.values():
            for base_col in feature_group:
                col_03 = f"{base_col}_03"
                col_12 = f"{base_col}_12"
                
                if col_03 in df.columns and col_12 in df.columns:
                    # Convert both columns to numeric
                    df[col_03] = pd.to_numeric(self._convert_to_numeric(df, col_03), errors='coerce')
                    df[col_12] = pd.to_numeric(self._convert_to_numeric(df, col_12), errors='coerce')
                    
                    # Absolute change
                    df[f"{base_col}_change"] = df[col_12] - df[col_03]
                    
                    # Percent change
                    df[f"{base_col}_pct_change"] = np.where(
                        df[col_03] != 0,
                        (df[col_12] - df[col_03]) / df[col_03],
                        0
                    )
                    
                    # Trend direction
                    df[f"{base_col}_trend"] = np.sign(df[f"{base_col}_change"]).astype(int)
                    
                    # Acceleration (change in rate of change)
                    if f"{base_col}_pct_change" in df.columns:
                        df[f"{base_col}_acceleration"] = df[f"{base_col}_pct_change"] / 9  # 9 years
        
        return df
    
    def create_education_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create education-related features"""
        df = df.copy()
        
        # Convert education columns to numeric
        edu_cols = [col for col in df.columns if any(edu in col for edu in self.key_features['education'])]
        for col in edu_cols:
            df[f"{col}_numeric"] = self._convert_to_numeric(df, col)
        
        # Education level relative to parents
        if all(col in df.columns for col in ['edu_gru_12', 'rameduc_m', 'rafeduc_m']):
            df['edu_vs_parents'] = (
                df['edu_gru_12_numeric'] - 
                ((df['rameduc_m_numeric'] + df['rafeduc_m_numeric']) / 2)
            )
        
        # Education stability
        if all(col in df.columns for col in ['edu_gru_03', 'edu_gru_12']):
            df['edu_stability'] = (df['edu_gru_03_numeric'] == df['edu_gru_12_numeric']).astype(int)
        
        return df
    
    def create_socioeconomic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create socioeconomic features"""
        df = df.copy()
        
        for suffix in self.temporal_suffixes:
            # Income features
            income_col = f'hincome{suffix}'
            if income_col in df.columns:
                df[income_col] = pd.to_numeric(df[income_col], errors='coerce').fillna(0)
                
                # Income quantiles
                df[f'income_quantile{suffix}'] = pd.qcut(
                    df[income_col], 
                    q=5, 
                    labels=False, 
                    duplicates='drop'
                )
                
                # Working hours relative to income
                work_hours = f'rjob_hrswk{suffix}'
                if work_hours in df.columns:
                    df[f'income_per_hour{suffix}'] = np.where(
                        df[work_hours] > 0,
                        df[income_col] / df[work_hours],
                        0
                    )
        
        return df
    
    def create_social_health_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create social and health interaction features"""
        df = df.copy()
        
        for suffix in self.temporal_suffixes:
            # Social engagement score
            social_cols = [
                f'rrfcntx_m{suffix}',
                f'rsocact_m{suffix}',
                f'n_living_child{suffix}'
            ]
            valid_cols = [col for col in social_cols if col in df.columns]
            
            if valid_cols:
                # Convert to numeric and normalize
                for col in valid_cols:
                    df[f"{col}_numeric"] = self._convert_to_numeric(df, col)
                    if df[f"{col}_numeric"].std() != 0:
                        df[f"{col}_norm"] = (
                            df[f"{col}_numeric"] - df[f"{col}_numeric"].mean()
                        ) / df[f"{col}_numeric"].std()
                
                # Create composite social score
                norm_cols = [f"{col}_norm" for col in valid_cols if f"{col}_norm" in df.columns]
                if norm_cols:
                    df[f'social_score{suffix}'] = df[norm_cols].mean(axis=1)
            
            # Health indicators
            health_cols = [f'n_depr{suffix}', f'bmi{suffix}']
            valid_health = [col for col in health_cols if col in df.columns]
            
            if valid_health:
                # Normalize health indicators
                for col in valid_health:
                    df[f"{col}_numeric"] = pd.to_numeric(df[col], errors='coerce')
                    if df[f"{col}_numeric"].std() != 0:
                        df[f"{col}_norm"] = (
                            df[f"{col}_numeric"] - df[f"{col}_numeric"].mean()
                        ) / df[f"{col}_numeric"].std()
                
                # Create health score
                health_norm = [f"{col}_norm" for col in valid_health if f"{col}_norm" in df.columns]
                if health_norm:
                    df[f'health_score{suffix}'] = df[health_norm].mean(axis=1)
        
        return df
    
    def create_cognitive_activity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create cognitive activity features"""
        df = df.copy()
        
        cognitive_cols = ['reads_12', 'games_12']
        valid_cols = [col for col in cognitive_cols if col in df.columns]
        
        if valid_cols:
            # Convert to numeric
            for col in valid_cols:
                df[f"{col}_numeric"] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            # Create cognitive activity score
            numeric_cols = [f"{col}_numeric" for col in valid_cols]
            df['cognitive_activity_score'] = df[numeric_cols].mean(axis=1)
            
            # Interaction with education
            if 'edu_gru_12' in df.columns:
                df['edu_cognitive_interaction'] = (
                    df['cognitive_activity_score'] * 
                    self._convert_to_numeric(df, 'edu_gru_12')
                )
        
        return df
    
    def create_age_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create age-related interaction features"""
        df = df.copy()
        
        for suffix in self.temporal_suffixes:
            age_col = f'age{suffix}'
            if age_col in df.columns:
                age_numeric = self._convert_to_numeric(df, age_col)
                
                # Age-health interaction
                if f'health_score{suffix}' in df.columns:
                    df[f'age_health_interaction{suffix}'] = age_numeric * df[f'health_score{suffix}']
                
                # Age-social interaction
                if f'social_score{suffix}' in df.columns:
                    df[f'age_social_interaction{suffix}'] = age_numeric * df[f'social_score{suffix}']
                
                # Age-education interaction
                edu_col = f'edu_gru{suffix}'
                if edu_col in df.columns:
                    df[f'age_education_interaction{suffix}'] = (
                        age_numeric * 
                        self._convert_to_numeric(df, edu_col)
                    )
        
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering steps"""
        print("Starting feature engineering pipeline v2...")
        print(f"Initial shape: {df.shape}")
        
        # Apply each feature engineering step
        df = self.create_temporal_changes(df)
        print("Created temporal changes")
        
        df = self.create_education_features(df)
        print("Created education features")
        
        df = self.create_socioeconomic_features(df)
        print("Created socioeconomic features")
        
        df = self.create_social_health_features(df)
        print("Created social and health features")
        
        df = self.create_cognitive_activity_features(df)
        print("Created cognitive activity features")
        
        df = self.create_age_interactions(df)
        print("Created age interactions")
        
        print(f"Final shape: {df.shape}")
        
        # Remove any infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Fill remaining NaN values with 0
        df = df.fillna(0)
        
        return df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering to new data"""
        return self.fit_transform(df)