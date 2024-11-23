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
                return df[col].map(edu_order).fillna(0)
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
                    
                    # Fill NaN values with 0
                    df[col_03] = df[col_03].fillna(0)
                    df[col_12] = df[col_12].fillna(0)
                    
                    # Absolute change
                    df[f"{base_col}_change"] = df[col_12] - df[col_03]
                    
                    # Percent change (handle division by zero)
                    df[f"{base_col}_pct_change"] = np.where(
                        (df[col_03] != 0) & np.isfinite(df[col_03]),
                        (df[col_12] - df[col_03]) / np.abs(df[col_03]),
                        0
                    )
                    
                    # Replace infinite values with max/min of non-infinite values
                    df[f"{base_col}_pct_change"] = df[f"{base_col}_pct_change"].replace([np.inf, -np.inf], 0)
                    
                    # Trend direction (-1, 0, 1)
                    df[f"{base_col}_trend"] = np.sign(df[f"{base_col}_change"]).fillna(0).astype(int)
                    
                    # Acceleration (change in rate of change)
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
            parent_edu = ((df['rameduc_m_numeric'] + df['rafeduc_m_numeric']) / 2).fillna(0)
            df['edu_vs_parents'] = df['edu_gru_12_numeric'] - parent_edu
        
        # Education stability
        if all(col in df.columns for col in ['edu_gru_03', 'edu_gru_12']):
            df['edu_stability'] = (
                df['edu_gru_03_numeric'].fillna(0) == df['edu_gru_12_numeric'].fillna(0)
            ).astype(int)
        
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
                    df[income_col].clip(lower=0), 
                    q=5, 
                    labels=False, 
                    duplicates='drop'
                ).fillna(0)
                
                # Working hours relative to income
                work_hours = f'rjob_hrswk{suffix}'
                if work_hours in df.columns:
                    work_hours_value = pd.to_numeric(df[work_hours], errors='coerce').fillna(0)
                    df[f'income_per_hour{suffix}'] = np.where(
                        work_hours_value > 0,
                        df[income_col] / work_hours_value,
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
                    df[f"{col}_numeric"] = pd.to_numeric(
                        self._convert_to_numeric(df, col), 
                        errors='coerce'
                    )
                    df[f"{col}_numeric"] = df[f"{col}_numeric"].fillna(0)
                    
                    # Normalize non-zero values
                    col_std = df[f"{col}_numeric"].std()
                    if col_std != 0:
                        df[f"{col}_norm"] = (
                            (df[f"{col}_numeric"] - df[f"{col}_numeric"].mean()) / col_std
                        ).fillna(0)
                    else:
                        df[f"{col}_norm"] = 0
                
                # Create composite social score
                norm_cols = [f"{col}_norm" for col in valid_cols if f"{col}_norm" in df.columns]
                if norm_cols:
                    df[f'social_score{suffix}'] = df[norm_cols].mean(axis=1).fillna(0)
            
            # Health indicators
            health_cols = [f'n_depr{suffix}', f'bmi{suffix}']
            valid_health = [col for col in health_cols if col in df.columns]
            
            if valid_health:
                # Normalize health indicators
                for col in valid_health:
                    df[f"{col}_numeric"] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                    col_std = df[f"{col}_numeric"].std()
                    if col_std != 0:
                        df[f"{col}_norm"] = (
                            (df[f"{col}_numeric"] - df[f"{col}_numeric"].mean()) / col_std
                        ).fillna(0)
                    else:
                        df[f"{col}_norm"] = 0
                
                # Create health score
                health_norm = [f"{col}_norm" for col in valid_health if f"{col}_norm" in df.columns]
                if health_norm:
                    df[f'health_score{suffix}'] = df[health_norm].mean(axis=1).fillna(0)
        
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
            df['cognitive_activity_score'] = df[numeric_cols].mean(axis=1).fillna(0)
            
            # Interaction with education
            if 'edu_gru_12' in df.columns:
                df['edu_cognitive_interaction'] = (
                    df['cognitive_activity_score'] * 
                    self._convert_to_numeric(df, 'edu_gru_12')
                ).fillna(0)
        
        return df
    
    def create_age_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create age-related interaction features"""
        df = df.copy()
        
        for suffix in self.temporal_suffixes:
            age_col = f'age{suffix}'
            if age_col in df.columns:
                age_numeric = pd.to_numeric(
                    self._convert_to_numeric(df, age_col), 
                    errors='coerce'
                ).fillna(0)
                
                # Age-health interaction
                if f'health_score{suffix}' in df.columns:
                    df[f'age_health_interaction{suffix}'] = (
                        age_numeric * df[f'health_score{suffix}']
                    ).fillna(0)
                
                # Age-social interaction
                if f'social_score{suffix}' in df.columns:
                    df[f'age_social_interaction{suffix}'] = (
                        age_numeric * df[f'social_score{suffix}']
                    ).fillna(0)
                
                # Age-education interaction
                edu_col = f'edu_gru{suffix}'
                if edu_col in df.columns:
                    df[f'age_education_interaction{suffix}'] = (
                        age_numeric * 
                        self._convert_to_numeric(df, edu_col)
                    ).fillna(0)
        
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
        
        # Final cleanup
        df = df.replace([np.inf, -np.inf], 0)
        df = df.fillna(0)
        
        print(f"Final shape: {df.shape}")
        return df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering to new data"""
        return self.fit_transform(df)

    
class FeatureEngineerHybrid:
    """Hybrid feature engineering combining best original and engineered features"""
    
    def __init__(self):
        # Key original features based on importance analysis
        self.important_original_features = [
            'edu_gru_12', 'edu_gru_03',  # Education
            'age_12', 'age_03',          # Age
            'hincome_12', 'hincome_03',  # Income
            'rrfcntx_m_12',              # Social contact
            'n_living_child_12',         # Family
            'reads_12', 'games_12',      # Cognitive activities
            'rameduc_m', 'rafeduc_m',    # Parental education
            'rjlocc_m_12',               # Occupation
            'j11_12',                    # Housing
            'rsocact_m_12',              # Social activities
            'n_depr_12', 'n_depr_03',    # Depression
            'bmi_12', 'bmi_03',          # Health
            'rjob_hrswk_12',             # Work hours
            'a34_12'                     # Language skills
        ]
        
        # Feature groups for engineering
        self.feature_groups = {
            'cognitive': ['reads_12', 'games_12', 'attends_class_12'],
            'social': ['rrfcntx_m_12', 'rsocact_m_12', 'n_living_child_12'],
            'health': ['n_depr_12', 'bmi_12', 'n_illnesses_12'],
            'economic': ['hincome_12', 'rjob_hrswk_12', 'hinc_business_12']
        }
    
    def _convert_to_numeric(self, df: pd.DataFrame, col: str) -> pd.Series:
        """Convert column to numeric, handling categorical variables"""
        if df[col].dtype == 'object':
            if 'edu' in col:
                edu_order = {
                    'no education': 0,
                    'incomplete primary': 1,
                    'primary': 2,
                    'incomplete secondary': 3,
                    'secondary': 4,
                    'preparatory or higher': 5
                }
                return df[col].map(edu_order).fillna(0)
            elif 'age' in col:
                age_map = {
                    '50-54': 52, '55-59': 57, '60-64': 62,
                    '65-69': 67, '70-74': 72, '75-79': 77,
                    '80-84': 82, '85+': 87
                }
                return df[col].map(age_map).fillna(df[col].map(age_map).median())
            else:
                return pd.Categorical(df[col]).codes
        return df[col]
    
    def create_cognitive_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create cognitive activity composite score"""
        df = df.copy()
        
        cognitive_cols = self.feature_groups['cognitive']
        valid_cols = [col for col in cognitive_cols if col in df.columns]
        
        if valid_cols:
            # Convert to numeric and normalize
            for col in valid_cols:
                df[f"{col}_norm"] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                if df[f"{col}_norm"].std() != 0:
                    df[f"{col}_norm"] = (df[f"{col}_norm"] - df[f"{col}_norm"].mean()) / df[f"{col}_norm"].std()
            
            norm_cols = [f"{col}_norm" for col in valid_cols]
            df['cognitive_activity_score'] = df[norm_cols].mean(axis=1)
            
        return df
    
    def create_social_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create social engagement composite score"""
        df = df.copy()
        
        social_cols = self.feature_groups['social']
        valid_cols = [col for col in social_cols if col in df.columns]
        
        if valid_cols:
            # Convert to numeric and normalize
            for col in valid_cols:
                df[f"{col}_norm"] = pd.to_numeric(self._convert_to_numeric(df, col), errors='coerce').fillna(0)
                if df[f"{col}_norm"].std() != 0:
                    df[f"{col}_norm"] = (df[f"{col}_norm"] - df[f"{col}_norm"].mean()) / df[f"{col}_norm"].std()
            
            norm_cols = [f"{col}_norm" for col in valid_cols]
            df['social_engagement_score'] = df[norm_cols].mean(axis=1)
        
        return df
    
    def create_health_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create health composite score"""
        df = df.copy()
        
        health_cols = self.feature_groups['health']
        valid_cols = [col for col in health_cols if col in df.columns]
        
        if valid_cols:
            # Convert to numeric and normalize
            for col in valid_cols:
                df[f"{col}_norm"] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                if df[f"{col}_norm"].std() != 0:
                    df[f"{col}_norm"] = (df[f"{col}_norm"] - df[f"{col}_norm"].mean()) / df[f"{col}_norm"].std()
            
            norm_cols = [f"{col}_norm" for col in valid_cols]
            df['health_status_score'] = -df[norm_cols].mean(axis=1)  # Negative because higher values indicate worse health
        
        return df
    
    def create_economic_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create economic stability score"""
        df = df.copy()
        
        economic_cols = self.feature_groups['economic']
        valid_cols = [col for col in economic_cols if col in df.columns]
        
        if valid_cols:
            # Convert to numeric and normalize
            for col in valid_cols:
                df[f"{col}_norm"] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                if df[f"{col}_norm"].std() != 0:
                    df[f"{col}_norm"] = (df[f"{col}_norm"] - df[f"{col}_norm"].mean()) / df[f"{col}_norm"].std()
            
            norm_cols = [f"{col}_norm" for col in valid_cols]
            df['economic_stability_score'] = df[norm_cols].mean(axis=1)
        
        return df
    
    def create_key_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create important feature interactions"""
        df = df.copy()
        
        # Education-Cognitive interaction
        if 'edu_gru_12' in df.columns and 'cognitive_activity_score' in df.columns:
            df['edu_cognitive_interaction'] = (
                self._convert_to_numeric(df, 'edu_gru_12') * 
                df['cognitive_activity_score']
            )
        
        # Age-Health interaction
        if 'age_12' in df.columns and 'health_status_score' in df.columns:
            age_numeric = self._convert_to_numeric(df, 'age_12')
            df['age_health_interaction'] = age_numeric * df['health_status_score']
        
        # Social-Economic interaction
        if 'social_engagement_score' in df.columns and 'economic_stability_score' in df.columns:
            df['social_economic_interaction'] = (
                df['social_engagement_score'] * 
                df['economic_stability_score']
            )
        
        return df
    
    def create_temporal_changes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal changes for key numeric features"""
        df = df.copy()
        
        # Define pairs of features to calculate changes
        temporal_pairs = [
            ('hincome_03', 'hincome_12'),
            ('n_depr_03', 'n_depr_12'),
            ('n_illnesses_03', 'n_illnesses_12')
        ]
        
        for col_03, col_12 in temporal_pairs:
            if col_03 in df.columns and col_12 in df.columns:
                base_name = col_03[:-3]
                df[f"{base_name}_change"] = (
                    pd.to_numeric(df[col_12], errors='coerce').fillna(0) - 
                    pd.to_numeric(df[col_03], errors='coerce').fillna(0)
                )
        
        return df
    
    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select original important features and add engineered features"""
        df = df.copy()
        
        # Keep only important original features
        original_features = [col for col in self.important_original_features if col in df.columns]
        selected_df = df[original_features].copy()
        
        # Add engineered features
        engineered_features = [
            'cognitive_activity_score',
            'social_engagement_score',
            'health_status_score',
            'economic_stability_score',
            'edu_cognitive_interaction',
            'age_health_interaction',
            'social_economic_interaction',
            'hincome_change',
            'n_depr_change',
            'n_illnesses_change'
        ]
        
        for col in engineered_features:
            if col in df.columns:
                selected_df[col] = df[col]
        
        return selected_df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply hybrid feature engineering pipeline"""
        print("Starting hybrid feature engineering pipeline...")
        print(f"Initial shape: {df.shape}")
        
        # Create composite scores
        df = self.create_cognitive_score(df)
        print("Created cognitive score")
        
        df = self.create_social_score(df)
        print("Created social score")
        
        df = self.create_health_score(df)
        print("Created health score")
        
        df = self.create_economic_score(df)
        print("Created economic score")
        
        # Create interactions and temporal features
        df = self.create_key_interactions(df)
        print("Created key interactions")
        
        df = self.create_temporal_changes(df)
        print("Created temporal changes")
        
        # Select final feature set
        df = self.select_features(df)
        print(f"Final shape: {df.shape}")
        
        # Handle any remaining missing values
        df = df.fillna(0)
        
        return df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering to new data"""
        return self.fit_transform(df)