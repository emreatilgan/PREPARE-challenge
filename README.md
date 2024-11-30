# Alzheimer's Disease Prediction Using Social Determinants

This repository contains code for predicting early signs of Alzheimer's disease and related dementias (AD/ADRD) using social determinants of health from the Mexican Health and Aging Study (MHAS).
https://www.drivendata.org/competitions/300/competition-nih-alzheimers-sdoh-2/

## Project Overview

The objective is to improve early prediction of AD/ADRD using social determinants of health data. The project uses data from a national longitudinal study of adults 50 years and older in Mexico, including information about:
- Demographics
- Economic circumstances
- Migration
- Physical limitations
- Self-reported health
- Lifestyle behaviors

## Project Structure

```
prepate-challenge/
├── data/
│   ├── raw/                  # Original data files
├── src/
│   ├── config.py            # Configuration parameters
│   ├── data/
│   │   ├── load_data.py     # Data loading utilities
│   │   └── preprocess.py    # Data preprocessing utilities
│   ├── features/
│   │   └── build_features.py # Feature engineering
│   ├── models/
│   │   ├── base.py          
│   │   ├── lgb_model.py      
│   │   ├── optimize.py         
│   │   └── optimize_hybrid.py
│   │   └── pipeline.py
│   │   └── predict.py
│   │   └── train.py
│   │   └── train_model.py
│   │   └── train_optimal.py
│   │   └── xgb_model.py
│   └── utils/
│       └── evaluation.py    # Evaluation metrics
├── notebooks/
│   ├── EDA.ipynb       # Exploratory data analysis
│   ├── EDA2.ipynb       # Exploratory data analysis
├── analysis/
├── models/
├── submissions/
├── requirements.txt
└── README.md
```

## Key Features

1. **Feature Engineering Pipeline**:
   - Temporal changes between 2003 and 2012
   - Composite health indicators
   - Social engagement metrics
   - Economic stability indicators

2. **Ensemble Modeling**:
   - Specialized models for different feature groups
   - LightGBM and XGBoost models
   - Cross-validation training
   - Feature importance tracking

3. **Model Groups**:
   - Demographics (education, age, family)
   - Social (engagement, relationships)
   - Health/Behavioral (health status, depression)
   - Economic (income, employment)

## Getting Started

1. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Place data files in `data/raw/`:
- train_features.csv
- test_features.csv
- train_labels.csv
- submission_format.csv

4. Run the pipeline:
```bash
python -m src.models.train_optimal
```

## Model Performance

The model achieves:
- Cross-validation RMSE on training data
- Feature importance analysis for interpretability
- Specialized performance for different feature groups

## Dependencies

- numpy
- pandas
- scikit-learn
- lightgbm
- xgboost
- matplotlib
- seaborn
- jupyter
