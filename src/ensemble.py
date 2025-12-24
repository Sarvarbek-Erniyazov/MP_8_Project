import json
import os
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from src.logger import logging_instance

def create_stacking_model(params_path="configs/best_params.json"):
    if not os.path.exists(params_path):
        logging_instance.error(f"Parametrlar fayli topilmadi: {params_path}")
        raise FileNotFoundError(f"{params_path} mavjud emas!")

    with open(params_path, 'r') as f:
        best_params = json.load(f)

    logging_instance.info("Stacking uchun bazaviy modellar yuklanmoqda...")

    
    base_learners = [
        ('lgbm', LGBMClassifier(**best_params['LightGBM'], verbosity=-1)),
        ('xgb', XGBClassifier(**best_params['XGBoost'], use_label_encoder=False, eval_metric='logloss')),
        ('rf', RandomForestClassifier(**best_params['Random_Forest'])),
        ('lr', LogisticRegression(**best_params['Logistic_Regression'], max_iter=1000))
    ]

    
    stack_model = StackingClassifier(
        estimators=base_learners,
        final_estimator=LogisticRegression(),
        cv=5, 
        n_jobs=-1,
        passthrough=False 
    )

    logging_instance.info("Stacking Ensemble arxitekturasi yaratildi.")
    return stack_model