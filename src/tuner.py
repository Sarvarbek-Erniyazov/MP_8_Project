import os
import json
import logging
import optuna
import pandas as pd
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from src.logger import logging_instance


optuna.logging.set_verbosity(optuna.logging.INFO)

def objective(trial, X, y, model_name):
    if model_name == "LightGBM":
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 800),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'verbosity': -1
        }
        model = LGBMClassifier(**params)
    
    elif model_name == "XGBoost":
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 800),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'max_depth': trial.suggest_int('max_depth', 3, 12)
        }
        model = XGBClassifier(**params, use_label_encoder=False, eval_metric='logloss')

    elif model_name == "Random_Forest":
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 5, 20)
        }
        model = RandomForestClassifier(**params)

    elif model_name == "Logistic_Regression":
        params = {
            'C': trial.suggest_float('C', 0.01, 10.0)
        }
        model = LogisticRegression(**params, max_iter=1000)

    
    return cross_val_score(model, X, y, cv=3, scoring='f1', n_jobs=-1).mean()

def run_all_tuning(train_path):
    if not os.path.exists(train_path):
        print(f"❌ Xatolik: {train_path} topilmadi!")
        return

    df = pd.read_csv(train_path)
    X = df.drop(columns=['is_canceled'])
    y = df['is_canceled']
    
    all_best_params = {}
    model_names = ["LightGBM", "XGBoost", "Random_Forest", "Logistic_Regression"]

    for name in model_names:
        logging_instance.info(f"--- {name} tuning boshlandi ---")
        print(f"🚀 {name} uchun eng yaxshi parametrlar qidirilmoqda...")
        
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, X, y, name), n_trials=30) 
        
        all_best_params[name] = study.best_params
        logging_instance.info(f"{name} yakuniy natija (F1): {study.best_value:.4f}")

    
    os.makedirs('configs', exist_ok=True)
    with open('configs/best_params.json', 'w') as f:
        json.dump(all_best_params, f, indent=4)
    
    print("\n" + "="*40)
    print("✅ Barcha modellar tuning qilindi!")
    print("📂 Parametrlar manzili: configs/best_params.json")
    print("="*40)

if __name__ == "__main__":
    run_all_tuning("data/engineered/train_final.csv")