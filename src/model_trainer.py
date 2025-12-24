import os
import joblib
import pandas as pd
import numpy as np
from src.logger import logging_instance


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, classification_report

def train_models(train_path, model_save_path):
    try:
        logging_instance.info("--- MODEL TRAINING BOSHLANDI ---")

        
        train_df = pd.read_csv(train_path)
        X_train = train_df.drop(columns=['is_canceled'])
        y_train = train_df['is_canceled']

        
        models = {
            "Logistic_Regression": LogisticRegression(max_iter=1000, C=0.1), 
            "Random_Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42), 
            "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42), 
            "LightGBM": LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42, verbose=-1) 
        }

        # 3. Cross-Validation (DLP isboti va Barqarorlikni tekshirish)
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        results = {}
        best_f1 = 0
        best_model_name = ""
        best_model_obj = None

        logging_instance.info(f"Cross-Validation boshlandi (5-folds).")

        for name, model in models.items():
            # F1-score mehmonxona loyihasi uchun eng muhim (Bekor qilishni aniq topish kerak)
            cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='f1')
            
            mean_score = cv_scores.mean()
            std_score = cv_scores.std()
            results[name] = mean_score

            logging_instance.info(f"{name}: CV F1-Score = {mean_score:.4f} (+/- {std_score:.4f})")
            print(f"📊 {name}: CV F1 = {mean_score:.4f}")

            
            if mean_score > best_f1:
                best_f1 = mean_score
                best_model_name = name
                best_model_obj = model

        
        logging_instance.info(f"Eng yaxshi model tanlandi: {best_model_name} (F1: {best_f1:.4f})")
        
        best_model_obj.fit(X_train, y_train)
        
        os.makedirs(model_save_path, exist_ok=True)
        model_file = os.path.join(model_save_path, "best_hotel_model.pkl")
        joblib.dump(best_model_obj, model_file)

        logging_instance.info(f"Model saqlandi: {model_file}")
        
        print("\n" + "="*40)
        print(f"🏆 G'OLIB MODEL: {best_model_name}")
        print(f"🎯 F1-SCORE: {best_f1:.4f}")
        print(f"📂 MANZIL: {model_file}")
        print("="*40)

        return best_model_name, best_f1

    except Exception as e:
        logging_instance.error(f"Trainingda xatolik: {str(e)}")
        raise e

if __name__ == "__main__":
    # Test qilish uchun
    TRAIN_CSV = "data/engineered/train_final.csv"
    MODEL_DIR = "models"
    train_models(TRAIN_CSV, MODEL_DIR)