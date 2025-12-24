import os
import joblib
import pandas as pd
from src.logger import logging_instance
from src.feature_eng import run_feature_engineering
from src.preprocessing import initiate_preprocessing
from src.evaluator import evaluate_model
from src.ensemble import create_stacking_model 

def main():
    try:
        logging_instance.info("Loyiha 'Ensemble Stacking Pipeline' bosqichiga o'tdi.")

        
        RAW_PATH = "data/raw/hotel_bookings_updated_2024.csv"
        OUTPUT_DIR = "data/engineered" 
        MODEL_DIR = "models"
        PARAMS_JSON = "configs/best_params.json"
        
        
        df_raw = pd.read_csv(RAW_PATH)
        df_fe = run_feature_engineering(df_raw, OUTPUT_DIR)
        train_df, test_df = initiate_preprocessing(df_fe, OUTPUT_DIR)

        
        logging_instance.info("--- STACKING ENSEMBLE TRAINING BOSHLANDI ---")
        X_train = train_df.drop(columns=['is_canceled'])
        y_train = train_df['is_canceled']

        model = create_stacking_model(PARAMS_JSON)
        
        print("🚀 Stacking model o'qitilmoqda (bu bir necha daqiqa olishi mumkin)...")
        model.fit(X_train, y_train)

        
        os.makedirs(MODEL_DIR, exist_ok=True)
        model_path = os.path.join(MODEL_DIR, "best_hotel_model.pkl")
        joblib.dump(model, model_path)
        logging_instance.info(f"Stacking model saqlandi: {model_path}")

        
        test_path = os.path.join(OUTPUT_DIR, "test_final.csv")
        test_f1 = evaluate_model(test_path, model_path)

        print("\n" + "="*55)
        print("🏆 FINAL STACKING ENSEMBLE NATIJASI:")
        print(f"🎯 TEST F1-SCORE: {test_f1:.4f}")
        print(f"💾 MODEL: {model_path}")
        print("="*55)

    except Exception as e:
        logging_instance.error(f"Loyiha ijrosida xatolik: {e}")
        print(f"❌ Xatolik: {e}")

if __name__ == "__main__":
    main()