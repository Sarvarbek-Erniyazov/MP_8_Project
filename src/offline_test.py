import pandas as pd
import joblib
import os
from sklearn.metrics import confusion_matrix, accuracy_score
from src.logger import logging_instance

def run_offline_test(test_data_path, model_path):
    try:
        logging_instance.info("--- OFFLINE TESTING BOSHLANDI ---")
        
        
        test_df = pd.read_csv(test_data_path)
        model = joblib.load(model_path)
        
        X_unseen = test_df.drop(columns=['is_canceled'])
        y_true = test_df['is_canceled']
        
        
        y_pred = model.predict(X_unseen)
        y_probs = model.predict_proba(X_unseen)[:, 1] 
        
        
        results = test_df.copy()
        results['prediction'] = y_pred
        results['probability_percent'] = (y_probs * 100).round(2)
        results['is_correct'] = results['prediction'] == results['is_canceled']
        
        
        acc = accuracy_score(y_true, y_pred)
        print(f"\n✅ Offline Test Accuracy: {acc:.4f}")
        
        
        errors = results[results['is_correct'] == False]
        print(f"⚠️ Jami xatolar soni: {len(errors)} ta (umumiy {len(test_df)} tadan)")
        
        
        print("\n🔍 Model shubha qilgan (xato topgan) holatlardan namuna:")
        print(errors[['lead_time', 'probability_percent', 'is_canceled']].head())
        
        
        os.makedirs('data/predictions', exist_ok=True)
        results.to_csv('data/predictions/offline_test_results.csv', index=False)
        logging_instance.info("Offline test natijalari 'data/predictions/' papkasiga saqlandi.")
        
    except Exception as e:
        logging_instance.error(f"Offline testingda xatolik: {e}")
        print(f"❌ Xatolik: {e}")

if __name__ == "__main__":
    run_offline_test("data/engineered/test_final.csv", "models/best_hotel_model.pkl")