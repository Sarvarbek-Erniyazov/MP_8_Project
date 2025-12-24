import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from src.logger import logging_instance

def evaluate_model(test_path, model_path):
    try:
        logging_instance.info("--- MODELNI BAHOLASH (EVALUATION) BOSHLANDI ---")

        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model topilmadi: {model_path}")
        
        model = joblib.load(model_path)
        test_df = pd.read_csv(test_path)

        X_test = test_df.drop(columns=['is_canceled'])
        y_test = test_df['is_canceled']

        
        y_pred = model.predict(X_test)
        
        
        f1 = f1_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)

        logging_instance.info(f"Test Set F1-Score: {f1:.4f}")
        logging_instance.info(f"Classification Report:\n{report}")

        
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Yuradi', 'Bekor qiladi'], 
                    yticklabels=['Yuradi', 'Bekor qiladi'])
        plt.xlabel('Bashorat')
        plt.ylabel('Haqiqiy holat')
        plt.title('Confusion Matrix: Model qayerda adashmoqda?')
        
        
        os.makedirs('reports', exist_ok=True)
        plt.savefig('reports/confusion_matrix.png')
        
        print("\n" + "!"*40)
        print("📊 YAKUNIY TEST NATIJALARI:")
        print(f"🎯 Test F1-Score: {f1:.4f}")
        print("\n📝 Classification Report:")
        print(report)
        print("!"*40)

        return f1

    except Exception as e:
        logging_instance.error(f"Baholashda xatolik: {str(e)}")
        raise e