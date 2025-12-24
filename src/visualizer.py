import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os
from sklearn.metrics import confusion_matrix, classification_report

def generate_reports():
    
    os.makedirs('reports', exist_ok=True)
    
    
    model_scores = {
        'Logistic Regression': 0.7191,
        'Random Forest': 0.8197,
        'LightGBM': 0.8293,
        'XGBoost': 0.8313,
        'STACKING (Final)': 0.8469
    }
    
    plt.figure(figsize=(10, 6))
    colors = ['gray', 'gray', 'gray', 'blue', 'green']
    sns.barplot(x=list(model_scores.keys()), y=list(model_scores.values()), palette=colors)
    plt.title('Modellararo F1-Score solishtiruvi', fontsize=15)
    plt.ylim(0.6, 0.9)
    plt.ylabel('F1-Score')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('reports/metrics_comparison.png', bbox_inches='tight')
    plt.close()

    
    model = joblib.load('models/best_hotel_model.pkl')
    
    xgb_model = model.named_estimators_['xgb']
    
    
    test_df = pd.read_csv('data/engineered/test_final.csv')
    feature_names = test_df.drop(columns=['is_canceled']).columns
    
    feat_importances = pd.Series(xgb_model.feature_importances_, index=feature_names)
    plt.figure(figsize=(10, 8))
    feat_importances.nlargest(10).sort_values().plot(kind='barh', color='skyblue')
    plt.title('Top 10 ta eng muhim faktor (Feature Importance)')
    plt.xlabel('Muhimlik darajasi')
    plt.savefig('reports/feature_importance.png', bbox_inches='tight')
    plt.close()

    
    y_true = test_df['is_canceled']
    y_pred = model.predict(test_df.drop(columns=['is_canceled']))
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Yakuniy modelning xato va muvaffaqiyat jadvali')
    plt.ylabel('Haqiqiy holat')
    plt.xlabel('Model bashorati')
    plt.savefig('reports/confusion_matrix.png', bbox_inches='tight')
    plt.close()

    print("✅ Barcha vizual hisobotlar 'reports/' papkasiga saqlandi!")

if __name__ == "__main__":
    generate_reports()