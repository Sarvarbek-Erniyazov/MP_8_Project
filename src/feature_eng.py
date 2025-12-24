import os
import pandas as pd
import numpy as np
from src.logger import logging_instance

def run_feature_engineering(df, output_path):
    try:
        logging_instance.info("--- FEATURE ENGINEERING BOSHLANDI ---")
        
        
        df['FE_total_people'] = df['adults'] + df['children'] + df['babies']
        df['FE_total_stay'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
        df['FE_price_per_person'] = df['adr'] / (df['FE_total_people'].replace(0, 1))
        
        
        df['FE_cancel_ratio'] = df['previous_cancellations'] / (df['previous_cancellations'] + df['previous_bookings_not_canceled'] + 1e-5)
        
        
        if 'arrival_date_month' in df.columns:
            summer_months = ['June', 'July', 'August']
            df['FE_is_summer'] = df['arrival_date_month'].apply(lambda x: 1 if x in summer_months else 0)

        logging_instance.info("Yangi mantiqiy ustunlar (FE_) muvaffaqiyatli qo'shildi.")

        
        os.makedirs(output_path, exist_ok=True)
        full_path = os.path.join(output_path, "engineered_data.csv")
        df.to_csv(full_path, index=False)
        
        logging_instance.info(f"Engineered CSV saqlandi: {full_path}")
        return df

    except Exception as e:
        logging_instance.error(f"Feature Engineeringda xatolik: {str(e)}")
        raise e