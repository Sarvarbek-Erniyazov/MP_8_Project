import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer

from sklearn.preprocessing import RobustScaler, OneHotEncoder 
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from category_encoders import TargetEncoder
from src.logger import logging_instance

def initiate_preprocessing(df, processed_data_path):
    try:
        logging_instance.info("--- ILMIY ASOSLANGAN PREPROCESSING (V2) BOSHLANDI ---")

        
        leaky_cols = ['reservation_status', 'reservation_status_date', 'arrival_date_year']
        df = df.drop(columns=[col for col in leaky_cols if col in df.columns])
        
        X = df.drop(columns=['is_canceled'])
        y = df['is_canceled']

        # Train-Test Split (DLP oltin qoidasi)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        
        high_card_cols = ['country', 'city']
        cat_cols = [c for c in X.select_dtypes(include='object').columns if c not in high_card_cols]
        num_cols = X.select_dtypes(exclude='object').columns.tolist()

        
        num_pipeline = Pipeline([
            ('imputer', KNNImputer(n_neighbors=5)),
            ('scaler', RobustScaler()) # Outlierlarga chidamli scaler
        ])

        cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        target_enc_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
            ('target_enc', TargetEncoder(smoothing=25)) # Smoothing oshirildi (overfittingdan himoya)
        ])

        preprocessor = ColumnTransformer([
            ('num', num_pipeline, num_cols),
            ('cat', cat_pipeline, cat_cols),
            ('high_card', target_enc_pipeline, high_card_cols)
        ])

        
        X_train_transformed = preprocessor.fit_transform(X_train, y_train)
        X_test_transformed = preprocessor.transform(X_test)
        
        ohe_features = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(cat_cols)
        all_feature_names = num_cols + list(ohe_features) + high_card_cols

        
        logging_instance.info("Feature Selection: Top 35 ta ustun tanlanmoqda (Deposit turlarini qamrab olish uchun).")
        selector = SelectKBest(score_func=mutual_info_classif, k=35)
        
        X_train_final = selector.fit_transform(X_train_transformed, y_train)
        X_test_final = selector.transform(X_test_transformed)
        
        selected_mask = selector.get_support()
        selected_features = [f for f, s in zip(all_feature_names, selected_mask) if s]

        
        os.makedirs(processed_data_path, exist_ok=True)
        
        train_df = pd.DataFrame(X_train_final, columns=selected_features)
        train_df['is_canceled'] = y_train.values
        
        test_df = pd.DataFrame(X_test_final, columns=selected_features)
        test_df['is_canceled'] = y_test.values

        
        train_df.to_csv(os.path.join(processed_data_path, "train_final.csv"), index=False)
        test_df.to_csv(os.path.join(processed_data_path, "test_final.csv"), index=False)
        joblib.dump(preprocessor, os.path.join(processed_data_path, "preprocessor_full.pkl"))
        joblib.dump(selector, os.path.join(processed_data_path, "feature_selector.pkl"))

        logging_instance.info(f"YAKUN: Preprocessing tugadi. {len(selected_features)} ta ustun saqlandi.")
        return train_df, test_df

    except Exception as e:
        logging_instance.error(f"Preprocessing v2 da xatolik: {str(e)}")
        raise e