import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import os

MODEL_PATH = "models/best_hotel_model.pkl"
PREPROCESSOR_PATH = "models/preprocessor_full.pkl"

app = FastAPI(title="Hotel Booking API", version="2.0")

try:
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    print("✅ Model va Preprocessor yuklandi!")
except Exception as e:
    print(f"❌ Yuklashda xatolik: {e}")

class BookingInput(BaseModel):
    hotel: str
    city: str
    lead_time: int
    arrival_date_month: str
    arrival_date_week_number: int
    arrival_date_day_of_month: int
    stays_in_weekend_nights: int
    stays_in_week_nights: int
    adults: int
    children: float
    babies: int
    meal: str
    country: str
    market_segment: str
    distribution_channel: str
    is_repeated_guest: int
    previous_cancellations: int
    previous_bookings_not_canceled: int
    reserved_room_type: str
    assigned_room_type: str
    booking_changes: int
    deposit_type: str
    agent: float
    company: float
    days_in_waiting_list: int
    customer_type: str
    adr: float
    required_car_parking_spaces: int
    total_of_special_requests: int

@app.post("/predict")
def predict(data: BookingInput):
    try:
        df = pd.DataFrame([data.dict()])
        
        
        df['FE_total_people'] = df['adults'] + df['children'] + df['babies']
        df['FE_total_stay'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
        df['FE_price_per_person'] = df['adr'] / (df['FE_total_people'].replace(0, 1))
        df['FE_cancel_ratio'] = df['previous_cancellations'] / (df['previous_cancellations'] + df['previous_bookings_not_canceled'] + 1e-5)
        summer_months = ['June', 'July', 'August']
        df['FE_is_summer'] = df['arrival_date_month'].apply(lambda x: 1 if x in summer_months else 0)

        
        processed_data = preprocessor.transform(df)
        
        
        if hasattr(model, 'feature_name_'):
            feature_names = model.feature_name_
            
            if isinstance(processed_data, pd.DataFrame):
                final_data = processed_data[feature_names]
            else:
                
                final_data = processed_data[:, :35]
        else:
            final_data = processed_data[:, :35]

        
        prediction = model.predict(final_data)[0]
        probability = model.predict_proba(final_data)[0][1]
        
        return {
            "status": "Canceled" if int(prediction) == 1 else "Not Canceled",
            "probability": f"{round(float(probability) * 100, 2)}%"
        }
    except Exception as e:
        print(f"🛑 Xatolik: {str(e)}")
        return {"error": str(e)}