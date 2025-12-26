import gradio as gr
import requests

def hotel_ui_predict(lead_time, month, country, deposit_type, adr, special_req): 
    url = "http://127.0.0.1:8000/predict"
    
    payload = {
        "hotel": "Resort Hotel", 
        "city": "Lisbon",        
        "lead_time": int(lead_time), 
        "arrival_date_month": month,
        "arrival_date_week_number": 30, 
        "arrival_date_day_of_month": 15,
        "stays_in_weekend_nights": 1, 
        "stays_in_week_nights": 2,
        "adults": 2, "children": 0, "babies": 0, "meal": "BB",
        "country": country, "market_segment": "Online TA",
        "distribution_channel": "TA/TO", "is_repeated_guest": 0,
        "previous_cancellations": 0, "previous_bookings_not_canceled": 0,
        "reserved_room_type": "A", "assigned_room_type": "A",
        "booking_changes": 0, "deposit_type": deposit_type, "agent": 9,
        "company": 0, "days_in_waiting_list": 0, "customer_type": "Transient",
        "adr": float(adr), "required_car_parking_spaces": 0, "total_of_special_requests": int(special_req)
    }
    
    try:
        response = requests.post(url, json=payload)
        res = response.json()
        if "error" in res:
            return f"🛑 API Xatosi: {res['error']}"
        return f"Model Bashorati: {res['status']}\nEhtimollik: {res['probability']}"
    except Exception as e:
        return f"❌ Xatolik: API bilan bog'lanib bo'lmadi! ({str(e)})"

demo = gr.Interface(
    fn=hotel_ui_predict,
    inputs=[
        gr.Number(label="Lead Time (Oldindan bron qilish kunlari)"),
        gr.Dropdown(["July", "August", "September", "October", "November", "December", "January", "February", "March", "April", "May", "June"], label="Kelish oyi"),
        gr.Textbox(label="Davlat kodi (PRT, GBR, USA)"),
        gr.Radio(["No Deposit", "Non Refund", "Refundable"], label="Depozit turi"),
        gr.Number(label="ADR (Kunlik narx)"),
        gr.Slider(0, 5, step=1, label="Maxsus so'rovlar soni")
    ],
    outputs="text",
    title="🏢 Hotel Booking Cancellation AI Predictor"
)

if __name__ == "__main__":
    demo.launch()