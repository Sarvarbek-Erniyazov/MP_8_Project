import joblib
import pandas as pd
import os

def live_prediction():
    try:
        model_path = "models/best_hotel_model.pkl"
        test_data_path = "data/engineered/test_final.csv"

        if not os.path.exists(model_path):
            print("❌ Model topilmadi!")
            return

        model = joblib.load(model_path)
        test_df = pd.read_csv(test_data_path)
        
        print("\n" + "="*50)
        print("🏨 MEHMONXONA BRONINI BASHORAT QILISH TIZIMI")
        print("="*50)
        
        while True:
            choice = input("\n[1] Tasodifiy mijozni tekshirish\n[2] Chiqish\nTanlovingiz: ")
            
            if choice == '1':
                sample = test_df.sample(1)
                real_status = int(sample['is_canceled'].values[0])
                X_sample = sample.drop(columns=['is_canceled'])
                
                # Bashorat
                prob = model.predict_proba(X_sample)[0][1]
                pred = model.predict(X_sample)[0]
                
                print("\n" + "-"*30)
                print(f"📊 Mijoz tahlili:")
                print(f"🔹 Lead Time (Standardizatsiyalangan): {sample['lead_time'].values[0]:.4f}")
                
                status_text = "🔴 BEKOR QILADI" if pred == 1 else "🟢 KELADI (Yashaydi)"
                print(f"\n🤖 MODEL BASHORATI: {status_text}")
                print(f"🎯 Ishonch darajasi: {prob*100:.2f}%")
                
                real_text = "Bekor qilgan" if real_status == 1 else "Kelgan"
                print(f"✅ HAQIQIY HOLAT: {real_text}")
                
                if int(pred) == real_status:
                    print("✨ Model TO'G'RI topdi!")
                else:
                    print("⚠️ Model bu safar adashdi.")
                print("-"*30)
                
            elif choice == '2':
                print("Dastur yakunlandi. Omad!")
                break
                
    except Exception as e:
        print(f"❌ Xatolik yuz berdi: {e}")

if __name__ == "__main__":
    live_prediction()