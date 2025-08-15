import numpy as np
import pickle
from flask import Flask, request, jsonify


# สร้าง Flask application
app = Flask(__name__)


# โหลดโมเดลและ scaler ที่บันทึกไว้
try:
    with open('iris_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    # เก็บชื่อสายพันธุ์ไว้เพื่อใช้ตอบกลับ
    iris_target_names = ['setosa', 'versicolor', 'virginica']
    
except FileNotFoundError:
    print("Error: ไม่พบไฟล์โมเดล 'iris_model.pkl' หรือ 'iris_scaler.pkl'")
    print("กรุณารันสคริปต์ฝึกโมเดลก่อน")
    model = None
    scaler = None


# สร้าง Endpoint สำหรับการทำนายผล (Prediction)


# postman http://127.0.0.1:5001/predict : {"features":[1,2,3,4]}
@app.route('/predict', methods=['POST'])
def predict():
    # รับข้อมูล JSON จาก request
    data = request.get_json(force=True)
    
    try:
        # แปลงข้อมูล features ให้เป็น numpy array
        features = np.array(data['features'])
        
        # ตรวจสอบมิติของข้อมูล (ต้องเป็น 2D array)
        if features.ndim == 1:
            features = features.reshape(1, -1) # แปลง [f1, f2, f3, f4] ให้เป็น [[f1, f2, f3, f4]]


        # ทำนายผล
        prediction_index = model.predict(features)
        
        # ดึงชื่อคลาสจากการทำนาย
        predicted_class_name = iris_target_names[prediction_index[0]]
        
        # สร้างผลลัพธ์ที่จะส่งกลับเป็น JSON
        result = {
            'input_features': data['features'],
            'predicted_class': predicted_class_name
        }
        
        return jsonify(result)


    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Endpoint หลักสำหรับทดสอบว่า API ทำงานหรือไม่
@app.route('/', methods=['GET'])
def index():
    return "<h1>Iris Prediction API</h1><p>Use the /predict endpoint with a POST request.</p>"


# รัน Flask server
if __name__ == '__main__':
    # app.run(debug=True) # ใช้สำหรับตอนพัฒนา
    app.run(host='0.0.0.0', port=5001) # ใช้สำหรับ production หรือให้เครื่องอื่นเรียกได้
