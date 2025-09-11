import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials, auth, firestore
import requests
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import pickle
import json
# from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.sequence import pad_sequences
from io import BytesIO
import base64
from PIL import Image
import pandas as pd
import joblib                                                                                                                                                                                                                                   

app = Flask(__name__)
CORS(app) # aktifkan CORS di semua route


# --- Init Firebase Admin ---
cred = credentials.Certificate("final-project-f2032-firebase-adminsdk-fbsvc-374faa76da.json")
firebase_admin.initialize_app(cred)

# Firebase Web API Key (ambil dari Firebase console > Project settings > General > Web API Key)
FIREBASE_API_KEY = os.getenv("FIREBASE_API_KEY", "AIzaSyCkuYpZiiwgPI5ouuBseQUlYJWC3MdAt9Q")

db = firestore.client()

# Load Model
model_aug = load_model('model_aug_ver5.keras')
model_cnn = load_model('gru_glove_improved_model.keras')

df = pd.read_csv('logistics.csv')


# API: Jumlah maintenance_cost per bulan berdasarkan last_maintenance_date
@app.route('/monthly-maintenance-cost', methods=['GET'])
def monthly_maintenance_cost():
    try:
        if 'Last_Maintenance_Date' not in df.columns or 'Maintenance_Cost' not in df.columns:
            return jsonify({'error': 'Kolom Last_Maintenance_Date atau Maintenance_Cost tidak ditemukan di CSV'}), 400
        # Pastikan format tanggal
        df['Last_Maintenance_Date'] = pd.to_datetime(df['Last_Maintenance_Date'], errors='coerce')
        df['month'] = df['Last_Maintenance_Date'].dt.to_period('M').astype(str)
        result = df.groupby('month')['Maintenance_Cost'].sum().round(2).to_dict()
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
# --- API Distribusi Maintenance Required dari CSV ---
@app.route('/maintenance-required-distribution', methods=['GET'])
def maintenance_required_distribution():
    try:
        # Pastikan kolom benar
        if 'Maintenance_Required' not in df.columns:
            return jsonify({'error': 'Kolom Maintenance_Required tidak ditemukan di CSV'}), 400
        count_0 = int((df['Maintenance_Required'] == 0).sum())
        count_1 = int((df['Maintenance_Required'] == 1).sum())
        return jsonify({
            'class_0': count_0,
            'class_1': count_1
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 2. Rata-rata biaya maintenance per tipe kendaraan
@app.route('/avg-maintenance-cost-by-vehicle-type', methods=['GET'])
def avg_maintenance_cost_by_vehicle_type():
    try:
        if 'Vehicle_Type' not in df.columns or 'Maintenance_Cost' not in df.columns:
            return jsonify({'error': 'Kolom Vehicle_Type atau Maintenance_Cost tidak ditemukan di CSV'}), 400
        result = df.groupby('Vehicle_Type')['Maintenance_Cost'].mean().round(2).to_dict()
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 3. Rata-rata konsumsi bahan bakar per tipe kendaraan
@app.route('/avg-fuel-consumption-by-vehicle-type', methods=['GET'])
def avg_fuel_consumption_by_vehicle_type():
    try:
        if 'Vehicle_Type' not in df.columns or 'Fuel_Consumption' not in df.columns:
            return jsonify({'error': 'Kolom Vehicle_Type atau Fuel_Consumption tidak ditemukan di CSV'}), 400
        result = df.groupby('Vehicle_Type')['Fuel_Consumption'].mean().round(2).to_dict()
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 4. Distribusi kondisi cuaca saat pengiriman
@app.route('/weather-condition-distribution', methods=['GET'])
def weather_condition_distribution():
    try:
        if 'Weather_Conditions' not in df.columns or 'Maintenance_Required' not in df.columns:
            return jsonify({'error': 'Kolom Weather_Conditions atau Maintenance_Required tidak ditemukan di CSV'}), 400
        grouped = df.groupby('Maintenance_Required')['Weather_Conditions'].value_counts().unstack(fill_value=0)
        # Ubah ke format dict: {class: {weather: count, ...}, ...}
        result = grouped.to_dict(orient='index')
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 4. Distribusi kondisi cuaca saat pengiriman
@app.route('/brake-condition-distribution', methods=['GET'])
def brake_condition_distribution():
    try:
        if 'Brake_Condition' not in df.columns or 'Maintenance_Required' not in df.columns:
            return jsonify({'error': 'Kolom Brake_Condition atau Maintenance_Required tidak ditemukan di CSV'}), 400
        grouped = df.groupby('Maintenance_Required')['Brake_Condition'].value_counts().unstack(fill_value=0)
        # Ubah ke format dict: {class: {brake: count, ...}, ...}
        result = grouped.to_dict(orient='index')    
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 5. Rata-rata waktu pengiriman per kondisi jalan
@app.route('/avg-delivery-time-by-road-condition', methods=['GET'])
def avg_delivery_time_by_road_condition():
    try:
        if 'Road_Conditions' not in df.columns or 'Delivery_Times' not in df.columns:
            return jsonify({'error': 'Kolom Road_Conditions atau Delivery_Times tidak ditemukan di CSV'}), 400
        result = df.groupby('Road_Conditions')['Delivery_Times'].mean().round(2).to_dict()
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 6. Jumlah kendaraan berdasarkan usia kendaraan
@app.route('/vehicle-age-distribution', methods=['GET'])
def vehicle_age_distribution():
    try:
        if 'Vehicle_Age' not in df.columns:
            return jsonify({'error': 'Kolom Vehicle_Age tidak ditemukan di CSV'}), 400
        # Kelompokkan usia kendaraan dalam rentang (binning)
        bins = [0, 3, 6, 10, 15, 20, 100]
        labels = ['0-3', '4-6', '7-10', '11-15', '16-20', '20+']
        df['Vehicle_Age_Group'] = pd.cut(df['Vehicle_Age'], bins=bins, labels=labels, right=True, include_lowest=True)
        result = df['Vehicle_Age_Group'].value_counts().sort_index().to_dict()
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route("/")
def home():
    return {"message": "Hello from Flask with Uvicorn!"}


# --- Login User (pakai Firebase REST API) ---
@app.route("/login", methods=["POST"])
def login():
    data = request.json
    
    email = data.get("email")
    password = data.get("password")

    url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={FIREBASE_API_KEY}"
    payload = {
        "email": email,
        "password": password,
        "returnSecureToken": True
    }
    
    print("Payload diterima:", payload)
    try:
        res = requests.post(url, json=payload)
        result = res.json()

        if "idToken" in result:
            return jsonify({
                "message": "Login success",
                "idToken": result["idToken"],
                "refreshToken": result["refreshToken"],
                "uid": result["localId"]
            }), 200
        else:
            return jsonify({"message": "Login failed", "error": result}), 401
    except Exception as e:
        return jsonify({"message": "Error login", "error": str(e)}), 400

# --- Verify ID Token ---
@app.route("/verify_token", methods=["POST"])
def verify_token():
    data = request.json
    id_token = data.get("idToken")

    try:
        decoded_token = auth.verify_id_token(id_token)
        uid = decoded_token['uid']
        email = decoded_token.get('email')
        return jsonify({"message": "Token valid", "uid": uid, "email": email}), 200
    except Exception as e:
        return jsonify({"message": "Invalid token", "error": str(e)}), 401

    text = str(text).lower()
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^a-z ]', '', text)
    tokens = re.findall(r'\b\w+\b', text)
    tokens_neg = []
    skip = False
    for i, word in enumerate(tokens):
        if skip:
            skip = False
            continue
        if word in negation_words and i+1 < len(tokens):
            tokens_neg.append(word + '_' + tokens[i+1])
            skip = True
        else:
            tokens_neg.append(word)
    tokens_clean = [
        word for word in tokens_neg
        if (word not in all_stopwords or word in otomotif_keywords) and len(word) > 2
    ]
    tokens_clean = [
        word if word in otomotif_keywords else lemmatizer.lemmatize(word)
        for word in tokens_clean
    ]
    return ' '.join(tokens_clean)

# --- NLP Prediction ---
@app.route('/nlp-predict', methods=['POST'])
def predictNlp():
    
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
        
    with open('label_mapping.json', 'r') as f:
        label_mapping = json.load(f)

    # y_train_aug = np.load('y_train_aug.npy')
    
    # Mengambil data dari form HTML
    data = request.json
    text = data["text"]
    vehicle_id = data["vehicle_id"]
    plate_number = data["plate_number"]
    driver_name = data["driver_name"]
    company_name = data["company_name"]
    
    maxlen = 80
    review_seq = tokenizer.texts_to_sequences([text])
    review_pad = pad_sequences(review_seq, maxlen=maxlen, padding='post')
    # Melakukan prediksi
    pred = model_cnn.predict(review_pad)
    # print('pred: ', pred)
    # Ambil index nilai tertinggi
    sorted_idx = np.argsort(pred[0])[::-1]
    top1_idx = sorted_idx[0]
    next2_idx = sorted_idx[1:3]
    # Gabungkan: nilai tertinggi + 2 terbesar berikutnya
    # top_idx = np.concatenate(([top1_idx], next2_idx))

    # Mapping top 3 indices to labels
    index_to_label = {v: k for k, v in label_mapping.items()}
    top3_labels = [index_to_label[int(idx)] for idx in next2_idx]
    # Remove 'OTHER' if present
    top3_labels = [label for label in top3_labels if label != 'OTHER']
    print('Top 3 labels (no OTHER):', top3_labels)
    y_pred = np.argmax(pred, axis=1)
    label = [index_to_label[int(idx)] for idx in y_pred]
    print("y Prob:", y_pred)
    print("Predicted label:", label)
    try:
        updatevehicle = updateReportedIssue(vehicle_id)
        print(updatevehicle,'hasil update vehicle')
        if not updatevehicle:
            return jsonify({"message": "Error updating vehicle last_maintenance"}), 500
        report_ref = db.collection("reported_issues").document()
        report_ref.set({
            # general info
            "plate_number": plate_number,
            "driver_name": driver_name,
            "company_name": company_name,
            # vehicle info
            "vehicle_id": vehicle_id,
            # nlp prediction
            "reported_issues": text,
            "component_problem": top3_labels,
            "status": "On Progress", # status default
            "created_at": firestore.SERVER_TIMESTAMP,
        }, merge=True)

        return jsonify({
                'component_problem': label,
                'other_component': top3_labels,
            })

    except Exception as e:
        return jsonify({"message": "Error creating report", "error": str(e)}), 500
    
# --- CRUD User ---
# --- Register User ---
@app.route("/register", methods=["POST"])
def register():
    data = request.json
    email = data.get("email")
    password = data.get("password")
    user_name = data.get("username")
    address = data.get("address")
    phone_number = data.get("phone")
    company_name = data.get("company")

    try:
        user = auth.create_user(email=email, password=password)
        
        db.collection("users").document(user.uid).set({
            "email": email,
            "username": user_name,
            "address": address,
            "phone_number": phone_number,
            "created_at": firestore.SERVER_TIMESTAMP,
            "role": "customer",  # default role
            "company_name": company_name
        }, merge=True)

        return jsonify({"message": "User created", "uid": user.uid, "email": user.email}), 201
    except Exception as e:
        return jsonify({"message": "Register failed", "error": str(e)}), 400
    
# -- Get User by UID --
@app.route("/users/<uid>", methods=["GET"])
def get_user(uid):
    try:
        # Ambil dokumen user dari Firestore
        user_ref = db.collection("users").document(uid).get()

        if user_ref.exists:
            return jsonify({
                "uid": uid,
                "data": user_ref.to_dict()
            }), 200
        else:
            return jsonify({"message": "User not found"}), 404

    except Exception as e:
        return jsonify({"message": "Error fetching user", "error": str(e)}), 500

# -- Update User by UID --
@app.route("/users/<uid>", methods=["PUT"])
def update_user(uid):
    data = request.json  # ambil data dari body request
    print(data,'data yang di update')
    try:
        user_ref = db.collection("users").document(uid)
        if not user_ref.get().exists:
            return jsonify({"message": "User not found"}), 404

        # Update hanya field yang dikirim
        user_ref.update(data)

        return jsonify({
            "message": "User updated successfully",
            "uid": uid,
            "updated_fields": data
        }), 200

    except Exception as e:
        return jsonify({"message": "Error updating user", "error": str(e)}), 500

# -- Delete User by UID --
@app.route("/users/<uid>", methods=["DELETE"])
def delete_user(uid):
    try:
        user_ref = db.collection("users").document(uid)

        if not user_ref.get().exists:
            return jsonify({"message": "User not found"}), 404

        # Hapus data user dari Firestore
        user_ref.delete()

        # Opsional: hapus juga dari Firebase Auth
        try:
            auth.delete_user(uid)
        except Exception as e:
            # Kalau gagal hapus di Auth, log error tapi tetap hapus dari Firestore
            print("Error deleting from Firebase Auth:", str(e))

        return jsonify({"message": f"User {uid} deleted successfully"}), 200

    except Exception as e:
        return jsonify({"message": "Error deleting user", "error": str(e)}), 500

# -- Get All Users --
@app.route("/users", methods=["GET"])
def get_all_users():
    try:
        users_ref = db.collection("users").stream()

        users = []
        for user in users_ref:
            users.append({
                "uid": user.id,
                **user.to_dict()
            })

        return jsonify({
            "count": len(users),
            "users": users
        }), 200

    except Exception as e:
        return jsonify({"message": "Error fetching users", "error": str(e)}), 500

# -- Update vehicle maintenance_status by vehicle_id --
def updateStatusMaintenance(vehicle_id, status):
    try:
        vehicle_ref = db.collection("vehicles").document(vehicle_id)
        if not vehicle_ref.get().exists:
            return jsonify({"message": "vehicle not found"}), 404
        
        vehicle_ref = db.collection("vehicles").document(vehicle_id)

        # Tambah 1 ke maintenance_status
        vehicle_ref.update({
            "maintenance_status": status
        })
        print("vehicle maintenance_status updated successfully")
        return True

    except Exception as e:
        print("Error updating vehicle maintenance_status", str(e))
        return False
    
# -- Maintenance Prediction --
@app.route("/maintenance", methods=["POST"])
def maintenanceVehicle():
    data = request.json
    vehicle_id = data.get("vehicle_id")
    usage_hours = data.get("usage_hours")
    load_capacity = data.get("load_cap")
    actual_load = data.get("actual_load")
    engine_temp = data.get("engine_temp")
    brake_condition = data.get("brake_condition")
    tire_pressure = data.get("tire_pressure")
    fuel_consumption = data.get("fuel_consume")
    vibration_levels = data.get("vibration_levels")
    oil_quality = data.get("oil_quality")
    anomalies_detected = data.get("anomalies_detected")
    battery_status = data.get("battery_status")
    route_info = data.get("route_info")
    weather_condition = data.get("weather_cond")
    road_condition = data.get("road_cond")
    plate_number = data.get("plate_number")
    company_name = data.get("company_name")
    vehicle_type = data.get("vehicle_type")
    vehicle_model = data.get("vehicle_model")
    manufacture_year = data.get("manufacture_year")
    maintenance_type = data.get("maintenance_type")
    maintenance_cost = data.get("maintenance_cost")
    deliverty_times = data.get("delivery_times")
    last_maintenance = data.get("last_maintenance")
    impact_on_efficiency = data.get("impact_on_efficiency")
    # print(data,'data yang masuk') # ok
    try:
        predict = predict_classification(data)
        print("Predictive Score: ", predict)
        if predict == False:
            return jsonify({"message": "Error in prediction"}), 500
        
        maintenance_ref = db.collection("maintenance").document()
        updatevehicle = updateStatusMaintenance(vehicle_id, predict)
        if not updatevehicle:
            return jsonify({"message": "Error updating vehicle maintenance_status"}), 500
            
        maintenance_ref.set({
            # general info
            "vehicle_id": vehicle_id,
            "created_at": firestore.SERVER_TIMESTAMP,
            # usage data
            "usage_hours": usage_hours,
            "load_capacity": load_capacity,
            "actual_load": actual_load,
            # components condition
            "engine_temp": engine_temp,
            "brake_condition": brake_condition,
            "tire_pressure": tire_pressure,
            "fuel_consumption": fuel_consumption,
            "vibration_levels": vibration_levels,
            "oil_quality": oil_quality,
            "anomalies_detected": anomalies_detected,
            "battery_status": battery_status,
            # environment cons
            "route_info": route_info,
            "weather_condition": weather_condition,
            "road_condition": road_condition,
            "plate_number": plate_number,
            "company_name": company_name,
            "vehicle_type": vehicle_type,
            "vehicle_model": vehicle_model,
            "manufacture_year": manufacture_year,
            # maintenance info
            "maintenance_type": maintenance_type,
            "maintenance_cost": maintenance_cost,
            "delivery_times": deliverty_times,
            "last_maintenance": last_maintenance,
            "impact_on_efficiency": impact_on_efficiency,
            # prediction
            # label
            "predictive_score": 0
        }, merge=True)

        return jsonify({"prediction": predict}), 201
    except Exception as e:
        return jsonify({"message": "Error maintenance predict", "error": str(e)}), 500
    
# -- CRUD Vehicle--
# --- Create New Vehicle ---
@app.route("/vehicles", methods=["POST"])
def addVehicle():

    data = request.json
    vehicle_type = data.get("vehicle_type")
    company_name = data.get("company_name")
    plate_number = data.get("plate_number")
    driver_name = data.get("driver_name")
    vehicle_model = data.get("vehicle_model")
    manufacture_year = data.get("manufacture_year")
    print(manufacture_year,'data yang masuk') # ok

    try:
        print(manufacture_year,'data yang masuk di try') # ok
        vehicle_ref = db.collection("vehicles").document()
        vehicle_ref.set({
            # general info
            "plate_number": plate_number,
            "driver_name": driver_name,
            "company_name": company_name,
            # vehicle info
            "vehicle_type": vehicle_type,
            "vehicle_model": vehicle_model,
            "manufacture_year": manufacture_year,
            # last service
            "last_maintenance": "",
            # regression prediction
            "maintenance_status": "",
            # nlp prediction
            "issues_reported": 0,
            # cnn prediction
            "tire_condition": 0,
            "created_at": firestore.SERVER_TIMESTAMP,
        }, merge=True)

        return jsonify({"message": "Vehicle created", "vehicle_id": vehicle_ref.id}), 201
    except Exception as e:
        return jsonify({"message": "Error creating vehicle", "error": str(e)}), 500

# -- Update vehicle by UID --
@app.route("/vehicles/<uid>", methods=["PUT"])
def update_vehicle(uid):
    data = request.json  # ambil data dari body request
    print(data,'data yang di update')
    try:
        vehicle_ref = db.collection("vehicles").document(uid)
        if not vehicle_ref.get().exists:
            return jsonify({"message": "vehicle not found"}), 404

        # Update hanya field yang dikirim
        vehicle_ref.update(data)

        return jsonify({
            "message": "vehicle updated successfully",
            "uid": uid,
            "updated_fields": data
        }), 200

    except Exception as e:
        return jsonify({"message": "Error updating vehicle", "error": str(e)}), 500

# -- Delete vehicle by UID --
@app.route("/vehicles/<uid>", methods=["DELETE"])
def delete_vehicle(uid):
    try:
        vehicle_ref = db.collection("vehicles").document(uid)

        if not vehicle_ref.get().exists:
            return jsonify({"message": "vehicle not found"}), 404

        # Hapus data vehicle dari Firestore
        vehicle_ref.delete()

        return jsonify({"message": f"vehicle {uid} deleted successfully"}), 200

    except Exception as e:
        return jsonify({"message": "Error deleting vehicle", "error": str(e)}), 500

# -- Get vehicle by UID --
@app.route("/vehicles/<uid>", methods=["GET"])
def get_vehicle(uid):
    try:
        # Ambil dokumen vehicle dari Firestore
        vehicle_ref = db.collection("vehicles").document(uid).get()

        if vehicle_ref.exists:
            return jsonify({
                "uid": uid,
                "data": vehicle_ref.to_dict()
            }), 200
        else:
            return jsonify({"message": "vehicle not found"}), 404

    except Exception as e:
        return jsonify({"message": "Error fetching user", "error": str(e)}), 500

# -- Get All Report by UID (VehicleId) --
@app.route("/vehicle-issues/<uid>", methods=["GET"])
def getReportByVehicleID(uid):
    try:
        # Ambil dokumen vehicle dari Firestore
        reports = db.collection("reported_issues").where("vehicle_id", "==", uid).stream()
        allReports = []
        found = False
        for report in reports:
            found = True
            data = report.to_dict()
            allReports.append({
                "id": report.id,
                **data
            })
            print(report.id, data)
        if not found:
            return jsonify({"message": "No report found for this vehicle_"}), 404
            
        
        return jsonify({
            "vehicle_id": uid,
            "data": allReports
        }), 200

    except Exception as e:
        return jsonify({"message": "Error fetching user", "error": str(e)}), 500
    
# -- Get All vehicles --
@app.route("/vehicles", methods=["GET"])
def get_all_vehicles():
    try:
        vehicles_ref = db.collection("vehicles").stream()

        vehicles = []
        for vehicle in vehicles_ref:
            vehicles.append({
                "uid": vehicle.id,
                **vehicle.to_dict()
            })

        return jsonify({
            "count": len(vehicles),
            "vehicles": vehicles
        }), 200

    except Exception as e:
        return jsonify({"message": "Error fetching vehicles", "error": str(e)}), 500

# -- Update vehicle reported_issues by vehicle_id --
def updateReportedIssue(vehicle_id):
    try:
        vehicle_ref = db.collection("vehicles").document(vehicle_id)
        if not vehicle_ref.get().exists:
            return jsonify({"message": "vehicle not found"}), 404
        
        vehicle_ref = db.collection("vehicles").document(vehicle_id)

        # Tambah 1 ke issues_reported
        vehicle_ref.update({
            "issues_reported": firestore.Increment(1)
        })
        print("vehicle reported_issues updated successfully")
        return True

    except Exception as e:
        print("Error updating vehicle last_maintenance", str(e))
        return False
    
# -- Update vehicle last_maintenance by vehicle_id --
def updateLastMaintenance(vehicle_id, last_maintenance):
    print(vehicle_id, last_maintenance,'update vehicle last maintenance')
    try:
        vehicle_ref = db.collection("vehicles").document(vehicle_id)
        if not vehicle_ref.get().exists:
            return jsonify({"message": "vehicle not found"}), 404
        
        vehicle_ref = db.collection("vehicles").document(vehicle_id)

        # Update last_maintenance field
        vehicle_ref.update({
            "last_maintenance": last_maintenance
        })
        print("vehicle last_maintenance updated successfully")
        return True

    except Exception as e:
        print("Error updating vehicle last_maintenance", str(e))
        return False
    
# CRUD Last Maintenance
# --- Add Last Maintenance ---
@app.route("/last-maintenance", methods=["POST"])
def addLastMaintenance():

    data = request.json
    vehicle_id = data.get("vehicle_id")
    plate_number = data.get("plate_number")
    last_maintenance = data.get("last_maintenance")
    maintenance_type = data.get("maintenance_type")
    maintenance_cost = data.get("maintenance_cost")
    downtime_maintenance = data.get("downtime_maintenance")
    failure_history = data.get("failure_history")
    impact_on_effciency = data.get("impact_on_efficiency")

    try:
        upadtevehicle = updateLastMaintenance(vehicle_id, last_maintenance)
        print(upadtevehicle,'hasil update vehicle')
        if not upadtevehicle:
            return jsonify({"message": "Error updating vehicle last_maintenance"}), 500
        last_maintenance_ref = db.collection("last_maintenance").document()
        last_maintenance_ref.set({
            "vehicle_id": vehicle_id,
            "plate_number": plate_number,
            "created_at": firestore.SERVER_TIMESTAMP,
            "last_maintenance": last_maintenance,
            "maintenance_type": maintenance_type,
            "maintenance_cost": maintenance_cost,
            "downtime_maintenance": downtime_maintenance,
            "failure_history": failure_history,
            "impact_on_effciency": impact_on_effciency,
        }, merge=True)


        return jsonify({"message": "Last Maintenace Add", "last_maintenance_id": last_maintenance_ref.id}), 201
    except Exception as e:
        return jsonify({"message": "Error creating Last Maintenace", "error": str(e)}), 500

# -- Get last_maintenance by vehicle UID --
@app.route("/vehicles/<uid>/last-maintenance", methods=["GET"])
def get_last_maintenance(uid):
    try:
        # Query berdasarkan vehicle_id dan ambil created_at paling baru
        query = (
            db.collection("last_maintenance")
            .where("vehicle_id", "==", uid)
            .order_by("created_at", direction=firestore.Query.DESCENDING)
            .limit(1)
        )

        docs = query.stream()
        last_doc = None
        for doc in docs:
            last_doc = doc

        if not last_doc:
            return jsonify({"message": "No maintenance record found"}), 404

        return jsonify({
            "last_maintenance": last_doc.to_dict(),
            "id": last_doc.id
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -- Update last_maintenance by UID --
@app.route("/last-maintenance/<uid>", methods=["PUT"])
def update_last_maintenance(uid):
    data = request.json
    try:
        last_maintenance_ref = db.collection("last_maintenance").document(uid)
        if not last_maintenance_ref.get().exists:
            return jsonify({"message": "last_maintenance not found"}), 404
        print(last_maintenance_ref.get().to_dict(), 'data lama')
        # updateVehicleLastMaintenance(vehicle_id, last_maintenance)
        # Update hanya field yang dikirim
        last_maintenance_ref.update(data)

        return jsonify({
            "message": "last_maintenance updated successfully",
            "uid": uid,
            "updated_fields": data
        }), 200

    except Exception as e:
        return jsonify({"message": "Error updating last_maintenance", "error": str(e)}), 500

# Cnn Prediction
def predictionCNN(file, target_size=(220, 220)):
    try:
        # Load image dari FileStorage langsung
        img = Image.open(file).convert("RGB")
        
        # Resize dan convert ke grayscale
        img_resized = img.resize(target_size).convert("L")  # grayscale

        # Convert ke array
        x = np.array(img_resized) / 255.0
        x = np.expand_dims(x, axis=(0, -1))  # shape (1, H, W, 1)

        # Predict
        y_pred_proba = model_aug.predict(x, batch_size=1)
        class_names = ['cracked', 'normal']
        y_pred_class_name = class_names[np.argmax(y_pred_proba[0])]

        # Convert image ke base64 agar bisa dikirim ke JSON
        buffered = BytesIO()
        img_resized.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return y_pred_class_name
    except Exception as e:
        raise ValueError(f"Error predicting image: {str(e)}")

# cnn Prediction Endpoint
@app.route("/cnn-predict/", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    
    vehicle_id = request.form.get("vehicle_id")
    plate_number = request.form.get("plate_number")
    driver_name = request.form.get("driver_name")
    company_name = request.form.get("company_name")
    position = request.form.get("position")
    status = "Ok"  # default status
    try:
        y_pred_class_name = predictionCNN(file, target_size=(220, 220))
        if y_pred_class_name:
            if y_pred_class_name == 'cracked':
                updateTireCondition(vehicle_id)
                status = "Needs Attention"
            tires_condition_ref = db.collection("tires_conditions").document()
            tires_condition_ref.set({
                "vehicle_id": vehicle_id,
                "plate_number": plate_number,
                "created_at": firestore.SERVER_TIMESTAMP,
                "driver_name": driver_name,
                "company_name": company_name,
                "tire_position": position,
                "tire_condition": y_pred_class_name,
                "status": status, # status default
            }, merge=True)

        return jsonify({
            "predicted_class": y_pred_class_name
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "message": "Error during prediction"
        }), 500

# -- Update vehicle reported_issues by vehicle_id --
def updateReportedIssue(vehicle_id):
    try:
        vehicle_ref = db.collection("vehicles").document(vehicle_id)
        if not vehicle_ref.get().exists:
            return jsonify({"message": "vehicle not found"}), 404
        
        vehicle_ref = db.collection("vehicles").document(vehicle_id)

        # Tambah 1 ke issues_reported
        vehicle_ref.update({
            "issues_reported": firestore.Increment(1)
        })
        print("vehicle reported_issues updated successfully")
        return True

    except Exception as e:
        print("Error updating vehicle report", str(e))
        return False
    
# -- Update vehicle tire_condition by vehicle_id --
def updateTireCondition(vehicle_id):
    try:
        vehicle_ref = db.collection("vehicles").document(vehicle_id)
        if not vehicle_ref.get().exists:
            return jsonify({"message": "vehicle not found"}), 404
        
        vehicle_ref = db.collection("vehicles").document(vehicle_id)

        # Tambah 1 ke issues_reported
        vehicle_ref.update({
            "tire_condition": firestore.Increment(1)
        })
        print("vehicle tire_condition updated successfully")
        return True

    except Exception as e:
        print("Error updating vehicle last_maintenance", str(e))
        return False
    
# -- Get All issues --
@app.route("/issues", methods=["GET"])
def get_all_issues():
    try:
        issues_ref = db.collection("reported_issues").stream()

        issues = []
        for issue in issues_ref:
            issues.append({
                "uid": issue.id,
                **issue.to_dict()
            })

        return jsonify({
            "count": len(issues),
            "issues": issues
        }), 200

    except Exception as e:
        return jsonify({"message": "Error fetching issues", "error": str(e)}), 500

# -- Get All tires --
@app.route("/tires", methods=["GET"])
def get_all_tires():
    try:
        tires_ref = db.collection("tires_conditions").stream()

        tires = []
        for tire in tires_ref:
            tires.append({
                "uid": tire.id,
                **tire.to_dict()
            })

        return jsonify({
            "count": len(tires),
            "tires": tires
        }), 200

    except Exception as e:
        return jsonify({"message": "Error fetching tires", "error": str(e)}), 500

# -- Get All tires --
@app.route("/maintenances", methods=["GET"])
def get_all_maintains():
    try:
        maintenances_ref = db.collection("maintenance").stream()

        maintenances = []
        for maintains in maintenances_ref:
            maintenances.append({
                "uid": maintains.id,
                **maintains.to_dict()
            })

        return jsonify({
            "count": len(maintenances),
            "maintenances": maintenances
        }), 200

    except Exception as e:
        return jsonify({"message": "Error fetching maintenances", "error": str(e)}), 500

# -- Get All Tires issues by UID (VehicleId) --
@app.route("/tires/<uid>", methods=["GET"])
def getTiresByVehicle(uid):
    try:
        # Ambil dokumen vehicle dari Firestore
        reports = db.collection("tires_conditions").where("vehicle_id", "==", uid).stream()
        allReports = []
        found = False
        for report in reports:
            found = True
            data = report.to_dict()
            allReports.append({
                "id": report.id,
                **data
            })
            print(report.id, data)
        if not found:
            return jsonify({"message": "No report found for this vehicle_"}), 404
            
        
        return jsonify({
            "vehicle_id": uid,
            "data": allReports
        }), 200

    except Exception as e:
        return jsonify({"message": "Error fetching user", "error": str(e)}), 500

def feature_creation(df):
    # Convert to datetime
    df['Last_Maintenance_Date'] = pd.to_datetime(df['Last_Maintenance_Date'])
    df['Last_Maintenance_Date'].dtype# Derivative Feature
    current_date = df["Last_Maintenance_Date"].max() 
    df["Days_Since_Last_Maintenance"] = (current_date - df["Last_Maintenance_Date"]).dt.days
    # Derivative Feature
    df["Vehicle_Age"] = current_date.year - df["Year_of_Manufacture"]
    return df

# API: Inference model best_model_classification.pkl dengan preprocessing sesuai feature engineering notebook
# @app.route('/predict-classification', methods=['POST'])
def predict_classification(data_input):
    # print("Received input data:", data_input) #aman
    # pass
    try:
        # Load pipeline model (sudah termasuk preprocessor dan XGBoost)
        model = joblib.load('best_model_classification.pkl')

        data = request.json
        # Konversi setiap value menjadi list agar bisa diproses ke DataFrame
        data = {
            "Make_and_Model": [data.get("vehicle_type")],
            "Year_of_Manufacture": [data.get("manufacture_year")],
            "Vehicle_Type": [data.get("vehicle_type")],
            "Usage_Hours": [data.get("usage_hours")],
            "Route_Info": [data.get("route_info")],
            "Load_Capacity": [data.get("load_cap")],
            "Actual_Load": [data.get("actual_load")],
            "Maintenance_Type": [data.get("maintenance_type")],
            "Maintenance_Cost": [data.get("maintenance_cost")],
            "Tire_Pressure": [data.get("tire_pressure")],
            "Fuel_Consumption": [data.get("fuel_consume")],
            "Battery_Status": [data.get("battery_status")],
            "Vibration_Levels": [data.get("vibration_levels")],
            "Oil_Quality": [data.get("oil_quality")],
            "Brake_Condition": [data.get("brake_condition")],
            "Weather_Conditions": [data.get("weather_cond")],
            "Road_Conditions": [data.get("road_cond")],
            "Delivery_Times": [data.get("delivery_times")],
            "Last_Maintenance_Date": [data.get("last_maintenance")],
            "Impact_on_Efficiency": [data.get("impact_on_efficiency")]
        }
        # print("Processed input data:", data)
        df_input = pd.DataFrame(data)
        # print("Input DataFrame:\n", df_input)
        df_input = feature_creation(df_input)
        # print("Data after feature creation 1:\n", df_input)
        # Convert DataFrame to dictionary for inspection or further use
        # df_input_dict = df_input.to_dict(orient="records")
        # print("df_input as dict:", df_input_dict)
        # Drop kolom leakage dan tidak dipakai (harus sama dengan training)
        drop_cols = [ "Last_Maintenance_Date"]
        for col in drop_cols:
            if col in df_input.columns:
                df_input = df_input.drop(columns=[col])

        # Pastikan urutan kolom sama dengan training
        feature_order = [
            'Make_and_Model', 'Year_of_Manufacture', 'Vehicle_Type', 'Usage_Hours', 'Route_Info',
            'Load_Capacity', 'Actual_Load', 'Maintenance_Type', 'Maintenance_Cost', 'Tire_Pressure',
            'Fuel_Consumption', 'Battery_Status', 'Vibration_Levels', 'Oil_Quality', 'Brake_Condition',
            'Weather_Conditions', 'Road_Conditions', 'Delivery_Times', 'Impact_on_Efficiency',
            'Days_Since_Last_Maintenance', 'Vehicle_Age'
        ]
        df_input = df_input[feature_order]
        # print("Data after feature creation 2:\n", df_input[feature_order])
        # Inference (pipeline sudah handle encoding/scaling)
        y_pred = model.predict(df_input)
        print("Prediction result:", y_pred)
        # Probabilitas prediksi (jika model mendukung)
        y_proba = model.predict_proba(df_input).tolist() if hasattr(model, 'predict_proba') else None
        label = y_pred[0].tolist()
        print(y_proba, 'y proba')
        print(label, 'label')
        if label == 0:
            result = "Good Condition"
        # elif label == 1:
        #     result = "Medium Risk"
        else:
            result = "Need Service"
        return result
    except Exception as e:
        print("Error in prediction:", str(e))
        return False
        
# if name dikomen di hugging face
if __name__ == "__main__":
    app.run(debug=True)