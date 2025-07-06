from flask import Flask, request, jsonify, render_template, session, redirect, url_for
import joblib
import pandas as pd
from flask_cors import CORS
import os
import json
from datetime import datetime
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
import uuid
import numpy as np

app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-change-this')
CORS(app)

# Crop information database
CROP_INFO = {
    'rice': {
        'name': 'Rice',
        'description': 'A staple food crop that grows well in warm, humid conditions with plenty of water.',
        'growing_season': '3-6 months',
        'water_requirement': 'High',
        'soil_type': 'Clay loam, silty loam',
        'temperature_range': '20-35°C',
        'harvest_time': '90-150 days',
        'nutritional_value': 'Rich in carbohydrates, moderate protein',
        'market_value': 'High',
        'image_url': '/static/images/rice.jpeg'
    },
    'maize': {
        'name': 'Maize (Corn)',
        'description': 'A versatile cereal crop used for food, feed, and industrial products.',
        'growing_season': '3-4 months',
        'water_requirement': 'Medium',
        'soil_type': 'Well-drained loamy soil',
        'temperature_range': '18-32°C',
        'harvest_time': '80-120 days',
        'nutritional_value': 'High in carbohydrates, moderate protein',
        'market_value': 'High',
        'image_url': '/static/images/Maize.jpeg'
    },
    'chickpea': {
        'name': 'Chickpea',
        'description': 'A legume crop rich in protein, commonly used in various cuisines.',
        'growing_season': '3-5 months',
        'water_requirement': 'Low to Medium',
        'soil_type': 'Well-drained sandy loam',
        'temperature_range': '15-30°C',
        'harvest_time': '90-150 days',
        'nutritional_value': 'High protein, fiber, vitamins',
        'market_value': 'Medium to High',
        'image_url': '/static/images/Chickpea.jpeg'
    },
    'kidneybeans': {
        'name': 'Kidney Beans',
        'description': 'A nutritious legume with kidney-shaped seeds, rich in protein and fiber.',
        'growing_season': '3-4 months',
        'water_requirement': 'Medium',
        'soil_type': 'Well-drained loamy soil',
        'temperature_range': '18-30°C',
        'harvest_time': '90-120 days',
        'nutritional_value': 'High protein, iron, fiber',
        'market_value': 'Medium',
        'image_url': '/static/images/Kidney Beans.jpeg'
    },
    'pigeonpeas': {
        'name': 'Pigeon Peas',
        'description': 'A drought-resistant legume crop, important for food security.',
        'growing_season': '4-6 months',
        'water_requirement': 'Low',
        'soil_type': 'Well-drained sandy loam',
        'temperature_range': '20-35°C',
        'harvest_time': '120-180 days',
        'nutritional_value': 'High protein, minerals',
        'market_value': 'Medium',
        'image_url': '/static/images/Pigeon Peas.jpeg'
    },
    'mothbeans': {
        'name': 'Moth Beans',
        'description': 'A drought-tolerant legume crop, suitable for arid regions.',
        'growing_season': '2-3 months',
        'water_requirement': 'Very Low',
        'soil_type': 'Sandy to loamy soil',
        'temperature_range': '25-35°C',
        'harvest_time': '60-90 days',
        'nutritional_value': 'High protein, minerals',
        'market_value': 'Low to Medium',
        'image_url': '/static/images/Moth Beans.jpeg'
    },
    'mungbean': {
        'name': 'Mung Bean',
        'description': 'A fast-growing legume crop, commonly used for sprouts.',
        'growing_season': '2-3 months',
        'water_requirement': 'Low to Medium',
        'soil_type': 'Well-drained loamy soil',
        'temperature_range': '20-35°C',
        'harvest_time': '60-90 days',
        'nutritional_value': 'High protein, vitamins',
        'market_value': 'Medium',
        'image_url': '/static/images/Mung Bean.jpeg'
    },
    'blackgram': {
        'name': 'Black Gram',
        'description': 'A nutritious legume crop, important in Indian cuisine.',
        'growing_season': '3-4 months',
        'water_requirement': 'Medium',
        'soil_type': 'Clay loam, black soil',
        'temperature_range': '20-35°C',
        'harvest_time': '90-120 days',
        'nutritional_value': 'High protein, iron',
        'market_value': 'Medium',
        'image_url': '/static/images/blackgram.jpg'
    },
    'lentil': {
        'name': 'Lentil',
        'description': 'A cool-season legume crop, rich in protein and fiber.',
        'growing_season': '3-4 months',
        'water_requirement': 'Low to Medium',
        'soil_type': 'Well-drained loamy soil',
        'temperature_range': '15-25°C',
        'harvest_time': '90-120 days',
        'nutritional_value': 'High protein, fiber, iron',
        'market_value': 'Medium to High',
        'image_url': '/static/images/Lentil.jpeg'
    },
    'pomegranate': {
        'name': 'Pomegranate',
        'description': 'A fruit tree known for its antioxidant-rich seeds.',
        'growing_season': '6-7 months',
        'water_requirement': 'Medium',
        'soil_type': 'Well-drained loamy soil',
        'temperature_range': '15-35°C',
        'harvest_time': '180-210 days',
        'nutritional_value': 'High antioxidants, vitamins',
        'market_value': 'High',
        'image_url': '/static/images/Pomegranate.jpeg'
    },
    'banana': {
        'name': 'Banana',
        'description': 'A tropical fruit crop, one of the most consumed fruits worldwide.',
        'growing_season': '9-12 months',
        'water_requirement': 'High',
        'soil_type': 'Rich, well-drained soil',
        'temperature_range': '20-35°C',
        'harvest_time': '270-365 days',
        'nutritional_value': 'High potassium, vitamins',
        'market_value': 'High',
        'image_url': '/static/images/Banana.jpeg'
    },
    'mango': {
        'name': 'Mango',
        'description': 'A tropical fruit tree, known as the king of fruits.',
        'growing_season': '4-6 months',
        'water_requirement': 'Medium to High',
        'soil_type': 'Well-drained loamy soil',
        'temperature_range': '20-35°C',
        'harvest_time': '120-180 days',
        'nutritional_value': 'High vitamins, antioxidants',
        'market_value': 'High',
        'image_url': '/static/images/Mango.jpeg'
    },
    'grapes': {
        'name': 'Grapes',
        'description': 'A perennial vine crop, used for fresh fruit and wine production.',
        'growing_season': '4-6 months',
        'water_requirement': 'Medium',
        'soil_type': 'Well-drained loamy soil',
        'temperature_range': '15-35°C',
        'harvest_time': '120-180 days',
        'nutritional_value': 'High antioxidants, vitamins',
        'market_value': 'High',
        'image_url': '/static/images/Grapes.jpeg'
    },
    'watermelon': {
        'name': 'Watermelon',
        'description': 'A refreshing summer fruit crop, high in water content.',
        'growing_season': '3-4 months',
        'water_requirement': 'High',
        'soil_type': 'Sandy loam soil',
        'temperature_range': '20-35°C',
        'harvest_time': '80-120 days',
        'nutritional_value': 'High water content, vitamins',
        'market_value': 'Medium to High',
        'image_url': '/static/images/Watermelon.jpeg'
    },
    'muskmelon': {
        'name': 'Muskmelon',
        'description': 'A sweet melon variety, rich in vitamins and minerals.',
        'growing_season': '3-4 months',
        'water_requirement': 'Medium',
        'soil_type': 'Well-drained loamy soil',
        'temperature_range': '20-35°C',
        'harvest_time': '80-120 days',
        'nutritional_value': 'High vitamins, minerals',
        'market_value': 'Medium',
        'image_url': '/static/images/Muskmelon.jpeg'
    },
    'apple': {
        'name': 'Apple',
        'description': 'A temperate fruit crop, one of the most popular fruits.',
        'growing_season': '4-6 months',
        'water_requirement': 'Medium',
        'soil_type': 'Well-drained loamy soil',
        'temperature_range': '15-25°C',
        'harvest_time': '120-180 days',
        'nutritional_value': 'High fiber, vitamins, antioxidants',
        'market_value': 'High',
        'image_url': '/static/images/Apple.jpeg'
    },
    'orange': {
        'name': 'Orange',
        'description': 'A citrus fruit crop, rich in vitamin C and antioxidants.',
        'growing_season': '6-8 months',
        'water_requirement': 'Medium to High',
        'soil_type': 'Well-drained sandy loam',
        'temperature_range': '15-30°C',
        'harvest_time': '180-240 days',
        'nutritional_value': 'High vitamin C, fiber',
        'market_value': 'High',
        'image_url': '/static/images/Orange.jpeg'
    },
    'papaya': {
        'name': 'Papaya',
        'description': 'A tropical fruit crop, known for its digestive enzymes.',
        'growing_season': '6-9 months',
        'water_requirement': 'Medium to High',
        'soil_type': 'Well-drained loamy soil',
        'temperature_range': '20-35°C',
        'harvest_time': '180-270 days',
        'nutritional_value': 'High enzymes, vitamins A and C',
        'market_value': 'Medium to High',
        'image_url': '/static/images/Papaya.jpeg'
    },
    'coconut': {
        'name': 'Coconut',
        'description': 'A versatile tropical crop, used for food, oil, and fiber.',
        'growing_season': '12-18 months',
        'water_requirement': 'High',
        'soil_type': 'Sandy coastal soil',
        'temperature_range': '20-35°C',
        'harvest_time': '360-540 days',
        'nutritional_value': 'High healthy fats, minerals',
        'market_value': 'High',
        'image_url': '/static/images/Coconut.jpeg'
    },
    'cotton': {
        'name': 'Cotton',
        'description': 'A fiber crop, primarily used for textile production.',
        'growing_season': '5-6 months',
        'water_requirement': 'Medium to High',
        'soil_type': 'Well-drained loamy soil',
        'temperature_range': '20-35°C',
        'harvest_time': '150-180 days',
        'nutritional_value': 'Fiber crop, not for consumption',
        'market_value': 'High',
        'image_url': '/static/images/Cotton.jpeg'
    },
    'jute': {
        'name': 'Jute',
        'description': 'A natural fiber crop, used for making ropes and textiles.',
        'growing_season': '4-5 months',
        'water_requirement': 'High',
        'soil_type': 'Alluvial soil',
        'temperature_range': '20-35°C',
        'harvest_time': '120-150 days',
        'nutritional_value': 'Fiber crop, not for consumption',
        'market_value': 'Medium',
        'image_url': '/static/images/jute.jpg'
    },
    'coffee': {
        'name': 'Coffee',
        'description': 'A tropical crop, used for making the popular beverage.',
        'growing_season': '3-4 years',
        'water_requirement': 'Medium',
        'soil_type': 'Well-drained volcanic soil',
        'temperature_range': '15-25°C',
        'harvest_time': '1080-1440 days',
        'nutritional_value': 'Stimulant, antioxidants',
        'market_value': 'High',
        'image_url': '/static/images/coffee.jpg'
    }
}

# Initialize database
def init_db():
    conn = sqlite3.connect('crop_predictions.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id TEXT PRIMARY KEY,
            nitrogen REAL,
            phosphorus REAL,
            potassium REAL,
            temperature REAL,
            humidity REAL,
            ph REAL,
            rainfall REAL,
            predicted_crop TEXT,
            confidence REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

# Load model, scaler, encoder, and feature names with error handling
try:
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    features = joblib.load('features.pkl')
    model_loaded = True
except Exception as e:
    print(f"Error loading model assets: {e}")
    model, scaler, label_encoder, features = None, None, None, None
    model_loaded = False

# Crop categorization helper
def get_crop_category(crop_name):
    cereals = ['rice', 'maize']
    legumes = ['chickpea', 'kidneybeans', 'pigeonpeas', 'mothbeans', 'mungbean', 'blackgram', 'lentil']
    fruits = ['pomegranate', 'banana', 'mango', 'grapes', 'watermelon', 'muskmelon', 'apple', 'orange', 'papaya', 'coconut']
    fiber_crops = ['cotton', 'jute', 'coffee']
    
    if crop_name in cereals:
        return 'cereal'
    elif crop_name in legumes:
        return 'legume'
    elif crop_name in fruits:
        return 'fruit'
    elif crop_name in fiber_crops:
        return 'fiber'
    else:
        return 'other'

# Make the function available to templates
app.jinja_env.globals.update(get_crop_category=get_crop_category)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/crop-info')
def crop_info():
    return render_template('crop_info.html', crops=CROP_INFO)

@app.route('/history')
def history():
    return render_template('history.html')

@app.route('/health')
def health():
    if not model_loaded or not all([model, scaler, label_encoder, features]):
        return jsonify({
            'error': 'Model not loaded. Please check the server logs for details.',
            'status': 'error'
        }), 500
    
    return jsonify({'status': 'healthy', 'model_loaded': model_loaded})

@app.route('/crop-prediction', methods=['POST'])
def predict():
    if not model_loaded or not all([model, scaler, label_encoder, features]):
        return jsonify({
            'error': 'Model not loaded. Please check the server logs for details.',
            'status': 'error'
        }), 500
    
    try:
        assert model is not None
        assert scaler is not None
        assert label_encoder is not None
        assert features is not None
        
        data = request.get_json()
        
        # Validate input data
        required_fields = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}', 'status': 'error'}), 400
            try:
                value = float(data[field])
                if value < 0:
                    return jsonify({'error': f'{field} cannot be negative', 'status': 'error'}), 400
                # Special validation for pH (0-14 range)
                if field == 'ph' and (value < 0 or value > 14):
                    return jsonify({'error': 'pH must be between 0 and 14', 'status': 'error'}), 400
            except ValueError:
                return jsonify({'error': f'{field} must be a valid number', 'status': 'error'}), 400
        
        input_df = pd.DataFrame([data])
        input_df = input_df[features]
        
        # Ensure DataFrame has proper feature names to avoid warnings
        input_df.columns = features

        # Scale the input
        input_scaled = scaler.transform(input_df)

        # Make prediction
        prediction_index = model.predict(input_scaled)[0]
        crop_name = label_encoder.inverse_transform([prediction_index])[0]
        
        # Get prediction probabilities
        probabilities = model.predict_proba(input_scaled)[0]
        confidence = max(probabilities) * 100
        
        # Debug information
        print(f"Input data: {data}")
        print(f"Scaled input: {input_scaled}")
        print(f"Prediction index: {prediction_index}")
        print(f"Predicted crop: {crop_name}")
        print(f"All probabilities: {probabilities}")
        print(f"Confidence: {confidence}")
        
        # Get top 3 predictions for debugging
        prob_array = np.array(probabilities)
        top_indices = prob_array.argsort()[-3:][::-1]
        top_crops = []
        for idx in top_indices:
            crop = label_encoder.inverse_transform([idx])[0]
            prob = probabilities[idx] * 100
            top_crops.append(f"{crop}: {prob:.1f}%")
        print(f"Top 3 predictions: {top_crops}")
        
        # Save prediction to database
        prediction_id = str(uuid.uuid4())
        conn = sqlite3.connect('crop_predictions.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO predictions 
            (id, nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall, predicted_crop, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (prediction_id, data['N'], data['P'], data['K'], data['temperature'], 
              data['humidity'], data['ph'], data['rainfall'], crop_name, confidence))
        conn.commit()
        conn.close()

        # Get crop details
        crop_details = CROP_INFO.get(crop_name, {})
        
        return jsonify({
            'prediction': crop_name,
            'confidence': round(confidence, 2),
            'message': f'Based on the provided soil and climate data, the recommended crop is: {crop_name} (Confidence: {confidence:.1f}%)',
            'status': 'success',
            'prediction_id': prediction_id,
            'crop_details': crop_details
        })
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 400

@app.route('/api/predictions', methods=['GET'])
def get_predictions():
    try:
        conn = sqlite3.connect('crop_predictions.db')
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM predictions 
            ORDER BY timestamp DESC 
            LIMIT 100
        ''')
        rows = cursor.fetchall()
        conn.close()
        
        predictions = []
        for row in rows:
            predictions.append({
                'id': row[0],
                'nitrogen': row[1],
                'phosphorus': row[2],
                'potassium': row[3],
                'temperature': row[4],
                'humidity': row[5],
                'ph': row[6],
                'rainfall': row[7],
                'predicted_crop': row[8],
                'confidence': row[9],
                'timestamp': row[10]
            })
        
        return jsonify({'predictions': predictions, 'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/api/crop-info/<crop_name>', methods=['GET'])
def get_crop_info(crop_name):
    if crop_name in CROP_INFO:
        return jsonify({'crop_info': CROP_INFO[crop_name], 'status': 'success'})
    else:
        return jsonify({'error': 'Crop not found', 'status': 'error'}), 404

@app.route('/api/stats', methods=['GET'])
def get_stats():
    try:
        conn = sqlite3.connect('crop_predictions.db')
        cursor = conn.cursor()
        
        # Total predictions
        cursor.execute('SELECT COUNT(*) FROM predictions')
        total_predictions = cursor.fetchone()[0]
        
        # Most predicted crops
        cursor.execute('''
            SELECT predicted_crop, COUNT(*) as count 
            FROM predictions 
            GROUP BY predicted_crop 
            ORDER BY count DESC 
            LIMIT 5
        ''')
        top_crops = [{'crop': row[0], 'count': row[1]} for row in cursor.fetchall()]
        
        # Average confidence
        cursor.execute('SELECT AVG(confidence) FROM predictions')
        avg_confidence = cursor.fetchone()[0] or 0
        
        conn.close()
        
        return jsonify({
            'total_predictions': total_predictions,
            'top_crops': top_crops,
            'average_confidence': round(avg_confidence, 2),
            'status': 'success'
        })
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/export-predictions', methods=['GET'])
def export_predictions():
    try:
        conn = sqlite3.connect('crop_predictions.db')
        df = pd.read_sql_query('SELECT * FROM predictions ORDER BY timestamp DESC', conn)
        conn.close()
        
        csv_data = df.to_csv(index=False)
        
        from flask import Response
        return Response(
            csv_data,
            mimetype='text/csv',
            headers={'Content-Disposition': 'attachment; filename=crop_predictions.csv'}
        )
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/crop-prediction/info', methods=['GET'])
def prediction_info():
    if not model_loaded or not features:
        return jsonify({
            'error': 'Model not loaded. Please check logs for details.',
            'status': 'error',
            'model_loaded': False
        }), 500
    
    return jsonify({
        'description': 'Crop Prediction API',
        'features_required': list(features) if features else [],
        'method': 'POST',
        'endpoint': '/crop-prediction',
        'model_loaded': True
    })

@app.route('/debug/model', methods=['GET'])
def debug_model():
    if not model_loaded or not all([model, scaler, label_encoder, features]):
        return jsonify({
            'error': 'Model not loaded',
            'status': 'error'
        }), 500
    
    try:
        # Get all possible crop labels
        all_crops = label_encoder.classes_.tolist() if label_encoder else []
        
        # Test with some sample data to see predictions
        sample_data = {
            'N': 50, 'P': 50, 'K': 50, 
            'temperature': 25, 'humidity': 70, 'ph': 6.5, 'rainfall': 100
        }
        
        input_df = pd.DataFrame([sample_data])
        input_df = input_df[features]
        input_scaled = scaler.transform(input_df) if scaler else None
        
        if input_scaled is None:
            return jsonify({
                'error': 'Scaler not available',
                'status': 'error'
            }), 500
        
        prediction_index = model.predict(input_scaled)[0] if model else None
        if prediction_index is None:
            return jsonify({
                'error': 'Model prediction failed',
                'status': 'error'
            }), 500
            
        crop_name = label_encoder.inverse_transform([prediction_index])[0] if label_encoder else None
        probabilities = model.predict_proba(input_scaled)[0] if model else []
        
        # Get all predictions with probabilities
        all_predictions = []
        if label_encoder and probabilities:
            for i, prob in enumerate(probabilities):
                crop = label_encoder.inverse_transform([i])[0]
                all_predictions.append({
                    'crop': crop,
                    'probability': round(prob * 100, 2)
                })
        
        # Sort by probability
        all_predictions.sort(key=lambda x: x['probability'], reverse=True)
        
        return jsonify({
            'model_loaded': True,
            'available_crops': all_crops,
            'total_crops': len(all_crops),
            'sample_prediction': {
                'input_data': sample_data,
                'predicted_crop': crop_name,
                'confidence': round(max(probabilities) * 100, 2) if probabilities else 0
            },
            'all_predictions': all_predictions[:10],  # Top 10
            'status': 'success'
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/test/predictions', methods=['GET'])
def test_predictions():
    if not model_loaded or not all([model, scaler, label_encoder, features]):
        return jsonify({
            'error': 'Model not loaded',
            'status': 'error'
        }), 500
    
    try:
        # Test with different input combinations
        test_cases = [
            {
                'name': 'Rice conditions',
                'data': {'N': 80, 'P': 40, 'K': 30, 'temperature': 28, 'humidity': 85, 'ph': 6.0, 'rainfall': 200}
            },
            {
                'name': 'Maize conditions', 
                'data': {'N': 60, 'P': 50, 'K': 40, 'temperature': 25, 'humidity': 70, 'ph': 6.5, 'rainfall': 150}
            },
            {
                'name': 'Chickpea conditions',
                'data': {'N': 40, 'P': 30, 'K': 20, 'temperature': 22, 'humidity': 60, 'ph': 7.0, 'rainfall': 100}
            },
            {
                'name': 'Mango conditions',
                'data': {'N': 70, 'P': 60, 'K': 50, 'temperature': 30, 'humidity': 75, 'ph': 6.8, 'rainfall': 180}
            }
        ]
        
        results = []
        for test_case in test_cases:
            input_df = pd.DataFrame([test_case['data']])
            input_df = input_df[features]
            input_scaled = scaler.transform(input_df) if scaler else None
            
            if input_scaled is None:
                continue
                
            prediction_index = model.predict(input_scaled)[0] if model else None
            if prediction_index is None:
                continue
                
            crop_name = label_encoder.inverse_transform([prediction_index])[0] if label_encoder else None
            probabilities = model.predict_proba(input_scaled)[0] if model else []
            confidence = max(probabilities) * 100 if probabilities else 0
            
            # Get top 3 predictions
            top_predictions = []
            if label_encoder and probabilities:
                prob_array = np.array(probabilities)
                top_indices = prob_array.argsort()[-3:][::-1]
                for idx in top_indices:
                    crop = label_encoder.inverse_transform([idx])[0]
                    prob = probabilities[idx] * 100
                    top_predictions.append(f"{crop}: {prob:.1f}%")
            
            results.append({
                'test_case': test_case['name'],
                'input_data': test_case['data'],
                'predicted_crop': crop_name,
                'confidence': round(confidence, 2),
                'top_3_predictions': top_predictions
            })
        
        return jsonify({
            'test_results': results,
            'status': 'success'
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/api/clear-history', methods=['POST'])
def clear_history():
    try:
        conn = sqlite3.connect('crop_predictions.db')
        cursor = conn.cursor()
        cursor.execute('DELETE FROM predictions')
        conn.commit()
        conn.close()
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/api/predictions/<prediction_id>', methods=['DELETE'])
def delete_prediction(prediction_id):
    try:
        conn = sqlite3.connect('crop_predictions.db')
        cursor = conn.cursor()
        cursor.execute('DELETE FROM predictions WHERE id = ?', (prediction_id,))
        conn.commit()
        conn.close()
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500

if __name__ == '__main__':
    init_db()
    app.run(debug=True) 