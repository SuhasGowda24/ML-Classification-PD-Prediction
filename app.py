import os
import pandas as pd
import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify

#load pkl
model_path = 'parkinsons_model.pkl'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}. Please run train_and_save_model.py first.")

model_pipeline = joblib.load(model_path)

def safe_read_csv(file_path):
    encodings = ["utf-8", "utf-16", "utf-16-le", "utf-16-be", "latin1"]
    
    for enc in encodings:
        try:
            return pd.read_csv(file_path, header=None, skiprows=1, sep=' ', encoding=enc)
        except Exception:
            continue

    raise ValueError("Could not read file. Unsupported encoding or not a stroke-data file.")

#data preprocessing function
def calculate_features(file_path):
    #Y coordinate, X coordinate, time stamp, button state, azimuth, altitude, pressure 
    # data = pd.read_csv(file_path, header=None, skiprows=1, sep=' ')
    data = safe_read_csv(file_path)

    data['time_diff'] = data[2].diff().fillna(0)
    data['y_diff'] = data[0].diff().fillna(0)
    data['x_diff'] = data[1].diff().fillna(0)
    data['distance'] = np.sqrt(data['x_diff']**2 + data['y_diff']**2)
    data['velocity'] = data['distance'] / data['time_diff']
    data['velocity'].replace([np.inf, -np.inf], 0, inplace=True)
    #time_diff, y_diff, x_diff, distance, velocity

    #convertion
    data['velocity_diff'] = data['velocity'].diff().fillna(0)
    data['acceleration'] = data['velocity_diff'] / data['time_diff']
    data['acceleration'].replace([np.inf, -np.inf], 0, inplace=True)
    data['acceleration_diff'] = data['acceleration'].diff().fillna(0)
    data['jerk'] = data['acceleration_diff'] / data['time_diff']
    data['jerk'].replace([np.inf, -np.inf], 0, inplace=True)
    total_duration = data[2].iloc[-1] - data[2].iloc[0]
    num_pen_lifts = (data[3] == 0).sum()

    on_surface_time = data[data[3] == 1]['time_diff'].sum()
    in_air_time = data[data[3] == 0]['time_diff'].sum()
    
    if on_surface_time > 0:
        ratio_air_surface = in_air_time / on_surface_time
    else:
        ratio_air_surface = 0.0

    pressure_mean = data[6].mean()
    pressure_std = data[6].std()
    pressure_max = data[6].max()

    features = {
        'velocity_mean': data['velocity'].mean(),
        'velocity_std': data['velocity'].std(),
        'velocity_max': data['velocity'].max(),
        'acceleration_mean': data['acceleration'].mean(),
        'acceleration_std': data['acceleration'].std(),
        'jerk_mean': data['jerk'].mean(),
        'total_duration': total_duration,
        'num_pen_lifts': num_pen_lifts,
        'ratio_air_surface': ratio_air_surface,
        'pressure_mean': pressure_mean,
        'pressure_std': pressure_std,
        'pressure_max': pressure_max,
    }
    return features

model_accuracy = 0.7417

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    prediction_accuracy = None
    message_prefix = "The model predicts that the patient is likely"
    if request.method == 'POST':
        # Get uploaded file and task type
        if 'svc_file' not in request.files:
            return "No file part", 400
        
        svc_file = request.files['svc_file']
        task_type = request.form['task_type']

        if svc_file.filename == '':
            return "No selected file", 400

        # Save the file to the uploads folder
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], svc_file.filename)
        svc_file.save(file_path)

        # Calculate features from the uploaded file
        features = calculate_features(file_path)
        features['task_type'] = task_type
        
        # Create a DataFrame for prediction
        input_df = pd.DataFrame([features])
        

        # Make a prediction using the loaded model pipeline
        predicted_proba = model_pipeline.predict_proba(input_df)[0]
        predicted_label = model_pipeline.predict(input_df)[0]
        
        # The probability of the predicted class is the accuracy for this specific prediction
        prediction_accuracy = predicted_proba[predicted_label]
        
        if predicted_label == 1:
            prediction = "Positive for Parkinson's Disease"
            message_prefix = "The model is"
        else:
            prediction = "Negative for Parkinson's Disease"
            message_prefix = "The model predicts that the patient is likely"

        os.remove(file_path)
            
    return render_template('index.html', prediction=prediction, accuracy=model_accuracy,prediction_accuracy=prediction_accuracy,message_prefix=message_prefix)

if __name__ == '__main__':
    app.run(debug=True)