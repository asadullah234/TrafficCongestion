from flask import Flask, render_template, request, jsonify, send_from_directory
from ultralytics import YOLO
import cv2
import os
import time
import threading
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import io
import base64
from werkzeug.utils import secure_filename
import numpy as np

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load YOLOv8 model
model = YOLO("best.pt")

# Global variables for real-time analysis
realtime_active = False
current_junction = 1
analysis_results = {}

# Define some colors for different vehicle classes (BGR format)
COLORS = {
    'car': (0, 255, 0),       # Green
    'bus': (255, 0, 0),       # Blue
    'truck': (0, 0, 255),     # Red
    'motorcycle': (255, 255, 0), # Cyan
    'bike': (255, 0, 255),    # Magenta
    'rickshaw': (0, 255, 255),# Yellow
    'pickup': (255, 165, 0)   # Orange
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def analyze_image(image_path, junction_id):
    img = cv2.imread(image_path)
    if img is None:
        return {"error": "Failed to load image"}
    
    results = model(img)
    vehicle_classes = ['car', 'bus', 'truck', 'bike', 'rickshaw', 'pickup']
    counts = {v: 0 for v in vehicle_classes}
    total = 0
    
    # Create a copy of the image for drawing bounding boxes
    img_with_boxes = img.copy()
    
    # Process detections
    for r in results:
        for box in r.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            
            if class_id < len(model.names):
                class_name = model.names[class_id].lower()
                
                if class_name in counts:
                    counts[class_name] += 1
                    total += 1
                    
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    color = COLORS.get(class_name, (255, 255, 255))
                    
                    # Draw bounding box
                    cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 2)
                    
                    # Add label with class name and confidence
                    label = f"{class_name}: {confidence:.2f}"
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
                    text_y = y1 - 5 if y1 - 5 > 5 else y1 + 20
                    
                    # Draw label background
                    cv2.rectangle(img_with_boxes, 
                                (x1, text_y - text_size[1] - 5), 
                                (x1 + text_size[0] + 5, text_y + 5), 
                                color, -1)
                    
                    # Add text
                    cv2.putText(img_with_boxes, label, (x1, text_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Calculate signal timing
    if total < 5:
        green_time = 5
    elif total < 10:
        green_time = 10
    elif total < 20:
        green_time = 20
    elif total < 30:
        green_time = 30
    else:
        green_time = 40
    
    congestion_pct = min((total / 50) * 100, 100)
    
    # Add summary information to the image
    overlay = img_with_boxes.copy()
    h, w = img_with_boxes.shape[:2]
    cv2.rectangle(overlay, (0, h-80), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, img_with_boxes, 0.3, 0, img_with_boxes)
    
    # Add summary text
    summary_text = f"Total Vehicles: {total} | Congestion: {congestion_pct:.1f}% | Green Time: {green_time}s"
    cv2.putText(img_with_boxes, summary_text, (10, h-50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Add legend for vehicle classes
    legend_y = h - 30
    legend_x = 10
    for i, cls in enumerate(vehicle_classes):
        if counts[cls] > 0:
            color = COLORS.get(cls, (255, 255, 255))
            legend_text = f"{cls}: {counts[cls]}"
            cv2.putText(img_with_boxes, legend_text, (legend_x, legend_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            legend_x += len(legend_text) * 8 + 10
            if legend_x > w - 100:
                legend_y += 20
                legend_x = 10
    
    # Save processed image
    processed_filename = f"processed_{junction_id}_{int(time.time())}.jpg"
    processed_path = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
    cv2.imwrite(processed_path, img_with_boxes)
    
    return {
        "original_image": os.path.basename(image_path),
        "processed_image": processed_filename,
        "counts": counts,
        "total_vehicles": total,
        "green_time": green_time,
        "congestion_pct": congestion_pct,
        "junction_id": junction_id
    }

def generate_chart(data):
    plt.figure(figsize=(10, 6))
    vehicle_types = list(data['counts'].keys())
    counts = list(data['counts'].values())
    
    bars = plt.bar(vehicle_types, counts)
    for i, bar in enumerate(bars):
        vehicle_type = vehicle_types[i]
        if vehicle_type in COLORS:
            bgr_color = COLORS[vehicle_type]
            rgb_color = (bgr_color[2]/255, bgr_color[1]/255, bgr_color[0]/255)
            bar.set_color(rgb_color)
    
    plt.xlabel('Vehicle Type')
    plt.ylabel('Count')
    plt.title('Vehicle Distribution')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    for i, v in enumerate(counts):
        if v > 0:
            plt.text(i, v + 0.1, str(v), ha='center')
    
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    
    return f'<img src="data:image/png;base64,{img_str}" class="img-fluid" alt="Vehicle Distribution Chart">'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    results = []
    for junction_id in range(1, 5):
        file_key = f'junction_{junction_id}'
        if file_key not in request.files:
            continue
            
        file = request.files[file_key]
        if file and allowed_file(file.filename):
            filename = secure_filename(f"junction_{junction_id}_{file.filename}")
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            result = analyze_image(filepath, junction_id)
            result['chart'] = generate_chart(result)
            results.append(result)
    
    return render_template('multiresult.html', results=results)

@app.route('/capture', methods=['POST'])
def capture_image():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return jsonify({"error": "Failed to capture image"}), 400
    
    filename = f"captured_{int(time.time())}.jpg"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    cv2.imwrite(filepath, frame)
    
    result = analyze_image(filepath, 1)
    result['chart'] = generate_chart(result)
    
    return render_template('multiresult.html', results=[result])

@app.route('/start_realtime', methods=['POST'])
def start_realtime():
    global realtime_active, current_junction
    realtime_active = True
    current_junction = 1
    return render_template('realtime_result.html')

@app.route('/get_realtime_data')
def get_realtime_data():
    global current_junction, realtime_active
    
    if not realtime_active:
        return jsonify({"status": "inactive"})
    
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return jsonify({"error": "Capture failed"}), 400
    
    filename = f"realtime_junction_{current_junction}.jpg"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    cv2.imwrite(filepath, frame)
    
    result = analyze_image(filepath, current_junction)
    result['chart'] = generate_chart(result)
    
    current_junction += 1
    if current_junction > 4:
        realtime_active = False
    
    return jsonify(result)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)