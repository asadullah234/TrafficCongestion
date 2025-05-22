from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from ultralytics import YOLO
import cv2
import os
import time
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import io
import base64
from werkzeug.utils import secure_filename
import numpy as np

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:asad@localhost:5432/Traffic'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

db = SQLAlchemy(app)

# Load YOLOv8 model
model = YOLO("best.pt")

# Global variables for real-time analysis
realtime_active = False
current_junction = 1
COLORS = {
    'car': (0, 255, 0),
    'bus': (255, 0, 0),
    'truck': (0, 0, 255),
    'bike': (255, 0, 255),
    'rickshaw': (0, 255, 188),
    'pickup': (255, 165, 0)
}

# Database Models
class TrafficAnalysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    junction_id = db.Column(db.Integer)
    total_vehicles = db.Column(db.Integer)
    green_time = db.Column(db.Integer)
    congestion_pct = db.Column(db.Float)
    counts_json = db.Column(db.JSON)
    processed_image = db.Column(db.String(200))

class TrafficJunction(db.Model):
    """Junction/Intersection master table"""
    __tablename__ = 'traffic_junctions'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    location = db.Column(db.String(200))
    max_capacity = db.Column(db.Integer, default=50)
    normal_green_time = db.Column(db.Integer, default=30)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    traffic_lights = db.relationship('TrafficLightStatus', backref='junction', lazy=True)

class TrafficLightStatus(db.Model):
    """Traffic light control and status table"""
    __tablename__ = 'traffic_light_status'
    
    id = db.Column(db.Integer, primary_key=True)
    junction_id = db.Column(db.Integer, db.ForeignKey('traffic_junctions.id'), nullable=False)
    direction = db.Column(db.String(20))  # North, South, East, West
    current_state = db.Column(db.String(10), default='RED')  # RED, YELLOW, GREEN
    last_changed = db.Column(db.DateTime, default=datetime.utcnow)
    green_duration = db.Column(db.Integer, default=30)
    yellow_duration = db.Column(db.Integer, default=5)
    red_duration = db.Column(db.Integer, default=35)
    is_operational = db.Column(db.Boolean, default=True)

# Helper Functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def create_tables_safely():
    """Create tables with error handling"""
    try:
        with app.app_context():
            db.create_all()
            print("✓ All tables created successfully!")
            initialize_default_data()
    except Exception as e:
        print(f"Error creating tables: {e}")
        print("Attempting to resolve conflicts...")
        try:
            with app.app_context():
                db.drop_all()
                db.create_all()
                print("✓ Tables recreated successfully after resolving conflicts!")
                initialize_default_data()
        except Exception as e2:
            print(f"Still having issues: {e2}")

def initialize_default_data():
    """Initialize some default data"""
    try:
        if TrafficJunction.query.first() is None:
            default_junctions = [
                TrafficJunction(name="Main Street & 1st Ave", location="Downtown", max_capacity=50),
                TrafficJunction(name="Highway 101 & Oak St", location="North Side", max_capacity=75),
                TrafficJunction(name="Park Ave & 5th St", location="City Center", max_capacity=40),
                TrafficJunction(name="Industrial Rd & Factory St", location="Industrial Zone", max_capacity=60)
            ]
            
            for junction in default_junctions:
                db.session.add(junction)
            
            db.session.commit()
            print("✓ Default data initialized!")
    except Exception as e:
        print(f"Error initializing default data: {e}")

def analyze_image(image_path, junction_id):
    """Analyze traffic in an image using YOLO model"""
    analysis_timestamp = datetime.utcnow()
    print(f"[{analysis_timestamp.strftime('%Y-%m-%d %H:%M:%S')}] Starting analysis for Junction {junction_id}")
    
    img = cv2.imread(image_path)
    if img is None:
        return {"error": "Failed to load image", "timestamp": analysis_timestamp.isoformat()}
    
    results = model(img)
    vehicle_classes = ['car', 'bus', 'truck', 'bike', 'rickshaw', 'pickup']
    counts = {v: 0 for v in vehicle_classes}
    total = 0
    img_with_boxes = img.copy()

    for r in results:
        if r.boxes is not None:
            for box in r.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                if class_id < len(model.names):
                    class_name = model.names[class_id].lower()

                    if class_name in counts:
                        counts[class_name] += 1
                        total += 1
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        color = COLORS.get(class_name, (255, 255, 255))
                        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 2)
                        label = f"{class_name}: {confidence:.2f}"
                        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
                        text_y = y1 - 5 if y1 - 5 > 5 else y1 + 20
                        cv2.rectangle(img_with_boxes,
                                    (x1, text_y - text_size[1] - 5),
                                    (x1 + text_size[0] + 5, text_y + 5),
                                    color, -1)
                        cv2.putText(img_with_boxes, label, (x1, text_y),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # Calculate green time based on traffic
    if total == 0:
        green_time = 20  # Default minimum green time
    elif total < 5:
        green_time = 20
    elif total < 10:
        green_time = 25
    elif total < 20:
        green_time = 30
    elif total < 30:
        green_time = 40
    else:
        green_time = 60  # Maximum green time for heavy traffic

    congestion_pct = min((total / 50) * 100, 100)

    # Add info overlay to image
    overlay = img_with_boxes.copy()
    h, w = img_with_boxes.shape[:2]
    cv2.rectangle(overlay, (0, h - 120), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, img_with_boxes, 0.3, 0, img_with_boxes)

    # Add timestamp to the image
    timestamp_text = f"Analyzed: {analysis_timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}"
    cv2.putText(img_with_boxes, timestamp_text, (10, h - 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

    summary_text = f"Total Vehicles: {total} | Congestion: {congestion_pct:.1f}% | Green Time: {green_time}s"
    cv2.putText(img_with_boxes, summary_text, (10, h - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

    # Save processed image
    processed_filename = f"processed_{junction_id}_{int(time.time())}.jpg"
    processed_path = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
    cv2.imwrite(processed_path, img_with_boxes)

    print(f"[{analysis_timestamp.strftime('%Y-%m-%d %H:%M:%S')}] Analysis complete: {total} vehicles detected at Junction {junction_id}")

    # Save to database
    try:
        analysis = TrafficAnalysis(
            timestamp=analysis_timestamp,
            junction_id=junction_id,
            total_vehicles=total,
            green_time=green_time,
            congestion_pct=congestion_pct,
            counts_json=counts,
            processed_image=processed_filename
        )
        db.session.add(analysis)
        db.session.commit()
        print(f"[{analysis_timestamp.strftime('%Y-%m-%d %H:%M:%S')}] Data saved to database successfully")
    except Exception as e:
        print(f"[{analysis_timestamp.strftime('%Y-%m-%d %H:%M:%S')}] Database error: {e}")
        db.session.rollback()

    return {
        "original_image": os.path.basename(image_path),
        "processed_image": processed_filename,
        "counts": counts,
        "total_vehicles": total,
        "green_time": green_time,
        "congestion_pct": congestion_pct,
        "junction_id": junction_id,
        "analysis_timestamp": analysis_timestamp.isoformat(),
        "analysis_time_formatted": analysis_timestamp.strftime('%Y-%m-%d %H:%M:%S UTC'),
        "status": "success"
    }

def generate_chart(data):
    """Generate vehicle distribution chart"""
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

# Routes
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
    
    if not ret or frame is None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, "No Camera Available", (150, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    filename = f"captured_{int(time.time())}.jpg"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    cv2.imwrite(filepath, frame)
    
    result = analyze_image(filepath, 1)
    
    if result.get('total_vehicles', 0) > 0:
        result['chart'] = generate_chart(result)
    else:
        result['chart'] = '<p class="text-muted">No vehicles detected</p>'
    
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
    
    if not ret or frame is None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, f"Junction {current_junction} - No Camera", (50, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    filename = f"realtime_junction_{current_junction}_{int(time.time())}.jpg"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    cv2.imwrite(filepath, frame)
    
    result = analyze_image(filepath, current_junction)
    
    if result.get('total_vehicles', 0) > 0:
        result['chart'] = generate_chart(result)
    else:
        result['chart'] = '<p class="text-muted">No vehicles detected</p>'
    
    current_junction += 1
    if current_junction > 4:
        realtime_active = False
    
    return jsonify(result)

@app.route('/get_analysis_status')
def get_analysis_status():
    """Get current analysis status for the frontend"""
    global current_junction, realtime_active
    return jsonify({
        'current_junction': current_junction,
        'realtime_active': realtime_active,
        'can_upload': realtime_active and current_junction <= 4
    })

@app.route('/process_junction', methods=['POST'])
def process_junction():
    """Process uploaded image for current junction"""
    global current_junction
    
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type"}), 400
        
        # Save uploaded file
        filename = secure_filename(f"junction_{current_junction}_{int(time.time())}_{file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Analyze the image
        result = analyze_image(filepath, current_junction)
        
        if result.get('error'):
            return jsonify(result), 500
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in process_junction: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/upload_junction_image', methods=['POST'])
def upload_junction_image():
    """Handle individual junction image upload for real-time analysis"""
    try:
        junction_id = request.form.get('junction_id', type=int)
        if not junction_id or junction_id < 1 or junction_id > 4:
            return jsonify({"error": "Invalid junction ID"}), 400
        
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type"}), 400
        
        filename = secure_filename(f"realtime_junction_{junction_id}_{int(time.time())}_{file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        result = analyze_image(filepath, junction_id)
        result['status'] = 'success'
        result['original_filename'] = file.filename
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in upload_junction_image: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/reset_realtime_analysis', methods=['POST'])
def reset_realtime_analysis():
    """Reset the real-time analysis session"""
    global realtime_active, current_junction
    realtime_active = True
    current_junction = 1
    return jsonify({"status": "reset", "message": "Analysis session reset successfully"})

@app.route('/get_traffic_chart_data')
def get_traffic_chart_data():
    """Get traffic data for chart visualization"""
    try:
        recent_analyses = TrafficAnalysis.query.order_by(TrafficAnalysis.timestamp.desc()).limit(20).all()
        
        chart_data = []
        for analysis in recent_analyses:
            chart_data.append({
                'junction_id': analysis.junction_id,
                'timestamp': analysis.timestamp.strftime('%H:%M'),
                'total_vehicles': analysis.total_vehicles,
                'congestion_pct': analysis.congestion_pct,
                'green_time': analysis.green_time
            })
        
        return jsonify(chart_data)
        
    except Exception as e:
        print(f"Error getting chart data: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    except FileNotFoundError:
        return jsonify({"error": f"File {filename} not found"}), 404

@app.route('/api/junctions')
def get_junctions():
    """Get all junctions"""
    junctions = TrafficJunction.query.all()
    return jsonify([{
        'id': j.id,
        'name': j.name,
        'location': j.location,
        'is_active': j.is_active
    } for j in junctions])

@app.route('/api/traffic_summary')
def traffic_summary():
    """Get current traffic summary across all junctions"""
    recent_analyses = TrafficAnalysis.query.filter(
        TrafficAnalysis.timestamp >= datetime.utcnow().replace(hour=0, minute=0, second=0)
    ).all()
    
    summary = {
        'total_vehicles_today': sum(a.total_vehicles for a in recent_analyses),
        'avg_congestion': sum(a.congestion_pct for a in recent_analyses) / len(recent_analyses) if recent_analyses else 0,
        'active_junctions': TrafficJunction.query.filter_by(is_active=True).count(),
        'total_analyses_today': len(recent_analyses)
    }
    
    return jsonify(summary)

@app.route('/get_current_junction')
def get_current_junction():
    """Get the current active junction number"""
    global current_junction, realtime_active
    return jsonify({
        'current_junction': current_junction,
        'realtime_active': realtime_active
    })

@app.route('/update_junction_status', methods=['POST'])
def update_junction_status():
    """Update the current junction after countdown completes"""
    global current_junction, realtime_active
    data = request.get_json()
    if data and 'next_junction' in data:
        current_junction = data['next_junction']
        if current_junction > 4:
            realtime_active = False
        return jsonify({'status': 'success', 'current_junction': current_junction})
    return jsonify({'status': 'error', 'message': 'Invalid request'}), 400

@app.route('/get_current_status')
def get_current_status():
    """Get current analysis status"""
    global current_junction, realtime_active
    return jsonify({
        'current_junction': current_junction,
        'realtime_active': realtime_active,
        'can_upload': not realtime_active or current_junction <= 4
    })

@app.route('/complete_junction', methods=['POST'])
def complete_junction():
    """Mark current junction as complete and move to next"""
    global current_junction, realtime_active
    current_junction += 1
    if current_junction > 4:
        realtime_active = False
    return jsonify({
        'status': 'success',
        'current_junction': current_junction,
        'analysis_complete': current_junction > 4
    })

if __name__ == '__main__':
    create_tables_safely()
    app.run(debug=True)