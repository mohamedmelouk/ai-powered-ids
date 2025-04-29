from flask import Flask, render_template, jsonify, send_from_directory
from ids_core import IDS
import threading
import os
import time
import sys

app = Flask(__name__)
ids = IDS()

def initialize_system():
    print("Initializing Intrusion Detection System...")
    
    # Create absolute paths to prevent any directory issues
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'models', 'random_forest_model.pkl')
    train_data_path = os.path.join(base_dir, 'data', 'KDDTrain+.txt')
    test_data_path = os.path.join(base_dir, 'data', 'KDDTest+.txt')

    # Create required directories if they don't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(train_data_path), exist_ok=True)

    # Check if training data exists
    if not os.path.exists(train_data_path):
        error_msg = (
            f"ERROR: Training data not found at {train_data_path}\n"
            "Please download the NSL-KDD dataset and place:\n"
            "1. KDDTrain+.txt\n"
            "2. KDDTest+.txt\n"
            "in the 'data' folder\n\n"
            "You can download the dataset from:\n"
            "https://www.unb.ca/cic/datasets/nsl.html"
        )
        print(error_msg)
        sys.exit(1)

    # Train and save model if it doesn't exist
    if not os.path.exists(model_path):
        print("Model not found. Training new model...")
        print("This may take 10-30 minutes depending on your system...")
        
        start_time = time.time()
        try:
            X_train, X_test, y_train, y_test = ids.load_data(train_data_path, test_data_path)
            ids.train_model(X_train, y_train)
            ids.evaluate_model(X_test, y_test)
            ids.save_model(model_path)
            print(f"Model successfully trained and saved in {time.time()-start_time:.2f} seconds")
        except Exception as e:
            print(f"Failed to train model: {str(e)}")
            sys.exit(1)
    else:
        print("Found existing model, loading...")
        try:
            ids.load_model(model_path)
            print("Model loaded successfully")
        except Exception as e:
            print(f"Failed to load model: {str(e)}")
            print("Attempting to train a new model...")
            initialize_system()  # Retry training

# Initialize the system
initialize_system()

# Start packet sniffing in background
try:
    sniff_thread = threading.Thread(target=ids.start_sniffing, daemon=True)
    sniff_thread.start()
    print("Packet capture started successfully")
except Exception as e:
    print(f"Failed to start packet capture: {str(e)}")
    print("Running in simulation mode with sample alerts")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/alerts')
def alerts():
    return render_template('alerts.html', alerts=ids.get_alerts())

@app.route('/api/alerts')
def api_alerts():
    return jsonify(ids.get_alerts())

@app.route('/clear-alerts', methods=['POST'])
def clear_alerts():
    ids.clear_alerts()
    return jsonify({"status": "success"})

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

if __name__ == '__main__':
    print("\nStarting web interface at http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)