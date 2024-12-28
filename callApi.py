from flask import Flask, jsonify
import threading
import time
import requests

# Initialize the Flask app
app = Flask(__name__)

# API details
API_URL = "https://connecteamserver.onrender.com/recommend"
API_DATA = {"user_id": "user_2pKF6P71tt4pFEappnDEtA44t9C"}

# Function to call the API
def call_api_periodically():
    while True:
        try:
            response = requests.post(API_URL, json=API_DATA)
            print(f"API Response: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"Error calling API: {e}")
        time.sleep(30)  # Wait for 20 seconds

# Start the background thread
def start_background_thread():
    thread = threading.Thread(target=call_api_periodically, daemon=True)
    thread.start()

@app.route('/')
def index():
    return jsonify({"message": "Flask app is running, and API calls are being made every 20 seconds!"})

# Run the app
if __name__ == '__main__':
    start_background_thread()
    app.run(debug=True, host='0.0.0.0', port=51000)
