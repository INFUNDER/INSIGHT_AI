import cv2
from flask import Flask, Response, redirect, render_template, request, jsonify, url_for
from flask_sqlalchemy import SQLAlchemy
import threading
import signal
import time
from pose_analysis import PoseAnalyzer
from speech_analysis import SpeechAnalyzer
import os
import numpy as np
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model


app = Flask(__name__)

model = load_model('formal_vs_casual.h5')

# Configure the database URI
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///interview_questions.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize the database
db = SQLAlchemy(app)

# Define models
class AmazonQues(db.Model):
    q_num = db.Column(db.Integer, primary_key=True, autoincrement=True)
    questions = db.Column(db.String(250), nullable=False)
    option_A = db.Column(db.String(250), nullable=False)
    option_B = db.Column(db.String(250), nullable=False)
    option_C = db.Column(db.String(250), nullable=False)
    option_D = db.Column(db.String(250), nullable=False)
    Answer = db.Column(db.String(2), nullable=False)

class MicrosoftQues(db.Model):
    q_num = db.Column(db.Integer, primary_key=True, autoincrement=True)
    questions = db.Column(db.String(250), nullable=False)
    option_A = db.Column(db.String(250), nullable=False)
    option_B = db.Column(db.String(250), nullable=False)
    option_C = db.Column(db.String(250), nullable=False)
    option_D = db.Column(db.String(250), nullable=False)
    Answer = db.Column(db.String(2), nullable=False)

# Create the database tables
with app.app_context():
    db.create_all()

class CameraFeed:
    def __init__(self):
        self.pose_analyzer = PoseAnalyzer()  # Assuming PoseAnalyzer is defined elsewhere
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.start_time = None

    def start(self):
        self.start_time = time.time()
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            elapsed_time = time.time() - self.start_time
            if elapsed_time > 90:
                print("90 seconds passed. Stopping camera feed.")
                break

            # Apply PoseAnalyzer to the frame (assuming analyze_pose is a valid method)
            frame = self.pose_analyzer.analyze_pose(frame)

            # Encode frame to JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            # if not ret:
            #     print("Failed to encode frame to JPEG.")
            #     continue
            
            # Convert to byte array for HTTP streaming
            frame = buffer.tobytes()

            # Yield the frame in multipart format for HTTP streaming
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        # Release camera after finishing
        self.cap.release()
class App:
    def __init__(self):
        self.speech_analyzer = SpeechAnalyzer()
        self.camera_feed = CameraFeed()
        self.stop_event = threading.Event()

    def start(self):
        # Start speech analysis in a separate thread
        speech_thread = threading.Thread(target=self.speech_analyzer.capture_and_analyze_speech, args=(self.stop_event,))
        speech_thread.daemon = True
        speech_thread.start()

        # Start the Flask app
        app.run(debug=True, use_reloader=False)
    # def signal_handler(self, signal, frame):
    #     print("\nInterrupt signal received. Stopping...")
    #     self.stop_event.set()


def handle_signal(signal, frame, app_instance):
    print("\nInterrupt signal received. Stopping...")
    app_instance.stop_event.set()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/course')
def course():
    return render_template('course.html')

# @app.route('/interview')
# def interview():
#     new_model = load_model('HappySadClassifier.h5')
#     new_yhat = new_model.predict(np.expand_dims(cv2.resize/255, 0))
#     if new_yhat>0.5:
#         print("Predicted Class is Sad")
#     else:
#         print("Predicted Class is Happy")

#     return render_template('interview.html')

@app.route('/interview')
def interview():
    return render_template('interview.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part in the request'}), 400

    image = request.files['image']
    if image.filename == '':
        return jsonify({'error': 'No selected image'}), 400

    # Read and process the image
    img = Image.open(BytesIO(image.read()))
    img = img.convert('RGB')  # Ensure image is in RGB format
    resize = tf.image.resize(img, (150, 150))

    try:
        predictions = model.predict(np.expand_dims(resize/255, 0))
        print(predictions)
        prediction = 'Formal' if predictions[0] < 0.5 else 'Casual'
        # if prediction == 'Formal':
        #     return render_template('aiinterview.html')
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'Prediction error'}), 500

    return jsonify({'prediction': prediction})

if __name__ == "__main__":
    app.run(debug=True)

@app.route('/coursedecp')
def coursedecp():
    amazon_questions = AmazonQues.query.all()
    return render_template('coursedecp.html', questions=amazon_questions)

@app.route('/coursedecp2')
def coursedecp2():
    amazon_questions = AmazonQues.query.all()
    return render_template('coursedecp2.html', questions=amazon_questions)

@app.route('/coursedecp3')
def coursedecp3():
    amazon_questions = AmazonQues.query.all()
    return render_template('coursedecp3.html', questions=amazon_questions)

@app.route('/aiinterview')
def aiinterview():
    # Start the App class when the route is accessed
    interview_app = App()
    
    # Start the app in a separate thread
    interview_thread = threading.Thread(target=interview_app.start)
    interview_thread.start()

    return render_template('aiinterview.html')


@app.route('/video_feed')
def video_feed():
    camera_feed = CameraFeed()
    return Response(camera_feed.start(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    interview_app = App()
    
    # Setup signal handler in the main thread
    signal.signal(signal.SIGINT, lambda s, f: handle_signal(s, f, interview_app))

    app.run(debug=True)

# playsinline