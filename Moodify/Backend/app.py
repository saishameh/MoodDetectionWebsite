from flask import Flask, render_template, jsonify, Response
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

app = Flask(__name__, template_folder='Frontend/templates', static_folder='Frontend/static')

# Load your emotion recognition model
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'facial_expression_model_architecture.h5')
emotion_model = load_model(model_path)

# Function to generate frames with emotion recognition
def generate_frames():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_resized = cv2.resize(gray, (48, 48))
        normalized_frame = gray_resized / 255.0
        input_data = np.expand_dims(np.expand_dims(normalized_frame, axis=-1), axis=0)

        emotion_probabilities = emotion_model.predict(input_data)[0]
        detected_emotion = np.argmax(emotion_probabilities)

        cv2.putText(frame, f'Emotion: {detected_emotion}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Route to render the main.html template
app = Flask(__name__, template_folder='Moodify/Frontend/templates', static_folder='Moodify/Frontend/static')

@app.route('/')
def home():
    return render_template('Main.html')

# Route to start emotion recognition
@app.route('/start_emotion_recognition')
def start_emotion_recognition():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
