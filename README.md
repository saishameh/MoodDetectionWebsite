
# Real-time Emotion Detection Web Application

## Overview
This project is a real-time emotion detection web application built with Flask and TensorFlow. The application captures video from the webcam, processes each frame using a pre-trained deep learning model, and dynamically updates the user interface based on the detected emotion.

## Features
- Real-time video feed from the webcam.
- Emotion detection using a pre-trained deep learning model.
- Web interface to view the video feed and detected emotions.
- HTML interface

## Prerequisites
- Python 3.x
- Flask
- OpenCV
- TensorFlow

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/real-time-emotion-detection.git

2. Navigate to the project directory:
   ```bash
   cd Moodify/Backend

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt

## Usage
1. Run the Flask application:
   ```bash
   python app.py

2. Open your web browser and go to http://localhost:5000/

3. Note: AI Button Functionality
   Currently, clicking the "AI" button on the main page doesn't open the camera and detect emotions. We are actively working on this feature.

4. Alternative Route: Video Feed
   To directly access the video feed and see real-time emotion detection, navigate to http://127.0.0.1:5000/video_feed in your web browser. Here, the camera will open, and emotions will be detected.

## License
This project is licensed under the License - see the LICENSE file for details.

Open source used: https://github.com/nebez/floppybird
