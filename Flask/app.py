from flask import Flask, request, render_template
import cv2
import mediapipe as mp
import numpy as np

app = Flask(__name__)

mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(static_image_mode=True)

def detect_pose(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose_detector.process(image_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        shoulder_left = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        shoulder_right = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        hip_left = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        hip_right = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

        shoulder_distance = abs(shoulder_left.x - shoulder_right.x)
        hip_distance = abs(hip_left.x - hip_right.x)

        if (shoulder_distance > 0.3 and hip_distance > 0.3 and 
            abs(shoulder_left.y - shoulder_right.y) < 0.1 and 
            abs(hip_left.y - hip_right.y) < 0.1):
            pose_name = "T-Pose"
        else:
            pose_name = "Unknown"

        for landmark in results.pose_landmarks.landmark:
            x = int(landmark.x * image.shape[1])
            y = int(landmark.y * image.shape[0])
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

    else:
        pose_name = "Unknown"

    return pose_name, image

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    pose_name = None
    if request.method == 'POST':
        file = request.files['file']
        if file:
            image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
            pose_name, processed_image = detect_pose(image)
            cv2.imwrite('static/processed_image.jpg', processed_image)

    return render_template('index.html', pose_name=pose_name)

if __name__ == '__main__':
    app.run(debug=True)
