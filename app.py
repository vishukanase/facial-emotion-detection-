import os
# from flask import  Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from io import BytesIO
import base64
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array


from flask import Flask, request, jsonify, render_template  # Added render_template
# ...

app = Flask(__name__, static_folder='static', template_folder='template')
CORS(app)

# New route to serve frontend
@app.route('/')
def login():
    return render_template('login.html')

@app.route('/index.html')
def dashboard():
    return render_template('index.html')

@app.route('/index2.html')
def index2():
    return render_template('index2.html')


# Initialize Flask app
#app = Flask(__name__)
# CORS(app)

# Ensure 'captured_images' directory exists
if not os.path.exists('captured_images'):
    os.makedirs('captured_images')

# Load pre-trained face detection model (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# Load pre-trained emotion recognition model
model_path = "model.h5"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

model = load_model(model_path)
emotion_classes = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Check model input shape
model_input_shape = model.input_shape
print(f"Model input shape: {model_input_shape}")

# Define routes
@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        # Get image data from the request 
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "No image provided in the request!"}), 400

        img_data = data['image']

        # Decode image to  base64 image
        img_data = img_data.split(',')[1]
        img_bytes = base64.b64decode(img_data)
        img = Image.open(BytesIO(img_bytes))
        img_np = np.array(img)  # Convert to numpy array (RGB format)

        # Check if the image is in RGBA (4 channels) and convert to RGB (3 channels)
        if img_np.shape[2] == 4:  # If the image has 4 channels (RGBA)
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)  # Convert to RGB

        # Detect faces in the image using the original color image
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV compatibility
        faces = face_cascade.detectMultiScale(img_cv, 1.1, 4)

        if len(faces) == 0:
            return jsonify({"message": "No face detected in the image!"}), 400

        results = []  # Store emotions for each detected face
        for i, (x, y, w, h) in enumerate(faces):
            # Crop the face region from the original RGB image
            face_img = img_np[y:y + h, x:x + w]

            # Resize and preprocess the image based on the model's input shape
            if model_input_shape[-1] == 1:  # Grayscale input expected
                face_img_gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
                face_img_resized = cv2.resize(face_img_gray, (48, 48))
                face_img_normalized = face_img_resized / 255.0
                face_img_array = np.expand_dims(face_img_normalized, axis=-1)  # Add channel dimension
            else:  # RGB input expected
                face_img_resized = cv2.resize(face_img, (48, 48))
                face_img_normalized = face_img_resized / 255.0

            face_img_array = np.expand_dims(face_img_normalized, axis=0)  # Add batch dimension

            # Log input shape for debugging
            print(f"Input shape for model: {face_img_array.shape}")

            # Predict emotion
            predictions = model.predict(face_img_array)
            print(f"Predictions: {predictions}")  # Debugging line to check model output
            predicted_emotion = emotion_classes[np.argmax(predictions)]

            # Save the cropped face image
            img_path = os.path.join('captured_images', f'cropped_face_{i + 1}.png')
            

            # Append emotion to results
            results.append({
                "face_number": i + 1,
                "emotion": predicted_emotion,
                "cropped_face_path": img_path
            })

        # Return all emotions and cropped face paths
        return jsonify({
            "message": "Faces cropped and emotions detected successfully!",
            "results": results
        })

    except Exception as e:
        # Log and return error
        app.logger.error(f"Error occurred: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
