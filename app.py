from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from tensorflow.lite.python.interpreter import Interpreter
from ultralytics import YOLO

app = Flask(__name__, template_folder='/content/templates')

# Configure upload folder
UPLOAD_FOLDER = '/content/static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)  # Create folder if it doesn't exist

# Paths to models and labels
PATH_TO_TFLITE_MODEL = '/content/custom_model_lite/detect.tflite'
PATH_TO_YOLO_MODEL = '/content/yolov8_full_train_v11/weights/best.pt'
PATH_TO_LABELS = '/content/sssd2-115_v2/train/label_map.pbtxt'

# Load TensorFlow Lite model
interpreter = Interpreter(model_path=PATH_TO_TFLITE_MODEL)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_height = input_details[0]['shape'][1]
input_width = input_details[0]['shape'][2]
float_input = (input_details[0]['dtype'] == np.float32)

# Load YOLO model using Ultralytics library
try:
    yolo_model = YOLO(PATH_TO_YOLO_MODEL)  # Use the ultralytics library to load YOLO model
    print("YOLO model loaded successfully!")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    yolo_model = None

# Load label map
labels = []
with open(PATH_TO_LABELS, 'r') as f:
    for line in f:
        if "name:" in line:
            labels.append(line.split('"')[1])

# Preprocessing function
def preprocess_image(image_path, height, width, float_input, input_mean=127.5, input_std=127.5):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height))
    input_data = np.expand_dims(image_resized, axis=0)
    if float_input:
        input_data = (np.float32(input_data) - input_mean) / input_std
    return input_data, image_rgb

# TFLite inference function
def detect_objects_tflite(image_path, interpreter, input_height, input_width):
    input_data, image_rgb = preprocess_image(image_path, input_height, input_width, float_input)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Retrieve results
    boxes = interpreter.get_tensor(output_details[1]['index'])[0]
    classes = interpreter.get_tensor(output_details[3]['index'])[0]
    scores = interpreter.get_tensor(output_details[0]['index'])[0]

    detections = []
    imH, imW, _ = image_rgb.shape
    for i in range(len(scores)):
        if ((scores[i] > 0.5) and (scores[i] <= 1.0)):
            ymin = int(max(1, (boxes[i][0] * imH)))
            xmin = int(max(1, (boxes[i][1] * imW)))
            ymax = int(min(imH, (boxes[i][2] * imH)))
            xmax = int(min(imW, (boxes[i][3] * imW)))

            # Draw bounding box
            cv2.rectangle(image_rgb, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

            # Prepare label with white background
            object_name = labels[int(classes[i])]
            label = f"{object_name}: {scores[i] * 100:.2f}%"

            # Calculate text size and position
            font_scale = 1  # Increased font size
            font_thickness = 2
            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            label_ymin = max(ymin, label_size[1] + 10)
            
            # Add white background behind the text
            cv2.rectangle(image_rgb, (xmin, label_ymin - label_size[1] - 10), 
                          (xmin + label_size[0], label_ymin + baseline - 10), (255, 255, 255), cv2.FILLED)
            # Add text on top of the white background
            cv2.putText(image_rgb, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)

            detections.append(f"{object_name} ({scores[i] * 100:.2f}%) [{xmin}, {ymin}, {xmax}, {ymax}]")
    return detections, image_rgb


# YOLO inference function
def detect_objects_yolo(image_path):
    if yolo_model is None:
        return [], None

    results = yolo_model.predict(image_path, save=False, imgsz=640)
    detections = results[0].boxes.data.cpu().numpy()

    image = cv2.imread(image_path)
    for detection in detections:
        xmin, ymin, xmax, ymax, confidence, class_id = map(int, detection[:6])
        label = f"{labels[class_id]}"

        # Draw bounding box
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)

        # Add text with white background and increased font size
        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        label_ymin = max(ymin, label_size[1] + 10)
        cv2.rectangle(image, (xmin, label_ymin - label_size[1] - 10), (xmin + label_size[0], label_ymin + baseline - 10), (255, 255, 255), cv2.FILLED)
        cv2.putText(image, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    return detections, image




@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return redirect(request.url)

    file = request.files['image']
    if file.filename == '':
        return redirect(request.url)

    # Save uploaded file
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Run TFLite detection
    tflite_detections, tflite_image = detect_objects_tflite(file_path, interpreter, input_height, input_width)
    tflite_result_path = os.path.join(app.config['UPLOAD_FOLDER'], f"tflite_{filename}")
    cv2.imwrite(tflite_result_path, cv2.cvtColor(tflite_image, cv2.COLOR_RGB2BGR))

    # Run YOLO detection
    yolo_detections, yolo_image = detect_objects_yolo(file_path)
    yolo_result_path = os.path.join(app.config['UPLOAD_FOLDER'], f"yolo_{filename}")
    if yolo_image is not None:
        cv2.imwrite(yolo_result_path, yolo_image)

    # Convert detections to renderable lists
    tflite_predictions = tflite_detections if tflite_detections else []
    yolo_predictions = []
    if yolo_detections is not None:
        for det in yolo_detections:
            class_id, confidence, xmin, ymin, xmax, ymax = int(det[5]), det[4], int(det[0]), int(det[1]), int(det[2]), int(det[3])
            yolo_predictions.append(f"{labels[class_id]} ({confidence * 100:.2f}%) [{xmin}, {ymin}, {xmax}, {ymax}]")

    # Prepare paths for rendering in HTML
    tflite_img_path = url_for('static', filename=f'uploads/tflite_{filename}')
    yolo_img_path = url_for('static', filename=f'uploads/yolo_{filename}')

    return render_template(
        'index.html',
        tflite_predictions=tflite_predictions,
        yolo_predictions=yolo_predictions,
        tflite_img_path=tflite_img_path,
        yolo_img_path=yolo_img_path
    )


if __name__ == '__main__':
    app.run(debug=True)
