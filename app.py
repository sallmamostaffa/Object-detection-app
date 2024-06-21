import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
import cv2

# Load the model
model = hub.load("https://www.kaggle.com/models/tensorflow/ssd-mobilenet-v2/TensorFlow2/ssd-mobilenet-v2/1")

# Labels dictionary
labels = {
    1: 'person', 2: 'cycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 
    10: 'traffic light', 11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat', 
    18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack', 
    28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 
    38: 'kite', 39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle', 
    46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 
    55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch', 
    64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 
    76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 
    85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush',
}

# Function to load image
def load_image(image_file):
    img = Image.open(image_file)
    return img

# Function to detect objects in the image
def detect_objects(image, model):
    image_np = np.array(image)
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis,...]

    # Run the model
    detections = model(input_tensor)

    detection_classes = detections['detection_classes'][0].numpy().astype(np.int64)
    detection_scores = detections['detection_scores'][0].numpy()
    detection_boxes = detections['detection_boxes'][0].numpy()

    detected_objects = []
    height, width, _ = image_np.shape
    for i in range(len(detection_classes)):
        if detection_scores[i] > 0.5:  
            class_name = labels.get(detection_classes[i], 'N/A')
            box = detection_boxes[i] * [height, width, height, width]
            detected_objects.append((class_name, detection_scores[i], box))

    return detected_objects

# Function to draw bounding boxes on the image
def draw_boxes(image, detected_objects):
    image_np = np.array(image)
    for obj in detected_objects:
        class_name, score, box = obj
        y1, x1, y2, x2 = box.astype(int)
        image_np = cv2.rectangle(image_np, (x1, y1), (x2, y2), (255, 0, 0), 2)
        image_np = cv2.putText(image_np, f"{class_name} ({score:.2f})", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return Image.fromarray(image_np)

# Main function to run the Streamlit app
def main():
    st.title("Object Detection App")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        col1 , col2 = st.columns(2)
        with col1:
            st.image(image, caption='Uploaded Image', use_column_width=True)
            st.write("")

        if st.button(label= "Analyse Image"):
            st.write("")
            detected_objects = detect_objects(image, model)
            annotated_image = draw_boxes(image, detected_objects)

            with col2:
                st.image(annotated_image, caption='Processed Image', use_column_width=True)

            st.write("Detected Objects:")
            for obj, score, box in detected_objects:
                st.write(f"{obj} ({score:.2f})")

if __name__ == "__main__":
    main()
