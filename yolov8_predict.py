import csv
import os
from ultralytics import YOLO

model = YOLO("/content/yolov8n-seg_openvino_model")

# Specify the folder path containing the images
folder_path = "/content/frames"

# Retrieve the file names in the folder
file_names = os.listdir(folder_path)

# Create a CSV file to save the results
csv_file = open("resultsnewNN.csv", "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["file name", "class name", "id", "score", "bbox", "mask"])

# COCO80 class names
class_names = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
    "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
]

# Access the class IDs, probabilities, bounding boxes, and masks
for file_name in file_names:
    image_path = os.path.join(folder_path, file_name)

    # Perform object detection on the image and get the results
    results = model.predict(source=image_path, conf=0.50)

    # Retrieve the class IDs, probabilities, bounding boxes, and masks
    boxes = results[0].boxes
    masks = results[0].masks
    class_ids = boxes.cls.cpu().numpy().astype(int).tolist()
    probabilities = boxes.conf.cpu().numpy().tolist()

    # Write the results to the CSV file
    for class_id, prob, bbox, mask in zip(class_ids, probabilities, boxes.xyxy, masks.xy):
        class_name = class_names[class_id]
        csv_writer.writerow([file_name, class_name, class_id, prob, bbox, mask])

# Close the CSV file
csv_file.close()

