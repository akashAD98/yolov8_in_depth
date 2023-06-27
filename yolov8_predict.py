import csv
import os
from ultralytics import YOLO

model = YOLO("yolov8n-seg.pt")

# Specify the folder path containing the images
folder_path = "/content/frames"

# Retrieve the file names in the folder
file_names = os.listdir(folder_path)

# Create a CSV file to save the results
csv_file = open("resultsnew.csv", "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["file name", "class name", "id", "score", "bbox", "mask"])

# Access the class names, class IDs, probabilities, bounding boxes, and masks
for file_name in file_names:
    image_path = os.path.join(folder_path, file_name)

    # Perform object detection on the image and get the results
    results = model.predict(source=image_path, conf=0.50)

    # Retrieve the class names, class IDs, probabilities, bounding boxes, and masks
    boxes = results[0].boxes
    masks = results[0].masks
    class_ids = boxes.cls.cpu().numpy().astype(int).tolist()
    probabilities = boxes.conf.cpu().numpy().tolist()

    # Write the results to the CSV file
    for class_id, prob, bbox, mask in zip(class_ids, probabilities, boxes.xyxy, masks.xy):
        class_name = model.names[int(class_id)]
        csv_writer.writerow([file_name, class_name, class_id, prob, bbox, mask])

# Close the CSV file
csv_file.close()
