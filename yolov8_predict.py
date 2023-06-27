import csv
import os
from ultralytics import YOLO

model = YOLO("/content/drive/MyDrive/sam/yolov8l-seg_openvino_model")

# Specify the folder path containing the images
folder_path = "/content/frame_new"

# Create a CSV file to save the results
csv_file = open("result_NEW4.csv", "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["file name", "class name", "id", "score", "bbox", "mask"])

# Class names
class_names = ['pistol', 'revolver', 'cigaretee', 'cigar']

# Access the class IDs, probabilities, bounding boxes, and masks
for file_name in os.listdir(folder_path):
    image_path = os.path.join(folder_path, file_name)

    # Perform object detection on the image and get the results
    results = model.predict(source=image_path, conf=0.50, save=True)

    # Retrieve the class IDs, probabilities, bounding boxes, and masks
    boxes = results[0].boxes
    masks = results[0].masks

    # Check if boxes and masks are not None
    if boxes is not None and masks is not None:
        class_ids = boxes.cls.cpu().numpy().astype(int).tolist()
        probabilities = boxes.conf.cpu().numpy().tolist()

        # Write the results to the CSV file
        for class_id, prob, bbox, mask in zip(class_ids, probabilities, boxes.xyxy, masks.xy):
            class_name = class_names[class_id]
            csv_writer.writerow([file_name, class_name, class_id, prob, bbox, mask])
    else:
        # Write empty values for the image with no detections
        csv_writer.writerow([file_name, "", "", "", "", ""])

        #continue

# Close the CSV file
csv_file.close()



##if empty just add empty values for it
