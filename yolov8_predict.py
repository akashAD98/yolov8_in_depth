import csv
from ultralytics import YOLO

model = YOLO("yolov8n-seg.pt")

# Perform object detection and get the results
results = model.predict(source="/content/frames", conf=0.50, save=True)

# Create a CSV file to save the results
csv_file = open("results.csv", "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["file name", "class name", "id", "score", "bbox", "mask"])

# Access the class names, class IDs, probabilities, bounding boxes, and masks
for i, r in enumerate(results):
    file_name = f"img{i+1}.jpg"  # Assign a unique file name for each result
    boxes = r.boxes
    masks = r.masks
    class_ids = boxes.cls.cpu().numpy().astype(int).tolist()
    probabilities = boxes.conf.cpu().numpy().tolist()

    # Write the results to the CSV file
    for class_id, prob, bbox, mask in zip(class_ids, probabilities, boxes.xyxy, masks.xy):
        class_name = model.names[int(class_id)]
        csv_writer.writerow([file_name, class_name, class_id, prob, bbox, mask])

# Close the CSV file
csv_file.close()
