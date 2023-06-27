from ultralytics import YOLO

model = YOLO("yolov8n-seg.pt")

# Perform object detection and get the results
results = model.predict(source="/content/bus.jpg", conf=0.50)

# Create a text file to save the results
output_file = open("results.txt", "w")

# Access the class names, class IDs, probabilities, bounding boxes, and masks
for r in results:
    boxes = r.boxes
    masks = r.masks
    class_ids = boxes.cls.cpu().numpy().astype(int).tolist()
    probabilities = boxes.conf.cpu().numpy().tolist()

    # Write the results to the text file
    for class_id, prob, bbox, mask in zip(class_ids, probabilities, boxes.xyxy, masks.xy):
        class_name = model.names[int(class_id)]
        output_file.write(f"Class Name: {class_name}, Class ID: {class_id}, Probability: {prob}\n")
        output_file.write(f"Bounding Box: {bbox}\n")
        output_file.write(f"Segmentation Mask:\n")
        output_file.write(f"{mask}\n")
        output_file.write("\n")

# Close the text file
output_file.close()
