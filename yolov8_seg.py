from ultralytics import YOLO

model = YOLO("yolov8n-seg.pt")

# Perform object detection and get the results
results = model.predict(source="/content/bus.jpg", conf=0.50)

# Access the class names, class IDs, and probabilities
boxes = results[0].boxes
class_ids = boxes.cls.cpu().numpy().astype(int).tolist()
probabilities = boxes.conf.cpu().numpy().tolist()

# Print the class names, class IDs, and probabilities
for class_id, prob in zip(class_ids, probabilities):
    class_name = model.names[int(class_id)]
    print(f"Class Name: {class_name}, Class ID: {class_id}, Probability: {prob}")
