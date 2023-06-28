import cv2
import csv
from ultralytics import YOLO

cap = cv2.VideoCapture("/content/frane")
model = YOLO("yolov8n-seg.pt")

results = model.track(frame, persist=True, save=True)
masks = results[0].masks

# Class names
class_names = ['person','bicycle','car','motorbike','aeroplane','bus','train','truck','boat','traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','sofa','pottedplant','bed','diningtable','toilet','tvmonitor','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush']

# Create a CSV file to save the results
csv_file = open("results_withmask.csv", "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Tracker ID", "Bounding Box", "Object ID", "Confidence", "Mask"])

for i, r in enumerate(results):
    for index, box in enumerate(r.boxes):
        tracker_id = box.id
        conf = box.conf
        object_id = int(box.cls.item())  # Convert tensor to scalar integer
        bbox_data = box.xywh.tolist()
        mask_data = masks.xy[index].tolist()

        # Get the class name
        class_name = class_names[object_id]

        # Save the results in the CSV file
        csv_writer.writerow([tracker_id, bbox_data, object_id, conf, mask_data])

        print("Tracker ID:", tracker_id)
        print("Bounding Box:", bbox_data)
        print("Object ID:", object_id)
        print("Class Name:", class_name)
        print("Confidence:", conf)
        print("Mask:", mask_data)

# Close the CSV file
csv_file.close()
