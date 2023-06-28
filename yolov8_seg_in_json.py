

import cv2
import json
from ultralytics import YOLO

cap = cv2.VideoCapture("/content/frane")
model = YOLO("yolov8n-seg.pt")

results = model.track(frame, persist=True, save=True)
masks = results[0].masks

# Class names
class_names = ['person','bicycle','car','motorbike','aeroplane','bus','train','truck','boat','traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','sofa','pottedplant','bed','diningtable','toilet','tvmonitor','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush']

results_list = []

for i, r in enumerate(results):
    for index, box in enumerate(r.boxes):
        tracker_id = int(box.id.item())
        conf = float(box.conf.item())
        object_id = int(box.cls.item())
        bbox_data = box.xywh.tolist()
        mask_data = masks.xy[index].tolist()

        # Get the class name
        class_name = class_names[object_id]

        # Create a result dictionary
        result_dict = {
            "tracker_id": tracker_id,
            "bounding_box": bbox_data,
            "object_id": object_id,
            "confidence": conf,
            "mask": mask_data
        }

        results_list.append(result_dict)

# Save the results in JSON format
json_results = json.dumps(results_list, indent=4)
with open("results_injson.json", "w") as json_file:
    json_file.write(json_results)
