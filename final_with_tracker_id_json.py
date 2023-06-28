import cv2
import json
import os
from ultralytics import YOLO

# Path to the folder containing the frames
folder_path = "/content/framenew"

model = YOLO("yolov8n-seg.pt")

# results = model.track(frame, persist=True, save=True)
# masks = results[0].masks

# Class names
class_names = ['person','bicycle','car','motorbike','aeroplane','bus','train','truck','boat','traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','sofa','pottedplant','bed','diningtable','toilet','tvmonitor','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush']

results_list = []

# Iterate over the files in the folder
for file_name in os.listdir(folder_path):
    # Get the frame number from the file name
    frame_no = int(os.path.splitext(file_name)[0])

    # Read the frame
    frame_path = os.path.join(folder_path, file_name)
    frame = cv2.imread(frame_path)

    # Perform object detection on the frame and get the results
    results = model.track(frame, persist=True, save=True)
    masks = results[0].masks

    # Process the results
    frame_results = {"frame_no": frame_no, "track_details": []}
    for i, r in enumerate(results):
        for index, box in enumerate(r.boxes):
            tracker_id = int(box.id.item())
            conf = float(box.conf.item())
            object_id = int(box.cls.item())
            bbox_data = box.xywh.tolist()

            # Get the class name
            class_name = class_names[object_id]

            # Create a track detail dictionary
            track_detail = {
                "track_id": tracker_id,
                "bounding_box": bbox_data,
                "object_id": object_id,
                "confidence": conf
            }

            frame_results["track_details"].append(track_detail)

    results_list.append(frame_results)

# Save the results in JSON format
json_results = json.dumps(results_list, indent=4)
with open("results_injson_frame_info_tracker.json", "w") as json_file:
    json_file.write(json_results)
