import csv
import os
from ultralytics import YOLO

# folder path containing the images
folder_path = '/content/frame_new'

#  enter the confidence threshold
confidence_threshold = 0.50

#saving results in run/ folder
save_results = False

# Create a YOLO object with the specified model path
model_path = "/content/drive/MyDrive/sam/yolov8l-seg_openvino_model"
model = YOLO(model_path)

# Create a CSV file to save the results
csv_file = open("result_NEW4.csv", "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["file name", "class name", "id", "score", "bbox", "mask"])

# Class names
class_names = ['pistol', 'revolver', 'cigaretee', 'cigar']

# Initialize counters
total_images = 0
images_with_detections = 0

try:
    # Access the class IDs, probabilities, bounding boxes, and masks
    for file_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, file_name)

        # Perform object detection on the image and get the results
        results = model.predict(source=image_path, conf=confidence_threshold, save=save_results)

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

            images_with_detections += 1
        else:
            # Write empty values for the image with no detections
            csv_writer.writerow([file_name, "", "", "", "", ""])

        total_images += 1

        # Print progress updates
        print(f"Processed {total_images} images")

    # Print summary
    print("Object detection completed!")
    print(f"Total images processed: {total_images}")
    print(f"Images with detections: {images_with_detections}")

except Exception as e:
    # Handle exceptions
    print(f"An error occurred: {str(e)}")

finally:
    # Close the CSV file
    csv_file.close()


