# yolov8_in_depth
Understand yolov8 structure,custom data traininig


YOLOv8 is the latest version of the YOLO (You Only Look Once) object detection and image segmentation model developed by Ultralytics . YOLOv8 is a cutting-edge, state- of-the-art SOTA model that builds on the success of previous YOLO and introduces new features and improvements to further boost performance and flexibility. It can be trained on large datasets and is able to run on various hardware platforms, from CPU to GPU.
Not only that, but a key feature of YOLOv8 is its scalability . and it is designed as a framework so  it supports all previous versions of YOLO, making it easy to switch between different versions and compare their performance.


```

YOLOv8 innovation and improvement points:

1. Backbone . The idea of ​​CSP is still used , but the C3 module in YOLOv5 is replaced by the C2f module to achieve further lightweight, and YOLOv8 still uses the SPPF module used in YOLOv5 and other architectures;

2. PAN-FPN . There is no doubt that YOLOv8 still uses the idea of ​​​​PAN, but by comparing the structure diagrams of YOLOv5 and YOLOv8, we can see that YOLOv8 deletes the convolution structure in the PAN-FPN upsampling stage in YOLOv5, and also replaces the C3 module with C2f module

3. Decoupled-Head . Did you smell something different? Yes, YOLOv8 went to Decoupled-Head;

4. Anchor-Free . YOLOv8 abandoned the previous Anchor-Base and used the idea of ​​Anchor-Free ;

5. Loss function . YOLOv8 uses VFL Loss as classification loss and DFL Loss+CIOU Loss as classification loss;

6. Sample matching . YOLOv8 abandoned the previous IOU matching or unilateral ratio allocation, but used the Task-Aligned Assigner matching method.

```
