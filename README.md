# Benchmark workspace for YoloV5/YoloV8 models

## Summary 
This GitHub repository provides inferencing scripts for YCB objects using YOLOv5 and YOLOv8 models, as well as benchmarking scripts to measure mAP@50,70,90 metrics for ONNX, OPEN, and TFLITE inferencing engines. You can modify the scripts to either 320 or 640 image size for inferencing. The repository includes 122 photos of 21 YCB objects for testing purposes, taken from different directions with Pepper's camera.

## Benchmarking result from Pepper

Here is the benchmarking wrap up [excel sheet](https://docs.google.com/spreadsheets/d/1i_GBcmwbnrE2ut2_CwAxFIMNMah9NNdmEpMLt8RWAOI/edit#gid=0)

Dependencies
```
torch
torchvision
ultralytics
onnx
onnxsim
onnx2tf 
onnx-graphsurgeon 
tflite_support
sng4onnx
```

## Steps to run inference for YOLOV5 and YOLOV8 (PC)

To run inference for YOLOv5 and YOLOv8 on a PC, users can follow the steps outlined in the repository, which involve running the respective inferencing scripts to generate a pkl file containing the inferencing results. The benchmarking results can be obtained by running the corresponding benchmarking scripts, which load the pkl files to calculate the mAP@50,70,90 metrics.

1. Run yolov5_inference.py/yolov8_infernece.py for inferencing the testing images.
2. Inferencing result will be saved as a pkl file (python dictionary) and it will appear in yolov5_result/yolov8_result folder.
3. Run benchmarking_yolov5.py/benchmarking_yolov8.py to load the pkl files for getting benchmarking results ie. mAP@50,70,90. 

## Steps to run inference for YOLOV5 and YOLOV8 (Pepper)

To run inference for YOLOv5 and YOLOv8 on Pepper, users should create a folder in Pepper's local directory and copy the required files to it. After running the inferencing scripts, the resulting pkl files should be copied to the PC for further benchmarking. The benchmarking folders can be copied to the yolov5_result and yolov8_result folders in the PC's directory, and the benchmarking scripts can be run to obtain the mAP@50,70,90 metrics.

1. Create a benchmarking folder in Pepper local directory.
2. Copy yolov5_inference.py/yolov8_infernece.py, the model folder, ycb_objects.txt and the testing_imagesv2 folder to the bechmarking folder in Pepper.
3. Run yolov5_inference.py/yolov8_infernece.py for inferencing the testing images.
4. Inferencing result will be saved as a pkl file (python dictionary) and it will appear in multiple becnhmarking folders. 
5. Copy those benchmarking folders to the yolov5_result, yolo8_result folders in your PC. (Example has been shown inside these folders in this repository)
6. Run benchmarking_yolov5.py/benchmarking_yolov8.py to load the pkl files for getting benchmarking results ie. mAP@50,70,90. 
