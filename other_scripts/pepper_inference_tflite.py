import time
import cv2
import numpy as np
from PIL import Image
import cv2
import os
from yolov8 import YOLOv8
from yolov8.utils import draw_bounding_box_opencv
from yolov8.utils import class_names as CLASSES
import argparse
from tflite_runtime.interpreter import Interpreter
from yolov8.utils import xywh2xyxy, nms, draw_detections
import shutil
import pickle

class Detector():
    def __init__(self, model, res):

        self.res = int(res)
        self.conf_threshold = 0.3
        self.iou_threshold = 0.5
        self.model = model
        
        self.tflit_detector = self.init_tflite(self.model)

    def init_tflite(self, model):
        # Load the TFLite model and allocate tensors.
        self.tflite_interpreter = Interpreter(model_path=model)
        self.tflite_interpreter.allocate_tensors()

    def extract_boxes(self, predictions, original_shape, input_height, input_width):
        # Extract boxes from predictions
        boxes = predictions[:, :4]

        # Scale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes, original_shape, input_height, input_width)

        # Convert boxes to xyxy format
        boxes = xywh2xyxy(boxes)

        return boxes

    def rescale_boxes(self, boxes, original_shape, input_height, input_width):
        img_height, img_width = original_shape
        # Rescale boxes to original image dimensions
        input_shape = np.array([input_width, input_height, input_width, input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([img_width, img_height, img_width, img_height])
        return boxes
    
    def post_process(self, outputs, orig_image, scale):
        
        boxes = outputs[0] # Bounding box coordinates of detected objects
        outputs = np.array([cv2.transpose(outputs[0])])
        rows = outputs.shape[1]
        boxes = []
        scores = []
        class_ids = []
        
        for i in range(rows):
            classes_scores = outputs[0][i][4:]
            (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
            if maxScore >= 0.25:
                outputs[0][i][0] = (outputs[0][i][0])
                outputs[0][i][1] = ((outputs[0][i][1])/640)*480
                outputs[0][i][2] = (outputs[0][i][2])
                outputs[0][i][3] = ((outputs[0][i][3])/640)*480   
                box = [
                    outputs[0][i][0] - (0.5* outputs[0][i][2]), outputs[0][i][1] - (0.5* outputs[0][i][3]),
                    outputs[0][i][0]+ (0.5* outputs[0][i][2]), outputs[0][i][1] + (0.5* outputs[0][i][3])]
                
                boxes.append(box)
                scores.append(maxScore)
                class_ids.append(maxClassIndex)
                print(box)

        result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)
        detections = []
        if len(result_boxes)>0:
            for i in range(len(result_boxes)):
                index = result_boxes[i]
                box = boxes[index]
                detection = {
                    'class_id': class_ids[index],
                    'class_name': CLASSES[class_ids[index]],
                    'confidence': scores[index],
                    'box': box,
                    'scale': scale}
                detections.append(detection)
                img = draw_bounding_box_opencv(orig_image, class_ids[index], scores[index], round(box[0]), round(box[1]),
                                round(box[2]), round(box[3]))
        else:
            img = orig_image
            print("no detection found")

        return img, detections
    
    def detect_tflite(self, orig_image):
        time_1 = time.time()
        [height, width, _] = orig_image.shape
        length = max((height, width))
        scale = length / self.res

        input_height =  self.tflite_interpreter.get_input_details()[0]['shape'][1]
        input_width =  self.tflite_interpreter.get_input_details()[0]['shape'][2]

        # Get input and output tensors.
        input_details = self.tflite_interpreter.get_input_details()
        output_details = self.tflite_interpreter.get_output_details()
        # Resize and pad the image to keep the aspect ratio and fit the expected size.
        resized_image = cv2.resize(orig_image,(input_width, input_height))
        np_features = np.array(resized_image / 255.0)

        input_type = input_details[0]['dtype']
        if input_type == np.int8:
            input_scale, input_zero_point = input_details[0]['quantization']
            print("Input scale:", input_scale)
            print("Input zero point:", input_zero_point)
            print()
            np_features = (np_features / input_scale) + input_zero_point
            np_features = np.around(np_features)

        # Convert features to NumPy array of expected type
        np_features = np_features.astype(input_type)

        # Add dimension to input sample (TFLite model expects (# samples, data))
        np_features = np.expand_dims(np_features, axis=0)

        # Create input tensor out of raw features
        self.tflite_interpreter.set_tensor(input_details[0]['index'], np_features)

        # Run inference
        self.tflite_interpreter.invoke()

        # output_details[0]['index'] = the index which provides the input
        outputs =  self.tflite_interpreter.get_tensor(output_details[0]['index'])
        dict_detection_time = {}

        # If the output type is int8 (quantized model), rescale data
        output_type = output_details[0]['dtype']
        if output_type == np.int8:
            output_scale, output_zero_point = output_details[0]['quantization']
            print("Raw output scores:", outputs)
            print("Output scale:", output_scale)
            print("Output zero point:", output_zero_point)
            print()
            outputs = output_scale * (outputs.astype(np.float32) - output_zero_point)

        if len(outputs) == 0:
            return orig_image

        img, detections = self.post_process(outputs, orig_image, scale)

        time_2=time.time()
        detection_time = time_2-time_1
        #print("Detected class ids: ", class_ids)
        print("Detection time TfLite:", detection_time)
        print("Object detected TfLite: ", len(detections))
        dict_detection_time['detection time'] = detection_time
        detections.append(dict_detection_time)
        
        return img, detections

if __name__ == '__main__':
    
    # TFLite Benchmarking - result: dictionary storing all performance metric
    model_name = "yolov8-l_640_fp32"
    parent_dir_tflite = "./benchmark_result_tflite"
    path = os.path.join(parent_dir_tflite, model_name)
    check_file_existence = os.path.exists(parent_dir_tflite)
    
    if (check_file_existence):
        shutil.rmtree(parent_dir_tflite) 
        
    os.mkdir(parent_dir_tflite)
    os.mkdir(path)
    model_path = "./models/YOLOv8_weights/640/tflite/" + model_name[:12]+ "_saved_model/"+ model_name + ".tflite" 
    ycb = Detector(model_path, 640)
    dir_path = r'./testing_imagesv2/test_modified/images'
    total_images  = (len([entry for entry in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, entry))]))
    df_result = {}
    for i in range(total_images):
        img_name = "./testing_imagesv2/test_modified/images/" + str(i) + ".jpg"
        img = cv2.imread(img_name)
        cv_image, detection = ycb.detect_tflite(img)
        df_result[img_name] = detection
        cv2.imwrite("benchmark_result_tflite/" + model_name + "/result_" + str(i) + ".jpg", cv_image)
    
    with open("benchmark_result_tflite/" + model_name + "/"+ model_name + "_tflite"+ '.pkl', 'wb') as fp:
        pickle.dump(df_result, fp)
        print('dictionary saved successfully to file')
    
    # with open("benchmark_result_opencv/" + model_name + "/"+ model_name + "_opencv" + '.pkl', 'rb') as fp:
    #     data = pickle.load(fp)
    #     print('YOLO benchmarking dictionary')
    #     print(data)