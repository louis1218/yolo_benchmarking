import argparse
import time
import cv2
import numpy as np
from cv_bridge import CvBridge
import rospy
import message_filters
from sensor_msgs.msg import Image as Image2
import os
import cv2
# import qi 
import pickle
from yolov8.utils import draw_bounding_box_opencv
from yolov8.utils import class_names as CLASSES
import onnxruntime
import shutil
from tflite_runtime.interpreter import Interpreter

class DarkNet_YCB():
    def __init__(self, model, model_format):
        
        rospy.init_node('YoloV8', anonymous=True)

        self.bridge = CvBridge()
        self.res = 640
        self.model = model
        self.conf_threshold = 0.3

        if "16" in self.model:
            self.half = True
        else:
            self.half = False
        
        if "tflite" in model_format:
            self.tflit_detector = self.init_tflite(self.model)
        
        if "opencv" in model_format:
            self.cv2_detector = cv2.dnn.readNetFromONNX(self.model)
            self.pub_cv2 = rospy.Publisher(
                        'yolov8_detector_cv', Image2, queue_size=1)
        
        if "onnx" in model_format:
            self.init_onnx()
            self.session_options = onnxruntime.SessionOptions()
            self.session_options.intra_op_num_threads = 4
            self.session_options.execution_mode = onnxruntime.ExecutionMode.ORT_PARALLEL
            self.session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.onnx_detector = onnxruntime.InferenceSession(self.model, self.session_options, providers=['CPUExecutionProvider'])
            self.pub_onnx = rospy.Publisher(
                        'yolov8_detector_onnx', Image2, queue_size=1)
       
    def init_tflite(self, model):
        # Load the TFLite model and allocate tensors.
        self.tflite_interpreter = Interpreter(model_path=model)
        self.tflite_interpreter.allocate_tensors()
        
    def init_onnx(self):
        model_inputs = self.onnx_detector.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

        model_outputs = self.onnx_detector.get_outputs()

        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

    def pre_process(self, image, res):
        [height, width, _] = image.shape
        length = max((height, width))
        img = np.zeros((length, length, 3), np.uint8)
        img[0:height, 0:width] = image
        scale = length / res

        blob = cv2.dnn.blobFromImage(img, scalefactor=1 / 255, size=(res, res), swapRB=True, crop=False)

        return blob, scale

    def post_process(self, outputs, orig_image, scale, threshold=0.3, nms_threshold=0.45, nms_flag=True):
        outputs = np.array([cv2.transpose(outputs[0])])
        rows = outputs.shape[1]

        boxes = []
        scores = []
        class_ids = []

        for i in range(rows):
            classes_scores = outputs[0][i][4:]
            (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
            if maxScore >= threshold:
                box = [
                    outputs[0][i][0] - (0.5 * outputs[0][i][2]), outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                    outputs[0][i][2], outputs[0][i][3]]
                boxes.append(box)
                scores.append(maxScore)
                class_ids.append(maxClassIndex)

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
                img = draw_bounding_box_opencv(orig_image, class_ids[index], scores[index], round(box[0] * scale), round(box[1] * scale),
                                round((box[0] + box[2]) * scale), round((box[1] + box[3]) * scale))
        else:
            img = orig_image
            print("No detection found")
            
        return img, detections
    
    def detect_onnx(self, image, res=640):

        blob, scale = self.pre_process(image, res)
        if self.half:
            blob = blob.astype(np.float16)

        time_1 = time.time()

        outputs = self.onnx_detector.run(self.output_names, {self.input_names[0]: blob})

        # convert output to numpy array
        if self.half:
            outputs = np.array(outputs, dtype=np.float16)[0]
        else:
            outputs = np.array(outputs, dtype=np.float32)[0]
        print(type(outputs))
        
        time_2=time.time()

        img, detections = self.post_process(outputs, image, scale, self.conf_threshold)

        dict_detection_time = {}
        detection_time = time_2 - time_1
        dict_detection_time['detection time'] = detection_time
        detections.append(dict_detection_time)
        print("Detection time ONNX:", detection_time)
        
        ros_image_yolo_onnx = self.bridge.cv2_to_imgmsg(img, "bgr8")
        self.pub_onnx.publish(ros_image_yolo_onnx)
        
        return img, detections 

    def detect_opencv(self, image, res=640):

        blob, scale = self.pre_process(image, res)

        time_1 = time.time()

        self.cv2_detector.setInput(blob)
        outputs = self.cv2_detector.forward()

        time_2=time.time()

        img, detections = self.post_process(outputs, image, scale, self.conf_threshold)

        dict_detection_time = {}
        detection_time = time_2 - time_1
        dict_detection_time['detection time'] = detection_time
        detections.append(dict_detection_time)
        print("Detection time OPENCV:", detection_time)
        
        ros_image_yolo_cv = self.bridge.cv2_to_imgmsg(img, "bgr8")
        self.pub_cv2.publish(ros_image_yolo_cv)
    
        return img, detections

    def post_process_tflite(self, outputs, orig_image, scale):
        
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

        img, detections = self.post_process_tflite(outputs, orig_image, scale)

        time_2=time.time()
        detection_time = time_2-time_1
        #print("Detected class ids: ", class_ids)
        print("Detection time TfLite:", detection_time)
        print("Object detected TfLite: ", len(detections))
        dict_detection_time['detection time'] = detection_time
        detections.append(dict_detection_time)
        
        return img, detections

if __name__ == '__main__':
    
    model_format = ["tflite", "opencv", "onnx"]
    
    for i in range(len(model_format)):
        # OpenCV Benchmarking - result: dictionary storing all performance metric
        model_name = "yolov8-s_640_fp32"
        parent_dir_opencv = "./benchmark_result_" + model_format[i]
        path = os.path.join(parent_dir_opencv, model_name)
        check_file_existence = os.path.exists(parent_dir_opencv)
        
        if (check_file_existence):
            shutil.rmtree(parent_dir_opencv) 
            
        os.mkdir(parent_dir_opencv)
        os.mkdir(path)
        if model_format[i] == "opencv" or model_format[i] == "onnx":
            model_path = "./models/YOLOv8_weights/640/" + model_name  + ".onnx"
        if model_format[i] == "tflite":
            model_path = "./models/YOLOv8_weights/640/" + model_name  + ".tflite"
        ycb = DarkNet_YCB(model_path, model_format[i])
        dir_path = r'./testing_imagesv2/test_modified/images'
        total_images  = (len([entry for entry in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, entry))]))
        df_result = {}
        for i in range(total_images):
            img_name = "./testing_imagesv2/test_modified/images/" + str(i) + ".jpg"
            img = cv2.imread(img_name)
            if model_format[i] == "tflite":
                cv_image, detection = ycb.detect_tflite(img)
            if model_format[i] == "opencv":
                cv_image, detection = ycb.detect_opencv(img)
            if model_format[i] == "onnx":
                cv_image, result_onnx = ycb.detect_onnx(img)
            df_result[img_name] = detection
            # cv2.imwrite("benchmark_result_opencv/" + model_name + "/result_" + str(i) + ".jpg", cv_image)
        
        with open("benchmark_result_" + + model_format[i] + "/" + model_name + "/"+ model_name + "_" + model_format[i]+ '.pkl', 'wb') as fp:
            pickle.dump(df_result, fp)
            print(model_format[i] + ' dictionary saved successfully to file')
        
        # with open("benchmark_result_opencv/" + model_name + "/"+ model_name + "_opencv" + '.pkl', 'rb') as fp:
        #     data = pickle.load(fp)
        #     print('YOLO benchmarking dictionary')
        #     print(data)
            
    # #ONNX Benchmarking - result: dictionary storing all performance metric
    # parent_dir_onnx = "./benchmark_result_onnx"
    # path = os.path.join(parent_dir_onnx, model_name)
    # check_file_existence = os.path.exists(parent_dir_onnx)
    
    # if (check_file_existence):
    #     shutil.rmtree(parent_dir_onnx)
        
    # os.mkdir(parent_dir_onnx)
    # os.mkdir(path)
    
    # model_path = "./models/YOLOv8_weights/640/" + model_name  + ".onnx" 
    # ycb = DarkNet_YCB(model_path)
    # dir_path = r'./testing_imagesv2/test_modified/images'
    # total_images  = (len([entry for entry in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, entry))]))
    # df_result = {}
    # result_onnx = []
    # for i in range(total_images):
    #     img_name = "./testing_imagesv2/test_modified/images/" + str(i) + ".jpg"
    #     img = cv2.imread(img_name)
    #     cv_image, result_onnx = ycb.detect_onnx(img)
    #     df_result[img_name] = result_onnx
    #     # cv2.imwrite("benchmark_result_onnx/" + model_name + "/result_" + str(i) + ".jpg", cv_image)
    
    # with open("benchmark_result_onnx/" + model_name + "/"+ model_name + "_onnx"+  '.pkl', 'wb') as fp:
    #     pickle.dump(df_result, fp)
    #     print('dictionary saved successfully to file')
    
    # with open("benchmark_result_onnx/" + model_name + "/"+ model_name + "_onnx"+  '.pkl', 'rb') as fp:
    #     data = pickle.load(fp)
    #     print('YOLO benchmarking dictionary')
    #     print(data)       
        
    # # TFLite Benchmarking - result: dictionary storing all performance metric
    # model_name = "yolov8-l_640_fp32"
    # parent_dir_tflite = "./benchmark_result_tflite"
    # path = os.path.join(parent_dir_tflite, model_name)
    # check_file_existence = os.path.exists(parent_dir_tflite)
    
    # if (check_file_existence):
    #     shutil.rmtree(parent_dir_tflite) 
        
    # os.mkdir(parent_dir_tflite)
    # os.mkdir(path)
    # model_path = "./models/YOLOv8_weights/640/tflite/" + model_name[:12]+ "_saved_model/"+ model_name + ".tflite" 
    # ycb = DarkNet_YCB(model_path, 640)
    # dir_path = r'./testing_imagesv2/test_modified/images'
    # total_images  = (len([entry for entry in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, entry))]))
    # df_result = {}
    # for i in range(total_images):
    #     img_name = "./testing_imagesv2/test_modified/images/" + str(i) + ".jpg"
    #     img = cv2.imread(img_name)
    #     cv_image, detection = ycb.detect_tflite(img)
    #     df_result[img_name] = detection
    #     cv2.imwrite("benchmark_result_tflite/" + model_name + "/result_" + str(i) + ".jpg", cv_image)
    
    # with open("benchmark_result_tflite/" + model_name + "/"+ model_name + "_tflite"+ '.pkl', 'wb') as fp:
    #     pickle.dump(df_result, fp)
    #     print('dictionary saved successfully to file')
    
    # # with open("benchmark_result_opencv/" + model_name + "/"+ model_name + "_opencv" + '.pkl', 'rb') as fp:
    # #     data = pickle.load(fp)
    # #     print('YOLO benchmarking dictionary')
    # #     print(data)
    
    