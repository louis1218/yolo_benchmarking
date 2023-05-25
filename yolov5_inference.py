import cv2
import time
import sys
import numpy as np
import os
import onnxruntime
import pickle
import shutil

class YOLOV5_inference():
    
    def __init__(self, model, model_format, res):
    
        self.res = res
        self.model = model
        self.conf_threshold = 0.3
        self.score_threshold = 0.3
        self.nms_threshold = 0.45         
        self.class_list = []
        
        if "16" in self.model:
            self.half = True
        else:
            self.half = False
        
        print("\n")   
        print("Model: ", str(self.model))
        print("Model Format: ", str(model_format))
        print("Half: " , str(self.half))
        print("\n")
        
        if model_format == "tflite": 
            self.tflit_detector = self.init_tflite(self.model)

        if model_format == "opencv": 
            self.cv2_detector = cv2.dnn.readNetFromONNX(self.model)

        if model_format == "onnx":
            self.session_options = onnxruntime.SessionOptions()
            self.session_options.intra_op_num_threads = 4
            self.session_options.execution_mode = onnxruntime.ExecutionMode.ORT_PARALLEL
            self.session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.onnx_detector = onnxruntime.InferenceSession(self.model, self.session_options, providers=['CPUExecutionProvider'])
            self.init_onnx()
    
    def build_model(self):
        net = cv2.dnn.readNet(self.model)
        # if is_cuda:
        #     print("Attempty to use CUDA")
        #     net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        #     net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
        # else:
        print("Running on CPU")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        return net

    def init_onnx(self):
        model_inputs = self.onnx_detector.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

        model_outputs = self.onnx_detector.get_outputs()

        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

    # def pre_process(frame):

    #     row, col, _ = frame.shape
    #     _max = max(col, row)
    #     result = np.zeros((_max, _max, 3), np.uint8)
    #     result[0:row, 0:col] = frame
    #     return result

    def detect_opencv(self, image, net):
        
        blob = self.pre_process(image, net)
        
        time_1 = time.time()

        net.setInput(blob)
        preds = net.forward()
        
        time_2=time.time()
        class_ids, confidences, boxes, detections = self.post_process(blob, preds[0])
        
        dict_detection_time = {}
        detection_time = time_2 - time_1
        dict_detection_time['detection time'] = detection_time
        print("Detection time OPENCV:", detection_time)
        detections.append(dict_detection_time)

        
        return class_ids, confidences, boxes, detections
    
    def detect_onnx(self, image, net):
        
        blob = self.pre_process(image, net)
        
        time_1 = time.time()
        
        outputs = self.onnx_detector.run(self.output_names, {self.input_names[0]: blob})
        # print(outputs[0][0].shape)
        # net.setInput(blob)
        # preds = net.forward()
        
        time_2=time.time()
        
        class_ids, confidences, boxes, detections = self.post_process(blob, outputs[0][0])
        
        dict_detection_time = {}
        detection_time = time_2 - time_1
        dict_detection_time['detection time'] = detection_time
        detections.append(dict_detection_time)
        print("Detection time ONNX:", detection_time)
        
        return class_ids, confidences, boxes, detections

    def pre_process(self,frame,net):
        
        row, col, _ = frame.shape
        _max = max(col, row)
        result = np.zeros((_max, _max, 3), np.uint8)
        result[0:row, 0:col] = frame        

        blob = cv2.dnn.blobFromImage(result, 1/255.0, (self.res, self.res), swapRB=True, crop=False)
        
        return blob

    # def load_capture():
    #     capture = cv2.VideoCapture("sample.mp4")
    #     return capture

    def load_classes(self):
        with open("./ycb_objects.txt", "r") as f:
            self.class_list = [cname.strip() for cname in f.readlines()]
        return self.class_list

    def post_process(self,input_image, output_data):
        
        class_ids = []
        confidences = []
        boxes = []
        scores = []
        class_ids = []

        rows = output_data.shape[0]
        _, _ , image_width, image_height= input_image.shape

        x_factor = image_width / self.res
        y_factor =  image_height / self.res

        for r in range(rows):
            row = output_data[r]
            confidence = row[4]
            if confidence >= self.conf_threshold:

                classes_scores = row[5:]
                _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
                class_id = max_indx[1]
                if (classes_scores[class_id] > .5):

                    confidences.append(confidence)

                    x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item() 

                    left = int((x - 0.5 * w) * x_factor)
                    top = int((y - 0.5 * h) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)
                    box = np.array([left, top, width, height])
                    boxes.append(box)
                    scores.append(confidence)
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, self.nms_threshold) 

        detections = []
        if len(indexes)>0:
            for i in range(len(indexes)):
                index = indexes[i]
                box = boxes[index]

                detection = {
                    'class_id': class_ids[index],
                    'class_name': (self.class_list[class_ids[index]]),
                    'confidence': scores[index],
                    'box': box,
                    'scale': 1}
                detections.append(detection)


        result_class_ids = []
        result_confidences = []
        result_boxes = []

        for i in indexes:
            result_confidences.append(confidences[i])
            result_class_ids.append(class_ids[i])
            result_boxes.append(boxes[i])

        return result_class_ids, result_confidences, result_boxes, detections


if __name__ == '__main__':

    res = 320
    ori_model_path = "./models/YOLOv5_weights_models/" + str(res) + "/"
    model_files = os.listdir(ori_model_path)
    model_format = ["opencv", "onnx"]
    result_dir = "./yolov5_result/" + str(res)
    check_file_existence = os.path.exists(result_dir)
        
    if (check_file_existence):
        shutil.rmtree(result_dir) 
        
    os.mkdir(result_dir)
        
    for model_name in model_files:
        file_name, file_extension = os.path.splitext(str((ori_model_path) + str(model_name)))
        model_format = file_extension[1:]
        execution_format = ["onnx", "opencv"]
        model_name_for_folder = file_name[35:]
 
        for exe in execution_format:
            benchmarking_dir = "./yolov5_result/" + str(res) + "/" + "benchmark_result_" + exe + "_" + model_name_for_folder
            check_file_existence = os.path.exists(benchmarking_dir)
            
            if (check_file_existence):
                shutil.rmtree(benchmarking_dir) 
            os.mkdir(benchmarking_dir)
            
            model_path = "./models/YOLOv5_weights_models/" + str(res) + "/" + model_name 
                     
            df_result = {}
            
            ycb = YOLOV5_inference(model_path, exe, res)
            
            class_list = ycb.load_classes()        

            colors = [(255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 0)]

            # is_cuda = len(sys.argv) > 1 and sys.argv[1] == "cuda"

            net = ycb.build_model()

            dir_path = r'./testing_imagesv2/test_modified/images'
            total_images  = (len([entry for entry in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, entry))]))

            for i in range(total_images):
                img_name = "./testing_imagesv2/test_modified/images/" + str(i) + ".jpg"
            # img_name = "./000003-color_16.png"
                img = cv2.imread(img_name)

            # inputImage = format_yolov5(img)
            # outs = detect(inputImage, net)
            # blob_image, outs = ycb.pre_process(img, net)

            # class_ids, confidences, boxes = ycb.post_process(blob_image, outs[0])
            
                if exe == "opencv":
                    class_ids, confidences, boxes, detection = ycb.detect_opencv(img, net)
                if exe == "onnx":
                    class_ids, confidences, boxes, detection = ycb.detect_onnx(img, net)
                    
                # class_ids, confidences, boxes, detection = ycb.detect_opencv(img, net)

                for (classid, confidence, box) in zip(class_ids, confidences, boxes):
                    color = colors[int(classid) % len(colors)]
                    cv2.rectangle(img, box, color, 2)
                    cv2.rectangle(img, (box[0], box[1] - 20), (box[0] + box[2], box[1]), color, -1)
                    cv2.putText(img, class_list[classid], (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,0))
                
                df_result[img_name] = detection
                # cv2.imwrite(benchmarking_dir+ "/"+ model_name + "_" + exe+ "_" + str(i)+ ".jpg", img)
                
            with open(benchmarking_dir+ "/"+ model_name + "_" + exe+ '.pkl', 'wb') as fp:
                pickle.dump(df_result, fp)
                print(exe + ' dictionary saved successfully to file')