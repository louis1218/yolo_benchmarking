import pickle
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore") 
import os
import re

names_prediction = ['002_master_chef_can',
'003_cracker_box',
'004_sugar_box',
'005_tomato_soup_can',
'006_mustard_bottle',
'007_tuna_fish_can',
'008_pudding_box',
'009_gelatin_box',
'010_potted_meat_can',
'011_banana',
'019_pitcher_base',
'021_bleach_cleanser',
'024_bowl',
'025_mug',
'035_power_drill',
'036_wood_block',
'037_scissors',
'040_large_marker',
'051_large_clamp',
'052_extra_large_clamp',
'061_foam_brick']

# Predictions
def createPredictiondf(dict_file, istflite):
    
    with open(dict_file, 'rb') as fp:
        data = pickle.load(fp)  
        
    df_pred = pd.DataFrame(columns=['testing photo','detection time','box','class_id','class_name','confidence','scale'])

    count = 0
    for k in range(len(data)):
        image_jpg = list(data.keys())[k]
        cur_dict = data[image_jpg]

        for i in range(len(cur_dict)-1):
            # print(photo[i])
            df_tmp = pd.DataFrame([cur_dict[i]])
            df_pred.loc[count,'testing photo'] = image_jpg[40:]
            df_pred.loc[count,'detection time'] = cur_dict[-1]['detection time']
            df_pred.loc[count,'class_id'] = cur_dict[i]['class_id']
            df_pred.loc[count,'class_name'] = cur_dict[i]['class_name']
            df_pred.loc[count,'confidence'] = cur_dict[i]['confidence']
            df_pred.loc[count,'scale'] = cur_dict[i]['scale']
            df_pred.loc[count,'box'] = (cur_dict[i]['box'])      

            count+=1

    df_pred.columns = ['pred jpg_filename', 'pred detection time', 'pred box', 'pred class_id', 'pred class_name',
            'pred confidence', 'pred scale']

    df_pred['pred xmin'] = -1
    df_pred['pred ymin'] = -1
    df_pred['pred xmax'] = -1
    df_pred['pred ymax'] = -1

    for i in range(len(df_pred)):
        df_pred.loc[i,'pred xmin'] = int(df_pred.loc[i,'pred box'][0])
        df_pred.loc[i,'pred ymin'] = int(df_pred.loc[i,'pred box'][1])
        if (istflite == True):
            df_pred.loc[i,'pred xmax'] = int(df_pred.loc[i,'pred box'][2])
            df_pred.loc[i,'pred ymax'] = int(df_pred.loc[i,'pred box'][3])
        else:
            df_pred.loc[i,'pred xmax'] = int(df_pred.loc[i,'pred box'][2]) + int(df_pred.loc[i,'pred box'][0])
            df_pred.loc[i,'pred ymax'] = int(df_pred.loc[i,'pred box'][3]) + int(df_pred.loc[i,'pred box'][1])
    df_pred.drop(columns=['pred box', 'pred scale'])
    df_pred = df_pred.reindex(columns=["pred jpg_filename", "pred class_id", "pred class_name", "pred detection time",
                                    "pred confidence","pred xmin","pred ymin","pred xmax","pred ymax"])
    df_pred['pred class_id'] = df_pred['pred class_id'].astype(int) 


    for i in range(len(df_pred)):
        df_pred.loc[i, 'pred jpg_filename'] = int((df_pred.loc[i, 'pred jpg_filename'])[:-4])
        
    df_pred = df_pred.sort_values(by=['pred jpg_filename'])    

    return df_pred


# Ground Truth
def createGroundTruthdf(csv_annotion, pkl_file, res):
    
    df_gt = pd.read_csv(csv_annotion)
    with open(pkl_file, 'rb') as fp:
        dict_gt = pickle.load(fp)

    df_gt['jpg_filename'] = 0
    for i in range(len(df_gt)):
        for key ,value in dict_gt.items():
            if df_gt.iloc[i,0] == key :
                df_gt.loc[i,'jpg_filename'] = value
                
    df_gt.drop(columns=['filename'])
    df_gt = df_gt.reindex(columns=["jpg_filename","width", "height", "class","xmin","ymin","xmax","ymax"])


    for i in range(len(df_gt)):
        df_gt.loc[i, 'jpg_filename'] = int((df_gt.loc[i, 'jpg_filename'])[:-4])
        
    df_gt = df_gt.sort_values(by=['jpg_filename'])
    df_gt.reset_index(drop=True, inplace=True)
    
    if res == 320:
        for i in range(len(df_gt)):
            df_gt.loc[i,'width'] = 320
            df_gt.loc[i,'height'] = 240
            df_gt.loc[i,'xmin'] = int(df_gt.loc[i,'xmin'] /2)
            df_gt.loc[i,'ymin'] = int(df_gt.loc[i,'ymin'] /2)
            df_gt.loc[i,'xmax'] = int(df_gt.loc[i,'xmax'] /2)
            df_gt.loc[i,'ymax'] = int(df_gt.loc[i,'ymax'] /2)        
    
    return df_gt

# Merge

def Mergedf(df_groundtruth,df_predictions):

    df_merge = df_predictions.copy()

    def intersection_over_union(boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
        if interArea == 0:
            return 0
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
        boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        # return the intersection over union value
        return iou

    df_merge['IOU'] = 0
    df_merge['gt class'] = -1
    df_merge['gt xmin'] = -1
    df_merge['gt ymin'] = -1
    df_merge['gt xmax'] = -1
    df_merge['gt ymax'] = -1

    df_merge['pred jpg_filename'] = df_merge['pred jpg_filename'].astype(int)


    df_predictions['pred class_name'] = df_predictions['pred class_name'].astype(str)
    df_merge['gt class'] = df_merge['gt class'].astype(str)

    for i in range(len(df_merge)):
        for j in range(len(df_groundtruth)):
            if (((df_groundtruth.loc[j,'jpg_filename']) == (df_merge.loc[i,'pred jpg_filename'])) and ((df_groundtruth.loc[j,'class']) == (df_merge.loc[i,'pred class_name']))):
                df_merge.loc[i,'gt class'] = df_groundtruth.loc[j,'class']
                df_merge.loc[i,'gt xmin'] = df_groundtruth.loc[j,'xmin']
                df_merge.loc[i,'gt ymin'] = df_groundtruth.loc[j,'ymin']
                df_merge.loc[i,'gt xmax'] = df_groundtruth.loc[j,'xmax']
                df_merge.loc[i,'gt ymax'] = df_groundtruth.loc[j,'ymax']
                boxA = []
                boxA.append(df_merge.loc[i,'gt xmin'])
                boxA.append(df_merge.loc[i,'gt ymin'])
                boxA.append(df_merge.loc[i,'gt xmax'])
                boxA.append(df_merge.loc[i,'gt ymax'])
                boxB = []
                boxB.append(df_merge.loc[i,'pred xmin'])
                boxB.append(df_merge.loc[i,'pred ymin'])
                boxB.append(df_merge.loc[i,'pred xmax'])
                boxB.append(df_merge.loc[i,'pred ymax'])
                iou = intersection_over_union(boxA, boxB)
                df_merge.loc[i,'IOU'] = iou            
                
                
    for i in range(len(df_merge)):
        if (df_merge.loc[i,'gt class'] == "-1"):
            for j in range(len(df_groundtruth)):
                if (df_merge.loc[i,'pred jpg_filename'] == df_groundtruth.loc[j,'jpg_filename']):
                    boxA = []
                    boxA.append(df_groundtruth.loc[j,'xmin'])
                    boxA.append(df_groundtruth.loc[j,'ymin'])
                    boxA.append(df_groundtruth.loc[j,'xmax'])
                    boxA.append(df_groundtruth.loc[j,'ymax'])
                    boxB = []
                    boxB.append(df_merge.loc[i,'pred xmin'])
                    boxB.append(df_merge.loc[i,'pred ymin'])
                    boxB.append(df_merge.loc[i,'pred xmax'])
                    boxB.append(df_merge.loc[i,'pred ymax'])
                    iou = intersection_over_union(boxA, boxB)
                    if (iou>=0.5):
                        df_merge.loc[i,'gt class'] = df_groundtruth.loc[j,'class']
                        df_merge.loc[i,'gt xmin'] = df_groundtruth.loc[j,'xmin']
                        df_merge.loc[i,'gt ymin'] = df_groundtruth.loc[j,'ymin']
                        df_merge.loc[i,'gt xmax'] = df_groundtruth.loc[j,'xmax']
                        df_merge.loc[i,'gt ymax'] = df_groundtruth.loc[j,'ymax']
                        df_merge.loc[i,'IOU'] = iou
                        
                        
    df_merge = df_merge[df_merge['gt class'] != "-1"]

    df_merge.reset_index(drop=True, inplace=True)

    df_merge['pred perf@50'] = 0
    df_merge['pred perf@55'] = 0
    df_merge['pred perf@60'] = 0
    df_merge['pred perf@65'] = 0
    df_merge['pred perf@70'] = 0
    df_merge['pred perf@75'] = 0
    df_merge['pred perf@80'] = 0
    df_merge['pred perf@85'] = 0
    df_merge['pred perf@90'] = 0
    df_merge['pred perf@95'] = 0

    for iou in range(50,100,5):
        for i in range(len(df_merge)):
            if ((df_merge.loc[i,'pred class_name'] == df_merge.loc[i,'gt class']) and (df_merge.loc[i,'IOU'] >= (iou/100))):
                df_merge.loc[i,str('pred perf@' + str(iou))] = "TP"
            elif ((df_merge.loc[i,'pred class_name'] == df_merge.loc[i,'gt class']) and (df_merge.loc[i,'IOU'] < (iou/100))):
                df_merge.loc[i,str('pred perf@' + str(iou))] = "FP"
            elif ((df_merge.loc[i,'pred class_name'] != df_merge.loc[i,'gt class']) and (df_merge.loc[i,'IOU'] >= (iou/100))):
                df_merge.loc[i,str('pred perf@' + str(iou))] = "FN"   
            
    df_merge.sort_values(by='pred class_name',inplace = True)
    
    return df_merge

def AveragePrecisionDict(df_merge):

    for i in range(1,22):
        locals()['df{0}'.format(i)] = df_merge[df_merge['pred class_name'] == names_prediction[i-1]]

    APdict = {}
    threshold = [50,70,90]

    for i in range (1,22):
        dict_threshold = {}
        for thres in range(50,100,5):
            if (len(locals()['df{0}'.format(i)])>0):
                locals()['df{0}'.format(i)] = locals()['df{0}'.format(i)].sort_values(by=['IOU'],ascending=False)
                locals()['df{0}'.format(i)][('Cumulative TP@'+str(thres))] = 0
                locals()['df{0}'.format(i)][('Cumulative FP@'+str(thres))] = 0
                locals()['df{0}'.format(i)][('Cumulative FN@'+str(thres))] = 0
                locals()['df{0}'.format(i)][('Precision@'+str(thres))] = 0
                locals()['df{0}'.format(i)][('Recall@'+str(thres))] = 0

                tp = 0
                fp = 0
                fn = 0
                locals()['df{0}'.format(i)].reset_index(drop=True, inplace=True)
                for k in range(len(locals()['df{0}'.format(i)])):
                    if (locals()['df{0}'.format(i)].loc[k,('pred perf@'+str(thres))] == "TP"):
                        tp+=1
                        locals()['df{0}'.format(i)].loc[k,('Cumulative TP@'+str(thres))] = tp
                        locals()['df{0}'.format(i)].loc[k,('Cumulative FP@'+str(thres))] = fp
                        locals()['df{0}'.format(i)].loc[k,('Cumulative FN@'+str(thres))] = fn
                    elif (locals()['df{0}'.format(i)].loc[k,('pred perf@'+str(thres))] == "FP"):
                        fp+=1
                        locals()['df{0}'.format(i)].loc[k,('Cumulative FP@'+str(thres))] = fp
                        locals()['df{0}'.format(i)].loc[k,('Cumulative TP@'+str(thres))] = tp
                        locals()['df{0}'.format(i)].loc[k,('Cumulative FN@'+str(thres))] = fn
                    elif (locals()['df{0}'.format(i)].loc[k,('pred perf@'+str(thres))] == "FN"):
                        fn+=1
                        locals()['df{0}'.format(i)].loc[k,('Cumulative FN@'+str(thres))] = fn
                        locals()['df{0}'.format(i)].loc[k,('Cumulative FP@'+str(thres))] = fp
                        locals()['df{0}'.format(i)].loc[k,('Cumulative TP@'+str(thres))] = tp
                    else:
                        locals()['df{0}'.format(i)].loc[k,('Cumulative TP@'+str(thres))] = tp
                        locals()['df{0}'.format(i)].loc[k,('Cumulative FP@'+str(thres))] = fp
                        locals()['df{0}'.format(i)].loc[k,('Cumulative FN@'+str(thres))] = fn

                for m in range(len(locals()['df{0}'.format(i)])):
                    locals()['df{0}'.format(i)].loc[m,('Recall@'+str(thres))] = locals()['df{0}'.format(i)].loc[m,('Cumulative TP@'+str(thres))] / ((locals()['df{0}'.format(i)].loc[m,('Cumulative TP@'+str(thres))])+(locals()['df{0}'.format(i)].loc[m,('Cumulative FN@'+str(thres))]))
                    locals()['df{0}'.format(i)].loc[m,('Precision@'+str(thres))] = locals()['df{0}'.format(i)].loc[m,('Cumulative TP@'+str(thres))] / ((locals()['df{0}'.format(i)].loc[m,('Cumulative TP@'+str(thres))])+(locals()['df{0}'.format(i)].loc[m,('Cumulative FP@'+str(thres))]))

                precisions = locals()['df{0}'.format(i)][('Precision@'+str(thres))].tolist()
                recalls = locals()['df{0}'.format(i)][('Recall@'+str(thres))].tolist()
                precisions.append(1)
                recalls.append(0)

                precisions = np.array(precisions)
                recalls = np.array(recalls)

                where_are_NaNs = np.isnan(precisions)
                precisions[where_are_NaNs] = 0

                where_are_NaNs = np.isnan(recalls)
                recalls[where_are_NaNs] = 0                    

                AP = np.sum((recalls[:-1] - recalls[1:]) * precisions[:-1])
                # print(AP)
                dict_threshold[str(thres)] = AP

                APdict[locals()['df{0}'.format(i)].loc[0,'pred class_name']] = dict_threshold
                
    return APdict
            
if __name__ == '__main__':
    
    
    onnx =1 
    opencv =1
    tflite = 1
    model_format = []
    res = 320
    
    csv_file = "./ground_truth/_annotations.csv"
    dict_file_gt = "./ground_truth/store_name.pkl"
    
    for f in os.listdir('./yolov5_result/'+str(res)):
        if re.match('benchmark_result_', f):
            folder = os.path.join(("./yolov5_result/" +str(res)), f)
            # print(folder)
            files = os.listdir(os.path.join(("./yolov5_result/"+str(res)), f))
            # print(f)
            
            if "tflite" in f:
                istflite = True
            else:
                istflite = False
                
            # print(istflite)
            
            for file in os.listdir(folder):
                if file.endswith(".pkl"):
                    print("model: " + str(file[:-4]))

                    df_predictions = createPredictiondf(os.path.join(folder, file),istflite)
                    df_groundtruth = createGroundTruthdf(csv_file,dict_file_gt, res)
                    df_merge = Mergedf(df_groundtruth, df_predictions)
                    APdict = AveragePrecisionDict(df_merge)
                    
                    # if "002_master_chef_can" in APdict:
                    #     del APdict["002_master_chef_can"]

                    if "019_pitcher_base" in APdict:
                        del APdict["019_pitcher_base"]

                    mAP = {}

                    for interval in range(50,100,5):
                        calculate_mAP_dict = {}

                        for key, value in APdict.items():
                            ap_of_each_conf = 0
                            for thres in range(interval,100,5):
                                ap_of_each_conf = ap_of_each_conf + (value[str(thres)])
                            calculate_mAP_dict[key] = ap_of_each_conf/len(APdict)

                            sum_of_all_classes = 0
                            for key, value in calculate_mAP_dict.items():
                                sum_of_all_classes = sum_of_all_classes + value 
                            mAP[interval] = sum_of_all_classes/len(calculate_mAP_dict)

                        Average_inference_time = df_merge.loc[:, 'pred detection time'].mean()
                        print("mAP@" + str(interval) , mAP[interval])
                        print("Average Inference time: ", Average_inference_time)
                    print("\n")