import os 
import pickle

store_name = {}

path_image = '/home/crossing/Desktop/rbzh-model-benchmark/testing_imagesv2/test/images'
files_image = os.listdir(path_image)

path_txt = '/home/crossing/Desktop/rbzh-model-benchmark/testing_imagesv2/test/labels'
files_txt = os.listdir(path_txt)

for index, file in enumerate(files_image):
    # print(file)
    store_name[file] = (str(index) + '.jpg')
    file_name = file[:-4]
    # print(file_name)
    os.rename(os.path.join(path_image, str(str(file_name)+'.jpg')), os.path.join('/home/crossing/Desktop/rbzh-model-benchmark/testing_imagesv2/test_modified/images', ''.join([str(index), '.jpg'])))
    os.rename(os.path.join(path_txt, str(str(file_name)+'.txt')), os.path.join('/home/crossing/Desktop/rbzh-model-benchmark/testing_imagesv2/test_modified/labels', ''.join([str(index), '.txt'])))
    index=1

print(store_name)

with open("/home/crossing/Desktop/rbzh-model-benchmark/testing_imagesv2/"  +  "store_name" + '.pkl', 'wb') as fp:
    pickle.dump(store_name, fp)
    print('dictionary saved successfully to file')
# for index, file in enumerate(files):
#      os.rename(os.path.join(path, file),os.path.join(path,''.join([str(index),'.jpg'])))
#      index = index+1