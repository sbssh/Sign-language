'''
Demo to extract needed data from kaggleMNIST24. 

Save-to-pkl Item: 
- image. numpy array. dtype = uint8, size = same as original; 
- cls. char, classification type, one of 'A~Z'.
'''

import os 
import numpy as np
import cv2 
import csv
import pickle

if __name__ == '__main__':
    
    save_data_root = r"\\105.1.1.1\Hand\SignLanguage\Dataset_1_kaggleMNIST24\exported_img2alphabet"
    csv_filename = r"\\105.1.1.1\Hand\SignLanguage\Dataset_1_kaggleMNIST24\downloads\sign_mnist_test.csv"
    csv_lines = csv.reader(open(csv_filename,'r'))
    for index, line_dataframe in enumerate(csv_lines):
        if index ==0 : continue # for table title
        # ['label', 'pixel1', 'pixel2', 'pixel3', 'pixel4', 'pixel5', 'pixel6', 'pixel7', 'pixel8', 'pixel9', 'pixel10', 'pixel11', 'pixel12', 'pixel13', ...]
        img_Arr = np.array(list(map(int, line_dataframe[1:]))).reshape(28, -1).astype(np.uint8)
        label = chr(65 + int(line_dataframe[0]))
        assert label != "J" and label != "Z" and int(line_dataframe[0]) < 26, "invalid alphabet : " + label
        # img_Arr = cv2.resize(img_Arr, dsize=None, fx = 10, fy = 10)
        # cv2.imshow("data_in", img_Arr)
        # cv2.waitKey(1) # 27400

        save_Dic = {"image": img_Arr, "cls": label}
        save_name = "kaggleMNIST24_test%05d.pkl"%(index)
        pickle.dump(save_Dic, open(os.path.join(save_data_root, save_name), 'wb'), -1)
        if index % 100 == 0:

            print("check data num: ", index)