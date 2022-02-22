'''
check the exported asl pickle data. 
'''
import os, glob
import numpy as np
import cv2 
import pickle

Dataset_1_kaggleMNIST24_root = r"\\105.1.1.1\Hand\SignLanguage\Dataset_1_kaggleMNIST24\exported_img2alphabet"


CHECKONE = Dataset_1_kaggleMNIST24_root

if __name__ == '__main__':

    pkl_filenames = glob.glob(os.path.join(CHECKONE, "*.pkl"))
    for pkl_filename in pkl_filenames:
        data_Dic = pickle.load(open(pkl_filename, 'rb'), encoding='latin1')

        img_Arr = cv2.resize(data_Dic['image'], dsize=(256,256))
        print("check image dtype: ", img_Arr.dtype)
        print("label :", data_Dic['cls'])
        cv2.imshow("img_Arr", img_Arr)
        cv2.waitKey()
