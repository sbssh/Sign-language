

import os
import pickle
import torch
from PIL import Image

from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms


class Data(Dataset):
    def __init__(self,root):
        self.imgs=[]
        self.labels=[]
        self.len_classes=len(os.listdir(root))#种类数目

        for file in os.listdir(root):
            for name in os.listdir(root+"//"+file):
                #加载pkl文件
                path=root+"//"+file+"//"+str(name)
               # print(path)
                data_pkl=open(path,'rb')
                data=pickle.load(data_pkl)#pickle.load对同一个文件只能load一次，多次load会报错

                #获得图片
                img_np=data[0]
                img_np=np.squeeze(img_np)#将numpy数组降维成RGB图片常见的维度，便于之后数据的使用
                img_PIL=Image.fromarray(img_np)
                # resize=transforms.Resize((227,227))#调整图片大小
                # img_PIL=resize(img_PIL)
                totensor=transforms.ToTensor()
                tensor_img=totensor(img_PIL)  #转换为tensor类型
                # print(tensor_img.shape)

                #获得标签
                label=data[1]
                label=np.array(label,dtype=float)
                tensor_label=torch.LongTensor(label)  #转换为tensor中的Long类型

                #将获得的图片与标签加入各自的列表
                self.imgs.append(tensor_img)
                self.labels.append(tensor_label)

    def __getitem__(self, item):#获得对应图片与标签
        return self.imgs[item],self.labels[item]

    def __len__(self):#获得总的图片数量
        return len(self.imgs)

    def num_classes(self):#获得种类数量
        return self.len_classes

if __name__=='__main__':

   #tensorbord日志
   from torch.utils.tensorboard import SummaryWriter
   writer=SummaryWriter("logs")

   #打开数据集
   data=Data('F://Sign lauguage recognition//test//data_test')
   img,label=data[0]
   print(img.shape)
   #日志记录图片
   writer.add_image("img",img)
   writer.close()
