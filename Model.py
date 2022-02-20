

import torch
from torch import nn
from torchvision.models import alexnet, AlexNet


class Model(AlexNet):


    def __init__(self, num_classes=1000, init_weights=False):
        super(Model, self).__init__()
        # 使用nn.Sequential()将网络打包成一个模块，精简代码
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256*5*5, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )

        if init_weights:
            self._initialize_weights()

        #定义前向传播的过程
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)  #正态分布初始化
                nn.init.constant_(m.bias, 0)    #初始化偏重为0



if __name__=='__main__':
    from Data import Data
    from torch.utils.data import DataLoader
    dataset=Data('F://Sign lauguage recognition//test//data_test')
    data_loader=DataLoader(dataset=dataset,batch_size=2,shuffle=True,num_workers=0,drop_last=False)
    model=Model(2)
    for data in data_loader:
         imgs,labels=data
         # print(imgs)
         outputs=model(imgs)
         print(outputs)
         # print(imgs.shape)
         # outputs=model(imgs)
