
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import AlexNet

from Data import Data
from Model import Model

#设置运算平台
device = "cuda" if torch.cuda.is_available() else "cpu"

#加载数据集
dataset=Data('F://Sign lauguage recognition//test//data')
data_loader=DataLoader(dataset=dataset,batch_size=64,shuffle=True,num_workers=0,drop_last=False)

#加载模型
model=Model(num_classes=dataset.num_classes()).to(device=device)

#加载已有的模型参数
last_epoch=1
model.load_state_dict(torch.load("./model_state_dict/第{}轮训练模型参数".format(last_epoch)))

#设置损失函数
loss_fn=nn.CrossEntropyLoss()

#设置优化器
learning_rate=0.01
optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)




if __name__=='__main__':

  #开始训练


  epoch=100#轮数
  for i in range(epoch):
      print(str(i+1)+"轮:\n")
      loss_sum=0.0#每轮交叉熵
      correct_num=0
      for batch,data in enumerate(data_loader):
        #获得图片与标签
         imgs,labels=data
         imgs,labels=imgs.to(device),labels.to(device)

         #获得输出
         outputs=model(imgs)


         #计算正确个数
         correct_num=correct_num+(outputs.argmax(1)==labels).sum()

         #计算交叉熵
         loss=loss_fn(outputs,labels)
         # print(labels)
         # print(outputs)
         # print("loss:{}".format(loss))
         loss_sum=loss_sum+loss

         #优化
         optimizer.zero_grad()   #优化器参数清零
         loss.backward()     #损失函数回传
         optimizer.step()   #优化器对模型进行优化

         #打印数据
         # if batch %20==0:
         #    print("loss:"+str(loss_sum)+"\n")
      #打印每轮正确率与交叉熵
      accuracy=correct_num/len(dataset)
      print("accuracy:{}".format(accuracy))
      print("loss_sum:"+str(loss_sum)+"\n\n")

      if i==epoch-1:
         #保存最后一轮训练的模型参数
        torch.save(model.state_dict(),"./model_state_dict/第{}轮训练模型参数".format(i+1))


