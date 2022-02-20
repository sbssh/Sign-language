
import torch
from torch import nn
from torch.utils.data import DataLoader
from Data import Data
from Model import Model
from torch.utils.tensorboard import SummaryWriter

#加载数据集
dataset=Data('F://Sign lauguage recognition//test//data_test')
data_loader=DataLoader(dataset=dataset,batch_size=2,shuffle=True,num_workers=0,drop_last=False)

#加载模型
model=Model(num_classes=dataset.num_classes())

#加载已有的模型参数
last_train_epoch=1
model.load_state_dict(torch.load("./model_state_dict/第{}轮训练模型参数".format(last_train_epoch)))

#用交叉熵作为损失函数
loss_fn=nn.CrossEntropyLoss()

#加载tensorboard日志
writer=SummaryWriter("logs")

if __name__=='__main__':
  #开始测试
   epoch=2
   for i in range(epoch):
      correct_num=0
      sum_loss=0
      step=0
      for data in data_loader:
         with torch.no_grad():
          #获得图片与标签
           imgs,labels=data

          #获得模型输出
           outputs=model(imgs)
           # print(outputs)

          #计算loss与正确数correct_num
           loss=loss_fn(outputs,labels)
           sum_loss=sum_loss+loss
           correct_num=correct_num+(outputs.argmax(1)==labels).sum()
           print(correct_num)

          #日志记录每步的loss
           step=step+1
           writer.add_scalar("loss_everystep",loss,i+step)

      #日志记录每轮的正确率
      accuracy=correct_num/len(dataset)
      writer.add_scalar("accuracy_epoch",accuracy,i)


      #打印该轮正确率与该轮loss总和
      print("该轮正确率为：{}".format(accuracy))
      print("该轮测试总的loss和为：{}".format(sum_loss))

   #关闭日志
   writer.close()
