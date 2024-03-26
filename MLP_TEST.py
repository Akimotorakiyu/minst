import torch
import torch.nn as nn
class AlexModel(nn.Module):#定义AlexNet的模型
    def __init__(self):#width_mult为输入图像的尺寸缩放比默认为1。
        super(AlexModel,self).__init__()#继承父类的初始化方法
        self.layer0=nn.Sequential(
            nn.Linear(50176,10),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.layer1=nn.Sequential(
            nn.Linear(10,10),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
       
        #定义全连接层,用于1-10分类得分
        self.fc=nn.Sequential(#nn.Dropout(0.5),
                              nn.Linear(10,10),
        )

    def forward(self, x):#前向传播
        x= torch.flatten(x,start_dim=1) #torch.flatten是功能函数不是类，展平为一元
        x = self.layer0(x)
        for i in range(2):
            x = self.layer1(x)
        x = self.fc(x)
        return x
