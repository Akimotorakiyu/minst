# -*- coding: utf-8 -*-
# @Author  : Dengxun
# @Time    : 2023/3/23 19:43
# @Function:Model
import torch
import torch.nn as nn
class AlexModel(nn.Module):#定义AlexNet的模型
    def __init__(self):#width_mult为输入图像的尺寸缩放比默认为1。
        super(AlexModel,self).__init__()#继承父类的初始化方法
        self.layer1=nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=2,kernel_size=3,stride=3),#(224 - 3 + 2*2) / 3 + 1 = 55包含96个大小为11×11的滤波器（其实是11×11×3），卷积步长为4，因此第一层输出大小为55×55×96，padding=2；
        )

        #定义全连接层,用于1-10分类得分
        self.fc=nn.Sequential(#nn.Dropout(0.5),
                              nn.Linear(10952, 10))

    def forward(self, x):#前向传播

        x = self.layer1(x)
        x = torch.flatten(x,start_dim=1)#torch.flatten是功能函数不是类，展平为一元
        x = self.fc(x)
        return x


