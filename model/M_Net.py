"""
    Matting network : M-Net

Author: Zhengwei Li
Date  : 2018/12/24
"""

import torch
import torch.nn as nn


class M_net(nn.Module):
    '''
        encoder + decoder
    '''

    def __init__(self, classes=2):

        super(M_net, self).__init__()
        # -----------------------------------------------------------------
        # encoder  
        # ---------------------
        self.en_conv_bn_relu_1 = nn.Sequential(nn.Conv2d(6, 64, 3, 1, 1, bias=True),
                                       nn.BatchNorm2d(64),
                                       nn.ReLU())
        self.max_pooling_1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True) 

        # 1/2
        self.en_conv_bn_relu_2 = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1, bias=True),
                                       nn.BatchNorm2d(128),
                                       nn.ReLU())
        self.max_pooling_2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)  

        # 1/4
        self.en_conv_bn_relu_3 = nn.Sequential(nn.Conv2d(128, 256, 3, 1, 1, bias=True),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU())
        self.max_pooling_3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        # 1/8
        self.en_conv_bn_relu_4 = nn.Sequential(nn.Conv2d(256, 512, 3, 1, 1, bias=True),
                                       nn.BatchNorm2d(512),
                                       nn.ReLU())
        self.max_pooling_4 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)  

        # 1/16
        self.en_conv_bn_relu_5 = nn.Sequential(nn.Conv2d(512, 512, 3, 1, 1, bias=True),
                                       nn.BatchNorm2d(512),
                                       nn.ReLU())
        # -----------------------------------------------------------------
        # decoder  
        # ---------------------
        # 1/16
        self.de_conv_bn_relu_5 = nn.Sequential(nn.Conv2d(512, 512, 5, 1, 2, bias=True),
                                       nn.BatchNorm2d(512),
                                       nn.ReLU())
        # 1/8
        self.deconv_4 = nn.ConvTranspose2d(512, 512, 5, 2, 2, 1, bias=True)
        self.up_pool_4 = nn.MaxUnpool2d(2, stride=2)
        self.de_conv_bn_relu_4 = nn.Sequential(nn.Conv2d(512, 256, 3, 1, 1, bias=True),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU())

        # 1/4
        self.deconv_3 = nn.ConvTranspose2d(256, 256, 5, 2, 2, 1, bias=True)
        self.up_pool_3 = nn.MaxUnpool2d(2, stride=2)
        self.de_conv_bn_relu_3 = nn.Sequential(nn.Conv2d(256, 128, 3, 1, 1, bias=True),
                                       nn.BatchNorm2d(128),
                                       nn.ReLU())

        # 1/2
        self.deconv_2 = nn.ConvTranspose2d(128, 128, 5, 2, 2, 1, bias=True)
        self.up_pool_2 = nn.MaxUnpool2d(2, stride=2)
        self.de_conv_bn_relu_2 = nn.Sequential(nn.Conv2d(128, 64, 3, 1, 1, bias=True),
                                       nn.BatchNorm2d(64),
                                       nn.ReLU())

        # 1/1
        self.deconv_1 = nn.ConvTranspose2d(64, 64, 5, 2, 2, 1, bias=True)
        self.up_pool_1 = nn.MaxUnpool2d(2, stride=2)
        self.de_conv_bn_relu_1 = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1, bias=True),
                                       nn.BatchNorm2d(32),
                                       nn.ReLU())
        self.conv_1 = nn.Sequential(nn.Conv2d(32, 32, 5, 1, 2, bias=True),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU())

        self.conv = nn.Conv2d(32, 1, 5, 1, 2, bias=True)
        #self.conv1 = nn.Conv2d(33, 1, 5, 1, 2, bias=True)


    def forward(self, input):

        unsure = input[:, -2:-1, :, :]
        
        # ----------------
        # encoder
        # --------
        x = self.en_conv_bn_relu_1(input)
        x, idx_1 = self.max_pooling_1(x)

        x = self.en_conv_bn_relu_2(x)
        x, idx_2 = self.max_pooling_2(x)

        x = self.en_conv_bn_relu_3(x)
        x, idx_3 = self.max_pooling_3(x)

        x = self.en_conv_bn_relu_4(x)
        x, idx_4 = self.max_pooling_4(x)
        x = self.en_conv_bn_relu_5(x)
        # ----------------
        # decoder
        # --------
        x = self.de_conv_bn_relu_5(x)
        x = self.up_pool_4(x, idx_4)
        x = self.de_conv_bn_relu_4(x)
        #x = self.deconv_1(x)
        
        x = self.up_pool_3(x, idx_3)
        x = self.de_conv_bn_relu_3(x)
        #x = self.deconv_2(x)

        x = self.up_pool_2(x, idx_2)
        x = self.de_conv_bn_relu_2(x)
        #x = self.deconv_3(x)

        x = self.up_pool_1(x, idx_1)
        x = self.de_conv_bn_relu_1(x)
        x = self.conv_1(x)
        #x = self.deconv_4(x)
        
        #x = torch.cat((x, unsure), 1)

        # raw alpha pred
        out = self.conv(x)

        return out 





