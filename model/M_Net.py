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
        # 1/2
        self.en_conv_bn_relu_1 = nn.Sequential(nn.Conv2d(6, 16, 3, 1, 1, bias=False),
                                       nn.BatchNorm2d(16),
                                       nn.ReLU())
        self.max_pooling_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, return_indices=True) 

        # 1/4
        self.en_conv_bn_relu_2 = nn.Sequential(nn.Conv2d(16, 32, 3, 1, 1, bias=False),
                                       nn.BatchNorm2d(32),
                                       nn.ReLU())
        self.max_pooling_2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, return_indices=True)  

        # 1/8
        self.en_conv_bn_relu_3 = nn.Sequential(nn.Conv2d(32, 64, 3, 1, 1, bias=False),
                                       nn.BatchNorm2d(64),
                                       nn.ReLU())
        self.max_pooling_3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, return_indices=True)

        # 1/16
        self.en_conv_bn_relu_4 = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1, bias=False),
                                       nn.BatchNorm2d(128),
                                       nn.ReLU())
        self.max_pooling_4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, return_indices=True)  

        '''
        self.en_conv_bn_relu_5 = nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1, bias=False),
                                       nn.BatchNorm2d(128),
                                       nn.ReLU())
        '''
        # -----------------------------------------------------------------
        # decoder  
        # ---------------------
        # 1/8
        self.de_conv_bn_relu_1 = nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1, bias=False),
                                       nn.BatchNorm2d(128),
                                       nn.ReLU())
        self.deconv_1 = nn.ConvTranspose2d(128, 128, 5, 2, 2, 1, bias=False)
        self.up_pool_1 = nn.MaxUnpool2d(2, stride=2)

        # 1/4
        self.de_conv_bn_relu_2 = nn.Sequential(nn.Conv2d(128, 64, 3, 1, 1, bias=False),
                                       nn.BatchNorm2d(64),
                                       nn.ReLU())
        self.deconv_2 = nn.ConvTranspose2d(64, 64, 5, 2, 2, 1, bias=False)
        self.up_pool_2 = nn.MaxUnpool2d(2, stride=2)

        # 1/2
        self.de_conv_bn_relu_3 = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1, bias=False),
                                       nn.BatchNorm2d(32),
                                       nn.ReLU())
        self.deconv_3 = nn.ConvTranspose2d(32, 32, 5, 2, 2, 1, bias=False)
        self.up_pool_3 = nn.MaxUnpool2d(2, stride=2)

        # 1/1
        self.de_conv_bn_relu_4 = nn.Sequential(nn.Conv2d(32, 16, 3, 1, 1, bias=False),
                                       nn.BatchNorm2d(16),
                                       nn.ReLU())
        self.deconv_4 = nn.ConvTranspose2d(16, 16, 5, 2, 2, 1, bias=False)
        self.up_pool_4 = nn.MaxUnpool2d(2, stride=2)


        self.conv = nn.Conv2d(16, 1, 5, 1, 2, bias=False)
        self.conv1 = nn.Conv2d(17, 1, 5, 1, 2, bias=False)


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
        # ----------------
        # decoder
        # --------
        x = self.de_conv_bn_relu_1(x)
        sixteenth = x.max(dim=1, keepdim=True)[0] # (b, 1, h/16, w/16)
        print('16th :', sixteenth.shape)
        x = self.up_pool_1(x, idx_4)
        #x = self.deconv_1(x)
        
        x = self.de_conv_bn_relu_2(x)
        eighth = x.max(dim=1, keepdim=True)[0] # (b, 1, h/8, w/8)
        print('8th :', eighth.shape)
        x = self.up_pool_2(x, idx_3)
        #x = self.deconv_2(x)

        x = self.de_conv_bn_relu_3(x)
        forth = x.max(dim=1, keepdim=True)[0] # (b, 1, h/4, w/4)
        print('4th :', forth.shape)
        x = self.up_pool_3(x, idx_2)
        #x = self.deconv_3(x)

        x = self.de_conv_bn_relu_4(x)
        half = x.max(dim=1, keepdim=True)[0] # (b, 1, h/2, w/2)
        print('2nd :', half.shape)
        x = self.up_pool_4(x, idx_1)
        #x = self.deconv_4(x)
        
        #x = torch.cat((x, unsure), 1)

        # raw alpha pred
        out = self.conv(x)

        return out , [half, forth, eighth, sixteenth]





