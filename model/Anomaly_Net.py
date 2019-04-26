import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.en_conv_bn_relu_1 = nn.Sequential(nn.Conv2d(4, 64, 2, 1, 1, bias=True),
                                       nn.BatchNorm2d(64),
                                       nn.ReLU())

        # 1/2
        self.en_conv_bn_relu_2 = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1, bias=True),
                                       nn.BatchNorm2d(128),
                                       nn.ReLU()) 

        # 1/4
        self.en_conv_bn_relu_3 = nn.Sequential(nn.Conv2d(128, 256, 3, 1, 1, bias=True),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU())
        self.max_pooling_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 1/8
        self.en_conv_bn_relu_4 = nn.Sequential(nn.Conv2d(256, 512, 3, 1, 1, bias=True),
                                       nn.BatchNorm2d(512),
                                       nn.ReLU())
        self.max_pooling_4 = nn.MaxPool2d(kernel_size=2, stride=2)  

        # 1/16
        self.en_conv_bn_relu_5 = nn.Sequential(nn.Conv2d(512, 512, 3, 1, 1, bias=True),
                                       nn.BatchNorm2d(512),
                                       nn.ReLU())

    def forward(self, x):
        x = self.en_conv_bn_relu_1(x)

        x = self.en_conv_bn_relu_2(x)
        x = self.max_pooling_2(x)

        x = self.en_conv_bn_relu_3(x)
        x = self.max_pooling_3(x)

        x = self.en_conv_bn_relu_4(x)
        x = self.max_pooling_4(x)

        code = self.en_conv_bn_relu_5(x)
        
        return code

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        # 1/16
        self.de_conv_bn_relu_5 = nn.Sequential(nn.Conv2d(512, 512, 5, 1, 2, bias=True),
                                       nn.BatchNorm2d(512),
                                       nn.ReLU())
        # 1/8
        self.deconv_4 = nn.ConvTranspose2d(512, 512, 5, 2, 2, 1, bias=True)
        self.de_conv_bn_relu_4 = nn.Sequential(nn.Conv2d(512, 256, 3, 1, 1, bias=True),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU())

        # 1/4
        self.deconv_3 = nn.ConvTranspose2d(256, 256, 5, 2, 2, 1, bias=True)
        self.de_conv_bn_relu_3 = nn.Sequential(nn.Conv2d(256, 128, 3, 1, 1, bias=True),
                                       nn.BatchNorm2d(128),
                                       nn.ReLU())

        # 1/2
        self.deconv_2 = nn.ConvTranspose2d(128, 128, 5, 2, 2, 1, bias=True)
        self.de_conv_bn_relu_2 = nn.Sequential(nn.Conv2d(128, 64, 3, 1, 1, bias=True),
                                       nn.BatchNorm2d(64),
                                       nn.ReLU())

        # 1/1
        self.deconv_1 = nn.ConvTranspose2d(64, 64, 5, 2, 2, 1, bias=True)
        self.de_conv_bn_relu_1 = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1, bias=True),
                                       nn.BatchNorm2d(32),
                                       nn.ReLU())
        self.conv = nn.Conv2d(32, 3, 5, 1, 2, bias=True)

    def forward(self, x):
        x = self.de_conv_bn_relu_5(x)

        x = self.deconv_4(x)
        x = self.de_conv_bn_relu_4(x)

        x = self.deconv_3(x)
        x = self.de_conv_bn_relu_3(x)

        x = self.deconv_2(x)
        x = self.de_conv_bn_relu_2(x)

        x = self.deconv_1(x)
        x = self.de_conv_bn_relu_1(x)

        out = self.conv(x)

        return out

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.depthwise_1 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 2, 1, groups=512, bias=False)
            nn.BatchNorm2d(512),
            nn.ReLU6(inplace=True)
        )
        self.pointwise_1 = nn.Sequential(
            nn.Conv2d(512, 256, 1, 1, 0, bias=False),
            nn.BatchNorm2d(256)
        )

        self.depthwise_2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 2, 1, groups=256, bias=False)
            nn.BatchNorm2d(256),
            nn.ReLU6(inplace=True)
        )
        self.pointwise_2 = nn.Sequential(
            nn.Conv2d(256, 128, 1, 1, 0, bias=False),
            nn.BatchNorm2d(128)
        )

        self.squeeze_1 = nn.Sequential(
            nn.Conv2d(128, 64, 1, 1, 0, bias=False)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.squeeze_2 = nn.Sequential(
            nn.Conv2d(64, 32, 1, 1, 0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.squeeze_3 = nn.Sequential(
            nn.Conv2d(32, 16, 1, 1, 0, bias=False)
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        self.classify = nn.Sequential(
            nn.Linear(256, 1, bias=False),
            nn.Sigmoid()
        )


    def forward(self, x):
        # (512, 16, 16) -> (256, 8, 8)
        x = self.depthwise_1(x)
        x = self.pointwise_1(x)

        # (256, 8, 8) -> (128, 4, 4)
        x = self.depthwise_2(x)
        x = self.pointwise_2(x)

        # (128, 4, 4) -> (64, 4, 4)
        x = self.squeeze_1(x)

        # (64, 4, 4) -> (32, 4, 4)
        x = self.squeeze_2(x)

        # (32, 4, 4) -> (16, 4, 4)
        x = self.squeeze_3(x)

        # flatten
        x = x.view(-1)

        probs = self.classify(x)

        return probs

class AnomalyNet(nn.Module):
    def __init__(self):
        super(AnomalyNet, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.classifier = Discriminator()

    def forward(self, x):
        code = self.encoder(x)
        replica = self.decoder(code)
        probs = self.classifier(code)

        return replica, probs