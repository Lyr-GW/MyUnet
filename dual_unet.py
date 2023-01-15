import torch.nn as nn
import torch

class DoubleConv(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, input):
        return self.double_conv(input)

class Dual_UNet(nn.Module):
    def __init__(self, in_ch, vsl_ch, out_ch):
        self.in_ch = in_ch 
        self.vsl_ch = vsl_ch 
        self.out_ch = out_ch 

        super(Dual_UNet, self).__init__()
        #encoder lesion
        self.conv1 = DoubleConv(in_ch, 64)   #通道数
        self.pool1 = nn.MaxPool2d(2)        #每次把图像缩小一半
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(512, 1024)

        #encoder vessel 
        self.conv6 = DoubleConv(vsl_ch, 64)   #通道数
        self.pool6 = nn.MaxPool2d(2)        #每次把图像缩小一半
        self.conv7 = DoubleConv(64, 128)
        self.pool7 = nn.MaxPool2d(2)
        self.conv8 = DoubleConv(128, 256)
        self.pool8 = nn.MaxPool2d(2)
        self.conv9 = DoubleConv(256, 512)
        self.pool9 = nn.MaxPool2d(2)
        self.conv10 = DoubleConv(512, 1024)

        #decoder
        self.up11 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv11 = DoubleConv(1024, 512)
        self.up12 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv12 = DoubleConv(512, 256)
        self.up13 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv13 = DoubleConv(256, 128)
        self.up14 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv14 = DoubleConv(128, 64)

        self.up15 = nn.ConvTranspose2d(64, out_ch, 1)

    def forward(self, x, y):
        #lesion
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)

        #vessel
        c6 = self.conv6(y)
        p6 = self.pool6(c6)
        c7 = self.conv7(p6)
        p7 = self.pool7(c7)
        c8 = self.conv8(p7)
        p8 = self.pool8(c8)
        c9 = self.conv9(p8)
        p9 = self.pool9(c9)
        c10 = self.conv10(p9)

        #简单加和 +
        sum1 = c1 + c6
        sum2 = c2 + c7
        sum3 = c3 + c8
        sum4 = c4 + c9
        sum5 = c5 + c10

        up_11 = self.up11(sum5)
        merge11 = torch.cat([up_11, sum4], dim=1)    #按维数1（列）拼接，列增加
        c11 = self.conv11(merge11)
        up_12 = self.up12(c11)
        merge12 = torch.cat([up_12, sum3], dim=1)
        c12 = self.conv12(merge12)
        up_13 = self.up13(c12)
        merge13 = torch.cat([up_13, sum2], dim=1)
        c13 = self.conv13(merge13)
        up_14 = self.up14(c13)
        merge14 = torch.cat([up_14, sum1], dim=1)
        c14 = self.conv14(merge14)
        up_15 = self.up15(c14)

        out = nn.Sigmoid()(up_15)
        return out
