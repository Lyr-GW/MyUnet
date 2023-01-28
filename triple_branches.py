import torch.nn as nn
import torch

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvBlock, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, input):
        out = self.layer1(input)
        return out

class ConvBlock_Last(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvBlock, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, input):
        out = self.layer1(input)
        return out

class Triple_Branches(nn.Module):
    def __init__(self):
        super(Triple_Branches, self).__init__()

        self.up_conv1 = ConvBlock(3, 64)
        self.down_conv1 = ConvBlock(3, 64)
        self.mid_conv1 = ConvBlock(128, 64)
        #
        self.up_conv2 = ConvBlock(64, 128)
        self.down_conv2 = ConvBlock(64, 128)
        self.mid_conv2 = ConvBlock(320, 128)

        self.up_conv3 = ConvBlock(128, 256)
        self.down_conv3 = ConvBlock(128, 256)
        self.mid_conv3 = ConvBlock(640, 256)

        #1-3层 下采通道数
        # self.up_conv4 = ConvBlock(256,2)
        # self.down_conv4 = ConvBlock(256,1)
        self.up_conv4 = ConvBlock(256,64)
        self.down_conv4 = ConvBlock(256,64)

        self.up_conv5 = ConvBlock_Last(64,2)
        self.down_conv5 = ConvBlock_Last(64,1)

    def forward(self, x):
        up_c1 = self.up_conv1(x)
        down_c1 = self.down_conv1(x)
        #中介融合通道
        mid_cat1 = torch.cat([up_c1,down_c1], dim=1)
        m_c1 = self.mid_conv1(mid_cat1)
        up_cat1 = up_c1+m_c1
        down_cat1 = down_c1+m_c1

        up_c2 = self.up_conv2(up_cat1) 
        down_c2 = self.down_conv2(down_cat1)
        #中介融合通道
        mid_cat2 = torch.cat([up_c2,down_c2,m_c1], dim=1)
        m_c2 = self.mid_conv2(mid_cat2)
        up_cat2 = up_c2+m_c2
        down_cat2 = down_c2+m_c2
        
        up_c3 = self.up_conv3(up_cat2) 
        down_c3 = self.down_conv3(down_cat2)
        #中介融合通道
        mid_cat3 = torch.cat([up_c3,down_c3,m_c2], dim=1)
        m_c3 = self.mid_conv3(mid_cat3)
        up_cat3 = up_c3+m_c3
        down_cat3 = down_c3+m_c3

        up_c4 = self.up_conv4(up_cat3)
        down_c4 = self.down_conv4(down_cat3)

        up_c5 = self.up_conv5(up_c4)
        down_c5 = self.down_conv5(down_c4)

        vsl_seg = nn.Sigmoid()(up_c5)
        lesion_seg = nn.Sigmoid()(down_c5)

        return vsl_seg, lesion_seg 
