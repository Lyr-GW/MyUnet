import torch.nn as nn
import torch
import torch.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvBlock, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )
    def forward(self, input):
        out = self.layer1(input)
        return out

class ConvBlock_Last(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvBlock_Last, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )
    def forward(self, input):
        out = self.layer1(input)
        return out

class Up(nn.Module):
    def __init__(self, in_channels) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])
 
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return x    

class Triple_Branches(nn.Module):
    def __init__(self):
        super(Triple_Branches, self).__init__()

        self.pool = nn.MaxPool2d(2)
        # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up = nn.ModuleList()
        self.up_m = nn.ModuleList()
        up_in_chs = [512, 256, 128, 64]
        up_m_in_chs = [512, 256, 128, 64]
        for i in range(4):
            print(up_in_chs[i])
            self.up.append(nn.ConvTranspose2d(up_in_chs[i], up_in_chs[i], kernel_size=2, stride=2))
            self.up_m.append(nn.ConvTranspose2d(up_m_in_chs[i], up_m_in_chs[i], kernel_size=2, stride=2))
            # self.up[i] = nn.ConvTranspose2d(up_in_chs[i], up_in_chs[i], kernel_size=2, stride=2)
            # self.up_m[i] = nn.ConvTranspose2d(up_m_in_chs[i], up_m_in_chs[i], kernel_size=2, stride=2)

        '''下采样'''
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

        self.up_conv4 = ConvBlock(256, 512)
        self.down_conv4 = ConvBlock(256, 512)
        self.mid_conv4 = ConvBlock(1280, 512)

        self.up_conv5 = ConvBlock(512, 512)
        self.down_conv5 = ConvBlock(512, 512)
        self.mid_conv5 = ConvBlock(1536, 512)

        # self.up_conv5 = ConvBlock(512, 1024)
        # self.down_conv5 = ConvBlock(512, 1024)
        # self.mid_conv5 = ConvBlock(2560, 1024)

        # self.up_conv6 = ConvBlock(1024, 2048)
        # self.down_conv6 = ConvBlock(1024, 2048)
        # self.mid_conv6 = ConvBlock(6120, 2048)
        
        # '''上采样'''
        # self.up_conv7 = ConvBlock(2048, 1024)
        # self.down_conv7 = ConvBlock(2048, 1024)
        # self.mid_conv7 = ConvBlock(4096, 1024)

        # self.up_conv8 = ConvBlock(1024, 512)
        # self.down_conv8 = ConvBlock(1024, 512)
        # self.mid_conv8 = ConvBlock(2048, 512)

        self.up_conv9 = ConvBlock(1024, 256)
        self.down_conv9 = ConvBlock(1024, 256)
        self.mid_conv9 = ConvBlock(1536, 256)

        self.up_conv10 = ConvBlock(512, 128)
        self.down_conv10 = ConvBlock(512, 128)
        self.mid_conv10 = ConvBlock(768, 128)

        self.up_conv11 = ConvBlock(256, 64)
        self.down_conv11 = ConvBlock(256, 64)
        self.mid_conv11 = ConvBlock(384, 64)
        
        self.up_conv12 = ConvBlock_Last(128, 1)
        self.down_conv12 = ConvBlock_Last(128, 1)
        self.mid_conv12 = ConvBlock_Last(130, 1)
    

    def forward(self, x):
        up_c1 = self.up_conv1(x)
        down_c1 = self.down_conv1(x)
        #中介融合通道
        mid_cat1 = torch.cat([up_c1,down_c1], dim=1)
        m_c1 = self.mid_conv1(mid_cat1)
        up_cat1 = up_c1+m_c1
        down_cat1 = down_c1+m_c1

        up_pool1 = self.pool(up_cat1)
        down_pool1 = self.pool(down_cat1)
        mid_pool1 = self.pool(m_c1)
        up_c2 = self.up_conv2(up_pool1) 
        down_c2 = self.down_conv2(down_pool1)
        #中间融合通道
        mid_cat2 = torch.cat([up_c2,down_c2,mid_pool1], dim=1)
        m_c2 = self.mid_conv2(mid_cat2)
        up_cat2 = up_c2+m_c2
        down_cat2 = down_c2+m_c2
        
        up_pool2 = self.pool(up_cat2)
        down_pool2 = self.pool(down_cat2)
        mid_pool2 = self.pool(m_c2)
        up_c3 = self.up_conv3(up_pool2) 
        down_c3 = self.down_conv3(down_pool2)
        #中间融合通道
        mid_cat3 = torch.cat([up_c3,down_c3,mid_pool2], dim=1)
        m_c3 = self.mid_conv3(mid_cat3)
        up_cat3 = up_c3+m_c3
        down_cat3 = down_c3+m_c3

        up_pool3 = self.pool(up_cat3)
        down_pool3 = self.pool(down_cat3)
        mid_pool3 = self.pool(m_c3)
        up_c4 = self.up_conv4(up_pool3)
        down_c4 = self.down_conv4(down_pool3)
        #中间融合通道
        mid_cat4 = torch.cat([up_c4, down_c4, mid_pool3], dim=1)
        m_c4 = self.mid_conv4(mid_cat4)
        up_cat4 = up_c4+m_c4
        down_cat4 = down_c4+m_c4

        up_pool4 = self.pool(up_cat4)
        down_pool4 = self.pool(down_cat4)
        mid_pool4 = self.pool(m_c4)

        up_c5 = self.up_conv5(up_pool4)
        down_c5 = self.down_conv5(down_pool4)
        mid_cat5 = torch.cat([up_c5, down_c5, mid_pool4], dim=1)
        m_c5 = self.mid_conv5(mid_cat5)
        up_cat5 = up_c5+m_c5
        down_cat5 = down_c5+m_c5


        # up_c5 = self.up_conv5(self.pool(up_cat4)))torch.cat([, down_c3], ddim=1), mce3m_c3torch.cat([, ucp_c2], ddim=1)torch.cat([, down_ c2]),d ddim=1) , mc2m_c2
        # down_c5 = self.down_conv5(self.pool(down_cat4))
        # #中间融合通道
        # mid_cat5 = torch.cat([up_c5, down_c5, self.pool(m_c4)], dim=1)
        # m_c5 = self.mid_conv5(mid_cat5)
        # up_cat5 = up_c5+m_c5
        # down_cat5 = down_c5+m_c5

        # up_c6 = self.up_conv6(self.pool(up_cat5))
        # down_c6 = self.down_conv6(self.pool(down_cat5))
        # #中间融合通道
        # mid_cat6 = torch.cat([up_c6, down_c6, self.pool(m_c5)], dim=1)
        # m_c6 = self.mid_conv6(mid_cat6)
        # up_cat6 = up_c6+m_c6
        # down_cat6 = down_c6+m_c6)torch.cat([, down_c3], ddim=1), mce3m_c3torch.cat([, ucp_c2], ddim=1)torch.cat([, down_ c2￼￼]),d ddim=1) , mc2m_c2

        # '''上采样'''
        # up_c7 = self.up_conv7(self.up(up_cat6))
        # down_c7 = self.down_conv7(self.up(down_cat6))
        # #中间融合通道
        # mid_cat7 = torch.cat([up_c7, down_c7, self.up(m_c6)], dim=1)
        # m_c7 = self.mid_conv7(mid_cat7)
        # up_cat7 = up_c7+m_c7
        # down_cat7 = down_c7+m_c7

        # up_c8 = self.up_conv8(self.up(up_cat7))
        # down_c8 = self.down_conv8(self.up(down_cat7))
        # #中间融合通道
        # mid_cat8 = torch.cat([up_c8, down_c8, self.up(m_c7)], dim=1)
        # m_c8 = self.mid_conv8(mid_cat8)
        # up_cat8 = up_c8+m_c8
        # down_cat8 = down_c8+m_c8

        up_c9 = self.up_conv9(torch.cat([self.up[0](up_cat5), up_cat4], dim=1))    # 跳层连接
        down_c9 = self.down_conv9(torch.cat([self.up[0](down_cat5), down_cat4], dim=1))
        #中间融合通道
        mid_cat9 = torch.cat([up_c9, down_c9, self.up_m[0](m_c5), m_c4], dim=1)
        m_c9 = self.mid_conv9(mid_cat9)
        up_cat9 = up_c9+m_c9
        down_cat9 = down_c9+m_c9

        up_c10 = self.up_conv10(torch.cat([self.up[1](up_cat9), up_c3], dim=1))    # 跳层连接
        down_c10 = self.down_conv10(torch.cat([self.up[1](down_cat9), down_c3], dim=1))
        # #中间融合通道
        mid_cat10 = torch.cat([up_c10, down_c10, self.up_m[1](m_c9), m_c3], dim=1)
        m_c10 = self.mid_conv10(mid_cat10)
        up_cat10 = up_c10+m_c10
        down_cat10 = down_c10+m_c10

        up_c11 = self.up_conv11(torch.cat([self.up[2](up_cat10), up_c2], dim=1))    # 跳层连接
        down_c11 = self.down_conv11(torch.cat([self.up[2](down_cat10), down_c2], dim=1))
        # #中间融合通道
        mid_cat11 = torch.cat([up_c11, down_c11, self.up_m[2](m_c10), m_c2], dim=1)
        m_c11 = self.mid_conv11(mid_cat11)
        up_cat11 = up_c11+m_c11
        down_cat11 = down_c11+m_c11

        # print(self.up(up_cat9).shape)
        # print(up_c1.shape)
        up_c12 = self.up_conv12(torch.cat([self.up[3](up_cat11), up_c1], dim=1))    # 跳层连接
        down_c12 = self.down_conv12(torch.cat([self.up[3](down_cat11), down_c1], dim=1))
        # #中间融合通道
        mid_cat12 = torch.cat([up_c12, down_c12, self.up_m[3](m_c11), m_c1], dim=1)
        m_c12 = self.mid_conv12(mid_cat12)
        up_cat12 = up_c12+m_c12
        down_cat12 = down_c12+m_c12

        vsl_seg = nn.Sigmoid()(up_c12)
        lesion_seg = nn.Sigmoid()(down_c12)

        return vsl_seg, lesion_seg 
