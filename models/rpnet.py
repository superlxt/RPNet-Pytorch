# ERFNet full model definition for Pytorch
# Sept 2017
# Eduardo Romera
#######################

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class DownsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()

        self.conv = nn.Conv2d(ninput, noutput-ninput, (3, 3), stride=2, padding=1, bias=True)
        self.conv2 = nn.Conv2d(16, 64, (1, 1), stride=1, padding=0, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2, return_indices=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)
        

    def forward(self, input):
        c=input
        a=self.conv(input)
        b,max_indices=self.pool(input)
        #print(a.shape,b.shape,c.shape,max_indices.shape,"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        output1 = torch.cat([a, b], 1)
        if b.shape[1]==16:
            b_c = self.conv2(b)
        else:
            b_c=b
        output = self.bn(output1)
        return F.relu(output), max_indices, b, b_c, output1
    

class non_bottleneck_1d (nn.Module):
    def __init__(self, chann, dropprob, dilated):        
        super().__init__()

        self.conv3x1_1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1,0), bias=True)

        self.conv1x3_1 = nn.Conv2d(chann, chann, (1,3), stride=1, padding=(0,1), bias=True)

        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)

        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1*dilated,0), bias=True, dilation = (dilated,1))

        self.conv1x3_2 = nn.Conv2d(chann, chann, (1,3), stride=1, padding=(0,1*dilated), bias=True, dilation = (1, dilated))

        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)

        self.dropout = nn.Dropout2d(dropprob)
        

    def forward(self, input):

        output = self.conv3x1_1(input)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = F.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if (self.dropout.p != 0):
            output = self.dropout(output)
        
        return F.relu(output+input), output    #+input = identity (residual connection)


class RPNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.initial_block = DownsamplerBlock(3,16)

        self.l0d1=non_bottleneck_1d(16, 0.03, 1)
        self.down0_25=DownsamplerBlock(16,64)

        
        self.l1d1=non_bottleneck_1d(64, 0.03, 1)
        self.l1d2=non_bottleneck_1d(64, 0.03, 1)
        self.l1d3=non_bottleneck_1d(64, 0.03, 1)
        self.l1d4=non_bottleneck_1d(64, 0.03, 1)
        self.l1d5=non_bottleneck_1d(64, 0.03, 1)

        self.down0_125=DownsamplerBlock(64,128)

        self.l2d1=non_bottleneck_1d(128, 0.3, 2)
        self.l2d2=non_bottleneck_1d(128, 0.3, 4)
        self.l2d3=non_bottleneck_1d(128, 0.3, 8)
        self.l2d4=non_bottleneck_1d(128, 0.3, 16)

        self.l3d1=non_bottleneck_1d(128, 0.3, 2)
        self.l3d2=non_bottleneck_1d(128, 0.3, 4)
        self.l3d3=non_bottleneck_1d(128, 0.3, 8)
        self.l3d4=non_bottleneck_1d(128, 0.3, 16)
        #Only in encoder mode:
        self.conv2d1 = nn.Conv2d(
                    128,
                    num_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True)
        self.conv2d2 = nn.Conv2d(
                    192,
                    num_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True)
        self.conv2d3 = nn.Conv2d(
                    36,
                    num_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True)
        self.conv2d4 = nn.Conv2d(
                    16,
                    num_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False)
        self.conv2d5 = nn.Conv2d(
                    64,
                    num_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False)

        self.main_unpool1 = nn.MaxUnpool2d(kernel_size=2)
        self.main_unpool2 = nn.MaxUnpool2d(kernel_size=2)


    def forward(self, input, predict=False):
        output, max_indices0_0, d, d_d, dd  = self.initial_block(input)
        output,y = self.l0d1(output)



    
        output, max_indices1_0,d1,d1_d1, ddd = self.down0_25(output)
        #print(d1.shape,'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
        d2 = self.main_unpool1(d1, max_indices1_0)
        d_1=d2-dd

        output,y = self.l1d1(output)
        output,y = self.l1d2(output)
        output,y = self.l1d3(output)
        output,y = self.l1d4(output)

        cc_2=self.conv2d4(d_1)


        output, max_indices2_0,d3,d3_d3, dddd = self.down0_125(output)
        d4 = self.main_unpool2(d3, max_indices2_0)
        d_2=d4-d1_d1
        cc_4=self.conv2d5(d_2)

        output,y = self.l2d1(output)
        output,y = self.l2d2(output)
        output,y = self.l2d3(output)
        output,y = self.l2d4(output)
        output,y = self.l3d1(output)
        output,y = self.l3d2(output)
        output,y = self.l3d3(output)
        output,y = self.l3d4(output)
        x1_81 = output
        x1_8 = self.conv2d1(output)

        x1_8_2 = torch.nn.functional.interpolate(x1_81, scale_factor=2, mode='bilinear')

        out4 = torch.cat((x1_8_2,d_2),1)
        x1_41 = self.conv2d2(out4)
        x1_4=x1_41+cc_4

        x1_4_2 = torch.nn.functional.interpolate(x1_4, scale_factor=2, mode='bilinear')
        out2 = torch.cat((x1_4_2, d_1), 1)
        x1_21 = self.conv2d3(out2)
        x1_2=x1_21+cc_2

        x1_1 = torch.nn.functional.interpolate(x1_2, scale_factor=2, mode='bilinear')

        return x1_1, x1_2, x1_4, x1_8



