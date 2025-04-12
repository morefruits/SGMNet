import torch
from torch import nn
import torch.nn.functional as F
import time
import timm
# from resnet import *
import torchvision.models as models

class SE(nn.Module):  

    def __init__(self): 
        super(SE, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1), nn.BatchNorm2d(128), nn.PReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=3, dilation=2, padding=2), nn.BatchNorm2d(128), nn.PReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=3, dilation=4, padding=4), nn.BatchNorm2d(128), nn.PReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=3, dilation=6, padding=6), nn.BatchNorm2d(128), nn.PReLU()
         )
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1), nn.BatchNorm2d(128),  nn.PReLU()
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(5 * 128, 512, kernel_size=1), nn.BatchNorm2d(512), nn.PReLU()
        )

    def forward(self, x):

        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        conv4 = self.conv4(x)
        conv5 = F.upsample(self.conv5(F.adaptive_avg_pool2d(x, 1)), size=x.size()[2:], mode='bilinear')

        x_fuse = self.fuse(torch.cat((conv1, conv2, conv3, conv4, conv5), 1))

        return x_fuse

class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()

        self.backbone_rgb = models.resnet34(pretrained=True)
        self.backbone_t = models.resnet34(pretrained=True)
        self.relu = nn.ReLU(inplace=True)

        cp = []
        cp.append(nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), self.relu, aggregation_scale(64, 64)))
        cp.append(nn.Sequential(nn.Conv2d(128, 64, 3, 1, 1), self.relu, aggregation_scale(64, 64)))
        cp.append(nn.Sequential(nn.Conv2d(256, 96, 3, 1, 1), self.relu, aggregation_scale(96, 64)))
        cp.append(nn.Sequential(nn.Conv2d(512, 128, 3, 1, 1), self.relu, aggregation_scale(128, 64)))
        self.CP = nn.ModuleList(cp)

        self.att_block2 = mfe(in_dim=64, sr_ratio=4)
        self.att_block3 = mfe(in_dim=128, sr_ratio=4)
        self.att_block4 = mfe(in_dim=256, sr_ratio=2)
        self.att_block5 = mfe(in_dim=512, sr_ratio=1)

        self.rgb_global = SE()
        self.t_global =SE()


    def load_pretrained_model(self, model_path):
        # resnet pretrained parameter
        pretrained_dict_res = torch.load(model_path)
        res_model_dict = self.backbone.state_dict()
        pretrained_dict_res = {k: v for k, v in pretrained_dict_res.items() if k in res_model_dict}
        res_model_dict.update(pretrained_dict_res)
        self.backbone.load_state_dict(res_model_dict)

    def forward(self, x, y):

        # B = x.shape[0]
        verbose = False
        feature_extract = []
        tmp_x = []
        # print(x.shape)
        ############################ stage 0 ###########################
        res1 = self.backbone_rgb.conv1(x)
        res1 = self.backbone_rgb.bn1(res1)
        res1 = self.backbone_rgb.relu(res1)
        # tmp_x.append(res1)
        res1 = self.backbone_rgb.maxpool(res1)  

        res2 = self.backbone_t.conv1(y)
        res2 = self.backbone_t.bn1(res2)
        res2 = self.backbone_t.relu(res2)
        # tmp_x.append(res1)
        res2 = self.backbone_t.maxpool(res2)

        ############################ stage 1 ###########################
        x1 = self.backbone_rgb.layer1(res1)  
        x2 = self.backbone_t.layer1(res2)
        x1, x2, x3 = self.att_block2(x1, x2)
        if verbose: print(x1.size())
        if verbose: print(x2.size())
        if verbose: print(x3.size())
        # rgb2 = x1
        # t2 = x2
        # tmp_x.append(x3)
        rgb2 = self.CP[0](x1)
        t2 = self.CP[0](x2)

        ########################### stage 2 ###########################
        x1 = self.backbone_rgb.layer2(x1)  
        x2 = self.backbone_t.layer2(x2)
        x1, x2, x3 = self.att_block3(x1, x2)
        # rgb3 = x1
        # t3 = x2
        rgb3 = self.CP[1](x1)
        t3 = self.CP[1](x2)

        ############################ stage 3 ###########################
        x1 = self.backbone_rgb.layer3(x1)  
        x2 = self.backbone_t.layer3(x2)
        x1,x2,x3 = self.att_block4(x1, x2)
        # rgb4 = x1
        # t4 = x2
        rgb4 = self.CP[2](x1)
        t4 = self.CP[2](x2)

        ############################ stage 4 ###########################
        x1 = self.backbone_rgb.layer4(x1) 
        x2 = self.backbone_t.layer4(x2)
        x1,x2,x3 = self.att_block5(x1, x2)
        # rgb5 = x1
        # t5 = x2
        rgb_global = self.rgb_global(x1)
        t_global = self.t_global(x2)
        rgb5 = self.CP[3](x1)
        t5 = self.CP[3](x2)

        return rgb2,rgb3,rgb4,rgb5,rgb_global,t2,t3,t4,t5,t_global

def convblock(in_, out_, ks, st, pad):
    return nn.Sequential(
        nn.Conv2d(in_, out_, ks, st, pad),
        nn.BatchNorm2d(out_),
        nn.ReLU(inplace=True)
    )
    
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

