import numpy as np
import torch
import torchvision
import torch.nn as nn

#from torchvision.models.video.resnet import BasicBlock, BasicStem, Conv3DSimple, VideoResNet

class Downsample(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride=2):
        super(Downsample, self).__init__(nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=(1, stride, stride), bias=False),
                                         nn.BatchNorm3d(num_features=out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))


class Bottleneck(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, is_inflate, is_downsample=False, has_downsample=False):
        super(Bottleneck, self).__init__()
        self.is_downsample = is_downsample
        self.has_downsample = has_downsample
        if is_inflate:
            self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=mid_channels, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0), bias=False)
        else:
            self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=mid_channels, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=False)
        if self.is_downsample:
            self.conv2 = nn.Conv3d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), bias=False)
        else:
            self.conv2 = nn.Conv3d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False)
        self.bn1 = nn.BatchNorm3d(num_features=mid_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bn2 = nn.BatchNorm3d(num_features=mid_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.conv3 = nn.Conv3d(in_channels=mid_channels, out_channels=out_channels, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=False)
        self.bn3 = nn.BatchNorm3d(num_features=out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        if self.has_downsample:
            if self.is_downsample:
                self.downsample = Downsample(in_channels=in_channels, out_channels=out_channels, stride=2)
            else:
                self.downsample = Downsample(in_channels=in_channels, out_channels=out_channels, stride=1)

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.has_downsample:
            residual = self.downsample(x)
        else:
            residual = x
        out = out + residual
        out = self.relu(out)

        return out


class ResNet_I3D(nn.Module):
    def __init__(self):
        super(ResNet_I3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=3, out_channels=64, kernel_size=(5, 7, 7), stride=(2, 2, 2), bias=False, padding=(2, 3, 3))
        self.bn1 = nn.BatchNorm3d(num_features=64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(2, 2, 2), padding=(0, 1, 1), dilation=1, ceil_mode=False)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0), dilation=1, ceil_mode=False)

        self.layer1 = nn.Sequential(Bottleneck(in_channels=64, mid_channels=64, out_channels=256, is_inflate=True, is_downsample=False, has_downsample=True),
                                    Bottleneck(in_channels=256, mid_channels=64, out_channels=256, is_inflate=True),
                                    Bottleneck(in_channels=256, mid_channels=64, out_channels=256, is_inflate=True))

        self.layer2 = nn.Sequential(Bottleneck(in_channels=256, mid_channels=128, out_channels=512, is_inflate=True, is_downsample=True, has_downsample=True),
                                    Bottleneck(in_channels=512, mid_channels=128, out_channels=512, is_inflate=False),
                                    Bottleneck(in_channels=512, mid_channels=128, out_channels=512, is_inflate=True),
                                    Bottleneck(in_channels=512, mid_channels=128, out_channels=512, is_inflate=False))

        self.layer3 = nn.Sequential(Bottleneck(in_channels=512, mid_channels=256, out_channels=1024, is_inflate=True, is_downsample=True, has_downsample=True),
                                    Bottleneck(in_channels=1024, mid_channels=256, out_channels=1024, is_inflate=False),
                                    Bottleneck(in_channels=1024, mid_channels=256, out_channels=1024, is_inflate=True),
                                    Bottleneck(in_channels=1024, mid_channels=256, out_channels=1024, is_inflate=False),
                                    Bottleneck(in_channels=1024, mid_channels=256, out_channels=1024, is_inflate=True),
                                    Bottleneck(in_channels=1024, mid_channels=256, out_channels=1024, is_inflate=False))

        self.layer4 = nn.Sequential(Bottleneck(in_channels=1024, mid_channels=512, out_channels=2048, is_inflate=False, is_downsample=True, has_downsample=True),
                                    Bottleneck(in_channels=2048, mid_channels=512, out_channels=2048, is_inflate=True),
                                    Bottleneck(in_channels=2048, mid_channels=512, out_channels=2048, is_inflate=False))


    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.max_pool(out)
        out = self.layer1(out)
        out = self.pool2(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        return out


class SimpleSpatialTemporalModule(nn.Module):
    def __init__(self, spatial_size=7, temporal_size=4):
        super(SimpleSpatialTemporalModule, self).__init__()

        self.spatial_size = spatial_size if not isinstance(spatial_size, int) else (spatial_size, spatial_size)
        self.temporal_size = temporal_size
        self.pool_size = (self.temporal_size, ) + self.spatial_size

        self.op = nn.AvgPool3d(self.pool_size, stride=1, padding=0)

    def forward(self, x):
        return self.op(x)


class ClsHead(nn.Module):
    def __init__(self, in_features=2048, out_features=400):
        super(ClsHead, self).__init__()
        #self.global_average_pooling = nn.AdaptiveAvgPool3d((1, 1, 1)) # Tạm thời không cần thiết
        self.fc_cls = nn.Linear(in_features=in_features, out_features=out_features, bias=True)

    def forward(self, x):
        #out = self.global_average_pooling(x) # Tạm thời không cần thiết
        #print(out.size())
        #out = out.flatten(1)
        out = x.flatten(1)
        out = self.fc_cls(out)

        return out


class I3D(nn.Module):
    def __init__(self):
        super(I3D, self).__init__()
        self.backbone = ResNet_I3D()
        self.spatial_temporal_module = SimpleSpatialTemporalModule(spatial_size=8)
        self.cls_head = ClsHead()

    def forward(self, x):
        out = self.backbone(x)
        out = self.spatial_temporal_module(out)
        #print(out.size())
        out = self.cls_head(out)

        return out


#i3d = I3D()
#
#i = 0
#for name, param in dict(i3d.state_dict()).items():
#    i += 1
#    print(name, param.numpy().shape)
#
#print(i)
#
#i3d.load_state_dict(torch.load("i3d_kinetics_rgb_r50_c3d_inflated3x1x1_seg1_f32s2_f32s2-b93cc877.pth")["state_dict"])
#
##for name, module in i3d._modules.items():
##    print(name, module)
#
#print(i3d)
#
#input = np.random.rand(1, 3, 32, 256, 256)
#input = torch.from_numpy(input).float()
#
#output = i3d(input)
#
#print(output)