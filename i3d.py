import torch
import torchvision
import torch.nn as nn

#from torchvision.models.video.resnet import BasicBlock, BasicStem, Conv3DSimple, VideoResNet

class Downsample(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride=2):
        super(Downsample, self).__init__(nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=(1, stride, stride), bias=False),
                                         nn.BatchNorm3d(out_channels))


class BasicBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, is_inflate, is_downsample=False):
        super(BasicBlock, self).__init__()
        self.is_downsample = is_downsample
        if is_inflate:
            self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=mid_channels, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0), bias=False)
        else:
            self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=mid_channels, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=False)
        self.bn1 = nn.BatchNorm3d(mid_channels)
        self.relu1 = nn.ReLU(inplace=True)
        if self.is_downsample:
            self.downsample = Downsample(in_channels, out_channels=out_channels)
            self.conv2 = nn.Conv3d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False)
        else:
            self.conv2 = nn.Conv3d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), bias=False)
        self.bn2 = nn.BatchNorm3d(mid_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv3d(in_channels=mid_channels, out_channels=out_channels, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=False)
        self.bn3 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.is_downsample:
            residual = self.downsample(x)
        else:
            residual = x

        out = out + residual
        out = self.relu(out)

        return out


class BackBone(nn.Module):
    def __init__(self):
        super(BackBone, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=3, out_channels=64, kernel_size=(5, 7, 7), stride=(2, 2, 2), bias=False, padding=(2, 3, 3))
        self.bn1 = nn.BatchNorm3d(num_features=64)
        self.relu1 = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(2, 2, 2), padding=(0, 1, 1))

        self.layer1 = nn.Sequential(BasicBlock(in_channels=64, mid_channels=64, out_channels=256, is_inflate=True, is_downsample=True),
                                    BasicBlock(in_channels=256, mid_channels=64, out_channels=256, is_inflate=True),
                                    BasicBlock(in_channels=256, mid_channels=64, out_channels=256, is_inflate=True))

        self.layer2 = nn.Sequential(BasicBlock(in_channels=256, mid_channels=128, out_channels=512, is_inflate=True, is_downsample=True),
                                    BasicBlock(in_channels=512, mid_channels=128, out_channels=512, is_inflate=False),
                                    BasicBlock(in_channels=512, mid_channels=128, out_channels=512, is_inflate=True),
                                    BasicBlock(in_channels=512, mid_channels=128, out_channels=512, is_inflate=False))

        self.layer3 = nn.Sequential(BasicBlock(in_channels=512, mid_channels=256, out_channels=1024, is_inflate=True, is_downsample=True),
                                    BasicBlock(in_channels=1024, mid_channels=256, out_channels=1024, is_inflate=False),
                                    BasicBlock(in_channels=1024, mid_channels=256, out_channels=1024, is_inflate=True),
                                    BasicBlock(in_channels=1024, mid_channels=256, out_channels=1024, is_inflate=False),
                                    BasicBlock(in_channels=1024, mid_channels=256, out_channels=1024, is_inflate=True),
                                    BasicBlock(in_channels=1024, mid_channels=256, out_channels=1024, is_inflate=False))

        self.layer4 = nn.Sequential(BasicBlock(in_channels=1024, mid_channels=512, out_channels=2048, is_inflate=False, is_downsample=True),
                                    BasicBlock(in_channels=2048, mid_channels=512, out_channels=2048, is_inflate=True),
                                    BasicBlock(in_channels=2048, mid_channels=512, out_channels=2048, is_inflate=False))


    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.max_pool(out)
        out = self.layer1(out)
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
        self.global_average_pooling = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc_cls = nn.Linear(in_features=in_features, out_features=out_features, bias=True)

    def forward(self, x):
        out = self.global_average_pooling(x)
        out = self.fc_cls(out)

        return out


class I3D(nn.Module):
    def __init__(self):
        super(I3D, self).__init__()
        self.backbone = BackBone()
        self.simplespatialtemporalmodule = SimpleSpatialTemporalModule()
        self.cls_head = ClsHead()

    def forward(self, x):
        out = self.backbone(x)
        out = self.simplespatialtemporalmodule(out)
        out = self.cls_head(out)

        return out


i3d = I3D()

i = 0
for name, param in dict(i3d.state_dict()).items():
    i += 1
    print(name, param.numpy().shape)

print(i)

i3d.load_state_dict(torch.load("i3d_kinetics_rgb_r50_c3d_inflated3x1x1_seg1_f32s2_f32s2-b93cc877.pth")["state_dict"])