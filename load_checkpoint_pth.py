import numpy as np
import torch

import torch.nn as nn

from torchvision.models.video.resnet import VideoResNet, BasicBlock, R2Plus1dStem, Conv2Plus1D

checkpoint_path = "i3d_kinetics_rgb_r50_c3d_inflated3x1x1_seg1_f32s2_f32s2-b93cc877.pth"
#checkpoint_path = "r2.5d_d34_l32.pth"
model_state_dict = torch.load(checkpoint_path)
name_list = []
#for name, tensor in model_state_dict["state_dict"].items():
#    print("/".join(name.split(".")), tensor.numpy().shape)
#    name_list.append("/".join(name.split(".")))

print(type(model_state_dict))

i = 0
for name in dict(model_state_dict["state_dict"]):
    i += 1
    print(name, model_state_dict["state_dict"][name].numpy().shape)
print(i)
#for name in name_list:
#    name_compo = name.split("/")
#    print(name_compo)
#
#length = lambda x: len(x.split("/"))
#max_length = np.max(list(map(length, name_list)))
#print(max_length)
#
##compo_0 = lambda x: x.split("/")[0]
##compo = list(map(compo_0, name_list))
##print(compo)
#order_dict = {}
#for i in range(max_length):
#    compo_i = lambda x: x.split("/")[i] if len(x.split("/")) >= (i + 1) else None
#    compo = list(map(compo_i, name_list))
#    #print(compo)
#    compo = list(set(list(filter(None, compo))))
#    print(compo)

def r2plus1d_34(num_classes, pretrained=False, progress=False, **kwargs):
    model = VideoResNet(block=BasicBlock,
                        conv_makers=[Conv2Plus1D] * 4,
                        layers=[3, 4, 6, 3],
                        stem=R2Plus1dStem)

    model.fc = nn.Linear(model.fc.in_features, out_features=num_classes)

    # Fix difference in PyTorch vs Caffe2 architecture
    # https://github.com/facebookresearch/VMZ/issues/89
    model.layer2[0].conv2[0] = Conv2Plus1D(128, 128, 288)
    model.layer3[0].conv2[0] = Conv2Plus1D(256, 256, 576)
    model.layer4[0].conv2[0] = Conv2Plus1D(512, 512, 1152)

    # We need exact Caffe2 momentum for BatchNorm scaling
    for m in model.modules():
        if isinstance(m, nn.BatchNorm3d):
            m.eps = 1e-3
            m.momentum = 0.9

    return model

model = r2plus1d_34(400)

i = 0
for name in dict(model.state_dict()):
    i += 1
    print(name, model.state_dict()[name].size())

print(i, len(dict(model.state_dict()).keys()))

print(model.layer1[0].conv1[1])

#or name, module in model._modules.items():
#   print(name, module.__class__.__name__)
#   if isinstance(module, nn.Sequential):
#       continue
#   else:
#       for subname, submodule in module._modules.items():
#           print(subname, submodule.__class__.__name__)
#           if len(module._modules.keys()):
#               for sub1name, sub1module in submodule._modules.items():
#                   print(sub1name, sub1module.__class__.__name__)
#                   if isinstance(sub1module, nn.Sequential):
#                       for sub2name, sub2module in sub1module._modules.items():
#                           print(sub2name, sub2module.__class__.__name__)
#                           if isinstance(sub2module, nn.Sequential):
#                               for sub3name, sub3module in sub2module._modules.items():
#                                   print(sub3name, sub3module.__class__.__name__)

#i = 0
#for name, module in model._modules.items():
#    i += 1
#    print(str(i), "name:", name, "module:", type(module), "len:", len(module._modules.keys()))
#    if len(module._modules.keys()) == 0:
#        continue
#
#    else:
#        for subname, submodule in module._modules.items():
#            print("subname:", subname, "submodule:", type(submodule), "len:", len(submodule._modules.keys()))
#            if len(submodule._modules.keys()) == 0:
#                continue
#            else:
#                for sub1name, sub1module in submodule._modules.items():
#                    print("sub1name:", sub1name, "sub1module:", type(sub1module), "len:", len(sub1module._modules.keys()))
#                    if len(sub1module._modules.keys()) == 0:
#                        continue
#                    else:
#                        for sub2name, sub2module in sub1module._modules.items():
#                            print("sub2name:", sub2name, "sub2module:", type(sub2module), "len:", len(sub2module._modules.keys()))
#                            if len(sub2module._modules.keys()) == 0:
#                                continue
#                            else:
#                                for sub3name, sub3module in sub2module._modules.items():
#                                    print("sub3name:", sub3name, "sub3module:", type(sub3module), "len:", len(sub3module._modules.keys()))