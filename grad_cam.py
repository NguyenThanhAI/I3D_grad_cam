import os
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import mmcv
import torch
import torch.nn as nn
from torch.nn.modules import Sequential
from torch.nn import functional as F
from torch.autograd import Function

from i3d import I3D


def show_frames_on_figure(frames):
    fig = plt.figure(figsize=(10, 10))
    rows = 4
    columns = 8
    for i, frame in enumerate(frames):
        fig.add_subplot(rows, columns, i+1)
        plt.imshow(frame)
        plt.title("Frame " + str(i + 1))
    plt.savefig(os.path.join("heatmap", folder_name, "input_frames.jpg"))
    plt.show()


def get_input_frames(args):
    cap = cv2.VideoCapture(args.video_path)

    frame_list = []

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame, (256, 256))
        frame_list.append(frame.copy()[:, :, ::-1])
    choice_list = range(len(frame_list) - 32 + 1)
    if args.frame_index is None:
        index = np.random.choice(list(choice_list))
    else:
        index = args.frame_index
        assert index < len(frame_list) - 31
    print("frame index:", index)
    frame_list = frame_list[index:index + 32]
    frames_to_show = frame_list.copy()
    show_frames_on_figure(frames_to_show)
    preprocess = lambda x: (x - np.array([123.675, 116.28, 103.53])[np.newaxis, np.newaxis, :]) / np.array([58.395, 57.12, 57.375])[np.newaxis, np.newaxis, :]
    frame_list = list(map(preprocess, frame_list))
    frame_list = np.stack(frame_list, axis=0)
    frame_list = np.transpose(frame_list, axes=(3, 0, 1, 2))
    #print("frame_list:", frame_list)
    frame_list = torch.from_numpy(frame_list)
    frame_list = frame_list.unsqueeze(0).float().requires_grad_(True)

    return frame_list

class Feature_Extractor():

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []


    def save_gradient(self, grad):
        self.gradients.append(grad)

    def get_gradient(self):
        return self.gradients

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
            #print(list(x.size()))
        return outputs, x


class GradCam:
    def __init__(self, model, target_layer_names="backbone", use_cuda=False):
        self.model = model
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.extractor = Feature_Extractor(self.model, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        if index is None:
            softmax = F.softmax(output, dim=1)
            #print(softmax)
            index = np.argmax(softmax.cpu().data.numpy())

        #print("index of gradcam", index)

        one_hot = np.zeros((1, softmax.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)

        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradient()[-1].cpu().data.numpy()

        target = features[-1]

        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(3, 4))[0, :, :]

        cam = np.sum(weights[:, :, np.newaxis, np.newaxis] * target, axis=0)

        cam_list = []

        for i in range(cam.shape[0]):
            cam_temp = cam[i].copy()
            cam_temp = np.maximum(cam_temp, 0)
            cam_temp = cv2.resize(cam_temp, (512, 512))
            cam_temp = (cam_temp - np.min(cam_temp)) / np.max(cam_temp)
            cam_temp = cv2.applyColorMap(np.uint8(255 * cam_temp), cv2.COLORMAP_JET)
            cam_list.append(cam_temp)

        mean_cam = np.mean(cam, axis=0)
        mean_cam = np.maximum(mean_cam, 0)
        mean_cam = cv2.resize(mean_cam, (512, 512))
        mean_cam = (mean_cam - np.min(mean_cam)) / np.max(mean_cam)
        mean_cam = cv2.applyColorMap(np.uint8(255 * mean_cam), cv2.COLORMAP_JET)

        cam_list.append(mean_cam)

        min_cam = np.min(cam, axis=0)
        min_cam = np.maximum(min_cam, 0)
        min_cam = cv2.resize(min_cam, (512, 512))
        min_cam = (min_cam - np.min(min_cam)) / np.max(min_cam)
        min_cam = cv2.applyColorMap(np.uint8(255 * min_cam), cv2.COLORMAP_JET)

        cam_list.append(min_cam)

        max_cam = np.max(cam, axis=0)
        max_cam = np.maximum(max_cam, 0)
        max_cam = cv2.resize(max_cam, (512, 512))
        max_cam = (max_cam - np.min(max_cam)) / np.max(max_cam)
        max_cam = cv2.applyColorMap(np.uint8(255 * max_cam), cv2.COLORMAP_JET)

        cam_list.append(max_cam)

        fig = plt.figure(figsize=(15, 15))
        rows = 2
        columns = 4

        for k, cam_map in enumerate(cam_list):
            fig.add_subplot(rows, columns, k + 1)
            im = plt.imshow(cam_map[:, :, ::-1], cmap=plt.cm.get_cmap("jet"))
            if k == 0:
                plt.title("1st cam")
            elif k == 1:
                plt.title("2nd cam")
            elif k == 2:
                plt.title("3rd cam")
            elif k == 3:
                plt.title("4th cam")
            elif k == 4:
                plt.title("Average cam")
            elif k == 5:
                plt.title("Min cam")
            else:
                plt.title("Max cam")
        cax = plt.axes([0.925, 0.1, 0.02, 0.8])
        plt.colorbar(cax=cax)
        plt.savefig(os.path.join("heatmap", folder_name, "cams.jpg"))
        plt.show()

        cam = np.mean(cam, axis=0)

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (512, 512))

        cam = cam - np.min(cam)
        cam = cam / np.max(cam)

        cam = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

        return cam


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint_path", type=str, default="i3d_kinetics_rgb_r50_c3d_inflated3x1x1_seg1_f32s2_f32s2-b93cc877.pth", help="Path to pretrained model")
    parser.add_argument("--video_path", type=str, default=r"E:\PythonProjects\action_understanding\convert_caffe_model_to_pytorch\yoga\YZ8VMXkzYeE_000088_000098.mp4", help="Path to test video")
    parser.add_argument("--num_classes", type=int, default=400, help="Num classes")
    parser.add_argument("--use_cuda", type=bool, default=False, help="Use GPU acceleration")
    parser.add_argument("--frame_index", type=int, default=30, help="Index of first frame of 32 consequent frames")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()

    folder_name = "_".join(args.video_path.split(os.sep)[-2:])
    if not os.path.exists(os.path.join("heatmap", folder_name)):
        os.makedirs(os.path.join("heatmap", folder_name), exist_ok=True)

    model = I3D()

    model.load_state_dict(torch.load(args.checkpoint_path)["state_dict"])

    grad_cam = GradCam(model=model, target_layer_names=["backbone"], use_cuda=args.use_cuda)

    input = get_input_frames(args)

    cam = grad_cam(input)

    cv2.imshow("", cam)
    cv2.waitKey(0)

