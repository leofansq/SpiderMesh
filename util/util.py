# By Yuxiang Sun, Dec. 4, 2020
# Email: sun.yuxiang@outlook.com

import numpy as np 
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
 
# 0:unlabeled, 1:car, 2:person, 3:bike, 4:curve, 5:car_stop, 6:guardrail, 7:color_cone, 8:bump 
def get_palette():
    unlabelled = [0,0,0]
    car        = [64,0,128]
    person     = [64,64,0]
    bike       = [0,128,192]
    curve      = [0,0,192]
    car_stop   = [128,128,0]
    guardrail  = [64,64,128]
    color_cone = [192,128,128]
    bump       = [192,64,0]
    palette    = np.array([unlabelled,car, person, bike, curve, car_stop, guardrail, color_cone, bump])
    return palette

def visualize(image_name, predictions, weight_name, name='mm'):
    palette = get_palette()
    for (i, pred) in enumerate(predictions):
        pred = predictions[i].cpu().numpy()
        img = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
        for cid in range(0, len(palette)):
            img[pred == cid] = palette[cid]
        img = Image.fromarray(np.uint8(img))
        img.save('runs/vis_results/pred_' + weight_name + '_' + image_name[i] + '_' + name + '.png')

def compute_results(conf_total):
    n_class =  conf_total.shape[0]
    consider_unlabeled = True  # must consider the unlabeled, please set it to True
    if consider_unlabeled is True:
        start_index = 0
    else:
        start_index = 1
    precision_per_class = np.zeros(n_class)
    recall_per_class = np.zeros(n_class)
    iou_per_class = np.zeros(n_class)
    for cid in range(start_index, n_class):
        if conf_total[start_index:, cid].sum() == 0:
            precision_per_class[cid] =  np.nan
        else:
            precision_per_class[cid] = float(conf_total[cid, cid]) / float(conf_total[start_index:, cid].sum()) # precision = TP/TP+FP
        if conf_total[cid, start_index:].sum() == 0:
            recall_per_class[cid] = np.nan
        else:
            recall_per_class[cid] = float(conf_total[cid, cid]) / float(conf_total[cid, start_index:].sum()) # recall = TP/TP+FN
        if (conf_total[cid, start_index:].sum() + conf_total[start_index:, cid].sum() - conf_total[cid, cid]) == 0:
            iou_per_class[cid] = np.nan
        else:
            iou_per_class[cid] = float(conf_total[cid, cid]) / float((conf_total[cid, start_index:].sum() + conf_total[start_index:, cid].sum() - conf_total[cid, cid])) # IoU = TP/TP+FP+FN

    return precision_per_class, recall_per_class, iou_per_class

def _ntuple(n):
    def parse(x):
        if isinstance(x, list) or isinstance(x, tuple):
            return x
        return tuple([x]*n)
    return parse
_pair = _ntuple(2)


class Conv2_5D_disp(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, pixel_size=16):
        super(Conv2_5D_disp, self).__init__()
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.kernel_size_prod = self.kernel_size[0] * self.kernel_size[1]
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.pixel_size = pixel_size
        assert self.kernel_size_prod % 2 == 1
        
        self.weight_0 = Parameter(torch.Tensor(out_channels, in_channels, *kernel_size))
        self.weight_1 = Parameter(torch.Tensor(out_channels, in_channels, *kernel_size))
        self.weight_2 = Parameter(torch.Tensor(out_channels, in_channels, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x, disp, camera_params):
        N, C, H, W = x.size(0), x.size(1), x.size(2), x.size(3)
        out_H = (H + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1
        out_W = (W + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1] + 1
        intrinsic, extrinsic = camera_params['intrinsic'], camera_params['extrinsic']
        
        x_col = F.unfold(x, self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride)  # (N, C*kh*kw, out_H*out_W)
        x_col = x_col.view(N, C, self.kernel_size_prod, out_H * out_W)
        
        disp_col = F.unfold(disp, self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride)  # (N, kh*kw, out_H*out_W)
        valid_mask = 1 - disp_col.eq(0.).to(torch.float32)
        valid_mask *= valid_mask[:, self.kernel_size_prod // 2, :].view(N, 1, out_H * out_W)
        disp_col *= valid_mask
        depth_col = (extrinsic['baseline'] * intrinsic['fx']).view(N, 1, 1).cuda() / torch.clamp(disp_col, 0.01, 256)
        valid_mask = valid_mask.view(N, 1, self.kernel_size_prod, out_H * out_W)
        
        center_depth = depth_col[:, self.kernel_size_prod // 2, :].view(N, 1, out_H * out_W)
        grid_range = self.pixel_size * self.dilation[0] * center_depth / intrinsic['fx'].view(N, 1, 1).cuda()
        
        mask_0 = torch.abs(depth_col - (center_depth + grid_range)).le(grid_range / 2).view(N, 1, self.kernel_size_prod, out_H * out_W).to(torch.float32)
        mask_1 = torch.abs(depth_col - (center_depth)).le(grid_range / 2).view(N, 1, self.kernel_size_prod, out_H * out_W).to(torch.float32)
        mask_1 = (mask_1 + 1 - valid_mask).clamp(min=0., max=1.)
        mask_2 = torch.abs(depth_col - (center_depth - grid_range)).le(grid_range / 2).view(N, 1, self.kernel_size_prod, out_H * out_W).to(torch.float32)
        
        output = torch.matmul(self.weight_0.view(-1, C * self.kernel_size_prod), (x_col * mask_0).view(N, C * self.kernel_size_prod, out_H * out_W))
        output += torch.matmul(self.weight_1.view(-1, C * self.kernel_size_prod), (x_col * mask_1).view(N, C * self.kernel_size_prod, out_H * out_W))
        output += torch.matmul(self.weight_2.view(-1, C * self.kernel_size_prod), (x_col * mask_2).view(N, C * self.kernel_size_prod, out_H * out_W))
        output = output.view(N, -1, out_H, out_W)
        if self.bias:
            output += self.bias.view(1, -1, 1, 1)
        return output
    
    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)


class Conv2_5D_depth(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False, pixel_size=1, is_graph=False):
        super(Conv2_5D_depth, self).__init__()
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.kernel_size_prod = self.kernel_size[0] * self.kernel_size[1]
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.pixel_size = pixel_size
        assert self.kernel_size_prod % 2 == 1
        self.is_graph = is_graph
        if self.is_graph:
            self.weight_0 = Parameter(torch.Tensor(out_channels, 1, *kernel_size))
            self.weight_1 = Parameter(torch.Tensor(out_channels, 1, *kernel_size))
            self.weight_2 = Parameter(torch.Tensor(out_channels, 1, *kernel_size))
        else:
            self.weight_0 = Parameter(torch.Tensor(out_channels, in_channels, *kernel_size))
            self.weight_1 = Parameter(torch.Tensor(out_channels, in_channels, *kernel_size))
            self.weight_2 = Parameter(torch.Tensor(out_channels, in_channels, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x, depth, camera_params):
        N, C, H, W = x.size(0), x.size(1), x.size(2), x.size(3)
        out_H = (H + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1
        out_W = (W + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1] + 1
        intrinsic = camera_params['intrinsic']
        x_col = F.unfold(x, self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride)  # N*(C*kh*kw)*(out_H*out_W)
        x_col = x_col.view(N, C, self.kernel_size_prod, out_H * out_W)
        depth_col = F.unfold(depth, self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride)  # N*(kh*kw)*(out_H*out_W)
        center_depth = depth_col[:, self.kernel_size_prod // 2, :]
        center_depth  = center_depth.view(N, 1, out_H * out_W)
        grid_range = self.pixel_size * center_depth / intrinsic['fx'].cuda().view(N, 1, 1)

        mask_0 = torch.abs(depth_col - (center_depth + grid_range)).le(grid_range / 2).view(N, 1, self.kernel_size_prod, out_H * out_W).to(torch.float32)
        mask_1 = torch.abs(depth_col - (center_depth)).le(grid_range / 2).view(N, 1, self.kernel_size_prod, out_H * out_W).to(torch.float32)
        mask_2 = torch.abs(depth_col - (center_depth - grid_range)).le(grid_range / 2).view(N, 1, self.kernel_size_prod, out_H * out_W).to(torch.float32)
        output = torch.matmul(self.weight_0.view(-1, C * self.kernel_size_prod), (x_col * mask_0).view(N, C * self.kernel_size_prod, out_H * out_W))
        output += torch.matmul(self.weight_1.view(-1, C * self.kernel_size_prod), (x_col * mask_1).view(N, C * self.kernel_size_prod, out_H * out_W))
        output += torch.matmul(self.weight_2.view(-1, C * self.kernel_size_prod), (x_col * mask_2).view(N, C * self.kernel_size_prod, out_H * out_W))
        output = output.view(N, -1, out_H, out_W)
        if self.bias:
            output += self.bias.view(1, -1, 1, 1)
        return output



    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)


def __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum, **kwargs):
    for name, m in feature.named_modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            conv_init(m.weight, **kwargs)
        elif isinstance(m, Conv2_5D_depth):
            conv_init(m.weight_0, **kwargs)
            conv_init(m.weight_1, **kwargs)
            conv_init(m.weight_2, **kwargs)
        elif isinstance(m, Conv2_5D_disp):
            conv_init(m.weight_0, **kwargs)
            conv_init(m.weight_1, **kwargs)
            conv_init(m.weight_2, **kwargs)
        elif isinstance(m, norm_layer):
            m.eps = bn_eps
            m.momentum = bn_momentum
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum, **kwargs):
    if isinstance(module_list, list):
        for feature in module_list:
            __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum, **kwargs)
    else:
        __init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum, **kwargs)



