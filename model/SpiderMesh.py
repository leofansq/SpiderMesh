from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from functools import partial

class SpiderMesh(nn.Module):

    def __init__(self, num_classes, num_layers=152):
        super(SpiderMesh, self).__init__()

        if num_layers == 18:
            resnet_raw_model1 = models.resnet18(pretrained=True)
            resnet_raw_model2 = models.resnet18(pretrained=True)
            self.channels = [64, 64, 128, 256, 512]
        elif num_layers == 34:
            resnet_raw_model1 = models.resnet34(pretrained=True)
            resnet_raw_model2 = models.resnet34(pretrained=True)
            self.channels = [64, 64, 128, 256, 512]
        elif num_layers == 50:
            resnet_raw_model1 = models.resnet50(pretrained=True)
            resnet_raw_model2 = models.resnet50(pretrained=True)
            self.channels = [64, 256, 512, 1024, 2048]
        elif num_layers == 101:
            resnet_raw_model1 = models.resnet101(pretrained=True)
            resnet_raw_model2 = models.resnet101(pretrained=True)
            self.channels = [64, 256, 512, 1024, 2048]
        elif num_layers == 152:
            resnet_raw_model1 = models.resnet152(pretrained=True)
            resnet_raw_model2 = models.resnet152(pretrained=True)
            self.channels = [64, 256, 512, 1024, 2048]

        ########  ENCODER  ########
        self.encoder_thermal_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False) 
        self.encoder_thermal_conv1.weight.data = torch.unsqueeze(torch.mean(resnet_raw_model1.conv1.weight.data, dim=1), dim=1)
        self.encoder_thermal_bn1 = resnet_raw_model1.bn1
        self.encoder_thermal_relu = resnet_raw_model1.relu
        self.encoder_thermal_maxpool = resnet_raw_model1.maxpool
        self.encoder_thermal_layer1 = resnet_raw_model1.layer1
        self.encoder_thermal_layer2 = resnet_raw_model1.layer2
        self.encoder_thermal_layer3 = resnet_raw_model1.layer3
        self.encoder_thermal_layer4 = resnet_raw_model1.layer4
 
        self.encoder_rgb_conv1 = resnet_raw_model2.conv1
        self.encoder_rgb_bn1 = resnet_raw_model2.bn1
        self.encoder_rgb_relu = resnet_raw_model2.relu
        self.encoder_rgb_maxpool = resnet_raw_model2.maxpool
        self.encoder_rgb_layer1 = resnet_raw_model2.layer1
        self.encoder_rgb_layer2 = resnet_raw_model2.layer2
        self.encoder_rgb_layer3 = resnet_raw_model2.layer3
        self.encoder_rgb_layer4 = resnet_raw_model2.layer4

        self.encoder_list = [self.encoder_thermal_layer4, self.encoder_rgb_layer4]
        self.dilate = 2
        for l in self.encoder_list:
            for m in l.children():
                m.apply(partial(self._nostride_dilate, dilate=self.dilate))
                self.dilate *= 2

        ######### DTM ########
        self.ISF0 = ISFM(self.channels[0])
        self.ISF1 = ISFM(self.channels[1])
        self.ISF2 = ISFM(self.channels[2])
        self.ISF3 = ISFM(self.channels[3])
        self.ISF4 = ISFM(self.channels[4])        

        ########  SRM  ########
        self.head_rgb = PSEM(num_classes)
        self.head_thermal = PSEM(num_classes)

        ########  Classifier  ########
        self.classifier_rgb = nn.Conv2d(256, num_classes, kernel_size=1, bias=True)
        self.classifier_thermal = nn.Conv2d(256, num_classes, kernel_size=1, bias=True)
        self.business_layer = [self.head_rgb, self.classifier_rgb, self.head_thermal, self.classifier_thermal]

    def _nostride_dilate(self, m, dilate):
        if isinstance(m, nn.Conv2d):
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)
 
    def forward(self, input):

        rgb = input[:,:3]
        thermal = input[:,3:]

        #####################################
        ####          Encoder            ####
        #####################################

        mm_list = []

        # (240, 320)
        rgb = self.encoder_rgb_conv1(rgb)
        rgb = self.encoder_rgb_bn1(rgb)
        rgb = self.encoder_rgb_relu(rgb)
        thermal = self.encoder_thermal_conv1(thermal)
        thermal = self.encoder_thermal_bn1(thermal)
        thermal = self.encoder_thermal_relu(thermal)

        rgb, thermal = self.ISF0(rgb, thermal)

        rgb = self.encoder_rgb_maxpool(rgb)
        thermal = self.encoder_thermal_maxpool(thermal)

        # (120, 160)
        rgb = self.encoder_rgb_layer1(rgb)
        thermal = self.encoder_thermal_layer1(thermal)

        mm_list.append(rgb+thermal)
        rgb, thermal = self.ISF1(rgb, thermal)

        # (60, 80)
        rgb = self.encoder_rgb_layer2(rgb) 
        thermal = self.encoder_thermal_layer2(thermal)

        mm_list.append(rgb+thermal)
        rgb, thermal = self.ISF2(rgb, thermal)

        # (30, 40)
        rgb = self.encoder_rgb_layer3(rgb)
        thermal = self.encoder_thermal_layer3(thermal)

        mm_list.append(rgb+thermal)
        rgb, thermal = self.ISF3(rgb, thermal)

        # (15, 20)
        rgb = self.encoder_rgb_layer4(rgb)
        thermal = self.encoder_thermal_layer4(thermal)

        rgb, thermal = self.ISF4(rgb, thermal)

        #####################################
        ####           SRM             ####
        #####################################

        rgb = self.head_rgb(rgb, mm_list)
        thermal = self.head_thermal(thermal, mm_list)

        #####################################
        ####        Classifier           ####
        #####################################

        _, _, h, w = input.shape
        pred_rgb = self.classifier_rgb(rgb+thermal)
        pred_rgb = F.interpolate(pred_rgb, size=(h, w), mode='bilinear', align_corners=True)

        pred_thermal = self.classifier_thermal(thermal)
        pred_thermal = F.interpolate(pred_thermal, size=(h, w), mode='bilinear', align_corners=True)


        return pred_rgb, pred_thermal

############ Spatial-aware Recursive Meshing Module (SRM) ###############
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rates=(12, 24, 36), hidden_channels=256, norm_act=nn.BatchNorm2d, pooling_size=None):
        super(ASPP, self).__init__()
        self.pooling_size = pooling_size

        self.map_convs = nn.ModuleList([
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.Conv2d(in_channels, hidden_channels, 3, bias=False, dilation=dilation_rates[0], padding=dilation_rates[0]),
            nn.Conv2d(in_channels, hidden_channels, 3, bias=False, dilation=dilation_rates[1], padding=dilation_rates[1]),
            nn.Conv2d(in_channels, hidden_channels, 3, bias=False, dilation=dilation_rates[2], padding=dilation_rates[2])])
        self.map_bn = norm_act(hidden_channels * 4)

        self.global_pooling_conv = nn.Conv2d(in_channels, hidden_channels, 1, bias=False)
        self.global_pooling_bn = norm_act(hidden_channels)

        self.red_conv = nn.Conv2d(hidden_channels * 4, out_channels, 1, bias=False)
        self.pool_red_conv = nn.Conv2d(hidden_channels, out_channels, 1, bias=False)
        self.red_bn = norm_act(out_channels)

        self.leak_relu = nn.LeakyReLU()

    def forward(self, x):
        # Map convolutions
        out = torch.cat([m(x) for m in self.map_convs], dim=1)
        out = self.map_bn(out)
        out = self.leak_relu(out)
        out = self.red_conv(out)

        # Global pooling
        pool = self._global_pooling(x)
        pool = self.global_pooling_conv(pool)
        pool = self.global_pooling_bn(pool)

        pool = self.leak_relu(pool)

        pool = self.pool_red_conv(pool)
        if self.training or self.pooling_size is None:
            pool = pool.repeat(1, 1, x.size(2), x.size(3))

        out += pool
        out = self.red_bn(out)
        out = self.leak_relu(out)
        return out

    def _global_pooling(self, x):
        if self.training or self.pooling_size is None:
            pool = x.view(x.size(0), x.size(1), -1).mean(dim=-1)
            pool = pool.view(x.size(0), x.size(1), 1, 1)
        else:
            pooling_size = (min(try_index(self.pooling_size, 0), x.shape[2]),
                            min(try_index(self.pooling_size, 1), x.shape[3]))
            padding = (
                (pooling_size[1] - 1) // 2,
                (pooling_size[1] - 1) // 2 if pooling_size[1] % 2 == 1 else (pooling_size[1] - 1) // 2 + 1,
                (pooling_size[0] - 1) // 2,
                (pooling_size[0] - 1) // 2 if pooling_size[0] % 2 == 1 else (pooling_size[0] - 1) // 2 + 1)
            pool = nn.functional.avg_pool2d(x, pooling_size, stride=1)
            pool = nn.functional.pad(pool, pad=padding, mode="replicate")
        return pool

class SpatialGate(nn.Module):
    """
    x ---> Scale
    """
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        scale = self.compress(x)
        scale = self.spatial(scale)
        scale = torch.sigmoid(scale)
        return scale

class PSEM(nn.Module):
    def __init__(self, classify_classes, norm_act=nn.BatchNorm2d, bn_momentum=0.1):
        super(PSEM, self).__init__()

        self.classify_classes = classify_classes
        self.aspp = ASPP(2048, 256, [2, 4, 8], norm_act=norm_act)

        self.reduce0 = nn.Sequential(
            nn.Conv2d(256, 16, 1, bias=False),
            norm_act(16, momentum=bn_momentum),
            nn.ReLU())
        self.reduce1 = nn.Sequential(
            nn.Conv2d(512, 32, 1, bias=False),
            norm_act(32, momentum=bn_momentum),
            nn.ReLU())
        self.reduce2 = nn.Sequential(
            nn.Conv2d(1024, 64, 1, bias=False),
            norm_act(64, momentum=bn_momentum),
            nn.ReLU())

        self.mmsa0 = SpatialGate()
        self.mmsa1 = SpatialGate()
        self.mmsa2 = SpatialGate()

        self.last_conv = nn.Sequential(nn.Conv2d(368, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       norm_act(256, momentum=bn_momentum),
                                       nn.ReLU(),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       norm_act(256, momentum=bn_momentum),
                                       nn.ReLU())

    def forward(self, f, cm_list):

        cm0, cm1, cm2 = cm_list[0], cm_list[1], cm_list[2]
        h, w = cm0.size(2), cm0.size(3)

        f = self.aspp(f)
        att = F.interpolate(f, size=(cm2.size(2), cm2.size(3)), mode='bilinear', align_corners=True)

        cm2 = self.reduce2(cm2)
        mmsa = self.mmsa2(att)
        cm2 = cm2 * mmsa
        att = F.interpolate(torch.cat([att, cm2], 1), size=(cm1.size(2), cm1.size(3)), mode='bilinear', align_corners=True)

        cm1 = self.reduce1(cm1)
        mmsa = self.mmsa1(att)
        cm1 = cm1 * mmsa
        att = F.interpolate(torch.cat([att, cm1], 1), size=(h, w), mode='bilinear', align_corners=True)

        cm0 = self.reduce0(cm0)
        mmsa = self.mmsa0(att)
        cm0 = cm0 * mmsa

        cm1 = F.interpolate(cm1, size=(h, w), mode='bilinear', align_corners=True)
        cm2 = F.interpolate(cm2, size=(h, w), mode='bilinear', align_corners=True)
        f = F.interpolate(f, size=(h, w), mode='bilinear', align_corners=True)
        f = torch.cat((f, cm0, cm1, cm2), dim=1)       
        
        f = self.last_conv(f)

        return f


############ Demand-guided Target Masking Module (DTM) ###############

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], debug=None):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels))
        self.pool_types = pool_types
        self.debug = debug
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class CrossSpatialGate(nn.Module):
    """
    x2 ---> Scale
    x1 * Scale + x2
    """
    def __init__(self):
        super(CrossSpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x1, x2):
        x2_compress = self.compress(x2)
        x2_out = self.spatial(x2_compress)
        scale = torch.sigmoid(x2_out) # broadcasting
        return x1 * scale

class ISFM(nn.Module):
    def __init__(self, channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ISFM, self).__init__()

        self.channel_rgb = ChannelGate(channels, reduction_ratio, pool_types)
        self.channel_thermal = ChannelGate(channels, reduction_ratio, pool_types)

        self.spatial_rgb = CrossSpatialGate()
        self.spatial_thermal = CrossSpatialGate()

    def forward(self, modal_rgb, modal_thermal):

        scale_rgb = self.channel_rgb(modal_rgb)
        scale_thermal = self.channel_thermal(modal_thermal)

        vote_rgb = scale_rgb * modal_rgb
        vote_thermal = scale_thermal * modal_thermal

        aux_rgb = self.spatial_rgb(vote_thermal, vote_rgb)
        aux_thermal = self.spatial_thermal(vote_rgb, vote_thermal)

        return aux_rgb+vote_rgb, aux_thermal+vote_thermal

def complexity_test():
    from fvcore.nn import FlopCountAnalysis, parameter_count_table

    num_minibatch = 2
    rgb = torch.randn(num_minibatch, 3, 480, 640)
    thermal = torch.randn(num_minibatch, 1, 480, 640)
    model = SCINet(9)
    input = torch.cat((rgb, thermal), dim=1)

    print("Para: ", parameter_count_table(model))

    flops = FlopCountAnalysis(model, input)
    print("FLOPs: ", flops.total())

if __name__ == '__main__':
    complexity_test()
