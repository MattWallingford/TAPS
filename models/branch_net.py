import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#from .utils import load_state_dict_from_url

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


# def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
#     """3x3 convolution with padding"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
#                      padding=dilation, groups=groups, bias=False, dilation=dilation)


# def conv1x1(in_planes, out_planes, stride=1):
#     """1x1 convolution"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
def conv1x1(task_sizes, in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return branch_conv(task_sizes, in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(task_sizes, in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return branch_conv(task_sizes, in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv7x7(task_sizes, in_planes, out_planes, stride=1, groups=1, dilation=1):
    """7x7 convolution with padding"""
    return branch_conv(task_sizes, in_planes, out_planes, kernel_size=7, stride=stride,
                     padding=3, groups=groups, bias=False, dilation=dilation)


class BinarizeIndictator(autograd.Function):
    @staticmethod
    def forward(ctx, indicator):
        # Get the subnetwork by sorting the scores and using the top k%
        out = (indicator >= .1).float()
        return out
    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None


class branch_conv(nn.Conv2d):
    def __init__(self, task_sizes, *args, **kwargs):
        #task_sizes = [0, 120, 316]
        ask_partitions = [0, 120]
        self.task_sizes = task_sizes
        num_tasks = len(task_sizes)
        super().__init__(*args, **kwargs)
        weight_shape = self.weight.shape
        self.register_parameter(name = 'shared_delta', param = torch.nn.Parameter(torch.zeros(weight_shape)))
        #self.register_parameter(name = 'shared_indicator', param = torch.nn.Parameter(torch.ones([1])*.15))
        # self.conv_weights = nn.ParameterList()
        # self.indicators = nn.ParameterList()
        for i in range(num_tasks):
            # self.conv_weights.append(torch.nn.Parameter(torch.zeros(weight_shape)))
            # self.indicators.append(torch.nn.Parameter(torch.ones([1])*.15))
            weight_name = 'branch_weight_' + str(i)
            indicator_name = 'branch_indicator_' + str(i)
            shared_name = 'shared_indicator_' + str(i)
            self.register_parameter(name = weight_name, param = torch.nn.Parameter(torch.zeros(weight_shape)))
            self.register_parameter(name = indicator_name, param = torch.nn.Parameter(torch.ones([1])*.15))
            self.register_parameter(name = shared_name, param = torch.nn.Parameter(torch.ones([1])*.15))
        self.params = dict(self.named_parameters())
        self.weight.requires_grad = False
        self.weight.grad = None

    def forward(self, x, task_num):
        indicator_name = 'branch_indicator_' + str(task_num)
        weight_name = 'branch_weight_' + str(task_num)
        shared_name = 'shared_indicator_' + str(task_num)
        branch_weight = self.params[weight_name]
        shared_indicator = self.params[shared_name]
        I = BinarizeIndictator.apply(self.params[indicator_name])
        I_shared = BinarizeIndictator.apply(shared_indicator)
        w = self.weight + self.shared_delta + I * branch_weight #I_shared * self.shared_delta #
        x = F.conv2d(x,w,self.bias,self.stride,self.padding,self.dilation,self.groups)
        return x

def set_task_partitions(self, task_sizes):
    self.task_sizes = task_sizes
    self.linear_layers = nn.ModuleList()
    for i in range(len(task_sizes)):
        self.linear_layers.append(torch.nn.Linear(self.feat_dim, task_sizes[i]))

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, task_sizes, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(task_sizes, inplanes, planes, stride)
        num_tasks = len(task_sizes)
        #self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(task_sizes, planes, planes)
        #self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.bn1s = torch.nn.ModuleList()
        self.bn2s = torch.nn.ModuleList()
        for i in range(num_tasks):
            self.bn1s.append(norm_layer(planes))
            self.bn2s.append(norm_layer(planes))
            # self.conv_weights.append(torch.nn.Parameter(torch.zeros(weight_shape)))
            # self.indicators.append(torch.nn.Parameter(torch.ones([1])*.15))
            

    def forward(self, x, task_num):
        identity = x

        out = self.conv1(x, task_num)
        out = self.bn1s[task_num](out)
        out = self.relu(out)

        out = self.conv2(out, task_num)
        out = self.bn2s[task_num](out)

        if self.downsample is not None:
            identity = self.downsample(x, task_num)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, task_sizes, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(task_sizes, inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(task_sizes, width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(task_sizes, width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, task_num):
        identity = x

        out = self.conv1(x,task_num)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out, task_num)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out, task_num)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x, task_num)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, task_sizes, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = conv7x7(task_sizes, 3, self.inplanes, stride=2)
        # self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
        #                        bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(task_sizes, block, 64, layers[0])
        self.layer2 = self._make_layer(task_sizes, block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(task_sizes, block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(task_sizes, block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, task_sizes, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = TaskSequential(
                conv1x1(task_sizes, self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(task_sizes, self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        self.feat_dim = 512 * block.expansion
        for _ in range(1, blocks):
            layers.append(block(task_sizes, self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return TaskSequential(*layers)


    def set_task_partitions(self, task_sizes):
        self.task_sizes = task_sizes
        self.linear_layers = nn.ModuleList()
        for i in range(len(task_sizes)):
            self.linear_layers.append(torch.nn.Linear(self.feat_dim, task_sizes[i]))

    def set_device(self,device):
        for layer in self.linear_layers.values():
            layer.to(device)
        
    def getIndicators(self):
        indicators = []
        for i in self.named_parameters():
            if 'indicator' in i[0]:
                indicators.append(i[1])
        return indicators
    
    def getTaskIndicators(self, j):
        indicators = []
        for i in self.named_parameters():
            if 'branch_indicator_' + str(j) in i[0]:
                indicators.append(i[1])
        return indicators

    def getAllTaskIndicators(self):
        indicators = []
        for i in self.named_parameters():
            if 'branch_indicator_' in i[0]:
                indicators.append(i[1])
        return indicators

    def _forward_impl(self, x, task_num):
        # See note [TorchScript super()]
        x = self.conv1(x, task_num)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x, task_num)
        x = self.layer2(x, task_num)
        x = self.layer3(x, task_num)
        x = self.layer4(x, task_num)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        #x = self.fc(x)
        fc = self.linear_layers[task_num]
        x = fc(x)

        return x

    def forward(self, x, task_num):
        return self._forward_impl(x, task_num)


def _resnet(task_sizes, arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(task_sizes, block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)



def resnet34(task_sizes, pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    print(progress)
    return _resnet(task_sizes, 'resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)



def resnet50(task_sizes, pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(task_sizes, 'resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)



def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                **kwargs)



def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                **kwargs)



def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                pretrained, progress, **kwargs)



def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                pretrained, progress, **kwargs)



def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)



def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)

class TaskSequential(nn.Sequential):
    def __init__(self, *args):
        super(TaskSequential, self).__init__(*args)

    def forward(self, input, task_num):
        for module in self:
            if isinstance(module, nn.BatchNorm2d):
                input = module(input)
            else:
                input = module(input, task_num)
        return input