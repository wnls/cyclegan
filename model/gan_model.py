import functools
import numpy as np
import torch
import torch.nn as nn
import torchvision
import itertools
import torch.nn.functional as F

def norm_relu_layer(out_channel, norm, relu):
    if norm == 'batchnorm':
        norm_layer = nn.BatchNorm2d(out_channel)
    elif norm == 'instancenorm':
        norm_layer = nn.InstanceNorm2d(out_channel)
    elif norm is None:
        norm_layer = nn.Dropout2d(0)  # Identity
    else:
        raise Exception("Norm not specified!")

    if relu is None:
        relu_layer = nn.ReLU()
    else:
        relu_layer = nn.LeakyReLU(relu, inplace=True)

    return norm_layer, relu_layer


def Conv_Norm_ReLU(in_channel, out_channel, kernel, padding=0, dilation=1, groups=1, stride=1, bias=True,
                   norm='batchnorm', relu=None):
    """
    Convolutional -- Norm -- ReLU Unit
    :param norm: 'batchnorm' --> use BatchNorm2D, 'instancenorm' --> use InstanceNorm2D, 'none' --> Identity()
    :param relu: None -> Use vanilla ReLU; float --> Use LeakyReLU(relu)

    :input (N x in_channel x H x W)
    :return size same as nn.Conv2D
    """
    norm_layer, relu_layer = norm_relu_layer(out_channel, norm, relu)

    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel, padding=padding, stride=stride,
                  dilation=dilation, groups=groups, bias=bias),
        norm_layer,
        relu_layer
    )


def Deconv_Norm_ReLU(in_channel, out_channel, kernel, padding=0, output_padding=0, stride=1, groups=1,
                     bias=True, dilation=1, norm='batchnorm'):
    """
    Deconvolutional -- Norm -- ReLU Unit
    :param norm: 'batchnorm' --> use BatchNorm2D, 'instancenorm' --> use InstanceNorm2D, 'none' --> Identity()
    :param relu: None -> Use vanilla ReLU; float --> Use LeakyReLU(relu)

    :input (N x in_channel x H x W)
    :return size same as nn.ConvTranspose2D
    """
    norm_layer, relu_layer = norm_relu_layer(out_channel, norm, relu=None)
    return nn.Sequential(
        nn.ConvTranspose2d(in_channel, out_channel, kernel, padding=padding, output_padding=output_padding,
                           stride=stride, groups=groups, bias=bias, dilation=dilation),
        norm_layer,
        relu_layer
    )


class ResidualLayer(nn.Module):
    """
    Residual block used in Johnson's network model:

    Our residual blocks each contain two 3×3 convolutional layers with the same number of filters on both
    layer. We use the residual block design of Gross and Wilber [2] (shown in Figure 1), which differs from
    that of He et al [3] in that the ReLU nonlinearity following the addition is removed; this modified design
    was found in [2] to perform slightly better for image classification.
    """

    def __init__(self, channels, kernel_size, final_relu=False, bias=False, norm='batchnorm'):
        super().__init__()
        self.kernel_size = kernel_size
        self.channels = channels
        self.padding = (self.kernel_size[0] - 1) // 2
        self.final_relu = final_relu

        norm_layer, relu_layer = norm_relu_layer(self.channels, norm, relu=None)
        self.layers = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, self.kernel_size, padding=self.padding, bias=bias),
            norm_layer,
            nn.ReLU(),
            nn.Conv2d(self.channels, self.channels, self.kernel_size, padding=self.padding, bias=bias),
            norm_layer
        )

    def forward(self, input):
        # input (N x channels x H x W)
        # output (N x channels x H x W)
        out = self.layers(input)
        if self.final_relu:
            return nn.ReLU(out + input)
        else:
            return out + input


class Generator(nn.Module):
    def __init__(self, type):
        super().__init__()
        self.type = type


class GeneratorJohnson(Generator):
    """
    The Generator architecture in < Perceptual Losses for Real-Time Style Transfer and Super-Resolution >
    by Justin Johnson, et al.
    """

    def __init__(self, image_channel=3, use_bias=True, norm='instancenorm', n_res_blocks=6):
        super().__init__('Johnson')
        model = []
        model += [Conv_Norm_ReLU(image_channel, 32, (7, 7), padding=3, stride=1, bias=use_bias, norm=norm),  # c7s1-32
                  Conv_Norm_ReLU(32, 64, (3, 3), padding=1, stride=2, bias=use_bias, norm=norm),  # d64
                  Conv_Norm_ReLU(64, 128, (3, 3), padding=1, stride=2, bias=use_bias, norm=norm)]  # d128
        for i in range(n_res_blocks):
            model += [ResidualLayer(128, (3, 3), final_relu=False, bias=use_bias)]  # R128
        model += [Deconv_Norm_ReLU(128, 64, (3, 3), padding=1, output_padding=1, stride=2, bias=use_bias, norm=norm), # u64
                  Deconv_Norm_ReLU(64, 32, (3, 3), padding=1, output_padding=1, stride=2, bias=use_bias, norm=norm), # u32
                  nn.Conv2d(32, 3, (7, 7), padding=3, stride=1, bias=use_bias),  # c7s1-3
                  nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        """
        :param input: (N x channels x H x W)
        :return: output: (N x channels x H x W) with numbers of range [-1, 1] (since we use tanh())
        """
        return self.model(input)

class Discriminator(nn.Module):
    def __init__(self, type):
        self.type = type
        super(Discriminator, self).__init__()


class DiscriminatorPatchGAN(Discriminator):
    """
    The Discriminator Architecture used in < Image-to-Image Translation with Conditional Adversarial
    Networks > by Philip Isola, et al.
    """

    def __init__(self, image_channel=3, kernel_size=4, use_bias=True, norm='instancenorm', sigmoid=False):
        super().__init__('PatchGAN')
        model = []
        model += [Conv_Norm_ReLU(image_channel, 64, kernel_size, padding=1, stride=2, bias=use_bias, relu=0.2, norm=None), # C64
                  Conv_Norm_ReLU(64, 128, kernel_size, padding=1, stride=2, bias=use_bias, relu=0.2, norm=norm), # C128
                  Conv_Norm_ReLU(128, 256, kernel_size, padding=1, stride=2, bias=use_bias, relu=0.2, norm=norm), # C256
                  Conv_Norm_ReLU(256, 512, kernel_size, padding=1, bias=use_bias, relu=0.2, norm=norm), # C512
                  nn.Conv2d(512, 1, kernel_size, padding=1, bias=use_bias)
                  ]
        if sigmoid:
            model += [nn.Sigmoid()]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        """
        :param input: (N x channels x H x W)
        :return: output: (N x channels x H/16 x W/16) of discrimination values
        """
        return self.model(input)


class ResidualBlock2(nn.Module):
    def __init__(self, in_features, norm_layer=nn.InstanceNorm2d):
        super(ResidualBlock2, self).__init__()

        conv_block = [nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      norm_layer(in_features),
                      nn.ReLU(inplace=True),
                      nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      norm_layer(in_features)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class GeneratorJohnson2(Generator):
    """
    Generator with 9 residual blocks and reflection padding.
    """

    def __init__(self, image_channel=3, norm='instancenorm', n_res_blocks=9):
        super().__init__('Johnson')
        if norm == 'batchnorm':
            norm_layer = nn.BatchNorm2d
        elif norm == 'instancenorm':
            norm_layer = nn.InstanceNorm2d
        else:
            raise Exception("Norm not specified!")

        # Downsample
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(image_channel, 64, 7),
                 norm_layer(64),
                 nn.ReLU(inplace=True)]

        in_channels = 64
        out_channels = in_channels * 2
        for i in range(2):
            model += [nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
                      norm_layer(out_channels),
                      nn.ReLU(inplace=True)]
            in_channels = out_channels
            out_channels = in_channels * 2

        # Residual blocks
        for i in range(n_res_blocks):
            model += [ResidualBlock2(in_channels, norm_layer=norm_layer)]

        # Upsample
        out_channels = in_channels // 2
        for i in range(2):
            model += [nn.ConvTranspose2d(in_channels, out_channels, 3, stride=2, padding=1, output_padding=1),
                      norm_layer(out_channels),
                      nn.ReLU(inplace=True)]
            in_channels = out_channels
            out_channels = in_channels // 2

        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(64, 3, 7),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)


    def forward(self, input):
        return self.model(input)



