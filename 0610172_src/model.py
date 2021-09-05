import math
import random
import functools
import operator

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function
class Generator(nn.Module):
    def __init__(
        self,
        size,
        style_dim,
        n_mlp,
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
        lr_mlp=0.01,
    ):
        super().__init__()

        self.size = size
        self.style_dim = style_dim
        layers = []
        layers.append(Sqrt_Reciprocal())

        layers.append(EqualLinear( style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'))
        layers.append(EqualLinear( style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'))
        layers.append(EqualLinear( style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'))
        layers.append(EqualLinear( style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'))
        layers.append(EqualLinear( style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'))
        layers.append(EqualLinear( style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'))
        layers.append(EqualLinear( style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'))
        layers.append(EqualLinear( style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'))
            

        self.style = nn.Sequential(*layers)

        self.input = ConstantInput(512)
        # self.conv1 = StyledConv(512, 512, 3, style_dim, blur_kernel=blur_kernel)
        self.conv1 = StyledConv(512, 512, 3, style_dim)
        self.to_rgb1 = ToRGB(512, style_dim, upsample=False)

        self.log_size = 7
        self.num_layers = 11

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        in_channel = 512
        self.noises.register_buffer(f'noise_{0}', torch.randn(*[1, 1, 4, 4]))
        self.noises.register_buffer(f'noise_{1}', torch.randn(*[1, 1, 8, 8]))
        self.noises.register_buffer(f'noise_{2}', torch.randn(*[1, 1, 8, 8]))
        self.noises.register_buffer(f'noise_{3}', torch.randn(*[1, 1, 16, 16]))
        self.noises.register_buffer(f'noise_{4}', torch.randn(*[1, 1, 16, 16]))
        self.noises.register_buffer(f'noise_{5}', torch.randn(*[1, 1, 32, 32]))
        self.noises.register_buffer(f'noise_{6}', torch.randn(*[1, 1, 32, 32]))
        self.noises.register_buffer(f'noise_{7}', torch.randn(*[1, 1, 64, 64]))
        self.noises.register_buffer(f'noise_{8}', torch.randn(*[1, 1, 64, 64]))
        self.noises.register_buffer(f'noise_{9}', torch.randn(*[1, 1, 128, 128]))
        self.noises.register_buffer(f'noise_{10}', torch.randn(*[1, 1, 128, 128]))
        self.noises.register_buffer(f'noise_{11}', torch.randn(*[1, 1, 256, 256]))

        self.convs.append(StyledConv(512,512,3,512,upsample=True))
        self.convs.append(StyledConv(512,512,3,512,upsample=False))
        self.to_rgbs.append(ToRGB(512, 512))
        self.convs.append(StyledConv(512,512,3,512,upsample=True))
        self.convs.append(StyledConv(512,512,3,512,upsample=False))
        self.to_rgbs.append(ToRGB(512 , 512))
        self.convs.append(StyledConv(512,512,3,512,upsample=True))
        self.convs.append(StyledConv(512,512,3,512,upsample=False))
        self.to_rgbs.append(ToRGB(512 , 512))
        self.convs.append(StyledConv(512,512,3,512,upsample=True))
        self.convs.append(StyledConv(512,512,3,512,upsample=False))
        self.to_rgbs.append(ToRGB(512 , 512))
        self.convs.append(StyledConv(512,256,3,512,upsample=True))
        self.convs.append(StyledConv(256,256,3,512,upsample=False))
        self.to_rgbs.append(ToRGB(256, 512))

        self.n_latent = 12

    def make_noise(self):
        noises = []
        noises.append(torch.randn(1, 1, 4, 4, device='cuda',requires_grad = True))
        noises.append(torch.randn(1, 1, 8, 8, device='cuda',requires_grad = True))
        noises.append(torch.randn(1, 1, 8, 8, device='cuda',requires_grad = True))
        noises.append(torch.randn(1, 1, 16, 16, device='cuda',requires_grad = True))
        noises.append(torch.randn(1, 1, 16, 16, device='cuda',requires_grad = True))
        noises.append(torch.randn(1, 1, 32, 32, device='cuda',requires_grad = True))
        noises.append(torch.randn(1, 1, 32, 32, device='cuda',requires_grad = True))
        noises.append(torch.randn(1, 1, 64, 64, device='cuda',requires_grad = True))
        noises.append(torch.randn(1, 1, 64, 64, device='cuda',requires_grad = True))
        noises.append(torch.randn(1, 1, 128, 128, device='cuda',requires_grad = True))
        noises.append(torch.randn(1, 1, 128, 128, device='cuda',requires_grad = True))
        return noises


    def get_latent(self, input):
        return self.style(input)

    def forward(
        self,
        styles,
        return_latents=False,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
        noise=None,
        randomize_noise=True,
    ):
        print('dhhdhdddddddddddddd=========================================================')
        if not input_is_latent:
            styles = [self.style(s) for s in styles]

        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [
                    getattr(self.noises, f'noise_{i}') for i in range(self.num_layers)
                ]


        latent = styles[0].unsqueeze(1).repeat(1, 12, 1)
        out = self.input(latent)
        out = self.conv1(out, latent[:, 0], noise=noise[0])
        skip = self.to_rgb1(out, latent[:, 1])

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(
            self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
        ):
            out = conv1(out, latent[:, i], noise=noise1)
            out = conv2(out, latent[:, i + 1], noise=noise2)
            skip = to_rgb(out, latent[:, i + 2], skip)

            i += 2

        image = skip
        if return_latents:
            return image, latent

        else:
            return image, None
class Discriminator(nn.Module):
    def __init__(self, size, channel_multiplier=2, blur_kernel=[1, 3, 3, 1]):
        super().__init__()
        convs = []
        convs.append(ConvLayer(3, 256, 1))
        convs.append(ResBlock(256, 512))
        convs.append(ResBlock(512, 512))
        convs.append(ResBlock(512, 512))
        convs.append(ResBlock(512, 512))
        convs.append(ResBlock(512, 512))

        self.convs = nn.Sequential(*convs)


        self.final_conv = ConvLayer(513, 512, 3)
        self.final_linear = nn.Sequential(
            nn.Linear(512 * 4 * 4, 512),
            nn.Linear(512, 1),
            
        )

    def forward(self, input):
        out = self.convs(input)

        B, C, H, W = out.shape
        stddev = out.reshape(4 ,1 ,1 ,512 ,4 ,4) # 4 X 1 X 1 X 512 X 4 X 4
        
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 0.0000001)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(4, 1, H, W)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)
        
        out = out.reshape(B, -1) # 4,8192
        out = self.final_linear(out)

        return out


def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):

    out = upfirdn2d_native(input, kernel, up, up, down, down, pad[0], pad[1], pad[0], pad[1])
    return out
# class MMConv2d_Upsample(nn.Module):
#     def __init__(
#         self,
#         channel_input,
#         channel_output,
#         filter_sze,
#         latent
        
#     ):
#         super().__init__()
#         self.xx = torch.randn(filter_sze, channel_input)
#         self.yy = torch.randn(filter_sze, channel_output)
#         self.channel_input = channel_input
#         self.channel_output = channel_output

#         self.filter_sze = filter_sze
#         self.MMM =torch.nn.Sequential(
#             torch.nn.Linear(channel_input, latent),
#             torch.nn.ReLU(),
#             torch.nn.Linear(latent, channel_output),
#         )
#         self.blur = UpsampleFunc([1, 3, 3, 1], pad=(int((4-filter_sze)/2)+1, int((3-filter_sze)/2)+1), upsample_factor=2, function = 'none')
#         fan_in = 512 * (filter_sze ** 2)
#         self.scale = 1 / math.sqrt(fan_in)
        
#         self.relu1 = torch.nn.ReLU()

#         self.parameter = nn.Parameter(torch.randn(1, channel_output,512, filter_sze, filter_sze),  requires_grad = True)
#         self.fc = nn.Linear(latent, 512, bias=True)

#     def forward(self, input, style):
    
#         S = self.filter_sze
#         CHOUT = self.channel_output
#         B, I, H, W = input.shape
#         input = input.reshape(1, B * I, H, W)
        
#         parameter = self.fc(style).reshape(B, 1, I, 1, 1) * self.scale * self.parameter
#         sqrt_torch = torch.sqrt(parameter.pow(2).sum([2, 3, 4]) + 0.0000001)# adding 0.0000001 to avoid denominator being 0
#         parameter = (parameter * torch.reciprocal(sqrt_torch).reshape(B, CHOUT, 1, 1, 1)).reshape(B * CHOUT, I, S, S)
#         parameter = parameter.reshape(B, CHOUT, I, S, S).transpose(1, 2).reshape(B * I, CHOUT, S, S)
        
#         out = F.conv_transpose2d(input, parameter, padding=0, stride=2, groups=B)
#         __, __, H, W = out.shape
#         out = out.reshape(B, CHOUT, H, W)
#         out = self.blur(out)


#         return out

    



# class Convolution(nn.Module):
#     def __init__(
#         self,
#         channel_input,
#         channel_output,
#         filter_sze,
#         latent,
#         upsample=False,
#     ):
#         super().__init__()
#         self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
#         if upsample == True:
#             self.conv = MMConv2d_Upsample(
#                 channel_input,
#                 channel_output,
#                 filter_sze,
#                 latent
#             )
#         else:
#             self.conv = MMConv2d(
#                 channel_input,
#                 channel_output,
#                 filter_sze,
#                 latent
                
#             )
    
#         self.noise = nn.Parameter(torch.zeros(1), requires_grad = True)
#         self.classifier = nn.Sequential(
#             nn.Linear(512 * 7 * 7, 4096),
#             nn.ReLU(True),
#             nn.Dropout(),
#             nn.Linear(4096, 4096),
#             nn.ReLU(True),
#             nn.Dropout(),
#             nn.Linear(4096, 2),
#         )
#         self.activate = torch.nn.LeakyReLU(0.2)

#     def forward(self, input, style, noise=None):
#         out = self.conv(input, style)
#         if noise is None:
#             B, _, H, W = out.shape
#             tet = torch.flatten(out, 1)
            
#             noise = out.new_empty(B, 1, H, W).normal_()

#         out = out + self.noise * noise
#         out = self.activate(out)

#         return out

def upfirdn2d_native(
    input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1
):
    input = input.permute(0, 2, 3, 1)
    _, in_h, in_w, minor = input.shape
    kernel_h, kernel_w = kernel.shape
    out = input.reshape(-1, in_h, 1, in_w, 1, minor)
    out = F.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])
    WX = in_w * up_x
    HY = in_h * up_y
    out = out.reshape(-1, HY, WX, minor)
    MMpad_x0 = max(pad_x0, 0)
    MMpad_x1 = max(pad_x1, 0)
    MMpad_y0 = max(pad_y0, 0)
    MMpad_y1 = max(pad_y1, 0)
    out = F.pad(out, [0, 0, MMpad_x0, MMpad_x1, MMpad_y0, MMpad_y1])
    out = out[
        :,
        min(pad_y0, 0) : out.shape[1] - min(pad_y1, 0),
        min(pad_x0, 0) : out.shape[2] - min(pad_x1, 0),
        :,
    ]
    YY = pad_y0 + pad_y1
    XX =  pad_x0 + pad_x1
    out = out.permute(0, 3, 1, 2)
    out = out.reshape(
        [-1, 1, HY + YY,WX + XX]
    )
    w = torch.flip(kernel, [0, 1]).reshape(1, 1, kernel_h, kernel_w)
    out = F.conv2d(out, w)
    out = out.reshape(
        -1,
        minor,
        HY + YY - kernel_h + 1,
        WX +XX - kernel_w + 1,
    )

    return out[:, :, ::down_y, ::down_x]    
def SampleB(input, filter, x1, y1, x2, y2, pad_x0, pad_x1, pad_y0, pad_y1):

    out = upfirdn2d_native(input, filter,x1, y1, x2, y2, pad_x0, pad_x1, pad_y0, pad_y1)
    return out
class UpsampleFunc(nn.Module):
    def __init__(self, filter, pad, upsample_factor=1, function = 'upsample'):
        super().__init__()

        filter = torch.tensor(filter, dtype=torch.float32)
        if filter.ndim == 1:
            filter = filter[None, :] * filter[:, None]
        filter = filter/filter.sum()
        if function ==  'upsample':
            filter = filter * 4
            pad = (2,1)


        if function !='upsample' and upsample_factor > 1:
            filter = filter * (upsample_factor ** 2)

        self.register_buffer('filter', filter)
        self.function = function
        self.pad = pad

    def forward(self, input):
        if self.function == 'upsample':
            out = SampleB(input,self.filter, x1=2,  y1=2, x2=1, y2=1, pad_x0=self.pad[0], pad_x1=self.pad[1],  pad_y0=self.pad[0], pad_y1=self.pad[1])
        else:
            out = SampleB(input,self.filter, x1=1,  y1=1, x2=1, y2=1, pad_x0=self.pad[0], pad_x1=self.pad[1],  pad_y0=self.pad[0], pad_y1=self.pad[1])

        return out
class FusedLeakyReLU(nn.Module):
    def __init__(self, channel, negative_slope=0.2, scale=1.41421356237):
        super().__init__()

        self.bias = nn.Parameter(torch.zeros(channel))
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input):
        return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)
def fused_leaky_relu(input, bias, negative_slope=0.2, scale=1.414):
    return scale * F.leaky_relu(input + bias.reshape((1, -1)+(1,)*(len(input.shape)-2)), negative_slope=negative_slope)
    
class Sqrt_Reciprocal(nn.Module):#take sqrt and reciprocal of tensor
    def __init__(self):
        super().__init__()

    def forward(self, input):
        mean_torch = torch.mean(input ** 2, dim=1, keepdim=True)
        sqrt_torch = torch.sqrt(mean_torch + 1e-8)# adding 0.0000001 to avoid denominator being 0
        reci_torch = torch.reciprocal(sqrt_torch)
        return input * reci_torch
# class MMConv2d(nn.Module):
#     def __init__(
#         self,
#         channel_input,
#         channel_output,
#         filter_sze,
#         latent
        
#     ):
#         super().__init__()
#         self.xx = torch.randn(filter_sze, channel_input)
#         self.yy = torch.randn(filter_sze, channel_output)

        
#         self.CON = nn.Conv2d(channel_input, channel_output, kernel_size=3, padding=1)
#         self.channel_input = channel_input
#         self.channel_output = channel_output
#         self.MMM =torch.nn.Sequential(
#             torch.nn.Linear(channel_input, latent),
#             torch.nn.ReLU(),
#             torch.nn.Linear(latent, channel_output),
#         )
#         self.filter_sze = filter_sze
#         self.scale = 1 / math.sqrt(channel_input * filter_sze ** 2)
        
#         self.classifier = nn.Sequential(
#             nn.Linear(128 * 5 * 5, 4096),
#             nn.ReLU(True),
#             nn.Dropout(),
#             nn.Linear(4096, 4096),
#             nn.ReLU(True),
#             nn.Dropout(),
#             nn.Linear(4096, 2),
#         )
#         self.relu1 = torch.nn.ReLU()
#         self.padding = int(filter_sze/2)
#         self.parameter = nn.Parameter(torch.randn(1, channel_output,channel_input, filter_sze, filter_sze),  requires_grad = True)
#         self.fc = nn.Linear(latent, channel_input, bias = True)


#     def forward(self, input, style):
    
#         CHOUT = self.channel_output
#         B, C, H, W = input.shape
        
#         style = self.fc(style).reshape(B, 1, C, 1, 1)
#         parameter = self.scale * self.parameter * style

            
#         sqrt_torch = torch.sqrt(parameter.pow(2).sum([2, 3, 4]) + 0.0000001)# adding 0.0000001 to avoid denominator being 0
#         demod = torch.reciprocal(sqrt_torch)
#         parameter = parameter * demod.reshape(B, CHOUT, 1, 1, 1)

#         parameter = parameter.reshape(B * CHOUT, C, self.filter_sze, self.filter_sze)

 
#         input = input.reshape(1, B * C, H, W)
#         out = F.conv2d(input, parameter, padding=self.padding, groups=B)
        
#         _, _, H, W = out.shape
#         out = out.reshape(B, CHOUT, H, W)

#         return out






class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()
        kernel = torch.tensor(kernel, dtype=torch.float32)

        if kernel.ndim == 1:
            kernel = kernel[None, :] * kernel[:, None]

        kernel /= kernel.sum()

        return kernel

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer('kernel', kernel)

        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)

        return out

class EqualConv2d(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_channel, in_channel, kernel_size, kernel_size))

        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = F.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

class EqualLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation
        self.tmp = (1 / math.sqrt(in_dim))
        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)

        else:
            out = F.linear(input, self.weight * self.scale, bias=self.bias * self.lr_mul)

        return out


class ScaledLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()
        self.negative_slope = 0.2

    def forward(self, input):
        out = F.leaky_relu(input, negative_slope=self.negative_slope)

        return out /0.70710678118


class ModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        demodulate=True,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample

        if upsample:
            p = 3- kernel_size 

            self.blur = UpsampleFunc(blur_kernel, pad=(int((p + 1)/2) + 1, int(p/2)+ 1), upsample_factor=2,function = 'none')

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = int(kernel_size/2)

        self.weight = nn.Parameter(torch.randn(1, out_channel, in_channel, kernel_size, kernel_size))
        #self.concon = nn.conv2d()
        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)
        self.demodulate = demodulate



    def forward(self, input, style):
        batch, in_channel, height, width = input.shape
        KK=self.kernel_size
        style = self.modulation(style).reshape(batch, 1, in_channel, 1, 1)
        weight = self.scale * self.weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.reshape(batch, self.out_channel, 1, 1, 1)

        weight = weight.reshape( batch * self.out_channel, in_channel, KK, KK)
        input = input.reshape(1, batch * in_channel, height, width)
        out = F.conv2d(input, weight, padding=self.padding, groups=batch)
        _, _, height, width = out.shape
        out = out.reshape(batch, self.out_channel, height, width)

        if self.upsample:
            input = input.reshape(1, batch * in_channel, height, width)
            weight = weight.reshape(batch, self.out_channel, in_channel, KK, KK)
            weight = weight.transpose(1, 2).reshape(batch * in_channel, self.out_channel, KK, KK)
            out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, HH, width = out.shape
            out = out.reshape(batch, self.out_channel, HH, width)
            out = self.blur(out)

        return out


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()

        return image + self.weight * noise


class ConstantInput(nn.Module):
    def __init__(self, channel=512, size=4):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out

class StyledConv(nn.Module):
    def __init__(
        self,
        channel_input,
        channel_output,
        filter_sze,
        latent,
        upsample=False,
    ):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        if upsample == True:
            self.conv = ModulatedConv2d(
                channel_input,
                channel_output,
                filter_sze,
                latent,
                upsample = True
            )
        else:
            self.conv = ModulatedConv2d(
                channel_input,
                channel_output,
                filter_sze,
                latent,
                upsample = False
                
            )

        self.noise =NoiseInjection()
#         self.activate = torch.nn.LeakyReLU(0.2)
        self.tptp = torch.nn.LeakyReLU(0.2)
        self.activate =FusedLeakyReLU(channel_output)

    def forward(self, input, style, noise=None):
        
        out = self.conv(input, style)
        out = self.noise(out, noise=noise)
        # out = out + self.bias
        out = self.activate(out)


        return out

class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        if upsample:
            self.upsample = UpsampleFunc([1, 3, 3, 1], pad = (2,1), function = 'upsample')
        self.conv = ModulatedConv2d(in_channel, 3, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)

            out = out + skip

        return out
class ConvLayer(nn.Sequential):
    def __init__(
        self,
        channel_input,
        channel_output,
        filter_sze,
        downsample=False,
        bias=True,
        activate=True,
    ):
        layers = []

        if downsample:
            stride = 2
            self.padding = 0

            if filter_sze == 3: 
                layers.append( UpsampleFunc([1, 3, 3, 1], pad=(2,2), function = 'none') )
            else:
                layers.append( UpsampleFunc([1, 3, 3, 1], pad=(1,2), function = 'none') )

        else:
            stride = 1
            self.padding = int(filter_sze/2)

        layers.append(
            EqualConv2d(
                channel_input,
                channel_output,
                filter_sze,
                padding=self.padding,
                stride=stride,
                bias=bias and not activate,
            )
        )
        
        if activate:
            if bias:
                layers.append(FusedLeakyReLU(channel_output))

            else:
                layers.append(ScaledLeakyReLU(0.2))


        super().__init__(*layers)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3, downsample=False,bias=True,activate=True)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True,bias=True,activate=True)
        self.skip = ConvLayer(in_channel, out_channel, 1, downsample=True, activate=False, bias=False)

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)*0.70710678118

        skip = self.skip(input)*0.70710678118
        out = (out + skip) 

        return out



