import torch
import torch.nn as nn
import torch.nn.functional as F 
import math
import numpy as np


# 1d convolution wrapper

def Conv1d(in_channels, out_channels, kernel_size, dropout=0, **kwargs):
	m = nn.Conv1d(in_channels, out_channels, kernel_size, **kwargs)
	nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
	if m.bias is not None:
		nn.init.constant_(m.bias, 0)
	return nn.utils.weight_norm(m)


# 1x1 1d convolution wrapper
def Conv1d1x1(in_channels, out_channels, bias=True):
	return Conv1d(in_channels, out_channels, kernel_size=1, padding=0,
				  dilation=1, bias=bias)

class NoiseInjection(nn.Module):
	def __init__(self):
		super().__init__()

		self.weight = nn.Parameter(torch.zeros(1))

	def forward(self, input):
		batch, _, length = input.shape
		noise = input.new_empty(batch, 1, length).normal_()

		return input + self.weight * noise

		
## linear style modulation layer
class EqualLinear(nn.Module):
	def __init__(self, in_dim, out_dim, bias_init=0, lr_mul=1, activation=False):
		super().__init__()

		self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

		self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

		self.scale = (1 / math.sqrt(in_dim)) * lr_mul
		self.lr_mul = lr_mul

		self.activation = activation

	def forward(self, input):
		out = F.linear(input, self.weight * self.scale, bias=self.bias * self.lr_mul)
		if self.activation:
			out = F.leaky_relu(out, .2)

		return out

class EqualConv1d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=0, dilation=1, bias=True):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_channel, in_channel, kernel_size))
        self.scale = 1 / math.sqrt(in_channel * kernel_size)

        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = F.conv1d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation
        )

        return out


# normal styled 1d convolution
class StyledConv1d(nn.Module):
	def __init__(self, in_channel, out_channel, style_dim, kernel_size, dilation):
		super().__init__()

		self.eps = 1e-8
		self.kernel_size = kernel_size
		self.in_channel = in_channel
		self.out_channel = out_channel

		fan_in = in_channel * kernel_size ** 2
		self.scale = 1 / math.sqrt(fan_in)
		self.padding = (kernel_size - 1) // 2 * dilation
		self.dilation = dilation 

		self.weight = nn.Parameter(torch.randn(1, out_channel, in_channel, kernel_size))

		self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)
		
	def forward(self, inputs, style):
		batch, in_channel, length = inputs.shape

		# modulate weights with style
		style = self.modulation(style).view(batch, 1, in_channel, 1)
		weight = self.scale * self.weight * style

		# demod weights
		demod = torch.rsqrt(weight.pow(2).sum([2, 3]) + 1e-8)
		weight = weight * demod.view(batch, self.out_channel, 1, 1)

		weight = weight.view(batch * self.out_channel, in_channel, self.kernel_size)

		# apply convolution
		inputs = inputs.view(1, batch * in_channel, length)
		out = F.conv1d(inputs, weight, padding=self.padding, dilation=self.dilation, groups=batch)
		_, _, length = out.shape
		out = out.view(batch, self.out_channel, length)

		return out

class StyledConvLayer(nn.Module):
	def __init__(self, resid_channels, gate_channels, style_dim, kernel_size, dilation, noise=False):
		super().__init__()

		# 1d styled convolution
		self.conv = StyledConv1d(resid_channels, gate_channels, style_dim, kernel_size, dilation)

		# output convolution
		self.conv1x1_out = Conv1d1x1(gate_channels // 2, resid_channels)

		# noise

		self.noise = noise

		if self.noise:
			self.noise_injection = NoiseInjection()

	def forward(self, x, style):
		residual = x

		x = self.conv(x, style)

		splitdim = 1
		xa, xb = x.split(x.size(splitdim) // 2, dim=splitdim)

		x = torch.tanh(xa) * torch.sigmoid(xb)
		if self.noise:
			x = self.noise_injection(x)

		# for residual connection
		x = (self.conv1x1_out(x) + residual) * math.sqrt(0.5)

		return x

class ResConvLayer(nn.Module):
	def __init__(self, resid_channels, gate_channels, kernel_size, dilation):
		super().__init__()

		# 1d styled convolution

		self.conv = EqualConv1d(resid_channels, gate_channels, kernel_size=kernel_size, dilation=dilation, padding=(kernel_size - 1) // 2 * dilation)

		# output convolution
		self.conv1x1_out = Conv1d1x1(gate_channels // 2, resid_channels)

	def forward(self, x):
		residual = x

		x = self.conv(x)

		splitdim = 1
		xa, xb = x.split(x.size(splitdim) // 2, dim=splitdim)

		x = torch.tanh(xa) * torch.sigmoid(xb)

		# for residual connection
		x = (self.conv1x1_out(x) + residual) * math.sqrt(0.5)

		return x

class DownConvLayer(nn.Module):
	def __init__(self, resid_channels, gate_channels):
		super().__init__()
		# 1d styled convolution
		self.conv = EqualConv1d(resid_channels, gate_channels, kernel_size=3, stride=2, padding=1)

		# output convolution
		self.conv1x1_out = Conv1d1x1(gate_channels // 2, resid_channels)
	def forward(self, inputs):
		x = self.conv(inputs)

		splitdim = 1
		xa, xb = x.split(x.size(splitdim) // 2, dim=splitdim)

		x = torch.tanh(xa) * torch.sigmoid(xb)

		x = self.conv1x1_out(x)* math.sqrt(0.5)


		return x


class UpConvLayer(nn.Module):
	def __init__(self, resid_channels, gate_channels):
		super().__init__()
		# 1d styled convolution
		self.conv = nn.ConvTranspose1d(resid_channels, gate_channels, stride=2, kernel_size=4, padding=1)

		# output convolution
		self.conv1x1_out = Conv1d1x1(gate_channels // 2, resid_channels)
	def forward(self, inputs):

		x = self.conv(inputs)

		splitdim = 1
		xa, xb = x.split(x.size(splitdim) // 2, dim=splitdim)

		x = torch.tanh(xa) * torch.sigmoid(xb)

		x = self.conv1x1_out(x)* math.sqrt(0.5)


		return x


## descriminator block
class WavenetBlock(nn.Module):
	def __init__(self, resid_channels = 512, out_channels = 512, gate_channels = 512, kernel_size = 3, n_layers = 10, dilation_factor = 2):
		super().__init__()


		#### input = (B, resid_channels, T)
		#### output = (B, out_channels, T*4)
		
		self.conv_layers = nn.ModuleList()

		for i in range(n_layers):
			# add residual layer
			dilation = dilation_factor ** i
			self.conv_layers.append(ResConvLayer(resid_channels, gate_channels, kernel_size, dilation))
			if i == 0:
				# if first layer, upsample
				self.conv_layers.append(DownConvLayer(resid_channels, gate_channels))
				self.conv_layers.append(ResConvLayer(resid_channels, gate_channels, kernel_size, dilation))
				self.conv_layers.append(DownConvLayer(resid_channels, gate_channels))

		if resid_channels != out_channels:
			self.conv_layers.append(Conv1d1x1(resid_channels, out_channels))

	def forward(self, x):
		for layer in self.conv_layers:
			x = layer(x)

		return x


## generator block
class StyledWavenetBlock(nn.Module):
	def __init__(self, resid_channels = 512, out_channels = 512, gate_channels = 512, style_dim = 512, kernel_size = 3, n_layers = 10, dilation_factor = 2, noise=False):
		super().__init__()


		#### input = (B, resid_channels, T)
		#### output = (B, out_channels, T*4)
		
		self.conv_layers = nn.ModuleList()

		for i in range(n_layers):
			# add residual layer
			dilation = dilation_factor ** i
			self.conv_layers.append(StyledConvLayer(resid_channels, gate_channels, style_dim, kernel_size, dilation, noise))
			if i == 0:
				# if first layer, upsample
				self.conv_layers.append(UpConvLayer(resid_channels, gate_channels))
				self.conv_layers.append(StyledConvLayer(resid_channels, gate_channels, style_dim, kernel_size, dilation, noise))
				self.conv_layers.append(UpConvLayer(resid_channels, gate_channels))

		if resid_channels != out_channels:
			self.conv_layers.append(Conv1d1x1(resid_channels, out_channels))

	def forward(self, x, style):

		for layer in self.conv_layers:
			if isinstance(layer, StyledConvLayer):
				x = layer(x, style)
			else:
				x = layer(x)

		return x
