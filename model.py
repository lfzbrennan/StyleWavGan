import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import math
import numpy as np

from layers import *


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()
        self.channel = channel
        self.size = size

        self.input = nn.Parameter(torch.randn(1, self.channel, self.size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1)

        return out

class Generator(nn.Module):
	def __init__(self, style_dim=256, gate_channels=512):
		super().__init__()

		# learning input embedding = 64x1024

		self.input_channels = 1024
		self.input_length = 64
		self.output_channels = 256

		self.gate_channels = gate_channels
		self.style_dim = style_dim

		self.n_mlp = 8
		self.lr_mlp = .01


		# learned constant input embedding
		self.input = ConstantInput(self.input_channels, self.input_length)

		self.mlp_layers = nn.ModuleList()

		for _ in range(self.n_mlp):
			self.mlp_layers.append(EqualLinear(self.style_dim, self.style_dim, self.lr_mlp, activation=True))

		self.style = nn.Sequential(*self.mlp_layers)


		# (n_layers, output_channel)
		self.block_map = [
			(5, 1024, True),  
			(7, 512, True), 
			(9, 512, True),
			(11, 512, True),
			(13, 256, False)
		]

		# base wavenet blocks
		self.wavenet_blocks = nn.ModuleList()

		for i, (layers, out_channels, noise) in enumerate(self.block_map):
			if i == 0:
				in_channels = self.input_channels
			else:
				in_channels = self.block_map[i-1][1]

			self.wavenet_blocks.append(StyledWavenetBlock(in_channels, out_channels, self.gate_channels, self.style_dim, n_layers=layers, noise=noise))

		self.output_linear = Conv1d1x1(self.block_map[-1][1], 1)

		# output layer -> 256 softmax in this case
		self.output = nn.Tanh()


	def forward(self, style):

		# if mixing styles, style is a list (used for inference)
		# if using one style, style is a single tensor (for training)
		# edit to accomodate this

		latents = [self.style(s) for s in style]
		#latents = self.style(style)
		
		# push through constant learned layer
		out = self.input(latents[0])

		# push through wavenet blocks
		for i, block in enumerate(self.wavenet_blocks):
			out = block(out, latents[i % len(latents)])
			print(out.shape)


		out = output_linear(out)


		# get output function and return output
		return self.output(out), latents




class Descriminator(nn.Module):
	def __init__(self, gate_channels=512):
		super().__init__()

		# learning input embedding = 64x1024

		self.input_channels = 256
		self.output_channels = 64

		self.gate_channels = gate_channels

		# (n_layers, output_channel)
		self.block_map = [
			(13, 256),
			(11, 512),
			(9, 512),
			(7, 512), 
			(5, 1024) 
			
		]

		self.input_linear = Conv1d1x1(1, self.block_map[0][1])

		# base wavenet blocks
		self.wavenet_blocks = nn.ModuleList()

		for i, (layers, out_channels) in enumerate(self.block_map):
			if i == 0:
				in_channels = self.input_channels
			else:
				in_channels = self.block_map[i-1][1]

			self.wavenet_blocks.append(WavenetBlock(in_channels, out_channels, self.gate_channels, n_layers=layers))

		self.final_layer = nn.Sequential(
			EqualLinear(self.output_channels * 1024, 1024, activation=True),
			EqualLinear(1024, 1)
			) 


	def forward(self, audio):

		audio = self.input_linear(block)
		# push through wavenet blocks
		for i, block in enumerate(self.wavenet_blocks):
			audio = block(audio)
			print(audio.shape)


		out = audio.view(audio.shape[0], -1)


		out = self.final_layer(out)

		return out