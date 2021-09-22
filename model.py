import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import quantize
from layers import WavenetBlock, StyledWavenetBlock, EqualLinear, Conv1d1x1

# constant learned input (used instead of variable input for classic GANs)
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

# generates an audio sample
class Generator(nn.Module):
	def __init__(self, style_dim, layers, channel_mult, input_length):
		super().__init__()

		# (n_layers, output_channel, noise)

		''' block map with default settings:
		 	- each layer increases length by 4x
				* if input_length = 2 ** 17, then input_channels is 2 ** 7
		self.block_map = [
			(5, 256, True),
			(7, 128, True),
			(9, 64, True),
			(11, 32, True),
			(13, 16, False)
		]'''

		self.layers = layers
		self.channel_mult = channel_mult

		self.block_map = [
		(5 + 2 * i, int(self.channel_mult * 16 * 2 ** (self.layers - i - 1)), False if i == self.layers - 1 else True)
		for i in range(self.layers)
		]

		# learning input embedding = 64x1024
		self.input_channels = self.block_map[0][1]
		self.input_length = input_length // (4 ** len(self.block_map))

		self.style_dim = style_dim

		self.n_mlp = 4
		self.lr_mlp = .01

		# learned constant input embedding
		self.input = ConstantInput(self.input_channels, self.input_length)

		self.mlp_layers = nn.ModuleList()

		for _ in range(self.n_mlp):
			self.mlp_layers.append(EqualLinear(self.style_dim, self.style_dim, self.lr_mlp, activation=True))

		self.style = nn.Sequential(*self.mlp_layers)

		# base wavenet blocks
		self.wavenet_blocks = nn.ModuleList()

		for i, (layers, out_channels, noise) in enumerate(self.block_map):
			if i == 0:
				in_channels = self.input_channels
			else:
				in_channels = self.block_map[i-1][1]

			self.wavenet_blocks.append(StyledWavenetBlock(in_channels, out_channels, self.style_dim, n_layers=layers, noise=noise))

		self.output_linear = Conv1d1x1(self.block_map[-1][1], 1)
		# output layer -> [-1, 1] tanh
		self.output = nn.Tanh()

		'''
		different output layers ->>>> this one uses smooth tanh but can be adapted
		as needed
		if self.output == "tanh":
			### tanh output
			self.output_linear = Conv1d1x1(self.block_map[-1][1], 1)
			# output layer -> [-1, 1] tanh
			self.output = nn.Tanh()


		elif self.output == "softmax":
			### softmax output

			self.output_linear = Conv1d1x1(self.block_map[-1][1], 256)
			# output layer -> [0, 255] softmax
			self.output = nn.Softmax()
		'''


	def forward(self, style):

		# if mixing styles, style is a list (used for inference)
		# if using one style, style is a single tensor (for training)
		# edit to accomodate this

		# assuming multiple styles
		#latents = [self.style(s) for s in style]

		# assuming single style
		latents = self.style(style)

		# push through constant learned layer
		out = self.input(latents)

		# push through wavenet blocks
		for i, block in enumerate(self.wavenet_blocks):
			# single styles
			out = block(out, latents)

			# multiple styles
			#out = block(out, latents[i % len(latents)])


		out = self.output(self.output_linear(out))


		# get output function and return output
		return quantize(out), latents

# descriminator
class Descriminator(nn.Module):
	def __init__(self, layers, channel_mult, input_length):
		super().__init__()

		# (n_layers, output_channel)

		'''block map with default settings:
			- each layer decreases length by 4x
				* if input_length = 2 ** 17, then output_linear_channels is 2 ** 7
		self.block_map = [
			(13, 16),
			(11, 32),
			(9, 64),
			(7, 128),
			(5, 256)
		]'''

		self.layers = layers
		self.channel_mult = channel_mult

		self.block_map = [
		(5 + 2 * (self.layers - i - 1), int(self.channel_mult * 16 * 2 ** i))
		for i in range(self.layers)
		]

		self.input_channels = self.block_map[0][1]
		self.output_linear_channels = input_length // (4 ** len(self.block_map))


		self.input_linear = Conv1d1x1(1, self.input_channels)

		# base wavenet blocks
		self.wavenet_blocks = nn.ModuleList()

		for i, (layers, out_channels) in enumerate(self.block_map):
			if i == 0:
				in_channels = self.input_channels
			else:
				in_channels = self.block_map[i-1][1]

			self.wavenet_blocks.append(WavenetBlock(in_channels, out_channels, n_layers=layers))

		self.final_layer = nn.Sequential(
			nn.LayerNorm(self.block_map[-1][1] * self.output_linear_channels),
			EqualLinear(self.block_map[-1][1] * self.output_linear_channels, 1)
			)

	def forward(self, audio):

		# first block
		audio = F.relu(self.input_linear(audio))
		# push through wavenet blocks
		for i, block in enumerate(self.wavenet_blocks):
			audio = block(audio)

		out = audio.view(audio.shape[0], -1)
		#print(out.shape)

		out = self.final_layer(out)

		return out
