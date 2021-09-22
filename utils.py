import numpy as np
import torch
import torch.nn as nn
import os
from scipy.io.wavfile import write

# save generator and discrimnator
def save_model(save_dir, g, d):
	if not os.path.exists(save_dir):
		os.mkdir(save_dir)
	torch.save(g.state_dict(), f"{save_dir}/g.pt")
	torch.save(d.state_dict(), f"{save_dir}/d.pt")

# save audio sample
def save_audio_sample(save_dir, data, as_short=False):
	if not os.path.exists(save_dir):
		os.mkdir(save_dir)
	if as_short: # 8 bits
		data = (data + 1) * 128 # 8 bit
		data = data.astype("uint8")
	write(f"{save_dir}/sample.wav", 16000, data.T)

# quantize (-1.0, 1.0) -> (-1.0, 1.0)
def quantize(data, mu=128):

	if mu >= 2147483648: # quantizing to higher than int32
		return data

	return torch.round(data * mu) / mu

def softmax_to_tanh(data):
	sparse_to_value = np.linspace(-1, 1, 256)

	return torch.tensor(sparse_to_value[torch.argmax(data, dim=1)])

def init_weights(m):
	if type(m) == nn.Conv1d:
		torch.nn.init.normal(m.weight)
		m.bias.data.fill_(0.0)
