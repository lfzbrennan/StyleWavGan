import numpy as np
import torch
import os
from scipy.io.wavfile import write

# save generator and discrimnator
def save_model(save_dir, g, d):
	if not os.path.exists(save_dir):
		os.mkdir(save_dir)
	torch.save(g.state_dict(), f"{save_dir}/g.pt")
	torch.save(d.state_dict(), f"{save_dir}/d.pt")

# save audio sample
def save_audio_sample(save_dir, data, mu=128):
	if not os.path.exists(save_dir):
		os.mkdir(save_dir)
	data_short = (data + 1) * mu
	data_short = data_short.astype("uint8")
	write(f"{save_dir}/sample.wav", 16000, data_short.T)

# quantize (-1.0, 1.0) -> (-1.0, 1.0)
def quantize(data, mu=128):
	return np.round(data * mu) / mu
