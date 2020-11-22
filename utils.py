import numpy as np 
import torch
import os
from scipy.io.wavfile import write

def save_model(save_dir, g, d):
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	torch.save(g.state_dict(), f"{save_dir}/g.pt")
	torch.save(d.state_dict(), f"{save_dir}/d.pt")
def save_audio_sample(save_dir, data, mu=128):
	data = (data + 1) * mu
	data = data.astype("uint8")
	write(f"{save_dir}/sample.wav", data, rate=16000)

def quantize(data, mu=128):
	data = round(data * mu) / mu
