import numpy as np 
import torch
import os

def save_model(save_dir, g, d):
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	torch.save(g.state_dict(), f"{save_dir}/g.pt")
	torch.save(d.state_dict(), f"{save_dir}/d.pt")

def quantize(data, mu=128):
	data = round(data * mu) / mu
