import torch
from torch.utils.data import Dataset
import numpy as np 
import random
import librosa
import glob
from utils import quantize

class FMADataset(Dataset):
	def __init__(self):
		self.files = glob.glob("../../datasets/fma_large/**/*.mp3")
		self.input_length = 2 ** 16
		self.sampling_rate = 22050
	def __len__(self):
		return len(self.files)
	def __getitem__(self, index):
		return load_audio(self.files[index])
	def load_audio(self, file):
		audio, _ = librosa.load(file, sr=self.sampling_rate)
		random_index = random.randint(0, len(audio) - self.input_length)
		audio = audio[random_index:random_index + self.input_length]
		return quantize(audio)