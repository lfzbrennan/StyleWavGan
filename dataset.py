
from torch.utils.data import Dataset
import numpy as np
import random
import librosa
import glob
from utils import quantize

class FMADataset(Dataset):
	def __init__(self, augment):
		self.files = glob.glob("../../datasets/fma_large/**/*.wav")
		self.input_length = 2 ** 17
		self.augment = augment
	def __len__(self):
		return len(self.files)
	def __getitem__(self, index):
		return self.load_audio(index)
	def load_audio(self, index):
		audio, _ = librosa.load(self.files[index])

		audio = self.augment.augment(audio)
		audio = self.pad(audio)

		random_index = random.randint(0, len(audio) - self.input_length)
		audio = audio[random_index:random_index + self.input_length]
		return self.process_audio(audio)
	def process_audio(self, audio):
		audio = audio.astype("float32")
		audio = (audio - 128) / 128
		return quantize(audio)
	def pad_audio(self, audio):
		if len(audio) > self.input_length:
			return audio
		audio = np.concatenate((audio, [0] * (self.input_length - len(audio))))
		return audio
