
from torch.utils.data import Dataset
import numpy as np
import random
import librosa
import glob
from utils import quantize

# dataset based on the FMA music dataset
class FMADataset(Dataset):
	def __init__(self, augment, input_length=2**17, file_root="../../datasets/fma_large/**/*.wav"):
		self.files = glob.glob(file_root, recursive=True)
		self.input_length = input_length
		self.augment = augment
	def __len__(self):
		return len(self.files)
	def __getitem__(self, index):
		return self.load_audio(index)

	# process and load chosen audio segment
	def load_audio(self, index):
		audio, _ = librosa.load(self.files[index])

		audio = self.augment.augment(audio)
		audio = self.pad(audio)

		random_index = random.randint(0, len(audio) - self.input_length)
		audio = audio[random_index:random_index + self.input_length]
		return self.process_audio(audio)

	# normalize audio from 8bit wav (0-255) to float32 (-1.0, 1.0)
	def process_audio(self, audio):
		audio = audio.astype("float32")
		audio = (audio - 128) / 128

		# quantize first
		return quantize(audio)

	# pad to audio length, makes sure audio segment self.input_length long
	def pad_audio(self, audio):
		if len(audio) > self.input_length:
			return audio[:self.input_length]
		audio = np.concatenate((audio, [0] * (self.input_length - len(audio))))
		return audio
