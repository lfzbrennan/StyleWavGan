
from torch.utils.data import Dataset
from abc import abstractmethod
import numpy as np
import random
from pydub import AudioSegment
import torch
import glob
from utils import quantize
import soundfile as sf

# stylewavgan interface to create custom datasets to use
class StyleWavGANDataset(Dataset):

	# initialize -> probably want to initialize audio length
	@abstractmethod
	def __init__(self, *args, **kwargs):
		pass

	# return length of dataset
	@abstractmethod
	def __len__(self):
		pass

	# default for PyTorch datasets
	def __getitem__(self, index):
		return self.load_audio(index)

	# return audio sample of length audio_length
	# expecting FloatTensor between [-1, 1] to be output
	@abstractmethod
	def load_audio(self, index):
		pass


class LibriSpeechDataset(StyleWavGANDataset):
	def __init__(self, augment, input_length, quantize=128):
		self.data_root = "../../../datasets/libri_360/**/*.flac"
		self.files = glob.glob(self.data_root, recursive=True)
		self.input_length = input_length
		self.augment = augment
		self.quantize = quantize
	def __len__(self):
		return len(self.files)
	def __getitem__(self, index):
		return self.load_audio(index)

	# process and load chosen audio segments
	def load_audio(self, index):
		audio, sr = sf.read(self.files[index])
		assert(sr == 16000)
		audio /= np.max(np.absolute(audio))
		audio = self.pad(audio)

		random_index = random.randint(0, len(audio) - self.input_length)
		audio = audio[random_index:random_index + self.input_length]

		return self.process_audio(audio)

	# normalize audio from 8bit wav (0-255) to float32 [-1.0, 1.0]
	def process_audio(self, audio):
		audio = torch.FloatTensor(audio)
		# quantize first
		return quantize(audio, self.quantize)

	# pad to audio length, makes sure audio segment self.input_length long
	def pad(self, audio):
		if len(audio) > self.input_length:
			return audio[:self.input_length]
		audio = np.concatenate((audio, [0] * (self.input_length - len(audio))))
		return audio

# dataset based on the FMA music dataset
class FMADataset(StyleWavGANDataset):
	def __init__(self, augment, input_length, quantize=128):
		self.data_root = "../../../datasets/fma_large/**/*.mp3"
		self.files = glob.glob(self.data_root, recursive=True)
		self.input_length = input_length
		self.augment = augment
		self.quantize = quantize
	def __len__(self):
		return len(self.files)
	def __getitem__(self, index):
		return self.load_audio(index)

	# process and load chosen audio segments
	def load_audio(self, index):
		audio = AudioSegment.from_mp3(self.files[index]).set_frame_rate(16000).get_array_of_samples()
		audio /= np.max(np.absolute(audio))
		audio = self.pad(audio)

		random_index = random.randint(0, len(audio) - self.input_length)
		audio = audio[random_index:random_index + self.input_length]

		return self.process_audio(audio)

	# normalize audio from 8bit wav (0-255) to float32 [-1.0, 1.0]
	def process_audio(self, audio):
		audio = torch.FloatTensor(audio)
		# quantize first
		return quantize(audio, self.quantize)

	# pad to audio length, makes sure audio segment self.input_length long
	def pad(self, audio):
		if len(audio) > self.input_length:
			return audio[:self.input_length]
		audio = np.concatenate((audio, [0] * (self.input_length - len(audio))))
		return audio
