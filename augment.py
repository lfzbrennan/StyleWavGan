import torch
import numpy as np
from librosa.effects import time_stretch

import random

# adaptive augment class based on StyleGan2 paper
class AdaptiveAugment():
	def __init__(self, aug_target=.6, aug_len=5e5, update_every=8, device=None):
		self.data_aug_target = aug_target
		self.ada_aug_len = aug_len
		self.update_every = update_every

		self.ada_update = 0
		self.ada_aug_buf = torch.tensor([0.0, 0.0], device=device)
		self.r_t_stat = 0
		self.ada_aug_p = 0


	# update learning rate based on discriminator performance
	def tune(self, real_pred):
		self.ada_aug_buf += torch.tensor(
			(torch.sign(real_pred).sum().item(), real_pred.shape[0]),
			device=real_pred.device,
		)
		self.ada_update += 1

		if self.ada_update % self.update_every == 0:
			pred_signs, n_pred = self.ada_aug_buf.tolist()

			self.r_t_stat = pred_signs / n_pred

			if self.r_t_stat > self.ada_aug_target:
				sign = 1

			else:
				sign = -1

			self.ada_aug_p += sign * n_pred / self.ada_aug_len
			self.ada_aug_p = min(1, max(0, self.ada_aug_p))
			self.ada_aug_buf.mul_(0)
			self.ada_update = 0

		return self.ada_aug_p


	# augment -> called before real and pred goes through discriminator
	def augment(self, audio):
		if random.random() > self.ada_aug_p:
			audio = self.amp_augment(audio)
			audio = self.freq_augment(audio)
			audio = audio
		return audio

	# augment the amplitude of the audio segment
	def amp_augment(self, audio):
		audio *= random.randrange(.5, 2)
		audio = np.clip(audio, 0, 255)
		return audio

	# augment frequency of audio segment
	def freq_augment(self, audio):
		audio = time_stretch(audio, rate=random.randrange(.5, 2))
		return audio
