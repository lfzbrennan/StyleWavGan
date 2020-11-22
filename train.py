from model import Generator, Descriminator
from dataset import FMADataset
from utils import save_model, quantize, save_audio_sample

import torch
import torch.nn as nn 
import torch.nn.functional as F 
from torch.optim import AdamW
from torch.utils.data import RandomSampler, DataLoader
import numpy as np
from tqdm import trange, tqdm

from logger import Logger

def des_loss(real_pred, fake_pred):
	real_loss = F.softplus(-real_pred)
	fake_loss = F.softplus(fake_pred)

	return real_loss.mean() + fake_loss.mean()

def gen_loss(fake_pred):
	loss = F.softplus(-fake_pred)

	return loss.mean()

def d_regularize(real_pred, real_img):
	grad_real, = autograd.grad(
		outputs=real_pred.sum(), inputs=real_img, create_graph=True
	)
	grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

	return grad_penalty

def g_regularize(fake_img, latents, mean_path_length, decay=0.01):
	noise = torch.randn_like(fake_img) / math.sqrt(
		fake_img.shape[2] * fake_img.shape[3]
	)
	grad, = autograd.grad(
		outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
	)
	path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

	path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

	path_penalty = (path_lengths - path_mean).pow(2).mean()

	return path_penalty, path_mean.detach(), path_lengths



def train(output_dir="outputs/train1"):
	logger = Logger(output_dir + "/out.log")

	device = torch.device("cuda")

	batch_size = 8
	epochs = 10
	learning_rate = 1e-3
	r1_weight = 10
	path_regularize = 2

	save_log_interval = 10
	mean_path_length = 0

	style_dim = 512
	gate_channels = 512

	global_step = 0

	num_regularize = 16
	print("Building models...")

	g = Generator(style_dim, gate_channels)
	g = nn.DataParallel(g, device_ids=[0, 1])
	g.to(device)

	g_optim = AdamW(g.parameters(), learning_rate)

	d = Descriminator(gate_channels)
	d = nn.DataParallel(d, device_ids=[2, 3])
	d.to(device)

	d_optim = AdamW(d.parameters(), learning_rate)

	print("Building dataloaders...")
	dataset = FMADataset()
	data_sampler = RandomSampler(dataset)
	dataloader = DataLoader(dataset, sampler=data_sampler, batch_size=batch_size, num_workers=8)


	avg_d_loss, avg_g_loss = 0.0, 0.0

	training_iterator = trange(0, epochs, desc="Epochs")
	print("Starting training...")
	for cur_epoch in training_iterator:
		epoch_iterator = tqdm(dataloader, desc="Iteration")
		for step, batch in enumerate(epoch_iterator):
			batch = batch.to(device)
			global_step += 1

			random_styles = torch.randn(batch_size, 1, style_dim)
			fake_audio, latents = g(random_styles)
			fake_audio = quantize(fake_audio)

			real_audio = batch

			fake_pred = d(fake_audio)
			real_pred = d(real_audio)

			d_loss = des_loss(real_pred, fake_pred)
			d.zero_grad()
			d_loss.backward()
			d_optim.step()

			g_loss = gen_loss(fake_pred)
			g.zero_grad()
			g_loss.backward()
			g_optim.step()

			avg_g_loss += g_loss.item()
			avg_d_loss += d_loss.item()

			if global_step % num_regularize == 0:

				## descriminator regularization
				real_audio.requires_grad = True

				real_pred = discriminator(real_audio)
				r1_loss = d_regularize(real_pred, real_audio)
				d.zero_grad()
				(r1_weight / 2 * r1_loss * num_regularize).backward()

				d_optim.step()

				## generator regularization

				path_loss, mean_path_length, path_lengths = g_regularize(fake_audio, latents, mean_path_length)
				g.zero_grad()
				weighted_path_loss = path_regularize * num_regularize * path_loss
				weighted_path_loss.backward()
				g_optim.step()

			if global_step % save_log_interval == 0 and global_step != 0:
				### log
				avg_d_loss /= save_log_interval
				avg_g_loss /= save_log_interval

				logger.log(f"D Loss: {avg_d_loss}\tG Loss: {avg_d_loss}\tEpoch: {cur_epoch}\tIteration: {step}/{len(data_loader)}")

				avg_d_loss, avg_g_loss = 0.0, 0.0

				### save
				save_dir = f"{output_dir}/checkpoint-{global_step}"
				save_model(save_dir, g.module, d.module)
				save_audio_sample(save_dir, fake_audio[0])


	save_model(f"{output_dir}/final", g.module, d.module)







if __name__ == "__main__":
	train(output_dir="outputs/train1")


