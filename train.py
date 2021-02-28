from model import Generator, Descriminator
from dataset import FMADataset
from utils import save_model, quantize, save_audio_sample
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from torch.optim import Adam
from torch.utils.data import RandomSampler, DataLoader
import math
from tqdm import trange, tqdm

from logger import Logger
from augment import AdaptiveAugment

# change gradient usage
def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

# descriminator loss
def des_loss(real_pred, fake_pred):
	real_loss = F.softplus(-real_pred)
	fake_loss = F.softplus(fake_pred)

	return real_loss.mean() + fake_loss.mean()

# generator loss
def gen_loss(fake_pred):
	loss = F.softplus(-fake_pred)

	return loss.mean()

# descriminator regularization loss
def d_regularize(real_pred, real_img):
	grad_real, = autograd.grad(
		outputs=real_pred.sum(), inputs=real_img
	)
	grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

	return grad_penalty

# generator regularlization loss
def g_regularize(fake_img, latents, mean_path_length, decay=0.01):
	noise = torch.randn_like(fake_img) / math.sqrt(
		fake_img.shape[2] * fake_img.shape[3]
	)
	grad, = autograd.grad(
		outputs=(fake_img * noise).sum(), inputs=latents
	)
	path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

	path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

	path_penalty = (path_lengths - path_mean).pow(2).mean()

	return path_penalty, path_mean.detach(), path_lengths

# main train function
def train(output_dir="outputs/train1"):
	if not os.path.exists(output_dir):
		os.mkdir(output_dir)
	logger = Logger(output_dir + "/out.log")

	# split models among cuda devices -> script wrote for 4 parallel 1080ti
	device1 = torch.device("cuda:0")
	device2 = torch.device("cuda:2")

	# initialize model and training parameters
	batch_size = 4
	epochs = 10
	learning_rate = .002
	r1_weight = 10
	path_regularize = 2
	augment_regularize = 128

	save_log_interval = 1000
	mean_path_length = 0

	style_dim = 128
	gate_channels = 256

	global_step = 0

	num_regularize_d = 16
	num_regularize_g = 4

	g_reg_ratio = num_regularize_g / (num_regularize_g + 1)
	d_reg_ratio = num_regularize_d / (num_regularize_d + 1)

	print("Building models...")

	# build models (and optimizers) and put on correct device in parallel
	g = Generator(style_dim, gate_channels)
	g = nn.DataParallel(g, device_ids=[0, 1])
	g.to(device1)

	g_optim = Adam(g.parameters(), learning_rate * g_reg_ratio, betas=(0, .99 ** g_reg_ratio))

	d = Descriminator(gate_channels)
	d = nn.DataParallel(d, device_ids=[2, 3])
	d.to(device2)

	d_optim = Adam(d.parameters(), learning_rate * d_reg_ratio, betas=(0, .99 ** d_reg_ratio))

	print("Building dataloaders...")
	# create dataloaders
	augment = AdaptiveAugment(device=device2)
	dataset = FMADataset(augment)
	data_sampler = RandomSampler(dataset)
	dataloader = DataLoader(dataset, sampler=data_sampler, batch_size=batch_size, num_workers=4)

	avg_d_loss, avg_g_loss = 0.0, 0.0

	training_iterator = trange(0, epochs, desc="Epochs")
	print("Starting training...")
	# training loop
	for cur_epoch in training_iterator:
		epoch_iterator = tqdm(dataloader, desc="Iteration")
		for step, batch in enumerate(epoch_iterator):
			global_step += 1
			batch = batch.to(device2)
			real_audio = torch.unsqueeze(batch, 1)

			# train discriminator
			requires_grad(d, True)
			requires_grad(g, False)

			random_styles = torch.randn(batch_size, 1, style_dim).to(device1)
			fake_audio, latents = g(random_styles)
			fake_audio = fake_audio.detach().cpu().numpy()
			fake_audio = quantize(fake_audio)
			fake_audio = torch.tensor(fake_audio, device=device2)

			real_pred = d(real_audio)
			fake_pred = d(fake_audio)

			d_loss = des_loss(real_pred, fake_pred)
			d.zero_grad()
			d_loss.backward()
			d_optim.step()

			avg_d_loss += d_loss.item()

			# update adaptive augmentor if applicable
			if global_step % augment_regularize:
				dataset.augment.tune(real_pred)

            # regualize descrimnator if applicable
			if global_step % num_regularize_d == 0:

				real_audio.requires_grad = True

				#real_pred = d(real_audio)
				r1_loss = d_regularize(real_pred, real_audio)
				d.zero_grad()
				(r1_weight / 2 * r1_loss * num_regularize_d).backward()

				d_optim.step()

			# train generator
			requires_grad(d, False)
			requires_grad(g, True)

			random_styles = torch.randn(batch_size, 1, style_dim).to(device1)
			fake_audio, latents = g(random_styles)
			fake_audio = fake_audio.detach().cpu().numpy()

			fake_audio = quantize(fake_audio)
			fake_audio = torch.tensor(fake_audio, device=device2)

			fake_pred = d(fake_audio)

			g_loss = gen_loss(fake_pred)
			g.zero_grad()
			g_loss.backward()
			g_optim.step()
			avg_g_loss += g_loss.item()

            # regularize generator if applicable
			if global_step % num_regularize_g == 0:

				path_loss, mean_path_length, path_lengths = g_regularize(fake_audio, latents, mean_path_length)
				g.zero_grad()
				weighted_path_loss = path_regularize * num_regularize_g * path_loss
				weighted_path_loss.backward()
				g_optim.step()

            # log and save sample if applicable
			if global_step % save_log_interval == 0 or global_step == 1:
				### log
				if global_step != 1:
					avg_d_loss /= save_log_interval
					avg_g_loss /= save_log_interval

				logger.log(f"D Loss: {avg_d_loss}\tG Loss: {avg_g_loss}\tEpoch: {cur_epoch}\tIteration: {step}/{len(dataloader)}")

				avg_d_loss, avg_g_loss = 0.0, 0.0

				### save
				save_dir = f"{output_dir}/checkpoint-{global_step}"
				save_model(save_dir, g.module, d.module)
				save_audio_sample(save_dir, fake_audio[0].detach().cpu().numpy())


	save_model(f"{output_dir}/final", g.module, d.module)







if __name__ == "__main__":
	train(output_dir="outputs/train3")
