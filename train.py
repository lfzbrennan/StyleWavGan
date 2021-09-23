from model import Generator, Descriminator
from dataset import FMADataset, LibriSpeechDataset
from utils import save_model, quantize, save_audio_sample, init_weights
import os
import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from torch.optim import Adam
from torch.utils import data
from torch.utils.data import RandomSampler, DataLoader
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from distributed import (
	get_rank,
	synchronize,
	reduce_loss_dict,
	reduce_sum,
	get_world_size,
)
import math
import numpy as np
from tqdm import trange, tqdm
import argparse

from logger import Logger
from augment import AdaptiveAugment

import warnings
warnings.filterwarnings("ignore")

#torch.autograd.set_detect_anomaly(True)


@contextlib.contextmanager
def no_weight_gradients():
    global weight_gradients_disabled

    old = weight_gradients_disabled
    weight_gradients_disabled = True
    yield
    weight_gradients_disabled = old

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
def d_regularize(real_pred, real_audio):
	batch_size = real_audio.shape[0]

	grad = autograd.grad(
		outputs=real_pred, inputs=real_audio,
		create_graph=True, only_inputs=True)[0]

	grad = grad.reshape(batch_size, -1)
	grad[torch.isnan(grad)] = 0

	grad_penalty = ((grad.norm(2, dim=1) - 1) ** 2).mean()
	return grad_penalty

# generator regularlization loss
def g_regularize(fake_audio, latents, mean_path_length, decay=0.01):

	noise = torch.randn_like(fake_audio, device=fake_audio.device) / math.sqrt(
		fake_audio.shape[2]
	)
	grad = autograd.grad(
		outputs=[(fake_audio * noise).sum()], inputs=[latents], create_graph=True, only_inputs=True)[0]


	path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1) + 1e-8)


	path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

	path_penalty = (path_lengths - path_mean).pow(2).mean()

	return path_penalty, path_mean.detach(), path_lengths

# main train function
def train(args):
	output_dir = args.output_dir
	if not os.path.exists(output_dir):
		os.mkdir(output_dir)

	logger = Logger(output_dir + "/out.log")

	if get_rank() == 0:
		for arg in vars(args):
			logger.log(f"{arg}: {getattr(args, arg)}")

	distributed = not args.not_distributed

	if distributed:
		device=torch.device("cuda")
	else:
		device=torch.device("cuda:1")

	if distributed:
		if get_rank() == 0: print("Creating Distibuted Process Group...")
		torch.cuda.set_device(args.local_rank)
		dist.init_process_group("nccl", init_method="env://")
		synchronize()

	# initialize some parameters
	mean_path_length = 0
	global_step = 0

	if get_rank() == 0: print("Building models...")

	# build models (and optimizers) and put on correct device in parallel
	g = Generator(args.style_dim, args.layers, args.channel_mult, input_length=args.audio_length).to(device)
	d = Descriminator(args.layers, args.channel_mult, input_length=args.audio_length).to(device)

	if args.load_from_cp != "none":
		g.load_state_dict(torch.load(f"{args.load_from_cp}/g.pt"))
		d.load_state_dict(torch.load(f"{args.load_from_cp}/d.pt"))
	else:
		g.apply(init_weights)
		d.apply(init_weights)

	g_optim = Adam(g.parameters(), args.lr, betas=(.5, .99))  # standard betas for stylegan2
	d_optim = Adam(d.parameters(), args.lr * args.d_lr_mult, betas=(.5, .99))

	if distributed:
		if get_rank() == 0: print("Parallelizing models...")
		g = DistributedDataParallel(g, device_ids=[args.local_rank], output_device=args.local_rank, broadcast_buffers=False)
		d = DistributedDataParallel(d, device_ids=[args.local_rank], output_device=args.local_rank, broadcast_buffers=False)

	if distributed:
		g_module = g.module
		d_module = d.module
	else:
		g_module = g
		d_module = d

	if get_rank() == 0: print("Building dataloaders...")
	# create dataloaders
	if args.augment:
		augmenter = AdaptiveAugment()
	dataset = FMADataset(augmenter if args.augment else None, args.audio_length, quantize=args.mu)
	if distributed:
		data_sampler = data.distributed.DistributedSampler(dataset, shuffle=True)
	else:
		data_sampler = RandomSampler(dataset)
	dataloader = data.DataLoader(dataset, sampler=data_sampler, batch_size=args.batch_size, num_workers=4)

	avg_g_loss, avg_d_loss = 0.0, 0.0
	avg_reg_g_loss, avg_reg_d_loss = 0.0, 0.0
	loss_dict = {}
	if get_rank() == 0: print("Starting training...")
	# training loop
	for cur_epoch in range(0, args.epochs):
		for step, batch in enumerate(dataloader):
			global_step += 1
			real_audio = torch.unsqueeze(batch.to(device), 1)
			real_audio.requires_grad_(True)
			#print(f"Real Max: {torch.max(real_audio)}\tMin: {torch.min(real_audio)}")

			requires_grad(d, True)
			requires_grad(g, False)

			random_styles = torch.randn((args.batch_size, 1, args.style_dim)).to(device)
			fake_audio, _ = g(random_styles)
			#print(f"Fake Max: {torch.max(fake_audio)}\tMin: {torch.min(fake_audio)}")
			fake_pred = d(fake_audio)
			real_pred = d(real_audio)

			d_loss = des_loss(real_pred, fake_pred)
			loss_dict["d"] = d_loss.clone()

			# update adaptive augmentor if applicable

			if args.augment and global_step % args.num_augment:
				dataset.augment.tune(real_pred)


			# regualize descrimnator if applicable

			if args.d_reg and global_step % args.num_d_reg == 0:

				r1_loss = args.d_reg(real_pred, real_audio)
				reg_d_loss = args.d_reg_weight * r1_loss * args.num_d_reg
				loss_dict["d_reg"] = reg_d_loss.clone()
				d_loss += reg_d_loss


			d_loss.backward()
			torch.nn.utils.clip_grad_norm(d.parameters(), 1)
			d_optim.step()
			d.zero_grad()
			d_optim.zero_grad()

			requires_grad(g, True)
			requires_grad(d, False)

			random_styles = torch.randn((args.batch_size, 1, args.style_dim)).to(device)
			fake_audio, _ = g(random_styles)

			fake_pred = d(fake_audio)

			g_loss = gen_loss(fake_pred)
			loss_dict["g"] = g_loss.clone()

			# regularize generator if applicable
			if args.g_reg and global_step % args.num_g_reg == 0:

				random_styles = torch.randn((args.batch_size, 1, args.style_dim), requires_grad=True).to(device)
				fake_audio, latents = g(random_styles)

				path_loss, mean_path_length, path_lengths = g_regularize(fake_audio, latents, mean_path_length)
				reg_g_loss = args.g_reg_weight * args.num_g_reg * path_loss
				loss_dict["g_reg"] = reg_g_loss.clone()
				g_loss += reg_g_loss

			g_loss.backward()
			torch.nn.utils.clip_grad_norm(g.parameters(), 1)
			g_optim.step()
			g.zero_grad()
			g_optim.zero_grad()

			loss_reduced = reduce_loss_dict(loss_dict)
			avg_d_loss += loss_reduced["d"].mean().item()
			avg_g_loss += loss_reduced["g"].mean().item()

			if args.d_reg and global_step % args.num_d_reg == 0:
				avg_reg_d_loss += loss_reduced["d_reg"].mean().item()

			if args.g_reg and global_step % args.num_g_reg == 0:
				avg_reg_g_loss += loss_reduced["g_reg"].mean().item()

			# for debugging
			'''
			if get_rank() == 0 and global_step % 10 == 0:
				print(f"D Loss: {d_loss}\tG Loss: {g_loss}\tEpoch: {cur_epoch + 1}\tIteration: {step + 1}/{len(dataloader)}")
				print(f"Real Max: {torch.max(real_audio)}\tReal Min: {torch.min(real_audio)}")
				print(f"Fake Max: {torch.max(fake_audio)}\tFake Min: {torch.min(fake_audio)}")
				#print(f"Real Pred: {real_pred.item()}\tFake Pred: {fake_pred.item()}")
			'''
			# log and save sample if applicable
			if get_rank() == 0:
				if global_step % args.save_log_interval == 0 or global_step == 1:
					### log
					if global_step != 1:
						avg_d_loss /= args.save_log_interval
						avg_g_loss /= args.save_log_interval
						avg_reg_d_loss /= args.save_log_interval
						avg_reg_g_loss /= args.save_log_interval

					logger.log(f"D Loss: {avg_d_loss}\tD Reg: {avg_reg_d_loss}\tG Loss: {avg_g_loss}\tG Reg: {avg_reg_g_loss}\tEpoch: {cur_epoch + 1}\tIteration: {step + 1}/{len(dataloader)}")

					avg_d_loss, avg_g_loss = 0.0, 0.0
					avg_reg_g_loss, avg_reg_d_loss = 0.0, 0.0

					### save
					save_dir = f"{output_dir}/checkpoint-{global_step}"
					save_model(save_dir, g_module, d_module)
					save_audio_sample(save_dir, fake_audio[0].detach().cpu().numpy())


	save_model(f"{output_dir}/final", g_module, d_module)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--local_rank", type=int, default=0, help="local rank for distributed training")
	parser.add_argument("--not_distributed", default=False, action="store_true", help="if non-distributive training")
	parser.add_argument("--batch_size", type=int, default=2, help="batch size for each gpu")
	parser.add_argument("--epochs", type=int, default=10, help="# training epochs")
	parser.add_argument("--audio_length", type=int, default=2**17, help="audio length in samples (default is 2**17 ~= 8 seconds for 16kHz sample)")
	parser.add_argument("--layers", type=int, default=5, help="# of wavenet layers (higher = bigger model)")
	parser.add_argument("--channel_mult", type=float, default=1, help="channel multiplier (higher = bigger model)")
	parser.add_argument("--lr", type=float, default=1e-4, help="base learning rate (for generator)")
	parser.add_argument("--d_lr_mult", type=float, default=1, help="learning rate multiplier for descriminator")
	parser.add_argument("--save_log_interval", type=int, default=1000, help="log, save models, and save sample every save_log_interval iterations")
	parser.add_argument("--style_dim", type=int, default=128, help="length of style embeddings")
	parser.add_argument("--d_reg", default=False, action="store_true", help="turn on desriminator regularization")
	parser.add_argument("--g_reg", default=False, action="store_true", help="turn on generator regularization")
	parser.add_argument("--d_reg_weight", type=float, default=1e-10, help="descriminator regularization weight")
	parser.add_argument("--g_reg_weight", type=float, default=2, help="generator regularization weight")
	parser.add_argument("--mu", type=int, default=128, help="mu for quantizing. 128 mu = 8 bit quantize")
	parser.add_argument("--augment", default=False, action="store_true", help="turn on adaptive data augmentation")
	parser.add_argument("--num_d_reg", type=int, default=20, help="step descriminator regularization step every num_d_reg interations")
	parser.add_argument("--num_g_reg", type=int, default=10, help="step generator regularization step every num_g_reg interations")
	parser.add_argument("--num_augment", type=int, default=1000, help="adjust adaptive augment step every num_augment interations")
	parser.add_argument("--load_from_cp", type=str, default="none", help="directory to start training from checkpoint")
	parser.add_argument("--output_dir", type=str, default="outputs/1", help="output directory for logs, saved models, and sampled")
	parser.add_argument("--description", type=str, default="Training", help="description of training for log file")
	args = parser.parse_args()
	train(args)
