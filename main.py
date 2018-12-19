import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim import SGD
import torchvision
import torchvision.transforms as transforms

import math
import os
import argparse
import numpy as np

from models.vgg import VGG
from attacker.pgd import Linf_PGD, L2_PGD
from attacker.cw import cw
from attacker.nes import nes
from attacker.OPT_attack_lf import opt_lf

def main():
	parser = argparse.ArgumentParser(description='Attack CIFAR10 models')
	parser.add_argument('-a', '--attack', required=True, type=str, help='name of attacker: {NES, CW, L2, Linf, OPT_lf}')
	parser.add_argument('-m', '--model', required=True, type=str, help='CIFAR10 models: {rse, adv_vi, vi, adv, plain}')
	parser.add_argument('-s', '--steps', required=True, type=int, help='number of steps for attacker')
	parser.add_argument('-e', '--eps', required=True, type=str, help='list of epsilon values separated by comma, i.e. 0.2 or 0.05,0.01')
	opt = parser.parse_args()
	max_norm = [float(s) for s in opt.eps.split(',')]

	batch_size = 1
	# choose attack method 
	if opt.attack == 'CW':
		attack_f = cw
	elif opt.attack == 'L2':
		attack_f = L2_PGD
	elif opt.attack == 'Linf':
		attack_f = Linf_PGD
	elif opt.attack == 'NES':
		attack_f = nes
	elif opt.attack == 'OPT_lf':
		attack_f = opt_lf
		batch_size = 1
	else:
		raise ValueError(f'invalid attack function:{opt.attack}')
	
	# load cifar10 dataset  
	root = "~/datasets"
	nclass = 10
	img_width = 32

	transform_test = transforms.Compose([
        	transforms.ToTensor(),
    ])
	testset = torchvision.datasets.CIFAR10(root, train=False, download=True, transform=transform_test)
	testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
shuffle=False, num_workers=2)

	# set the model
	model_out = "../cifar10-checkpoint/"	
	if opt.model == 'rse':
		noise_init = 0.2
		noise_inner = 0.1
		model_out += "cifar10_vgg_rse.pth"
		from models.vgg_rse import VGG
		net = nn.DataParallel(VGG('VGG16', nclass, noise_init, noise_inner, img_width=img_width).cuda())
	elif opt.model == 'adv_vi':
		sigma_0 = 0.08
		N = 50000
		init_s = 0.08
		model_out += "cifar10_vgg_adv_vi.pth"
		from models.vgg_vi import VGG
		net = nn.DataParallel(VGG(sigma_0, N, init_s, 'VGG16', nclass, img_width=img_width).cuda())
	elif opt.model == 'vi':
		sigma_0 = 0.15
		N = 50000
		init_s = 0.15
		model_out += "cifar10_vgg_vi.pth"
		from models.vgg_vi import VGG
		net = nn.DataParallel(VGG(sigma_0, N, init_s, 'VGG16', nclass, img_width=img_width).cuda())
	elif opt.model == 'adv':
		model_out += "cifar10_vgg_adv.pth"
		from models.vgg import VGG
		net = nn.DataParallel(VGG('VGG16', nclass).cuda())
	elif opt.model == 'plain':
		model_out += "cifar10_vgg_plain.pth"
		from models.vgg import VGG
		net = nn.DataParallel(VGG('VGG16', nclass).cuda())
	else:
		raise ValueError(f'invalid cifar10 model: {opt.model}')
	
	net.load_state_dict(torch.load(model_out))
	net.cuda()
	net.eval()
	loss_f = nn.CrossEntropyLoss()	
	cudnn.benchmark = True
	softmax = nn.Softmax(dim=1)
	for eps in max_norm:
		print(f'Using attack {opt.attack} on CIFAR10 vgg_{opt.model} for eps = {eps}:')
		success = 0
		count = 0
		total = 0
		max_iter = 100
		distortion = 0.0
		for it, (x, y) in enumerate(testloader):
			if it+1 > max_iter:
				continue
			#if it<75:
			#	continue
			print(f'batch {it+1}/{max_iter}')
			#print(x,y)
			#print(torch.max(x),torch.min(x))
			x, y = x.cuda(), y.cuda()
			#y_adv = (y+1)%10 

			y_out = torch.max(softmax(net(x)[0]), dim=1)[1]
			#filter the wrong predictions
			x = x[y_out.eq(y)]
			y = y[y_out.eq(y)]
			#print(x,y)
			if x.size()[0]==0:
				continue
			#print(x[y_out.eq(y)].size()) 
			#pdb.set_trace()
			with torch.no_grad():
				x_adv = attack_f(x, y, net, opt.steps, eps)
				pred = torch.max(softmax(net(x_adv)[0]), dim=1)[1]
				is_adv = pred.eq(y) == 0 
				#y_out_str = ''.join(map(str, y_out.cpu().numpy()))
				pred_str = ''.join(map(str, pred.cpu().numpy()))
				is_adv_str = ''.join(map(str, is_adv.cpu().numpy()))
				y_true_str = ''.join(map(str, y.cpu().numpy()))
				print(f'\ttrue image labels (y_true)    : {y_true_str}')
				#print(f'\toriginal images predicted     : {y_out_str}')
				#print(y_adv.item())
				print(f'\tadversarial examples predicted: {pred_str}')
				print(f'\tis adversarial?                 {is_adv_str}')
				new_success = torch.sum(is_adv).item()
				new_distortion = distance2(x_adv, x, is_adv.cpu().numpy(), opt.attack)
				print(f'\tsuccess = {new_success}\n\ttotal={x.size()[0]}\n\tdistortion = {new_distortion}')
				if new_distortion and new_distortion[0]<=0.2:
					success += new_success
				total += x.size()[0]
				if len(new_distortion) > 0:
					count += 1
					distortion += np.mean(new_distortion)
		#if count == 0:
		#	print('Found no adversarial examples!!!')
		#else:
		#	print(f'Average distortion: {distortion/count}, average success rate = {success/(max_iter * batch_size)}')
		print(success,total)
def distance2(x_adv, x, is_adv, attack):
	diff = (x_adv - x).view(x.size(0), -1).cpu().numpy()
	l2 = np.sum(diff*diff, axis=1) 
	linf = np.max(np.abs(diff), axis=1)
	if attack in ('CW', 'L2'):
		return l2[is_adv==1]
	elif attack in ('Linf','NES','OPT_lf'):
		return linf[is_adv==1]
	else:
		return [] 

def distance(x_adv, x, is_adv, attack):
	if is_adv.nonzero().size(0) == 0:
		return 0.0
	indices = torch.t(is_adv.nonzero()).squeeze(0)
	print(f'\tonly consider distortion of the adversarial images: {indices.cpu().numpy()}')
	diff = (x_adv - x).view(x.size(0), -1)
	if attack in ('CW', 'L2', 'NES'):
		dist = torch.gather(torch.sum(diff*diff, dim=1), dim=0, index=indices)
		return torch.mean(torch.sqrt(dist)).item()
	elif attack in ('Linf'):
		dist = torch.gather(torch.max(torch.abs(diff), dim=1)[0], dim=0, index=indices)
		return torch.mean(dist).item()


if __name__ == "__main__":
	main()

