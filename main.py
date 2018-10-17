import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim import SGD
import torchvision
import torchvision.transforms as transforms

from models.vgg import VGG
from attacker.pgd import Linf_PGD, L2_PGD
from attacker.cw import cw

def main(attack, steps):
	#max_norm=0,0.005,0.01,0.015,0.02,0.025,0.03,0.035,0.04,0.045,0.05,0.055,0.06,0.065,0.07
	#max_norm=0,0.002,0.004,0.006,0.008,0.01,0.012,0.014,0.016,0.018,0.02
	#max_norm=0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.7,0.8,0.9,1.0
	max_norm = [0,0.01,0.02]
	n_ensemble = [50]

	# choose attack method 
	if attack == 'CW':
		attack_f = cw
	elif attack == 'L2':
		attack_f = L2_PGD
	elif attack == 'Linf':
		attack_f = Linf_PGD
	else:
		raise ValueError(f'invalid attack function:{attack}')
	
	# load cifar10 dataset  
	root = "~/datasets"
	nclass = 10
	img_width = 32
	batch_size = 100
	transform_test = transforms.Compose([
        	transforms.ToTensor(),
    ])
	testset = torchvision.datasets.CIFAR10(root, train=False, download=True, transform=transform_test)
	testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
shuffle=True, num_workers=2)

	# set the model
	noise_init = 0.2
	noise_inner = 0.1
	model_out = "/home/thmle/checkpoint/cifar10_vgg_rse.pth"
	from models.vgg_rse import VGG
	net = nn.DataParallel(VGG('VGG16', nclass, noise_init, noise_inner, img_width=img_width).cuda())
	net.load_state_dict(torch.load(model_out))
	net.cuda()
	net.eval()
	loss_f = nn.CrossEntropyLoss()	
	cudnn.benchmark = True
	
	print('#norm, accuracy')
	for eps in max_norm:
		correct = [0] * len(n_ensemble)
		total = 0
		max_iter = 10
		distortion = 0.0
		batch = 0
		for it, (x, y) in enumerate(testloader):
			x, y = x.cuda(), y.cuda()
			x_adv = attack_f(x, y, net, steps, eps)
			pred = ensemble_inference(net, n_ensemble, x_adv, nclass)
			for i, p in enumerate(pred):
				correct[i] += torch.sum(p.eq(y)).item()
			total += y.numel()
			distortion += distance(x_adv, x, attack)
			batch +=1 
			print(f'batch {batch}/{max_iter}: distortion = {distortion/batch:.5f} total_dis = {distortion}')
			if it >= max_iter:
				break
		for i, c in enumerate(correct):
			correct[i] = str(c / total)
		print(f'{distortion/batch},' + ','.join(correct))

def ensemble_inference(net, n_ensemble, x_in, nclass):
	batch = x_in.size(0)
	prev = 0
	prob = torch.FloatTensor(batch, nclass).zero_().cuda()
	answer = []
	softmax = nn.Softmax(dim=1)
	with torch.no_grad():
		for n in n_ensemble:
			for _ in range(n - prev):
				p = softmax(net(x_in)[0])
				prob.add_(p)
			answer.append(prob.clone())	
			prev = n
		for i, a in enumerate(answer):
			answer[i] = torch.max(a, dim=1)[1]
	return answer

def distance(x_adv, x, attack):
    diff = (x_adv - x).view(x.size(0), -1)
    if attack in ('CW', 'L2'):
        out = torch.sqrt(torch.sum(diff * diff) / x.size(0)).item()
        return out
    elif attack in ('Linf'):
        out = torch.mean(torch.max(torch.abs(diff), 1)[0]).item()
        return out


if __name__ == "__main__":
	attack = 'CW'
	#steps=( 3 4 6 9 13 18 26 38 55 78 112 162 234 336 483 695 1000 )
	steps = 3
	main(attack, steps)

