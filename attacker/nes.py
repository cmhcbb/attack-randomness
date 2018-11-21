import torch
import pdb

def nes_grad_est(x, y, net, sigma=1e-5, n = 10):
	g = torch.zeros(x.size()).cuda()
	N = x.size(0)
	y = y.clone().long().view(-1,1)
	g = g.view(N, -1)
	print(f'g.shape={g.shape}')
	for _ in range(n):
		u = torch.randn(x.size()).cuda()
		out1, _ = net(x+sigma*u)
		out2, _ = net(x-sigma*u)
		out1 = out1.gather(1, y)
		#pdb.set_trace()
		out2 = out1.gather(1, y)
		#print(f'out1.shape= {out1.shape}')
		#o = out1 * u.view(N, -1)
		#print(f'o.shape={o.shape}')
		u1 = u.clone().view(N, -1).cpu()
		o1 = out1.cpu() * u1
		print(f'o1.shape= {o1.shape}')
		g +=  out1 * u.view(N, -1)
		g -=  out2 * u.view(N, -1)
	g = g.view(x.size())
	return 1/(2*sigma*n) * g


def nes(x_in, y_out, net, steps, eps):
	if eps == 0:
		return x_in
	if net.training:
		net.eval()
	x_adv = x_in.clone()
	lr = 0.01

	for i in range(steps):
		print(f'\trunning step {i+1}/{steps} ...')
		x_adv = x_adv - lr * torch.sign(nes_grad_est(x_in, y_out, net))
		diff = x_adv - x_in
		diff.clamp_(-eps, eps)
		diff.clamp_(0.0, 1.0)
		x_adv = x_in + diff
	
	return x_adv
