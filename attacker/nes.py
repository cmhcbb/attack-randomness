import torch

def nes_grad_est(x, net, sigma=1e-5, n = 20):
	g = torch.zeros(x.size()).cuda()
	for _ in range(n):
		u = torch.randn(g.size()).cuda()
		out1, _ = net(x+sigma*u)
		out2, _ = net(x-sigma*u)
		g += torch.max(out1).item() * u
		g -= torch.max(out2).item() * u
	return 1/(2*sigma*n) * g


def nes(x_in, y_true, net, steps, eps):
	if eps == 0:
		return x_in
	if net.training:
		net.eval()
	x_adv = x_in.clone()
	lr = 0.01

	for i in range(steps):
		print(f'\trunning step {i+1}/{steps} ...')
		x_adv = x_adv + lr * torch.sign(nes_grad_est(x_in, net))
		diff = x_adv - x_in
		diff.clamp_(-eps, eps)
		diff.clamp_(0.0, 1.0)
		x_adv = x_in + diff
	
	return x_adv
