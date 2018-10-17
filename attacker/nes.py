import torch

def nes_grad_est(x, p, sigma, n, N):
	g = torch.zeros(n)
	for _ in range(n):
		u = torch.random.normal(zeros, I)
		g += p(x + sigma * u) * u
		g -= p(x - sigma * u) * u
	return 1/(2*sigma*n) * g


def nes(x_in, y_true, net, steps, eps):
	if eps = 0:
		return x_in
	training = net.training 
	if training:
		net.eval()
	index = y_true.cpu().view(-1,1)
	x_adv = x_in.clone().requries_grad_()
	eps = torch.tensor(eps).view(1,1,1,1).cuda()
	
	for _ in range(steps):
		x_adv = x_adv - theta * sign(nes_grad_est(x_in, p, sigma, n, N))
		x_adv.clamp_(x_in - eps, x_in + eps)

	return x_adv
