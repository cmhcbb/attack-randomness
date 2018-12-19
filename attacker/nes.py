import torch

def nes_grad_est(x, y, net, sigma=1e-3, n = 10):
	g = torch.zeros(x.size()).cuda()
	g = g.view(x.size()[0],-1)
	y = y.view(-1,1)
	for _ in range(n):
		u = torch.randn(x.size()).cuda()
		out1, _ = net(x+sigma*u)
		out2, _ = net(x-sigma*u)
		out1 = torch.gather(out1,1,y)
		#pdb.set_trace()
		out2 = torch.gather(out2,1,y)
		#print(out1.size(),u.size(),u.view(x.size()[0],-1).size())
		#print(out1[0][y],out2[0][y])
		g +=  out1 * u.view(x.size()[0],-1)
		g -=  out2 * u.view(x.size()[0],-1)
	g=g.view(x.size())
	return 1/(2*sigma*n) * g


def nes(x_in, y, net, steps, eps):
	if eps == 0:
		return x_in
	if net.training:
		net.eval()
	x_adv = x_in.clone()
	lr = 0.01
	for i in range(steps):
		#print(f'\trunning step {i+1}/{steps} ...')
		print(net(x_adv)[0][0][y])
		step_adv = x_adv - lr * torch.sign(nes_grad_est(x_adv, y, net))
		diff = step_adv - x_in
		diff.clamp_(-eps, eps)
		x_adv = x_in + diff
		x_adv.clamp_(0.0, 1.0)

	
	return x_adv
