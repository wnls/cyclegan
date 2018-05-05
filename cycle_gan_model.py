import torch
# import modules
import itertools

class CycleGANModel:

	def __init__(self, args):
		# Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
		self.G_A = Generator(?)
		self.G_B = Generator(?)
		self.D_A = Discriminator(?)
		self.D_B = Discriminator(?)

		self.optimizer_G = torch.optim.Adam(itertools.chain(self.G_A.parameters(), 
											self.G_B.parameters()), 
											lr=args.lr, betas=(args.beta1, 0.999))
		self.optimizer_D = torch.optim.Adam(itertools.chain(self.D_A.parameters(), 
											self.D_B.parameters()), 
											lr=args.lr, betas=(args.beta1, 0.999))

		self.gan_loss_fn = torch.nn.MSELoss()
		self.cycle_loss_fn = torch.nn.L1Loss()


	def optimize_parameters():
		# loss
		loss_G_A
		loss_G_B
		loss_D_A
		loss_D_B
		# backward

		# update
		self.optimizer_G.step()
		self.optimizer_D.step()


class Generator(nn.Module):
	def __init__(self):

	def forward(self, input):


class Discriminator(nn.Module):
	def __init__(self):

	def forward(self, input):
