import torch
# import modules
import itertools
from .gan_model import *

class CycleGANModel:

    def __init__(self, args):
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.G_A = GeneratorJohnson()
        self.G_B = GeneratorJohnson()
        self.D_A = DiscriminatorPatchGAN()
        self.D_B = DiscriminatorPatchGAN()

        self.optimizer_G = torch.optim.Adam(itertools.chain(self.G_A.parameters(),
                                            self.G_B.parameters()),
                                            lr=args.lr, betas=(args.beta1, 0.999))
        self.optimizer_D = torch.optim.Adam(itertools.chain(self.D_A.parameters(),
                                            self.D_B.parameters()),
                                            lr=args.lr, betas=(args.beta1, 0.999))

        self.gan_loss_fn = torch.nn.MSELoss()
        self.cycle_loss_fn = torch.nn.L1Loss()


    def train(self, input):
        self.G_A.train()
        self.G_B.train()
        self.D_A.train()
        self.D_B.train()

        A, B = input

        # loss
        B_gen, loss_G_A, loss_G_B, loss_D_A, loss_D_B =  self.compute_loss(A, B)

        # backward and update
        self.optimizer_G.zero_grad()
        loss_G = loss_G_A + loss_G_B
        loss_G.backward()
        self.optimizer_G.step()

        self.optimizer_D.zero_grad()
        loss_D_A.backward()
        loss_D_B.backward()
        self.optimizer_D.step()

        return {'G_A': loss_G_A, 'G_B': loss_G_B, 'D_A': loss_D_A, 'D_B': loss_D_B,
                'total': loss_G_A+loss_G_B+loss_D_A+loss_D_B}

    def eval(self, input):
        self.G_A.eval()
        self.G_B.eval()
        self.D_A.eval()
        self.D_B.eval()

        A, B = input
        B_gen, loss_G_A, loss_G_B, loss_D_A, loss_D_B = self.compute_loss(A, B)

        # TODO save B_gen

        return {'G_A': loss_G_A, 'G_B': loss_G_B, 'D_A': loss_D_A, 'D_B': loss_D_B,
                'total': loss_G_A + loss_G_B + loss_D_A + loss_D_B}

    def compute_loss(self, A, B):
        B_gen = self.G_A(A)
        A_cyc = self.G_B(B_gen)
        loss_G_A = self.gan_loss_fn(self.D_A(B_gen), torch.ones_like(B)) + self.cycle_loss_fn(A_cyc, A)  # TODO lambda_a

        A_gen = self.G_B(B)
        B_cyc = self.G_A(A_gen)
        loss_G_B = self.gan_loss_fn(self.D_B(A_gen), torch.ones_like(A)) + self.cycle_loss_fn(B_cyc, B)  # TODO lambda_b

        loss_D_A = self.gan_loss_fn(self.D_A(B), torch.ones_like(A)) + self.gan_loss_fn(self.D_A(B_gen),
                                                                                        torch.zeros_like(
                                                                                            A))  # TODO *0.5
        loss_D_B = self.gan_loss_fn(self.D_B(A), torch.ones_like(A)) + self.gan_loss_fn(self.D_B(A_gen),
                                                                                        torch.zeros_like(A))
        return B_gen, loss_G_A, loss_G_B, loss_D_A, loss_D_B

    def load_state(self, state, lr=None):
        print('Using pretrained model...')
        self.G_A.load_state_dict(state['G_A'])
        self.G_B.load_state_dict(state['G_B'])
        self.D_A.load_state_dict(state['D_A'])
        self.D_B.load_state_dict(state['D_B'])
        self.optimizer_G.load_state_dict(state['optimG'])
        self.optimizer_D.load_state_dict(state['optimD'])

        # set model lr to new lr
        if lr is not None:
            for param_group in self.optimizer_G.param_groups:
                before = param_group['lr']
                param_group['lr'] = lr
            for param_group in self.optimizer_D.param_groups:
                before = param_group['lr']
                param_group['lr'] = lr
            print('optim lr: before={} / after={}'.format(before, lr))


    def save_state(self):
        return {'G_A': self.G_A.state_dict(),
                'G_B': self.G_B.state_dict(),
                'optimG': self.optimizer_G.state_dict(),
                'optimD': self.optimizer_D.state_dict()}

