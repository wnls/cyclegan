import torch
# import modules
import itertools
import os
from .gan_model import *
import scipy.misc
import random
from torch.nn import init

class CycleGANModel:

    def __init__(self, args):
        self.start_epoch = 0
        self.args = args

        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        if args.G == 'res6':
            self.G_A = GeneratorJohnson()
            self.G_B = GeneratorJohnson()
        elif args.G == 'res9':
            self.G_A = GeneratorJohnson2(n_res_blocks=9)
            self.G_B = GeneratorJohnson2(n_res_blocks=9)

        self.D_A = DiscriminatorPatchGAN()
        self.D_B = DiscriminatorPatchGAN()

        self.optimizer_G = torch.optim.Adam(itertools.chain(self.G_A.parameters(),
                                            self.G_B.parameters()),
                                            lr=args.lr, betas=(args.beta1, 0.999))
        self.optimizer_D = torch.optim.Adam(itertools.chain(self.D_A.parameters(),
                                            self.D_B.parameters()),
                                            lr=args.lr, betas=(args.beta1, 0.999))

        self.scheduler_G = torch.optim.lr_scheduler.LambdaLR(self.optimizer_G, lr_lambda=self.lr_lambda)
        self.scheduler_D = torch.optim.lr_scheduler.LambdaLR(self.optimizer_D, lr_lambda=self.lr_lambda)

        self.init_type = args.init_type
        if args.init_type is not None:
            self.G_A.apply(self.init_weights)
            self.G_B.apply(self.init_weights)
            self.D_A.apply(self.init_weights)
            self.D_B.apply(self.init_weights)

        self.gan_loss_fn = torch.nn.MSELoss()
        self.cycle_loss_fn = torch.nn.L1Loss()
        self.idt_loss_fn = torch.nn.L1Loss()

        self.lambda_A = args.lambda_A
        self.lambda_B = args.lambda_B
        self.lambda_idt = args.lambda_idt

        self.A_gen_buffer = ImageBuffer()
        self.B_gen_buffer = ImageBuffer()

    def init_weights(self, m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if self.init_type == 'normal':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif self.init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=0.02)
            elif self.init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            else:
                raise NotImplementedError('initialization method [%s] not implemented' % self.init_type)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)

    def lr_lambda(self, epoch):
        return 1.0 - max(0, epoch + self.start_epoch - self.args.lr_decay_start) / (self.args.lr_decay_n + 1)

    def update_scheduler(self):
        self.scheduler_G.step()
        self.scheduler_D.step()
        print('learning rate = %.7f' % self.optimizer_G.param_groups[0]['lr'])

    def set_start_epoch(self, epoch):
        self.start_epoch = epoch

    def to(self, device):
        self.G_A.to(device)
        self.G_B.to(device)
        self.D_A.to(device)
        self.D_B.to(device)

        for state in itertools.chain(self.optimizer_G.state.values(), self.optimizer_D.state.values()):
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    def train(self, input, save, out_dir_img, epoch):
        # self.G_A.train()
        # self.G_B.train()
        # self.D_A.train()
        # self.D_B.train()

        A, B, A_gt, B_gt, index_A, index_B = input

        ############################
        # G loss
        ############################
        self.optimizer_G.zero_grad()

        # Identity loss
        if self.lambda_idt > 0:
            loss_G_A_idt = self.idt_loss_fn(self.G_A(B), B) * self.lambda_B * self.lambda_idt
            loss_G_B_idt = self.idt_loss_fn(self.G_B(A), A) * self.lambda_A * self.lambda_idt
        else:
            loss_G_A_idt = 0
            loss_G_B_idt = 0

        # GAN loss D_A(G_A(A))
        B_gen = self.G_A(A)
        loss_G_A = self.gan_loss(self.D_A(B_gen), 1)

        # GAN loss D_B(G_B(B))
        A_gen = self.G_B(B)
        loss_G_B = self.gan_loss(self.D_B(A_gen), 1)

        # Forward cycle loss
        A_cyc = self.G_B(B_gen)
        loss_cyc_A = self.cycle_loss_fn(A_cyc, A) * self.lambda_A

        # Backward cycle loss
        B_cyc = self.G_A(A_gen)
        loss_cyc_B = self.cycle_loss_fn(B_cyc, B) * self.lambda_B

        # Combine
        loss_G = loss_G_A + loss_G_B + loss_cyc_A + loss_cyc_B + loss_G_A_idt + loss_G_B_idt

        loss_G.backward()
        self.optimizer_G.step()

        # self.save_image((A, B_gen, A_cyc, B, A_gen, B_cyc), out_dir_img, "train_ep_%d" % epoch)
        ############################
        # D loss
        ############################
        self.optimizer_D.zero_grad()

        # D_A real loss
        loss_D_A_real = self.gan_loss(self.D_A(B), 1)

        # D_A fake loss
        B_gen_pool = self.B_gen_buffer.push_and_pop(B_gen).detach()
        loss_D_A_fake = self.gan_loss(self.D_A(B_gen_pool), 0)

        loss_D_A = (loss_D_A_real + loss_D_A_fake) * 0.5

        # D_B real loss
        loss_D_B_real = self.gan_loss(self.D_B(A), 1)

        # D_B fake loss
        A_gen_pool = self.A_gen_buffer.push_and_pop(A_gen).detach()
        loss_D_B_fake = self.gan_loss(self.D_B(A_gen_pool), 0)

        loss_D_B = (loss_D_B_real + loss_D_B_fake) * 0.5

        #TODO can add?
        loss_D = loss_D_A + loss_D_B
        loss_D.backward()
        self.optimizer_D.step()

        # save image
        if save:
            self.save_image((A, B_gen, A_cyc, A_gt, B, A_gen, B_cyc, B_gt),
                            out_dir_img, "train_ep_%d_A%d_B%d" % (epoch, index_A, index_B))

        return {'G': loss_G,
                'G_A': loss_G_A, 'Cyc_A': loss_cyc_A, 'G_A_idt': loss_G_A_idt,
                'G_B': loss_G_B,  'Cyc_B': loss_cyc_B,  'G_B_idt': loss_G_B_idt,
                'D': loss_D, 'D_A': loss_D_A, 'D_B': loss_D_B}

    def eval(self, input, save, out_dir_img, epoch):
        # self.G_A.eval()
        # self.G_B.eval()
        # self.D_A.eval()
        # self.D_B.eval()

        with torch.no_grad():
            A, B, A_gt, B_gt, index_A, index_B = input


            ############################
            # G loss
            ############################

            # Identity loss
            if self.lambda_idt > 0:
                loss_G_A_idt = self.idt_loss_fn(self.G_A(B), B) * self.lambda_B * self.lambda_idt
                loss_G_B_idt = self.idt_loss_fn(self.G_B(A), A) * self.lambda_A * self.lambda_idt
            else:
                loss_G_A_idt = 0
                loss_G_B_idt = 0

            # GAN loss D_A(G_A(A))
            B_gen = self.G_A(A)
            loss_G_A = self.gan_loss(self.D_A(B_gen), 1)

            # GAN loss D_B(G_B(B))
            A_gen = self.G_B(B)
            loss_G_B = self.gan_loss(self.D_B(A_gen), 1)

            # Forward cycle loss
            A_cyc = self.G_B(B_gen)
            loss_cyc_A = self.cycle_loss_fn(A_cyc, A) * self.lambda_A

            # Backward cycle loss
            B_cyc = self.G_A(A_gen)
            loss_cyc_B = self.cycle_loss_fn(B_cyc, B) * self.lambda_B

            # Combine
            loss_G = loss_G_A + loss_G_B + loss_cyc_A + loss_cyc_B + loss_G_A_idt + loss_G_B_idt

            ############################
            # D loss
            ############################

            # D_A real loss
            loss_D_A_real = self.gan_loss(self.D_A(B), 1)

            # D_A fake loss
            B_gen_pool = self.B_gen_buffer.push_and_pop(B_gen).detach()
            loss_D_A_fake = self.gan_loss(self.D_A(B_gen_pool), 0)

            loss_D_A = (loss_D_A_real + loss_D_A_fake) * 0.5

            # D_B real loss
            loss_D_B_real = self.gan_loss(self.D_B(A), 1)

            # D_B fake loss
            A_gen_pool = self.A_gen_buffer.push_and_pop(A_gen).detach()
            loss_D_B_fake = self.gan_loss(self.D_B(A_gen_pool), 0)

            loss_D_B = (loss_D_B_real + loss_D_B_fake) * 0.5

            # TODO can add?
            loss_D = loss_D_A + loss_D_B

            # TODO batch avg?

        # save image
        if save:
            self.save_image((A, B_gen, A_cyc, A_gt, B, A_gen, B_cyc, B_gt),
                            out_dir_img, "val_ep_%d_A%d_B%d" % (epoch, index_A, index_B))

        return {'G': loss_G,
                'G_A': loss_G_A, 'Cyc_A': loss_cyc_A, 'G_A_idt': loss_G_A_idt,
                'G_B': loss_G_B, 'Cyc_B': loss_cyc_B, 'G_B_idt': loss_G_B_idt,
                'D': loss_D, 'D_A': loss_D_A, 'D_B': loss_D_B}

    def test(self, images, i, out_dir_img, collage=False):
        A, B, A_gt, B_gt, index_A, index_B = images
        B_gen = self.G_A(A)
        if collage:
            A_gen = self.G_B(B)
            A_cyc = self.G_B(B_gen)
            B_cyc = self.G_A(A_gen)

            self.save_image((A, B_gen, A_cyc, A_gt, B, A_gen, B_cyc, B_gt), out_dir_img, "test_%d" % (i+1))
        else:
            self.save_image(B_gen, out_dir_img, "test_%d" % (i+1), test=True)

    def compute_loss(self, A, B):
        B_gen = self.G_A(A)
        A_cyc = self.G_B(B_gen)
        A_gen = self.G_B(B)
        B_cyc = self.G_A(A_gen)


        loss_G_A = self.gan_loss(self.D_A(B_gen), 1) + self.cycle_loss_fn(A_cyc, A) * self.lambda_A


        loss_G_B = self.gan_loss(self.D_B(A_gen), 1) + self.cycle_loss_fn(B_cyc, B) * self.lambda_B

        loss_D_A = self.gan_loss(self.D_A(B), 1) + self.gan_loss(self.D_A(B_gen), 0) * 0.5 # TODO
        loss_D_B = self.gan_loss(self.D_B(A), 1) + self.gan_loss(self.D_B(A_gen), 0)
        return B_gen, loss_G_A, loss_G_B, loss_D_A, loss_D_B

    def gan_loss(self, out, label):
        return self.gan_loss_fn(out, torch.ones_like(out) if label else torch.zeros_like(out))

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
                'D_A': self.D_A.state_dict(),
                'D_B': self.D_B.state_dict(),
                'optimG': self.optimizer_G.state_dict(),
                'optimD': self.optimizer_D.state_dict()}

    def save_image(self, input, filepath, fname, test=False):
        """ input is a tuple of the images we want to compare """
        # A, B_gen, A_cyc, B, A_gen, B_cyc = input


        if test:
            B_gen= input
            img = self.tensor2image(B_gen)
            path = os.path.join(filepath, 'B_gen_%s.png' % fname)
            scipy.misc.imsave(path, img.squeeze().transpose(1,2,0))
            print('saved %s' % path)
        else:
            A, B_gen, A_cyc, A_gt, B, A_gen, B_cyc, B_gt = input

            sources = torch.cat((A, B))
            gens = torch.cat((B_gen, A_gen))
            cycles = torch.cat((A_cyc, B_cyc))
            gts = torch.cat((A_gt, B_gt))

            merged = self.tensor2image(self.merge_images(sources, gens, cycles, gts))

            path = os.path.join(filepath, '%s.png' % fname)
            scipy.misc.imsave(path, merged)
            print('saved %s' % path)

    def tensor2image(self, input):
        image_data = input.data
        image = 127.5 * (image_data.cpu().float().numpy() + 1.0)
        return image.astype(np.uint8)

    def merge_images(self, sources, gens, cycles, gts):
        row, _, h, w = sources.size()
        # row, _, h, w = sources.shape
        # row = int(np.sqrt(batch_size))
        merged = torch.zeros(3, row * h, w * 4)
        for idx, (s, gt, g, c) in enumerate(zip(sources, gts, gens, cycles)):
            i = idx
            # i = (idx + 1) // row
            # j = idx % row
            # merged[:, i * h:(i + 1) * h, (j * 2) * w:(j * 2 + 1) * w] = s
            # merged[:, i * h:(i + 1) * h, (j*2+1) * w:(j * 2 + 2) * w] = t
            # merged[:, i * h:(i + 1) * h, (j*2+2) * w:(j * 2 + 3) * w] = c
            merged[:, i * h:(i + 1) * h, 0:w] = s
            merged[:, i * h:(i + 1) * h, w:2 * w] = gt
            merged[:, i * h:(i + 1) * h, 2 * w:3 * w] = g
            merged[:, i * h:(i + 1) * h, 3 * w:4 * w] = c
        return merged.permute(1, 2, 0)

class ImageBuffer():
    """
    Buffer to store [buffer_size] generated images.
    """
    def __init__(self, buffer_size=50):
        assert (buffer_size > 0), "Buffer size must > 0"
        self.buffer_size = buffer_size
        self.buffer = []

    def push_and_pop(self, images):
        """
        Query buffer and return Tensor of same size randomly picked
        from image buffer.
        :param data:
        :return:
        """
        result = []
        for image in images:
            image = torch.unsqueeze(image, 0)
            if len(self.buffer) < self.buffer_size:
                self.buffer.append(image)
                result.append(image)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.buffer_size-1)
                    result.append(self.buffer[i].clone())
                    self.buffer[i] = image
                else:
                    result.append(image)
        return torch.cat(result, 0)



