import torch
import itertools
import os
from . import networks
from . import base_model
from collections import OrderedDict
from utils.image_pool import ImagePool


class CycleGANModel(base_model.BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser (Option) -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or
                               test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A and lambda_B for the
        following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        base_model.BaseModel.__init__(self, opt)

        # specify the training losses you want to print out.
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'D_B', 'G_B', 'cycle_B']
        # specify the images you want to save/display.
        visual_names_a = ['real_A', 'fake_B', 'rec_A']
        visual_names_b = ['real_B', 'fake_A', 'rec_B']

        self.visual_names = visual_names_a + visual_names_b  # combine visualizations for A and B

        # specify the models you want to save to the disk...
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load generators
            self.model_names = ['G_A', 'G_B']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_generator(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_generator(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_discriminator(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_discriminator(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=opt.lr,
                                                betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                                                lr=opt.lr,
                                                betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

        self.real_A = None
        self.real_B = None
        self.fake_A = None
        self.fake_B = None
        self.rec_A = None
        self.rec_B = None
        self.loss_D_A = None
        self.loss_D_B = None
        self.loss_G_A = None
        self.loss_G_B = None
        self.loss_cycle_A = None
        self.loss_cycle_B = None
        self.loss_G = None

    def set_input(self, values: dict):
        """Unpack input data from the data loader and perform necessary pre-processing steps.

        Parameters:
            values (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        a_to_b = self.opt.direction == 'AtoB'
        self.real_A = values['A' if a_to_b else 'B'].to(self.device)
        self.real_B = values['B' if a_to_b else 'A'].to(self.device)
        self.image_paths = values['A_paths' if a_to_b else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))

    def backward_d_basic(self, net_d, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            net_d (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = net_d(real)
        loss_d_real = self.criterionGAN(pred_real, True)

        # Fake
        pred_fake = net_d(fake.detach())
        loss_d_fake = self.criterionGAN(pred_fake, False)

        # Combined loss and calculate gradients
        loss_d = (loss_d_real + loss_d_fake) * 0.5
        loss_d.backward()
        return loss_d

    def backward_d_a(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_b = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_d_basic(self.netD_A, self.real_B, fake_b)

    def backward_d_b(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_a = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_d_basic(self.netD_B, self.real_A, fake_a)

    def backward_g(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_a = self.opt.lambda_A
        lambda_b = self.opt.lambda_B

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_a
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_b
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.

        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_g()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights

        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_d_a()      # calculate gradients for D_A
        self.backward_d_b()      # calculate gradients for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights
