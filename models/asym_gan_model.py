from argparse import ArgumentParser
import torch
import itertools
from . import networks
from . import base_model
from . import style_loss_net
from utils.image_pool import ImagePool


class AsymGANModel(base_model.BaseModel):
    """
    This class implements the AsymGAN model, for learning image-to-image translation between domains of asynchronous
    complexity.

    The model training requires '--dataset_mode unaligned_pairs' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    [1] Y. Li, S. Tang, R. Zhang, Y. Zhang, J. Li, and S. Yan, “Asymmetric GAN for Unpaired Image-to-Image
        Translation,” IEEE Trans. Image Process., vol. 28, no. 12, pp. 5881–5896, 2019, doi: 10.1109/TIP.2019.2922854.
    """
    @staticmethod
    def modify_commandline_options(parser: ArgumentParser, is_train: bool = True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser (Option) -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or
                               test-specific options.

        Returns:
            the modified parser.

        For AsymGAN, in addition to GAN losses, we introduce lambda_A, lambda_B and Lambda_E for the
        following losses.
        A (source domain), B (target domain), Z (auxiliary data).
        Generators: G_A: A -> B; E: A -> Z; G_B: B, Z -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A; D_Z: E(A) vs. Z.

        Z Adversarial Loss: Lambda_1 * E[log D_Z(Z)] - E[log(1 - D_Z(E(A))]                    (Eqn. (7) in the paper)
        A Cycle Loss:  lambda_2 * ||G_B(G_A(A), E(A)) - A||                                    (Eqn. (8) in the paper)
        B Cycle Loss:  lambda_3 * ||G_A(G_B(B, Z)) - B||                                       (Eqn. (10) in the paper)
        Z Cycle Loss:  Lambda_4 * ||E(G_B(B, Z)) - Z||                                         (Eqn. (11) in the paper)
        A Adversarial Loss against Y: Lambda_5 * E[log(D_X(X)) + log(1 - D_X(G_B(G_A(A), Z))]  (Eqn. (13) in the paper)
        A Adversarial loss against Z: Lambda_6 * E[log(D_X(X)) + log(1 - D_X(G_B(B, E(A)))]    (Eqn. (14) in the paper)
        Content Loss: Lambda_7 * avg(E[||φ_j(G_B(Y, E(A))) - φ_j(Y)||^2])                      (Eqn. (15) in the paper)
        Style Loss: Lambda_8 * E[P_φj(G_B(Y, E(A))) - P_φj(X)]_F^2                             (Eqn. (16) in the paper)
        Total Variation Loss: Lambda_9 * (ϕ(G_A(A)) + ϕ(G_B(G_A(A), E(A))) + ϕ(G_B(G_A(A), E))
                                    + ϕ(G_B(Y, E)) + ϕ(G_A(G_B(B, E))) + ϕ(G_A(G_B(B, E(A))))) (Eqn. (18) in the paper)
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_1', type=float, default=1.0, help='weight for equation 7 in paper')
            parser.add_argument('--lambda_2', type=float, default=1.0, help='weight for equation 8 in paper')
            parser.add_argument('--lambda_3', type=float, default=1.0, help='weight for equation 10 in paper')
            parser.add_argument('--lambda_4', type=float, default=10.0, help='weight for equation 11 in paper')
            parser.add_argument('--lambda_5', type=float, default=1.0, help='weight for equation 13 in paper')
            parser.add_argument('--lambda_6', type=float, default=1.0, help='weight for equation 14 in paper')
            parser.add_argument('--lambda_7', type=float, default=0.2, help='weight for equation 15 in paper')
            parser.add_argument('--lambda_8', type=float, default=0.1, help='weight for equation 16 in paper')
            parser.add_argument('--lambda_9', type=float, default=10.0, help='weight for equation 17 in paper')

            parser.add_argument('--netGB', type=str, default='resnet_with_aux',
                                help='specify generator architecture [resnet_with_aux]')
            parser.add_argument('--netGE', type=str, default='stunted_resnet',
                                help='specify generator architecture [stunted_resnet]')
            parser.add_argument('--netDE', type=str, default='patch20', help='specify generator architecture [patch20]')

        return parser

    def __init__(self, opt):
        """Initialize the AsymGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        base_model.BaseModel.__init__(self, opt)

        # specify the training losses you want to print out.
        self.loss_names = ['loss_D_A', 'loss_D_B', 'loss_D_Z', 'loss_E', 'loss_G_A', 'loss_G_B', 'loss_cycle_A',
                           'loss_cycle_B', 'loss_cycle_Z', 'loss_content', 'loss_style', 'loss_total_variation',
                           'loss_G']
        # specify the images you want to save/display.
        visual_names_a = ['real_A', 'fake_B', 'rec_A']
        visual_names_b = ['real_B', 'fake_A', 'rec_B']

        self.visual_names = visual_names_a + visual_names_b  # combine visualizations for A and B

        # specify the models you want to save to the disk...
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'E', 'D_A', 'D_B', 'D_E']
        else:  # during test time, only load generators
            self.model_names = ['G_A', 'G_B', 'E']

        # define networks (both Generators and discriminators)
        # Some of the naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X), E (E), D_Z (D_Z)
        self.netG_A = networks.define_generator(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                                not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_generator(opt.output_nc, opt.input_nc, opt.ngf, opt.netGB, opt.norm,
                                                not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netE = networks.define_generator(opt.input_nc, opt.input_nc, opt.ngf, opt.netGE, opt.norm,
                                              not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_discriminator(opt.output_nc, opt.ndf, opt.netD,
                                                        opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain,
                                                        self.gpu_ids)
            self.netD_B = networks.define_discriminator(opt.input_nc, opt.ndf, opt.netD,
                                                        opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain,
                                                        self.gpu_ids)
            self.netD_Z = networks.define_discriminator(opt.input_nc, opt.ndf, opt.netDE,
                                                        opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain,
                                                        self.gpu_ids)

        if self.isTrain:
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_Z_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.styleLossNet = style_loss_net.StyleLossNet()

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(),
                                                                self.netG_B.parameters(),
                                                                self.netE.parameters()),
                                                lr=opt.lr,
                                                betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(),
                                                                self.netD_B.parameters(),
                                                                self.netD_Z.parameters()),
                                                lr=opt.lr,
                                                betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

        self.real_A = None
        self.real_B = None
        self.real_Z = None
        self.fake_A = None
        self.fake_B = None
        self.fake_Z = None
        self.rec_A = None
        self.rec_A_real_Z = None
        self.rec_B = None
        self.rec_B_fake_Z = None
        self.fake_A_fake_Z = None
        self.rec_Z = None
        self.loss_D_A = None
        self.loss_D_B = None
        self.loss_D_Z = None
        self.loss_E = None
        self.loss_G_A = None
        self.loss_G_B = None
        self.loss_cycle_A = None
        self.loss_cycle_B = None
        self.loss_cycle_Z = None
        self.loss_content = None
        self.loss_style = None
        self.loss_total_variation = None
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
        self.real_Z = torch.randn(self.real_A.size()).to(self.device)  # TODO See if the size is right
        self.image_paths = values['A_paths' if a_to_b else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""

        # Forward pass, as shown in Fig 2 (1) in the paper...
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.fake_Z = self.netE(self.real_A)    # E(A)
        self.rec_A = self.netG_B.forward_with_parts(self.fake_B, self.fake_Z)          # G_B(G_A(X), Z(X))
        self.rec_A_real_Z = self.netG_B.forward_with_parts(self.fake_B, self.real_Z)   # G_B(G_A(X), E)

        # Backward pass, as shown in Fig 2 (2) in the paper...
        self.fake_A = self.netG_B.forward_with_parts(self.real_B, self.real_Z)         # G_B(Y, Z)
        self.rec_B = self.netG_A(self.fake_A)                                          # G_A(G_B(Y, Z))
        self.fake_A_fake_Z = self.netG_B.forward_with_parts(self.real_B, self.fake_Z)  # G_B(Y, E(X))
        self.rec_B_fake_Z = self.netG_A(self.fake_A_fake_Z)                            # G_A(G_B(Y, E(X)))
        self.rec_Z = self.netE(self.fake_A)                                            # E(G_B(Y, Z))

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

    def backward_d_z(self):
        """Calculate GAN loss for discriminator D_Z"""
        fake_z = self.fake_Z_pool.query(self.fake_Z)
        self.loss_D_Z = self.backward_d_basic(self.netD_Z, self.real_Z, fake_z)

    def backward_g(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_1 = self.opt.lambda_1
        lambda_2 = self.opt.lambda_2
        lambda_3 = self.opt.lambda_3
        lambda_4 = self.opt.lambda_4
        lambda_5 = self.opt.lambda_5
        lambda_6 = self.opt.lambda_6
        lambda_7 = self.opt.lambda_7
        lambda_8 = self.opt.lambda_8
        lambda_9 = self.opt.lambda_9

        # E[log D_Z(Z)] - E[log(1 - D_Z(E(A))]  (Eqn. (7) in the paper)
        self.loss_E = lambda_1 * self.criterionGAN(self.netD_Z(self.fake_Z), True)

        # ||G_B(G_A(A), E(A)) - A||  (Eqn. (8) in the paper)
        self.loss_cycle_A = lambda_2 * self.criterionCycle(self.rec_A, self.real_A)

        # ||G_A(G_B(B, Z)) - B|| (Eqn. (10) in the paper)
        self.loss_cycle_B = lambda_3 * self.criterionCycle(self.rec_B, self.real_B)

        # ||E(G_B(B, Z)) - Z|| (Eqn. (11) in the paper)
        self.loss_cycle_Z = lambda_4 * self.criterionCycle(self.rec_Z, self.real_Z)

        # E[log(D_X(X)) + log(1 - D_X(G_B(G_A(A), Z))]  (Eqn. (13) in the paper)
        self.loss_G_A = lambda_5 * self.criterionGAN(self.netD_A(self.fake_B), True)

        # E[log(D_X(X)) + log(1 - D_X(G_B(B, E(A)))]    (Eqn. (14) in the paper)
        self.loss_G_B = lambda_6 * self.criterionGAN(self.netD_B(self.fake_A), True)

        # Get style and content losses.
        # Content Loss: avg(E[||φ_j(G_B(Y, E(A))) - φ_j(Y)||^2]) (Eqn. (15) in the paper)
        # Style Loss: E[P_φj(G_B(Y, E(A))) - P_φj(X)]_F^2 (Eqn. (16) in the paper)
        # Total Variation Loss: (ϕ(G_A(A)) + ϕ(G_B(G_A(A), E(A))) + ϕ(G_B(G_A(A), E))
        #                           + ϕ(G_B(Y, E)) + ϕ(G_A(G_B(B, E))) + ϕ(G_A(G_B(B, E(A))))) (Eqn. (18) in the paper)
        self.loss_style, self.loss_content, self.loss_total_variation = \
            self.styleLossNet.calculate_losses(self.real_A, self.fake_A, self.fake_A_fake_Z, self.rec_A,
                                               self.real_B, self.rec_B_fake_Z, self.rec_B)

        self.loss_content = lambda_7 * self.loss_content
        self.loss_style = lambda_8 * self.loss_style
        self.loss_total_variation = lambda_9 * self.loss_total_variation

        # combined loss and calculate gradients
        self.loss_G = self.loss_D_Z + self.loss_cycle_A + self.loss_cycle_B + self.loss_cycle_Z + self.loss_G_A + \
            self.loss_G_B + self.loss_content + self.loss_style + self.loss_total_variation
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.

        # G_A, G_B and E: discriminators require no gradients when optimizing generators...
        self.set_requires_grad([self.netD_A, self.netD_B, self.netD_Z], False)
        self.optimizer_G.zero_grad()  # set G_A, G_B and E's gradients to zero
        self.backward_g()             # calculate gradients for G_A, G_B and E
        self.optimizer_G.step()       # update G_A, G_B and E's weights

        # D_A, D_B and D_Z
        self.set_requires_grad([self.netD_A, self.netD_B, self.netD_Z], True)
        self.optimizer_D.zero_grad()   # set D_A, D_B and D_Z's gradients to zero
        self.backward_d_a()      # calculate gradients for D_A
        self.backward_d_b()      # calculate gradients for D_B
        self.backward_d_z()      # calculate gradients for D_Z
        self.optimizer_D.step()  # update D_A, D_B and D_Z's weights
