import warnings

import cv2
import math
import numpy as np
import torch
import torch.nn.functional as F

from .SRC import SRC_Loss
from .apim_loss import AdaptivePIMLoss
from .base_model import BaseModel
from . import networks
from .dn import init_dense_instance_norm, use_dense_instance_norm
from .hDCE import PatchHDCELoss
from .losses import NCC_Loss, NMI_Loss
from .patch_alignment_loss import PatchAlignmentLoss
from .patchnce import PatchNCELoss, FDL_loss
import util.util as util
import models.voxelmorph.torchvoxelmorph as vxm
from models.voxelmorph.torchvoxelmorph.layers import SpatialTransformer
from PIL import Image
from torchvision.transforms import Compose, CenterCrop, ToTensor, Normalize
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from collections import deque

from .utils import SCLossCriterion


def open_image_to_torch(path, size):
    t = Image.open(path)
    transforms = []
    transforms.append(CenterCrop(size))
    transforms.append(ToTensor())
    transforms.append(Normalize(mean=[0.5], std=[0.5]))
    transforms = Compose(transforms)
    t = transforms(t)
    t = t.unsqueeze(0)
    return t


def smooothing_loss(y_pred):
    dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
    dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])

    dx = torch.mul(dx, dx)
    dy = torch.mul(dy, dy)
    d = torch.mean(dx) + torch.mean(dy)
    return d / 2.0


class REGISTRATIONModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """
        parser.add_argument('--CUT_mode', type=str, default="CUT", choices='(CUT, cut, FastCUT, fastcut)')

        parser.add_argument('--lambda_GAN', type=float, default=1, help='weight for GAN loss：GAN(G(X))')

        parser.add_argument('--dce_idt', type=util.str2bool, nargs='?', const=True, default=True,
                            help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--nce_layers', type=str, default='0,4,8,12,16',
                            help='compute NCE loss on which layers')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'],
                            help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")
        parser.add_argument('--isDecay', type=bool, default=True,
                            help='gradually decrease the weight for the supervised training branch')
        parser.add_argument('--use_lambda_pair', action='store_true',
                            help='If specified, use lambda_pair to blend between unregistered and registered images.')


        # ------------------ Added Arguments for Multi-Level Adversarial Losses ------------------ #
        parser.add_argument('--lambda_pixel', type=float, default=1, help='Weight for pixel-level L1 loss after deformation')
        parser.add_argument('--w_barrier', type=float, default=5)
        parser.add_argument('--lamba_MMR', type=float, default=0.0)
        parser.add_argument('--lambda_smooth', type=float, default=1)
        parser.add_argument('--lambda_pixel_L1', type=float, default=1, help='Weight for pixel-level L1 loss after deformation')
        parser.add_argument('--self_regularization', type=float, default=0.03, help='loss between input and generated image')


        parser.add_argument('--lambda_patch', type=float, default=1.0,
                            help='Weight for patch-level APIM and PATCH GAN loss after deformation')
        parser.add_argument('--lambda_HDCE', type=float, default=0.2, help='weight for HDCE loss: HDCE(G(X), X)')
        parser.add_argument('--lambda_SRC', type=float, default=0.1, help='weight for SRC loss: SRC(G(X), X)')
        parser.add_argument('--use_curriculum', action='store_true')
        parser.add_argument('--HDCE_gamma', type=float, default=50)
        parser.add_argument('--HDCE_gamma_min', type=float, default=10)
        parser.add_argument('--step_gamma', action='store_true')
        parser.add_argument('--step_gamma_epoch', type=int, default=200)
        parser.add_argument('--no_Hneg', action='store_true')
        parser.add_argument('--lambda_apim', type=float, default=0.0, help='weight for APIM loss')
        parser.add_argument('--asp_loss_mode', type=str, default='linear_top',
                            help='"scheduler_lookup" options for the APIM loss.')
        parser.add_argument('--lambda_misalignment', type=float, default=1.0,
                            help='weight for patch misalignment loss')


        parser.add_argument('--lambda_global', type=float, default=1.0,
                            help='weight for Global loss.')
        parser.add_argument('--lambda_content', type=float, default=0.01,
                            help='weight for content loss.')
        parser.add_argument('--lambda_style', type=float, default=5.0,
                            help='weight for style loss.')
        parser.add_argument('--lambda_FDL', type=float, default=0.001,
                            help='weight for NCE loss: NCE(G(X), X)')

        # ----------------------------------------------------------------------------------------------- #

        parser.set_defaults(pool_size=0)  # no image pooling

        opt, _ = parser.parse_known_args()

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.N_EPOCHS = opt.n_epochs + opt.n_epochs_decay
        # ------------------ Modified loss_names to include multi-level adversarial losses ------------------ #
        self.loss_names = ['G', 'D_fake', 'D_real']  # Existing generator and discriminator losses
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

        if opt.lambda_HDCE > 0.0:
            self.loss_names.append('HDCE')
            self.loss_names.append('HDCE_hat')
            if opt.dce_idt and self.isTrain:
                self.loss_names += ['HDCE_Y']

        if opt.dce_idt and self.isTrain:
            self.visual_names += ['idt_B']

        if self.isTrain:
            # Adding multi-level adversarial loss names
            if self.opt.lambda_pixel > 0.0:
                self.loss_names += ['smooth', 'FDL', 'pixel', 'patch', 'global']
                self.visual_names += ['dvf', 'registered']
            else:
                self.loss_names += ['patch', 'global']

        if self.isTrain:
            self.model_names = ['G', 'F', 'R', 'D']  # Existing discriminator
        else:  # during test time, only load G
            self.model_names = ['G']

        if opt.lambda_FDL > 0.0:
            self.loss_names += ['FDL']
            self.criterionFDL = FDL_loss().to(self.device)

        if opt.lambda_SRC > 0.0:
            self.loss_names.append('SRC')

        if opt.lambda_apim > 0.0:
            self.loss_names.append('APIM')
            self.criterionAPIM = AdaptivePIMLoss(opt).to(self.device)

        if opt.lambda_misalignment > 0.0:
            self.loss_names.append('patch_alignment')

        if self.opt.lambda_pixel > 0.0 and opt.lambda_pixel_L1 > 0.0:
            self.loss_names.append('pixel_L1')

        if opt.lambda_content > 0.0:
            self.loss_names.append('content')

        if opt.lambda_style > 0.0:
            self.loss_names.append('style')



        # ------------------ Define Additional Loss Functions ------------------ #
        self.criterionL1 = torch.nn.L1Loss().to(self.device)  # Pixel-level L1 Loss after deformation
        self.criterionStyle = SCLossCriterion(self.device)    # Global-level Style Loss after deformation

        # ------------------------------------------------------------------------ #

        self.epoch = 0
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias,
                                      opt.no_antialias_up, self.gpu_ids, opt)
        self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout,
                                      opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

        nb_features = [
            [16, 32, 32, 64, 64, 64],  # encoder
            [64, 64, 64, 32, 32, 32, 16]  # decoder
        ]
        vol_shape = (opt.crop_size, opt.crop_size)
        self.netR = vxm.networks.VxmDense(ndims=2, nb_unet_features=nb_features, int_steps=7, bidir=True,
                                         int_downsize=2, inshape=vol_shape).cuda()
        self.netR.train()
        self.spatialTransformer = SpatialTransformer(vol_shape).cuda()
        self.window_size = 100
        self.score_threshold = -np.inf
        self.sliding_window = deque(maxlen=self.window_size)

        self.every_epoch_score = []
        self.pre_epoch = 0
        self.loss_median_score = None

        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                         opt.n_layers_D, opt.normD,
                                         opt.init_type, opt.init_gain,
                                         opt.no_antialias, self.gpu_ids, opt)

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = PatchNCELoss(opt).to(self.device)
            self.criterionSRC = []
            self.criterionHDCE = []
            self.criterionR = []
            self.criterionGlobal = SCLossCriterion(self.device)
            self.criterionNMI = NMI_Loss([0.05, 0.1, 0.15, 0.2, 0.25, 0.3,
                                         0.35, 0.4, 0.45, 0.5, 0.55, 0.6,
                                         0.65, 0.7, 0.75, 0.8, 0.85, 0.9,
                                         0.95], device=self.device)
            self.criterionMisalignment = PatchAlignmentLoss()


            for nce_layer in self.nce_layers:
                # self.criterionNCE.append(PatchNCELoss(opt).to(self.device))
                self.criterionSRC.append(SRC_Loss(opt).to(self.device))
                self.criterionHDCE.append(PatchHDCELoss(opt=opt).to(self.device))
                self.criterionR.append(SRC_Loss(opt).to(self.device))

            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            # self.criterionNCC = MS_SSIM().to(self.device)

            # ------------------ Modified Optimizer Definitions to Include New Losses ------------------ #
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_R = torch.optim.Adam(self.netR.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            # ---------------------------------------------------------------------------------------------------------------- #

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_R)
            self.optimizers.append(self.optimizer_D)


    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        self.set_input(data)
        bs_per_gpu = self.real_A.size(0) // max(len(self.opt.gpu_ids), 1)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()                     # compute fake images: G(A)
        if self.opt.isTrain:
            # self.compute_D_loss().backward()  # calculate gradients for D
            self.compute_G_loss().backward()  # calculate gradients for G

            if self.opt.netF == 'mlp_sample':
                self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr,
                                                    betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)

            if self.opt.lambda_apim > 0.0:
                self.calculate_APIM_loss(self.real_B, self.real_B) * 0.05

    def visual_tensor2im(self, tensors, nce_layer):
        new_tensors = [item.detach().cpu().float().numpy() for item in tensors]
        subplot_lines = new_tensors[0].shape[0]
        subplot_columns = len(new_tensors)
        plt.subplots(subplot_lines, subplot_columns)
        for i in range(subplot_lines):
            for j in range(subplot_columns):
                plt.subplot(subplot_lines, subplot_columns, i * subplot_columns + j + 1)
                # if j == subplot_columns - 1:
                #     plt.imshow(new_tensors[j][i].squeeze())
                # else:

                plt.imshow(new_tensors[j][i].transpose(1, 2, 0))
                plt.axis('off')
        plt.title(f'Layer {nce_layer}')
        plt.show()

    def calculate_FDL_loss(self, src, tgt):
        # 提取特征
        total_fdl_loss = self.criterionFDL(src, tgt, 1) * self.opt.lambda_FDL
        return total_fdl_loss

    def calculate_L1_loss(self, src, tgt, mask=None):
        diff = torch.abs(src - tgt)
        if mask is None:
            return torch.mean(diff)
        elif torch.sum(mask) == 0:
            return torch.tensor(0)
        else:
            norm_factor = 1 / (torch.sum(mask))
            return norm_factor * torch.sum(diff * mask)

    def calculate_HDCE_loss(self, src, tgt, weight=None):
        n_layers = len(self.nce_layers)

        feat_q_pool = tgt
        feat_k_pool = src

        total_HDCE_loss = 0.0
        for f_q, f_k, crit, nce_layer, w in zip(feat_q_pool, feat_k_pool, self.criterionHDCE, self.nce_layers, weight):
            if self.opt.no_Hneg:
                w = None
            loss = crit(f_q, f_k, w) * self.opt.lambda_HDCE
            total_HDCE_loss += loss.mean()

        return total_HDCE_loss / n_layers

    def optimize_parameters(self):
        # Forward pass
        self.forward()

        if self.isTrain:
            # ------------------ Update Discriminator ------------------ #
            self.set_requires_grad(self.netD, True)
            self.optimizer_D.zero_grad()
            self.loss_D = self.compute_D_loss()
            self.loss_D.backward()
            self.optimizer_D.step()
            self.set_requires_grad(self.netD, False)
            # ------------------------------------------------------- #

            # ------------------ Update Generator G and Registration R ------------------ #
            self.optimizer_G.zero_grad()
            self.optimizer_R.zero_grad()

            if self.opt.netF == 'mlp_sample':
                self.optimizer_F.zero_grad()

            self.loss_G = self.compute_G_loss()
            self.loss_G.backward()

            self.optimizer_G.step()
            self.optimizer_R.step()

            if self.opt.netF == 'mlp_sample':
                self.optimizer_F.step()
            # --------------------------------------------------------------------------------- #

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.real = torch.cat((self.real_A, self.real_B), dim=0)
        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])

        self.fake = self.netG(self.real)
        self.fake_B = self.fake[:self.real_A.size(0)]
        self.idt_B = self.fake[self.real_A.size(0):]

        # ------------------ Apply Deformation (Registration) ------------------ #
        y_output = self.netR(self.fake_B, self.real_B)
        self.deformation_field = y_output[2]
        self.registered = self.spatialTransformer(self.fake_B, self.deformation_field)
        test_image = open_image_to_torch("./deform256.jpg", 256).cuda()
        self.dvf = self.spatialTransformer(test_image, self.deformation_field)
        # ---------------------------------------------------------------------------- #

    def get_lambda_pair(self):
        if self.opt.use_lambda_pair:
            total_epochs = self.opt.n_epochs + self.opt.n_epochs_decay
            t = (self.epoch - 1) / float(total_epochs)
            lambda_pair = 1.0 - math.sin(t * math.pi / 2)
        else:
            lambda_pair = 0.0
        return lambda_pair
    # ------------------ Modified compute_D_loss to use registered images ------------------ #
    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator using deformed generated images"""
        # Use the deformed (registered) fake images for discriminator
        lambda_pair = self.get_lambda_pair()

        fake = self.registered.detach()
        fake_B = self.fake_B.detach()
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake_reg = self.netD(fake)
        pred_fake_unreg = self.netD(fake_B) if self.opt.use_lambda_pair else None
        self.loss_D_fake_reg = self.criterionGAN(pred_fake_reg, False).mean()
        self.loss_D_fake_unreg = self.criterionGAN(pred_fake_unreg, False).mean() if self.opt.use_lambda_pair else None
        self.loss_D_fake = (1 - lambda_pair) * self.loss_D_fake_reg + lambda_pair * self.loss_D_fake_unreg if self.opt.use_lambda_pair else self.loss_D_fake_reg

        # Real
        pred_real = self.netD(self.real_B)
        self.loss_D_real = self.criterionGAN(pred_real, True).mean()

        # Combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D
    # ---------------------------------------------------------------------------------------------------------------- #

    def calculate_R_loss(self, src, tgt, only_weight=False, epoch=None):
        n_layers = len(self.nce_layers)

        feat_q_pool = tgt
        feat_k_pool = src

        total_SRC_loss = 0.0
        weights = []
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionR, self.nce_layers):
            loss_SRC, weight = crit(f_q, f_k, only_weight, epoch)
            total_SRC_loss += loss_SRC * self.opt.lambda_SRC
            weights.append(weight)
        return total_SRC_loss / n_layers, weights

    # ------------------ Modified compute_G_loss to include multi-level adversarial losses ------------------ #
    def compute_G_loss(self):
        """Calculate multiple adversarial and non-adversarial losses for the generator"""
        fake = self.registered  # Use deformed (registered) fake images for adversarial losses

        lambda_pair = self.get_lambda_pair()
        # print(lambda_pair)
        # ------------------ Adversarial Loss ------------------ #
        if self.opt.lambda_GAN > 0.0:
            if self.opt.lambda_pixel > 0.0:
                pred_fake_reg = self.netD(fake)
                pred_fake_unreg = self.netD(self.fake_B) if self.opt.use_lambda_pair else None
                self.loss_G_GAN = ((1 - lambda_pair) * self.criterionGAN(pred_fake_reg, True).mean()
                               + lambda_pair * self.criterionGAN(pred_fake_unreg, True).mean()) \
                              * self.opt.lambda_GAN if self.opt.use_lambda_pair else self.criterionGAN(pred_fake_reg, True).mean() * self.opt.lambda_GAN
            else:
                pred_fake = self.netD(self.fake_B)
                self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0

        # ------------------ Pixel-Level L1 Loss ------------------ #
        if self.opt.lambda_pixel > 0.0:
            if self.opt.use_lambda_pair:
                loss_pixel_L1_unreg = self.criterionL1(self.fake_B, self.real_B)
                loss_pixel_L1_reg = self.criterionL1(self.registered, self.real_B)
                self.loss_pixel_L1 = (1 - lambda_pair) * loss_pixel_L1_reg + lambda_pair * loss_pixel_L1_unreg
            else:
                self.loss_pixel_L1 = self.criterionL1(self.registered, self.real_B)

            self.loss_pixel_L1 *= self.opt.lambda_pixel_L1

            # Corrected typo: lambda_MMR
            if self.opt.lambda_MMR > 0.0:
                nmi_loss = self.criterionNMI(self.registered, self.real_B)
                barrier = self.opt.w_barrier - nmi_loss
                # Ensure barrier > 0 to avoid log domain error
                barrier = torch.clamp(barrier, min=1e-6)
                self.loss_MMR = -torch.log(barrier) * self.opt.lambda_MMR
            else:
                self.loss_MMR = 0.0

            # Corrected typo: smoothing_loss
            self.loss_smooth = smooothing_loss(self.deformation_field) * self.opt.lambda_smooth

            self.loss_pixel = self.loss_pixel_L1 + self.loss_MMR + self.loss_smooth
            if self.opt.self_regularization > 0.0:
                self.loss_SR = self.opt.self_regularization * self.calculate_SR_loss(self.real_A, self.fake_B)
            else:
                self.loss_SR = 0.0

            # If self.loss_SR is intended to be part of loss_pixel, include it
            self.loss_pixel += self.loss_SR
            self.loss_pixel = self.loss_pixel * self.opt.lambda_pixel

        else:
            self.loss_pixel = 0.0


        # ------------------ Patch-Level APIM and PATCH GAN Loss ------------------ #

        if self.opt.lambda_patch > 0.0:

            if self.opt.lambda_apim > 0.0:
                # Adaptive PIM Loss
                if self.opt.lambda_pixel > 0.0:
                    self.loss_APIM = self.calculate_APIM_loss(self.registered, self.real_B)
                else:
                    self.loss_APIM = self.calculate_APIM_loss(self.fake_B, self.real_B)
            else:
                self.loss_APIM = 0.0

            if self.opt.lambda_misalignment > 0.0:
                if self.opt.use_lambda_pair:
                    if self.opt.lambda_pixel > 0.0:
                        self.loss_patch_unreg = lambda_pair * self.criterionMisalignment(self.real_B, self.fake_B)
                        self.loss_patch_reg = (1 - lambda_pair) * self.criterionMisalignment(self.real_B, self.registered)
                        self.loss_patch_alignment = self.loss_patch_unreg + self.loss_patch_reg
                    else:
                        self.loss_patch_alignment = self.criterionMisalignment(self.real_B, self.fake_B)
                else:
                    if self.opt.lambda_pixel > 0.0:
                        self.loss_patch_alignment = self.criterionMisalignment(self.real_B, self.registered)
                    else:
                        self.loss_patch_alignment = self.criterionMisalignment(self.real_B, self.fake_B)
                self.loss_patch_alignment = self.loss_patch_alignment * self.opt.lambda_misalignment

            else:
                self.loss_patch_alignment = 0.0

            if self.opt.lambda_HDCE > 0.0:
                fake_B_feat = self.netG(self.fake_B, self.nce_layers, encode_only=True)  # Use registered fake_B
                if self.opt.flip_equivariance:
                    fake_B_feat = [torch.flip(fq, [3]) for fq in fake_B_feat]
                real_A_feat = self.netG(self.real_A, self.nce_layers, encode_only=True)
                fake_B_pool, sample_ids = self.netF(fake_B_feat, self.opt.num_patches, None)
                real_A_pool, _ = self.netF(real_A_feat, self.opt.num_patches, sample_ids)
                self.loss_HDCE_Y = 0.0
                if self.opt.dce_idt:
                    idt_B_feat = self.netG(self.idt_B, self.nce_layers, encode_only=True)
                    if self.opt.flip_equivariance and self.flipped_for_equivariance:
                        idt_B_feat = [torch.flip(fq, [3]) for fq in idt_B_feat]
                    real_B_feat = self.netG(self.real_B, self.nce_layers, encode_only=True)

                    idt_B_pool, _ = self.netF(idt_B_feat, self.opt.num_patches, sample_ids)
                    real_B_pool, _ = self.netF(real_B_feat, self.opt.num_patches, sample_ids)
                    _, weight_idt = self.calculate_R_loss(real_B_pool, idt_B_pool, only_weight=True, epoch=self.epoch)
                    self.loss_HDCE_Y = self.calculate_HDCE_loss(real_B_pool, idt_B_pool, weight_idt)
                    self.loss_HDCE_Y_hat = self.calculate_HDCE_loss(idt_B_pool, real_B_pool, weight_idt)

                else:
                    idt_B_pool, real_B_pool = None, None
                    self.loss_HDCE_Y = 0.0

                self.loss_SRC, weight = self.calculate_R_loss(real_A_pool, fake_B_pool, epoch=self.epoch)

                self.loss_HDCE = self.calculate_HDCE_loss(real_A_pool, fake_B_pool, weight)
                self.loss_HDCE_hat = self.calculate_HDCE_loss(fake_B_pool, real_A_pool, weight)
                loss_HDCE_both = (self.loss_HDCE + self.loss_HDCE_Y + self.loss_HDCE_hat + self.loss_HDCE_Y_hat) / 4
            else:
                loss_HDCE_both = 0.0

            self.loss_patch = self.loss_APIM + loss_HDCE_both + self.loss_SRC
            self.loss_patch = self.loss_patch * self.opt.lambda_patch
        else:
            self.loss_patch = 0.0

        # ------------------ Global-Level Style Loss ------------------ #
        if self.opt.lambda_global > 0.0:

            if self.opt.lambda_content > 0.0 or self.opt.lambda_style > 0.0:
                if self.opt.lambda_pixel > 0.0:
                    target = fake  # Assuming 'fake' is intended here
                else:
                    target = self.fake_B
                self.loss_content, self.loss_style = self.criterionStyle(self.real_A, self.real_B, self.fake_B, target)
                self.loss_content *= self.opt.lambda_content
                self.loss_style *= self.opt.lambda_style
            else:
                self.loss_content, self.loss_style = 0.0, 0.0

            if self.opt.lambda_FDL > 0.0:
                target_fdl = self.registered if self.opt.lambda_pixel > 0.0 else self.fake_B
                self.loss_FDL = self.calculate_FDL_loss(self.real_B, target_fdl)
            else:
                self.loss_FDL = 0.0

            self.loss_global = self.loss_content + self.loss_style + self.loss_FDL
            self.loss_global = self.loss_global * self.opt.lambda_global

        else:
            self.loss_global = 0.0

        # ------------------ Aggregate All Losses ------------------ #
        self.loss_G = self.loss_G_GAN + self.loss_pixel + self.loss_patch + self.loss_global
        # --------------------------------------------------------- #

        return self.loss_G
    # ---------------------------------------------------------------------------------------------------------------- #

    def calculate_APIM_loss(self, src, tgt):
        n_layers = len(self.nce_layers)
        feat_q = self.netG(tgt, self.nce_layers, encode_only=True)

        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        feat_k = self.netG(src, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)

        total_loss = 0.0
        for f_q, f_k, nce_layer in zip(feat_q_pool, feat_k_pool, self.nce_layers):
            loss = self.criterionAPIM(f_q, f_k, self.epoch) * self.opt.lambda_apim

            total_loss += loss.mean()

        return total_loss / n_layers

    def calculate_L1_loss_soft(self, src, tgt, mask=None, weight_map=None, epoch=None):
        """
        Calculate L1 loss with soft weights, scheduled based on epoch.

        Args:
            src (torch.Tensor): Source tensor (predicted image) [batch, c, h, w]
            tgt (torch.Tensor): Target tensor (ground truth image) [batch, c, h, w]
            mask (torch.Tensor, optional): Optional mask tensor [batch, 1, h, w] for valid regions (default: None)
            weight_map (torch.Tensor, optional): Optional soft weight map tensor [batch, 1, h, w] (default: None)
            epoch (int, optional): Current training epoch (default: None)

        Returns:
            torch.Tensor: Weighted L1 loss
        """
        if epoch is None:
            raise ValueError("Epoch must be provided for scheduled weighting.")

        # 计算源图像和目标图像之间的差异
        diff = torch.abs(src - tgt)

        # 计算归一化的训练进度 t (0 到 1)
        t = (epoch - 1) / self.opt.n_epochs + self.opt.n_epochs_decay  # 确保 total_epochs 是 opt.n_epochs + opt.n_epochs_decay

        # 线性混合权重：从均匀权重过渡到 weight_map
        if weight_map is not None:
            # 混合权重
            mixed_weight = t * weight_map + (1 - t) * torch.ones_like(weight_map)
            # 确保权重在 [0, 1] 之间
            mixed_weight = torch.clamp(mixed_weight, min=0.0, max=1.0)
        else:
            mixed_weight = torch.ones_like(diff)

        # 应用混合权重
        weighted_diff = diff * mixed_weight

        # 如果提供了 mask，则仅在有效区域计算损失
        if mask is not None:
            if torch.sum(mask) == 0:
                return torch.tensor(0.0, device=src.device)
            norm_factor = 1.0 / torch.sum(mask)
            weighted_diff = weighted_diff * mask
            loss = norm_factor * torch.sum(weighted_diff)
        else:
            loss = torch.mean(weighted_diff)

        return loss

    def init_dense_instance_norm_for_whole_model(
        self,
        y_anchor_num,
        x_anchor_num,
    ):
        init_dense_instance_norm(
            self.netG,
            y_anchor_num=y_anchor_num,
            x_anchor_num=x_anchor_num,
        )

    def inference_with_anchor(self, X, y_anchor, x_anchor, padding=1, **kwargs):
        self.eval()
        with torch.no_grad():
            X = X.to(self.device)
            self.netG.train()
            Y_fake_no_anchor = self.netG(X)
            self.netG.eval()
            Y_fake = self.netG.forward_with_anchor(
                X, y_anchor=y_anchor, x_anchor=x_anchor, padding=padding, **kwargs,
            )
        return Y_fake, Y_fake_no_anchor

    def use_dense_instance_norm_for_whole_model(self):
        use_dense_instance_norm(self.netG, padding=1)

    def calculate_SR_loss(self,src,tgt):
        #rgb_mean_src = torch.mean(src,dim=1) #mean over color channel
        #rgb_mean_tgt = torch.mean(tgt,dim=1)
        diff_chan = src-tgt
        batch_mean = torch.mean(diff_chan,dim=0)
        rgb_sum = torch.sum(batch_mean,0)
        batch_mean2 = torch.mean(torch.mean(rgb_sum,0),0)

        return batch_mean2