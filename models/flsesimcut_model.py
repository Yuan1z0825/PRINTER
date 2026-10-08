import numpy as np
import torch

from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks, losses
from .patchnce import PatchNCELoss
import util.util as util


class FLSESIMCUTModel(BaseModel):
    """ This class implements CUT and FastCUT model, described in the paper
    Contrastive Learning for Unpaired Image-to-Image Translation
    Taesung Park, Alexei A. Efros, Richard Zhang, Jun-Yan Zhu
    ECCV, 2020

    The code borrows heavily from the PyTorch implementation of CycleGAN
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """
        parser.add_argument('--CUT_mode', type=str, default="CUT", choices='(CUT, cut, FastCUT, fastcut)')

        parser.add_argument('--attn_layers', type=str, default='4, 7, 9', help='compute spatial loss on which layers')
        parser.add_argument('--patch_size', type=int, default=64, help='patch size to calculate the attention')
        parser.add_argument('--patch_nums', type=float, default=256, help='select how many patches for shape consistency, -1 use all')
        parser.add_argument('--loss_mode', type=str, default='cos', help='which loss type is used, cos | l1 | info')
        parser.add_argument('--use_norm', action='store_true', help='normalize the feature map for FLSeSim')
        parser.add_argument('--learned_attn', action='store_true', help='use the learnable attention map')
        parser.add_argument('--augment', action='store_true', help='use data augmentation for contrastive learning')
        parser.add_argument('--lambda_spatial', type=float, default=10.0, help='weight for spatially-correlative loss')
        parser.add_argument('--lambda_spatial_idt', type=float, default=0.0, help='weight for idt spatial loss')
        parser.add_argument('--T', type=float, default=0.07, help='temperature for similarity')


        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=False, help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--nce_layers', type=str, default='0,4,8,12,16', help='compute NCE loss on which layers')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'], help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--self_regularization', type=float, default=0.03, help='loss between input and generated image')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")

        parser.set_defaults(pool_size=0)  # no image pooling

        opt, _ = parser.parse_known_args()

        # Set default parameters for CUT and FastCUT
        if opt.CUT_mode.lower() == "cut":
            parser.set_defaults(nce_idt=True, lambda_NCE=1.0)
        elif opt.CUT_mode.lower() == "fastcut":
            parser.set_defaults(
                nce_idt=False, lambda_NCE=10.0, flip_equivariance=True,
                n_epochs=150, n_epochs_decay=50
            )
        else:
            raise ValueError(opt.CUT_mode)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'NCE',  'SR']
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

        if opt.nce_idt and self.isTrain:
            self.loss_names += ['NCE_Y']
            self.visual_names += ['idt_B']

        if self.isTrain:
            self.model_names = ['G', 'F', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']

        self.attn_layers = [int(i) for i in self.opt.attn_layers.split(',')]

        self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
        self.netPre = losses.VGG16().to(self.device)

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = []
            self.criterionSpatial = losses.SpatialCorrelativeLoss(opt.loss_mode, opt.patch_nums, opt.patch_size, opt.use_norm,
                                    opt.learned_attn, gpu_ids=self.gpu_ids, T=opt.T).to(self.device)
            self.normalization = losses.Normalization(self.device)

            if opt.learned_attn:
                self.netF_SR = self.criterionSpatial
                self.model_names.append('F_SR')
                self.loss_names.append('spatial')
            else:
                self.set_requires_grad([self.netPre], False)

            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))

            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
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
            self.compute_D_loss().backward()                  # calculate gradients for D
            self.compute_G_loss().backward()                   # calculate graidents for G
            if self.opt.lambda_NCE > 0.0:
                self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)

            if self.opt.learned_attn:
                self.optimizer_F_SR = torch.optim.Adam([{'params': list(filter(lambda p:p.requires_grad, self.netPre.parameters())), 'lr': self.opt.lr*0.0},
                                        {'params': list(filter(lambda p:p.requires_grad, self.netF_SR.parameters()))}],
                                         lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F_SR)
                self.optimizer_F_SR.zero_grad()

    def optimize_parameters(self):
        # forward
        self.forward()

        if self.opt.learned_attn:
            self.set_requires_grad([self.netF_SR, self.netPre], True)
            self.optimizer_F_SR.zero_grad()
            self.loss_spatial = self.compute_F_SR_loss()
            self.loss_spatial.backward()
            self.optimizer_F_SR.step()


        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.zero_grad()

        if self.opt.learned_attn:
            self.set_requires_grad([self.netF_SR, self.netPre], False)

        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.step()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        if self.opt.isTrain and self.opt.augment:
            self.aug_A = input['A_aug' if AtoB else 'B_aug'].to(self.device)
            self.aug_B = input['B_aug' if AtoB else 'A_aug'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.real = torch.cat((self.real_A, self.real_B), dim=0) if self.opt.nce_idt and self.opt.isTrain else self.real_A
        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])

        self.fake = self.netG(self.real)
        self.fake_B = self.fake[:self.real_A.size(0)] #A生成的虚假的B
        if self.opt.nce_idt:
            self.idt_B = self.fake[self.real_A.size(0):]

    def compute_F_SR_loss(self):
        """
        Calculate the contrastive loss for learned spatially-correlative loss
        """
        norm_real_A, norm_real_B, norm_fake_B = self.normalization((self.real_A + 1) * 0.5), self.normalization((self.real_B + 1) * 0.5), self.normalization((self.fake_B.detach() + 1) * 0.5)
        if self.opt.augment:
            norm_aug_A, norm_aug_B = self.normalization((self.aug_A + 1) * 0.5), self.normalization((self.aug_B + 1) * 0.5)
            norm_real_A = torch.cat([norm_real_A, norm_real_A], dim=0)
            norm_fake_B = torch.cat([norm_fake_B, norm_aug_A], dim=0)
            norm_real_B = torch.cat([norm_real_B, norm_aug_B], dim=0)
        self.loss_spatial = self.Spatial_Loss(self.netPre, norm_real_A, norm_fake_B, norm_real_B)

        return self.loss_spatial

    def Spatial_Loss(self, net, src, tgt, other=None):
        """given the source and target images to calculate the spatial similarity and dissimilarity loss"""
        n_layers = len(self.attn_layers)
        feats_src = net(src, self.attn_layers, encode_only=True)
        feats_tgt = net(tgt, self.attn_layers, encode_only=True)
        if other is not None:
            feats_oth = net(torch.flip(other, [2, 3]), self.attn_layers, encode_only=True)
        else:
            feats_oth = [None for _ in range(n_layers)]

        total_loss = 0.0
        for i, (feat_src, feat_tgt, feat_oth) in enumerate(zip(feats_src, feats_tgt, feats_oth)):
            loss = self.criterionSpatial.loss(feat_src, feat_tgt, feat_oth, i)
            total_loss += loss.mean()

        if not self.criterionSpatial.conv_init:
            self.criterionSpatial.update_init_()

        return total_loss / n_layers

    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        fake_B = self.fake_B_pool.query(self.fake_B).detach()
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(fake_B)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        # Real
        self.pred_real = self.netD(self.real_B)
        loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_real = loss_D_real.mean()

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D

    def compute_G_loss(self):
        """Calculate GAN and NCE loss for the generator"""
        fake = self.fake_B
        # First, G(A) should fake the discriminator
        if self.opt.lambda_GAN > 0.0:
            pred_fake = self.netD(fake)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0

        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE = self.calculate_NCE_loss(self.real_A, self.fake_B)
        else:
            self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0

        if self.opt.self_regularization > 0.0:  
            self.loss_SR = self.opt.self_regularization * self.calculate_SR_loss(self.real_A, self.fake_B)
        else:
            self.loss_SR = 0.0

        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            self.loss_NCE_Y = self.calculate_NCE_loss(self.real_B, self.idt_B)
            loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
        else:
            loss_NCE_both = self.loss_NCE

        l_sptial = self.opt.lambda_spatial
        l_spatial_idt = self.opt.lambda_spatial_idt

        norm_real_A = self.normalization((self.real_A + 1) * 0.5)
        norm_fake_B = self.normalization((self.fake_B + 1) * 0.5)
        norm_real_B = self.normalization((self.real_B + 1) * 0.5)
        self.loss_G_s = self.Spatial_Loss(self.netPre, norm_real_A, norm_fake_B, None) * l_sptial if l_sptial > 0 else 0
        # identity loss
        if l_spatial_idt > 0:
            norm_fake_idt_B = self.normalization((self.idt_B + 1) * 0.5)
            self.loss_G_s_idt_B = self.Spatial_Loss(self.netPre, norm_real_B, norm_fake_idt_B, None) * l_spatial_idt
        else:
            self.loss_G_s_idt_B = 0



        self.loss_G = self.loss_G_GAN + loss_NCE_both + self.loss_SR + self.loss_G_s + self.loss_G_s_idt_B
        return self.loss_G

    def calculate_NCE_loss(self, src, tgt):
        n_layers = len(self.nce_layers)
        feat_q = self.netG(tgt, self.nce_layers, encode_only=True)

        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        feat_k = self.netG(src, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers

    def calculate_SR_loss(self,src,tgt):
        #rgb_mean_src = torch.mean(src,dim=1) #mean over color channel
        #rgb_mean_tgt = torch.mean(tgt,dim=1)
        diff_chan = src-tgt
        batch_mean = torch.mean(diff_chan,dim=0)
        rgb_sum = torch.sum(batch_mean,0)
        batch_mean2 = torch.mean(torch.mean(rgb_sum,0),0)

        return batch_mean2
