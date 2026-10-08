import torch
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks, utils
from torchvision import models
import numpy as np

from .frequency_loss import Gauss_Pyramid_Conv
from .patch_alignment_loss import PatchAlignmentLoss
from .content_loss import VGGLoss
from .patchnce import FocalNCELoss


class PPTModel(BaseModel):


    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """
        parser.add_argument('--CUT_mode', type=str, default="CUT", choices='(CUT, cut, FastCUT, fastcut)')

        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--lambda_DINO', type=float, default=1.0, help='weight for DINO loss: DINO(G(X), Y)')
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

        parser.add_argument('--out_dim', default=65536, type=int, help="""Dimensionality of
            the DINO head output. For complex and large datasets large values (like 65k) work well.""")
        parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
                            help="Whether to use batch normalizations in projection head (Default: False)")
        parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
                            help="""Whether or not to weight normalize the last layer of the DINO head.
            Not normalizing leads to better performance but can make the training unstable.
            In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")

        parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.),
                            help="""Scale range of the cropped image before resizing, relatively to the origin image.
            Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
            recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
        parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
                            help="""Initial value for the teacher temperature: 0.04 works well in most cases.
            Try decreasing it if the training loss does not decrease.""")
        parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
            of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
            starting with the default value of 0.04 and increase this slightly if needed.""")
        parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int,
                            help='Number of warmup epochs for the teacher temperature (Default: 30).')
        parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
            gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
            help optimization for larger ViT architectures. 0 for disabling.""")
        parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
            during which we keep the output layer fixed. Typically doing so during
            the first epoch helps training. Try increasing this value if the loss does not decrease.""")
        parser.add_argument('--local_crops_number', type=int, default=0, help="""Number of small
            local views to generate. Set this parameter to 0 to disable multi-crop training.
            When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
        parser.add_argument('--arch', default='vit_small', type=str,
                            choices=['vit_tiny', 'vit_small', 'vit_base', 'xcit', 'deit_tiny', 'deit_small'],
                            help="""Name of architecture to train. For quick experiments with ViTs,
            we recommend using vit_tiny or vit_small.""")
        parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")
        parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
            of input square patches - default 16 (for 16x16 patches). Using smaller
            values leads to better performance but requires more memory. Applies only
            for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
            mixed precision training (--use_fp16 false) to avoid unstabilities.""")

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

    def name(self):
        return 'PPTModel'

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.nce_layers = '0,4,8,12,16'
        self.dropout = False
        self.init_gain = 0.02
        self.no_antialias = False
        self.no_antialias_up = False
        self.num_patches = 256
        self.n_layers_D = opt.n_layers_D

        self.isTrain = opt.isTrain
        self.loss_names = ['G', 'D', 'NCE', 'NCE_Y', 'NCE_Y_hat', 'patch_alignment', 'content', 'freq']
        self.visual_names = ['real_A', 'fake_B', 'real_B', 'idt_B']
        if opt.nce_idt and self.isTrain:
            self.loss_names += ['NCE_Y']
            self.visual_names += ['idt_B']

        if self.isTrain:
            self.model_names = ['G', 'F']
        else:  # during test time, only load G
            self.model_names = ['G']

        self.nce_layers = [int(i) for i in self.nce_layers.split(',')]

        # load/define networks
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)

        self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

        self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type,
                                      opt.init_gain, opt.no_antialias, self.gpu_ids, opt)


        if self.isTrain:

            self.fake_AB_pool = ImagePool(opt.pool_size)

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = FocalNCELoss(opt)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionVGG = VGGLoss(self.gpu_ids)
            self.P = Gauss_Pyramid_Conv(num_high=5)
            self.gp_weights = [1.0] * 6
            self.criterionMisalignment = PatchAlignmentLoss()

            # initialize optimizers
            self.schedulers = []
            self.optimizers = []

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))



    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        bs_per_gpu = data["A"].size(0) // max(len(self.opt.gpu_ids), 1)
        self.set_input(data)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()  # compute fake images: G(A)
        if self.opt.isTrain:
            self.backward_D()  # calculate gradients for D
            self.backward_G()  # calculate graidents for G
            self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr,
                                                betas=(self.opt.beta1, self.opt.beta2))
            self.optimizers.append(self.optimizer_F)

    def set_input(self, input):
        if self.isTrain:
            self.real_A = input['A']
            self.real_B = input['B']
            if len(self.gpu_ids) > 0:
                self.real_A = input['A'].cuda(self.gpu_ids[0], non_blocking=True)
                self.real_B = input['B'].cuda(self.gpu_ids[0], non_blocking=True)
            self.image_paths = input['A_paths']
            if 'current_epoch' in input:
                self.current_epoch = input['current_epoch']
            if 'current_iter' in input:
                self.current_iter = input['current_iter']
        else:
            self.real_A = input['A']
            self.real_B = input['B']
            if len(self.gpu_ids) > 0:
                self.real_A = input['A'].cuda(self.gpu_ids[0], non_blocking=True)
                self.real_B = input['B'].cuda(self.gpu_ids[0], non_blocking=True)
            self.image_paths = input['A_paths']

    def forward(self):
        self.real = torch.cat((self.real_A, self.real_B), dim=0)
        self.fake = self.netG(self.real, layers=[])
        self.fake_B = self.fake[:self.real_A.size(0)]
        self.idt_B = self.fake[self.real_A.size(0):]

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        fake = self.fake_B.detach()
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(fake)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        # Real
        self.pred_real = self.netD(self.real_B)
        loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_real = loss_D_real.mean()

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D.backward()

    def backward_G(self):

        """Calculate GAN and NCE loss for the generator"""
        feat_real_A = self.netG(self.real_A, self.nce_layers, encode_only=True)
        feat_fake_B = self.netG(self.fake_B, self.nce_layers, encode_only=True)
        feat_real_B = self.netG(self.real_B, self.nce_layers, encode_only=True)
        feat_idt_B = self.netG(self.idt_B, self.nce_layers, encode_only=True)

        pred_fake = self.netD(self.fake_B)
        pred_real = self.netD(self.real_B)

        # advsarial loss
        self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean()

        # contrastive loss
        self.loss_NCE = self.calculate_NCE_loss(feat_real_A, feat_fake_B, self.netF)
        self.loss_NCE_Y = self.calculate_NCE_loss(feat_real_B, feat_idt_B, self.netF)
        self.loss_NCE_Y_hat = self.calculate_NCE_loss(feat_idt_B, feat_real_B, self.netF)
        self.loss_contrast = self.loss_NCE + self.loss_NCE_Y + self.loss_NCE_Y_hat

        # patch alignment loss
        self.loss_patch_alignment = self.criterionMisalignment(self.real_B, self.fake_B)

        # content loss
        self.loss_content = self.criterionVGG(self.fake_B, self.real_B)

        # freqency loss
        p_fake_B = self.P(self.fake_B)
        p_real_B = self.P(self.real_B)
        loss_pyramid = [self.criterionL1(pf, pr) for pf, pr in zip(p_fake_B, p_real_B)]
        weights = self.gp_weights
        loss_pyramid = [l * w for l, w in zip(loss_pyramid, weights)]
        self.loss_freq = torch.mean(torch.stack(loss_pyramid))

        # total loss
        self.loss_G = self.loss_G_GAN + self.loss_freq + self.loss_contrast + self.loss_patch_alignment + self.loss_content
        return self.loss_G.backward()

    # no backprop gradients
    def test(self):
        with torch.no_grad():
            self.fake_B = self.netG(self.real_A)

    def optimize_parameters(self):
        self.forward()

        # update D
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        # update G
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        # update F
        self.optimizer_F.step()

    def get_current_errors(self):
        return OrderedDict([('loss_G_GAN', self.loss_G_GAN.item()),
                            ('loss_freq', self.loss_freq.item()),
                            ('loss_D', self.loss_D.item()),
                            ('loss_contrast', self.loss_contrast.item()),
                            ('loss_content', self.loss_content.item()),
                            ('loss_patch_alignment', self.loss_patch_alignment.item()),
                            ])


    def calculate_NCE_loss(self, feat_src, feat_tgt, netF):
        n_layers = len(feat_src)
        feat_q = feat_tgt

        feat_k = feat_src
        feat_k_pool, sample_ids = netF(feat_k, self.num_patches, None)
        feat_q_pool, _ = netF(feat_q, self.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k in zip(feat_q_pool, feat_k_pool):
            loss = self.criterionNCE(f_q, f_k)
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers