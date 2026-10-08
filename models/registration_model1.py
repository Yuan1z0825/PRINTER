import numpy as np
import torch
from .base_model import BaseModel
from . import networks
from .dn import init_dense_instance_norm, use_dense_instance_norm
from .losses import NCC_Loss
from .patchnce import PatchNCELoss
import util.util as util
import models.voxelmorph.torchvoxelmorph as vxm
from models.voxelmorph.torchvoxelmorph.layers import SpatialTransformer
from PIL import Image
from torchvision.transforms import Compose, CenterCrop, ToTensor, Normalize
import matplotlib.pyplot as plt


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
    return d/2.0

class REGISTRATIONModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """
        parser.add_argument('--CUT_mode', type=str, default="CUT", choices='(CUT, cut, FastCUT, fastcut)')

        parser.add_argument('--lambda_GAN', type=float, default=1, help='weight for GAN loss：GAN(G(X))')
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


        # self.loss_names = ['G', 'D', 'NCE', 'R', 'smooth', 'local', 'NccScore']
        self.loss_names = ['G', 'D', 'NCE', 'R', 'smooth', 'local']
        # self.visual_names = ['real_A', 'fake_B', 'real_B', 'dvf', 'registered', 'regA', 'ncc']
        self.visual_names = ['real_A', 'fake_B', 'real_B', 'dvf', 'registered', 'regA']
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

        if opt.nce_idt and self.isTrain:
            self.loss_names += ['NCE_Y']
            self.visual_names += ['idt_B']

        if self.isTrain:
            self.model_names = ['G', 'F', 'R']
        else:  # during test time, only load G
            self.model_names = ['G', 'R']

        self.epoch = 0
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
        nb_features = [
            [16, 32, 32, 64, 64, 64],  # encoder
            [64, 64, 64, 32, 32, 32, 16]  # decoder
        ]
        vol_shape = (opt.crop_size, opt.crop_size)
        self.netR = vxm.networks.VxmDense(ndims=2, nb_unet_features=nb_features, int_steps=7, bidir=True, int_downsize=2, inshape=vol_shape).cuda()
        self.netR.train()
        self.spatialTransformer = SpatialTransformer(vol_shape).cuda()


        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = []

            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))

            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            self.criterionNCC = NCC_Loss(self.device, name='ncc', kernel_var=[9,9], kernel_type='mean')
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_R = torch.optim.Adam(self.netR.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
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
            #self.compute_D_loss().backward()                  # calculate gradients for D
            self.compute_G_loss().backward()                   # calculate graidents for G
            if self.opt.lambda_NCE > 0.0:
                self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)

    def visual_tensor2im(self, tensors, score):
        new_tensors = [item.detach().cpu().float().numpy() for item in tensors]
        subplot_lines = new_tensors[0].shape[0]
        subplot_columns = len(new_tensors)
        plt.subplots(subplot_lines, subplot_columns)
        for i in range(subplot_lines):
            for j in range(subplot_columns):
                plt.subplot(subplot_lines, subplot_columns, i * subplot_columns + j + 1)
                if j == subplot_columns - 1:
                    plt.imshow(new_tensors[j][i].squeeze(), cmap='gray',)
                else:
                    plt.imshow(new_tensors[j][i].transpose(1, 2, 0))

        plt.suptitle('NCC: ' + str(score.item()))
        plt.show()

    def calculate_L1_loss(self, src, tgt, mask=None):
        diff = torch.abs(src - tgt)
        if mask is None:
            return torch.mean(diff)
        elif torch.sum(mask) == 0:
            return torch.tensor(0)
        else:
            norm_factor = 1 / (torch.sum(mask))
            return norm_factor * torch.sum(diff * mask)

    def optimize_parameters(self):

        # print(self.epoch)
        # forward pass
        self.forward()

        # update Discriminator (D)
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()

        self.set_requires_grad(self.netD, False)

        # update Generator (G) and Registration (R)
        y_output = self.netR(self.real_A, self.real_B)
        self.regA = self.spatialTransformer(self.real_A, y_output[2])

        # y_output contains deformed source, target, and deformation field
        # 计算fake_B和real_B的NCC
        # self.loss_NCC = self.criterionNCC(self.regA, self.real_B)

        # self.loss_NccScore = self.loss_NCC[0]

        # 可视化fake_B, real_B, deformed source, loss_NCC
        # self.visual_tensor2im([self.fake_B, self.real_B, y_output[0], self.loss_NCC[1]], self.loss_NCC[0])
        # self.ncc = self.loss_NCC[1]
        # self.loss_NCC[0] = torch.clip(self.loss_NCC[0], 0, 1)

        y_pred = [self.spatialTransformer(self.fake_B, y_output[2]), y_output[2]]

        # Visualization (optional for debugging)
        self.registered = y_pred[0]  # Deformed B
        test_image = open_image_to_torch("./deform256.jpg", 256).cuda()
        self.dvf = self.spatialTransformer(test_image, y_pred[1])

        # Reset optimizers for G, R, and F
        self.optimizer_G.zero_grad()
        self.optimizer_R.zero_grad()

        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.zero_grad()

        # Compute Generator (G) loss
        self.loss_G = self.compute_G_loss()

        self.loss_local = self.calculate_NCE_loss(self.real_B, y_output[0]) * 0.25
        # self.loss_R = self.calculate_L1_loss_soft(y_pred[0], self.real_B, weight_map=None)*1.0 + self.loss_local*1.0 \
        #               + self.calculate_L1_loss_soft(self.idt_B, y_pred[0], weight_map=None)*1.0
        self.loss_R = self.calculate_L1_loss(y_pred[0], self.real_B)*1.0 + self.calculate_L1_loss(self.idt_B, y_pred[0])*1.0 + self.loss_local*1.0


        self.loss_smooth = smooothing_loss(y_pred[1]) * 0.1
        all_registration_loss = self.loss_R + self.loss_smooth
        self.loss_G = self.loss_G + all_registration_loss  #* self.loss_NCC[0]
        self.loss_G.backward()

        self.optimizer_G.step()
        self.optimizer_R.step()


        # 更新特征网络 (netF) 如果需要
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
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.real = torch.cat((self.real_A, self.real_B), dim=0) #if self.opt.nce_idt and self.opt.isTrain else self.real_A
        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])

        self.fake = self.netG(self.real)
        self.fake_B = self.fake[:self.real_A.size(0)]
        #if self.opt.nce_idt:
        self.idt_B = self.fake[self.real_A.size(0):]

    def compute_D_loss(self):
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

        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            self.loss_NCE_Y = self.calculate_NCE_loss(self.real_B, self.idt_B)
            loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
        else:
            loss_NCE_both = self.loss_NCE

        self.loss_G = self.loss_G_GAN + loss_NCE_both
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

    def calculate_L1_loss_soft(self, src, tgt, mask=None, weight_map=None):
        """
        Calculate L1 loss with soft weights.
        :param src: Source tensor (predicted image) [batch, c, h, w]
        :param tgt: Target tensor (ground truth image) [batch, c, h, w]
        :param mask: Optional mask tensor [batch, 1, h, w] for valid regions (default: None)
        :param weight_map: Optional soft weight map tensor [batch, 1, h, w] (default: None)
        :return: Weighted L1 loss
        """

        # 计算源图像和目标图像之间的差异
        diff = torch.abs(src - tgt)

        # 如果没有提供软权重，则默认使用均匀权重（即所有像素都一样）
        if weight_map is None:
            weight_map = torch.ones_like(diff)  # 所有像素权重都为 1
        else:
            # 确保 weight_map 的值在合理范围内，通常是 [0, 1] 之间
            weight_map = torch.clamp(weight_map, min=0.0, max=1.0)

        # 如果提供了 mask，则仅在有效区域计算损失
        if mask is None:
            weighted_diff = diff * weight_map  # 计算加权损失
            return torch.mean(weighted_diff)  # 计算加权平均损失
        elif torch.sum(mask) == 0:
            return torch.tensor(0, device=src.device)  # 如果 mask 全部为 0，返回 0
        else:
            norm_factor = 1 / torch.sum(mask)  # 对有效区域进行归一化
            weighted_diff = diff * weight_map * mask  # 在有效区域内计算加权损失
            return norm_factor * torch.sum(weighted_diff)  # 返回加权损失的均值


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