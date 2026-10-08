import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from collections import OrderedDict
from PIL import Image
from torchvision.transforms import Compose, CenterCrop, ToTensor, Normalize
import numpy as np

# 以下为你工程中已有的模块（请根据实际情况修改导入路径）
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks, utils
from .frequency_loss import Gauss_Pyramid_Conv
from .losses import NMI_Loss
from .patch_alignment_loss import PatchAlignmentLoss
from .content_loss import VGGLoss
from .patchnce import PatchNCELoss 
import models.voxelmorph.torchvoxelmorph as vxm
from .registration_model import open_image_to_torch as reg_open_image_to_torch
from .voxelmorph.torchvoxelmorph.layers import SpatialTransformer
from .networks import ResnetBlock, init_net
def distributed_sinkhorn(style_features, prototypes, sinkhorn_iterations=3, tau=0.05):
    """
    对风格特征和原型计算相似度矩阵，并利用 Sinkhorn 算法得到平衡分配。
    style_features: (B, style_dim)
    prototypes: (num_prototypes, style_dim)
    返回：平衡分配矩阵 L (B, num_prototypes) 和每个样本的离散索引
    """
    B = style_features.size(0)
    num_prototypes = prototypes.size(0)
    # 归一化
    prototypes_norm = F.normalize(prototypes, dim=1)  # (num_prototypes, style_dim)
    style_norm = F.normalize(style_features, dim=1)  # (B, style_dim)
    # 相似度矩阵 logits (B, num_prototypes)
    logits = torch.matmul(style_norm, prototypes_norm.t())
    L = torch.exp(logits / logits.max())
    L /= L.sum()  # 全局归一化

    for _ in range(sinkhorn_iterations):
        # 按行归一化（每个样本）
        L = L / (L.sum(dim=1, keepdim=True) + 1e-6)
        L = L / num_prototypes
        # 按列归一化（每个原型）
        L = L / (L.sum(dim=0, keepdim=True) + 1e-6)
        L = L / B
    L = L * B  # 恢复尺度

    # 采用 Gumbel Softmax 得到硬分配（one-hot）
    L_hard = F.gumbel_softmax(L, tau=0.5, hard=True)
    indexs = torch.argmax(L_hard, dim=1)
    return L_hard, indexs, logits


def momentum_update(old_value, new_value, momentum):
    """
    动量更新：old_value * momentum + new_value * (1 - momentum)
    """
    return momentum * old_value + (1 - momentum) * new_value


###############################################
# 辅助函数及模块定义
###############################################

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
    dx = dx * dx
    dy = dy * dy
    d = torch.mean(dx) + torch.mean(dy)
    return d / 2.0


def adaptive_instance_normalization(content_feat, gamma, beta, eps=1e-5):
    """
    自适应实例归一化（AdaIN）：先归一化内容特征，再利用风格给出的尺度（gamma）与偏置（beta）进行调制
    """
    size = content_feat.size()
    content_mean = content_feat.view(size[0], size[1], -1).mean(2).view(size[0], size[1], 1, 1)
    content_std = content_feat.view(size[0], size[1], -1).std(2).view(size[0], size[1], 1, 1) + eps
    normalized = (content_feat - content_mean) / content_std
    return normalized * gamma + beta



class Downsample(nn.Module):
    """
    下采样模块：使用平均池化实现 2 倍下采样
    """

    def __init__(self):
        super(Downsample, self).__init__()
        self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        return self.pool(x)


class Upsample(nn.Module):
    """
    上采样模块：使用 nearest neighbor 上采样
    """

    def __init__(self, channels):
        super(Upsample, self).__init__()

    def forward(self, x):
        return nn.Upsample(scale_factor=2, mode='nearest')(x)



class PrototypeAggregator(nn.Module):
    def __init__(self, content_dim, style_dim, num_prototypes):
        """
        利用内容特征预测在各个原型上的分布，从而生成一个加权的风格向量。
        content_dim: 内容特征维度（例如全局平均池化后的维度）
        style_dim: 风格向量维度（与原型维度一致）
        num_prototypes: 原型数量
        """
        super(PrototypeAggregator, self).__init__()
        # 先将内容特征映射到 style_dim，再输出 num_prototypes 个 logits
        self.fc = nn.Sequential(
            nn.Linear(content_dim, style_dim),
            nn.ReLU(inplace=True),
            nn.Linear(style_dim, num_prototypes)
        )

    def forward(self, content_feat, prototypes):
        # content_feat: (B, content_dim, H, W) 或 (B, content_dim)
        if content_feat.dim() == 4:
            pooled = torch.mean(content_feat, dim=[2, 3])  # shape (B, content_dim)
        else:
            pooled = content_feat  # 假定已为 (B, content_dim)
        logits = self.fc(pooled)  # shape (B, num_prototypes)
        weights = F.softmax(logits, dim=1)  # 得到对各原型的权重分布
        # 使用权重对所有原型进行加权求和，得到最终风格向量 (B, style_dim)
        style_vector = torch.matmul(weights, prototypes)
        return style_vector, weights

class ResnetEncoder(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect', no_antialias=False, no_antialias_up=False, opt=None):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetEncoder, self).__init__()
        norm_layer = networks.get_norm_layer(norm_type=norm_layer)

        self.opt = opt
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d or norm_layer.func == DenseInstanceNorm
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        if self.opt.model == 'registration' or self.opt.model == 'ffpecut' or 'ffpe' in self.opt.model or 'test' in self.opt.model:
            self.SAB = SpatialAttention()
            model += [self.SAB]
            print('Spatial Attention Block is added')
        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            if(no_antialias):
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True)]
            else:
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True),
                          Downsample()]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        self.model = nn.Sequential(*model)

    def forward(self, input, layers=[], encode_only=False):
        if -1 in layers:
            layers.append(len(self.model))
        if len(layers) > 0:
            
            feat = input
            feats = []
            for layer_id, layer in enumerate(self.model):
                #print(layer_id, layer)
                feat = layer(feat)
     
                if layer_id in layers:
                    # print("%d: adding the output of %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                    feats.append(feat)
                else:
                    # print("%d: skipping %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                    pass
                if layer_id == layers[-1] and encode_only:
                    return feats  # return intermediate features alone; stop in the last layers

            return feat, feats  # return both output and intermediate features
        else:
            """Standard forward"""
            fake = self.model(input)
            return fake

class ResnetDecoder(nn.Module):
    """
    ResNet 解码器：接收编码器输出的内容特征，在输入时利用风格向量经过全连接层映射得到的参数对内容特征进行 AdaIN 调制，
    然后通过上采样模块还原生成目标图像。

    其中：
      - content_dim：与编码器输出通道数一致（默认 256）；
      - n_downsampling：与编码器对应的下采样次数（默认 2）；
      - ngf：初始特征数（默认 64）。
    """

    def __init__(self, output_nc, content_dim=256, ngf=64, n_downsampling=2, norm=nn.BatchNorm2d,
                 no_antialias_up=False, style_dim=8):
        super(ResnetDecoder, self).__init__()
        norm_layer = networks.get_norm_layer(norm_type=norm)
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        self.style_fc = nn.Sequential(
            nn.Linear(style_dim, content_dim * 2),
            nn.ReLU(inplace=True)
        )
        self.n_downsampling = n_downsampling
        self.no_antialias_up = no_antialias_up
        model = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            out_channels = int(ngf * mult / 2)
            if no_antialias_up:
                model += [
                    nn.ConvTranspose2d(ngf * mult, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
                    norm_layer(out_channels),
                    nn.ReLU(True)]
            else:
                model += [Upsample(ngf * mult),
                          nn.Conv2d(ngf * mult, out_channels, kernel_size=3, stride=1, padding=1),
                          norm_layer(out_channels),
                          nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                  nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, content_feat, style_feat):
        params = self.style_fc(style_feat)  # 输出尺寸 (B, content_dim*2)
        content_channels = content_feat.size(1)
        gamma, beta = params[:, :content_channels], params[:, content_channels:]
        gamma = gamma.view(-1, content_channels, 1, 1)
        beta = beta.view(-1, content_channels, 1, 1)
        # 对内容特征进行 AdaIN 调制
        t = adaptive_instance_normalization(content_feat, gamma, beta)
        x = self.model(t)
        return x


class StylePrototypeLearner(nn.Module):
    def __init__(self, style_dim, num_prototypes=10, gamma=0.9, sinkhorn_iterations=3):
        """
        style_dim: 风格向量的维度
        num_prototypes: 原型个数
        gamma: 动量更新系数
        """
        super(StylePrototypeLearner, self).__init__()
        self.num_prototypes = num_prototypes
        self.gamma = gamma
        self.sinkhorn_iterations = sinkhorn_iterations
        # 初始化原型，采用可学习的参数（不参与梯度更新，后续通过动量更新）
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, style_dim), requires_grad=False)
        nn.init.kaiming_normal_(self.prototypes, nonlinearity='relu')

    def forward(self, style_features):
        """
        输入风格向量 shape: (B, style_dim)
        输出量化后的风格向量 quantized_style: (B, style_dim)
               以及用于监控的 logits 和索引信息
        """
        # 得到离散分配和对应索引
        L, indexs, logits = distributed_sinkhorn(style_features, self.prototypes,
                                                 sinkhorn_iterations=self.sinkhorn_iterations)
        # 利用动量更新调整原型
        with torch.no_grad():
            prototypes_data = self.prototypes.data.clone()
            for i in range(self.num_prototypes):
                mask = (indexs == i)
                if mask.sum() > 0:
                    # 计算分配到该原型的样本均值
                    assigned_features = style_features[mask]
                    new_value = F.normalize(assigned_features.mean(dim=0, keepdim=True), dim=1)
                    prototypes_data[i:i + 1] = momentum_update(prototypes_data[i:i + 1], new_value, self.gamma)
            # 归一化更新后的原型
            self.prototypes.data.copy_(F.normalize(prototypes_data, dim=1))
        # 用离散分配对原型进行加权求和，得到量化后的风格向量
        quantized_style = torch.matmul(L, self.prototypes)
        return quantized_style, logits, indexs


###############################################
# 风格编码器（与原实现一致）
###############################################

class StyleEncoder(nn.Module):
    """
    风格编码器：提取目标图像的全局风格信息，通过全局平均池化和全连接层输出风格向量
    """

    def __init__(self, input_nc=3, style_dim=8):
        super(StyleEncoder, self).__init__()
        self.conv1 = nn.Conv2d(input_nc, 64, kernel_size=7, stride=1, padding=3)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, style_dim)

    def forward(self, x):
        feat = self.relu1(self.conv1(x))
        feat = self.relu2(self.conv2(feat))
        feat = self.relu3(self.conv3(feat))
        pooled = self.pool(feat)
        pooled = pooled.view(pooled.size(0), -1)
        style_vector = self.fc(pooled)
        return style_vector


###############################################
# 修改后的双分支生成器
###############################################

class DualBranchGenerator(nn.Module):
    """
    双分支生成器：利用内容编码器提取内容特征，
    利用风格编码器获得风格信息更新原型，
    并通过聚合器从内容特征中预测出一个基于原型的风格向量，
    保证训练和推理时均使用基于原型的风格条件，从而提高泛化性。
    """

    def __init__(self, opt, gpu_ids):
        super(DualBranchGenerator, self).__init__()
        self.gpu_ids = gpu_ids
        self.n_layers_D = opt.n_layers_D

        self.content_encoder = ResnetEncoder(opt.input_nc, opt.output_nc, opt.ngf, norm_layer=opt.normG, use_dropout=not opt.no_dropout,
                                    no_antialias=opt.no_antialias, no_antialias_up=opt.no_antialias_up, n_blocks=9, opt=opt)
        init_net(self.content_encoder, opt.init_type, opt.init_gain, opt.gpu_ids)                            
        self.style_encoder = StyleEncoder(opt.input_nc, opt.style_dim)
        self.decoder = ResnetDecoder(opt.output_nc, content_dim=opt.content_dim, ngf=opt.ngf, n_downsampling=2,
                                     norm=opt.normG, no_antialias_up=opt.no_antialias_up, style_dim=opt.style_dim)
        self.style_proto_learner = StylePrototypeLearner(style_dim=opt.style_dim, num_prototypes=opt.num_prototypes)
        # 假设 content_encoder 的全局特征维度等于 content_dim
        self.proto_aggregator = PrototypeAggregator(opt.content_dim, opt.style_dim, opt.num_prototypes)



    def forward(self, content_img, style_img=None, encode_only=False, layers=[]):
        if encode_only:
            return self.content_encoder(content_img, layers=layers, encode_only=True)
        else:
            # 提取内容特征（若为多尺度，可额外返回中间特征）
            content_feat = self.content_encoder(content_img)
            # 当提供 style_img 时，通过风格编码器提取风格特征，并更新风格原型
            if style_img is not None:
                style_feat = self.style_encoder(style_img)
                _ = self.style_proto_learner(style_feat)
                # 可选：添加辅助损失，要求聚合器预测的风格向量与 style_feat 在某些投影空间内保持一致
            # 无论训练或推理，均通过聚合器利用内容特征和当前学习到的原型生成风格向量
            # print(content_feat.shape)  # 10, 3, 512 ,512       10 256 128 128
            # print(self.style_proto_learner.prototypes.shape) # 10 8
            agg_style, proto_weights = self.proto_aggregator(content_feat, self.style_proto_learner.prototypes)
            # 解码器利用 AdaIN 将内容特征和聚合得到的风格向量结合，生成目标图像
            out = self.decoder(content_feat, agg_style)
            return out


###############################################
# PPTPLUSModel 定义（基于双分支生成器，其他部分基本保持不变）
###############################################

class CUTRModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """ Configures options specific for CUT model """
        parser.add_argument('--CUT_mode', type=str, default="CUT", choices='(CUT, cut, FastCUT, fastcut)')

        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=False,
                            help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        # 对于双分支生成器，nce_layers设置为"0,1,2"对应内容编码器的三层特征
        parser.add_argument('--nce_layers', type=str, default='0,1,2', help='compute NCE loss on which layers')
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
        parser.add_argument('--use_lambda_pair', action='store_true',
                            help='If specified, use lambda_pair to blend between unregistered and registered images.')

        parser.add_argument('--lambda_pixel', type=float, default=1,
                            help='Weight for pixel-level L1 loss after deformation')
        parser.add_argument('--w_barrier', type=float, default=5)
        parser.add_argument('--lamba_MMR', type=float, default=1)
        parser.add_argument('--lambda_smooth', type=float, default=1)
        parser.add_argument('--lambda_pixel_L1', type=float, default=1,
                            help='Weight for pixel-level L1 loss after deformation')
        parser.add_argument('--self_regularization', type=float, default=0.03,
                            help='loss between input and generated image')

        parser.add_argument('--style_dim', type=int, default=8)
        parser.add_argument('--content_dim', type=int, default=256)
        parser.add_argument('--num_prototypes', type=int, default=20)

        parser.set_defaults(pool_size=0)  # no image pooling

        opt, _ = parser.parse_known_args()

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
        # 使用与双分支生成器相匹配的 nce_layers
        self.nce_layers = '0,1,2'
        self.nce_layers = [int(i) for i in self.nce_layers.split(',')]
        self.dropout = False
        self.init_gain = 0.02
        self.no_antialias = False
        self.no_antialias_up = False
        self.num_patches = 256
        self.n_layers_D = opt.n_layers_D

        self.isTrain = opt.isTrain
        self.loss_names = ['G', 'D', 'NCE', 'NCE_Y',
                           'pixel', 'pixel_L1', 'MMR', 'smooth']
        self.visual_names = ['real_A', 'fake_B', 'real_B', 'idt_B', 'registered', 'dvf']
        if opt.nce_idt and self.isTrain:
            self.loss_names += ['NCE_Y']
            self.visual_names += ['idt_B']

        if self.isTrain:
            self.model_names = ['G', 'F', 'D', 'R']
        else:
            self.model_names = ['G', 'R']

        self.epoch = 0

        # 使用双分支生成器（采用基于 ResNet 的编码器/解码器）
        self.netG = DualBranchGenerator(opt, opt.gpu_ids).to(self.device)
        self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout,
                                      opt.init_type, opt.init_gain, opt.no_antialias, opt.gpu_ids, opt)

        self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD,
                                      opt.init_type, opt.init_gain, opt.no_antialias, opt.gpu_ids, opt)

        nb_features = [
            [16, 32, 32, 64, 64, 64],  # encoder
            [64, 64, 64, 32, 32, 32, 16]  # decoder
        ]
        vol_shape = (opt.crop_size, opt.crop_size)
        self.netR = vxm.networks.VxmDense(ndims=2, nb_unet_features=nb_features, int_steps=7,
                                          bidir=True, int_downsize=2, inshape=vol_shape).cuda()
        self.netR.train()
        self.spatialTransformer = SpatialTransformer(vol_shape).cuda()

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)

            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = PatchNCELoss(opt)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionVGG = VGGLoss(self.gpu_ids)
            self.P = Gauss_Pyramid_Conv(num_high=5)
            self.gp_weights = [1.0] * 6
            self.criterionMisalignment = PatchAlignmentLoss()
            self.criterionNMI = NMI_Loss([0.05, 0.1, 0.15, 0.2, 0.25, 0.3,
                                          0.35, 0.4, 0.45, 0.5, 0.55, 0.6,
                                          0.65, 0.7, 0.75, 0.8, 0.85, 0.9,
                                          0.95], device=self.device)
            self.schedulers = []
            self.optimizers = []

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_R = torch.optim.Adam(self.netR.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_R)

            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

    def data_dependent_initialize(self, data):
        bs_per_gpu = data["A"].size(0) // max(len(self.opt.gpu_ids), 1)
        self.set_input(data)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()  # 计算生成图像
        if self.opt.isTrain:
            self.backward_D()  # 计算判别器梯度
            self.backward_G()  # 计算生成器梯度
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
        # 使用 real_A（内容图）与 real_B（目标风格图）生成转换图像 fake_B
        if self.isTrain:
            self.fake_B = self.netG(self.real_A, self.real_B, encode_only=False)
            # 身份映射：使用 real_B 本身作为内容与风格输入生成 idt_B
            self.idt_B = self.netG(self.real_B, self.real_B, encode_only=False)
        else:
            self.fake_B = self.netG(self.real_A, encode_only=False)
            self.idt_B = self.netG(self.real_B, encode_only=False)

        # 配准模块：对 fake_B 进行配准，使其更贴近 real_B
        y_output = self.netR(self.real_A, self.real_B)
        self.deformation_field = y_output[2]
        self.registered = self.spatialTransformer(self.fake_B, self.deformation_field)
        test_image = open_image_to_torch("./deform256.jpg", 256).cuda()
        B = self.fake_B.shape[0]
        expanded_test_img = test_image.expand(B, -1, -1, -1)
        self.dvf = self.spatialTransformer(expanded_test_img, self.deformation_field)

    def get_image_paths(self):
        return self.image_paths

    def backward_D(self):
        """计算判别器损失，使用经过配准的 fake_B"""
        lambda_pair = self.get_lambda_pair()

        fake = self.registered.detach()
        fake_B = self.fake_B.detach()
        pred_fake_reg = self.netD(fake)
        pred_fake_unreg = self.netD(fake_B) if self.opt.use_lambda_pair else None
        self.loss_D_fake_reg = self.criterionGAN(pred_fake_reg, False).mean()
        self.loss_D_fake_unreg = self.criterionGAN(pred_fake_unreg, False).mean() if self.opt.use_lambda_pair else None
        self.loss_D_fake = (
                                   1 - lambda_pair) * self.loss_D_fake_reg + lambda_pair * self.loss_D_fake_unreg if self.opt.use_lambda_pair else self.loss_D_fake_reg

        pred_real = self.netD(self.real_B)
        self.loss_D_real = self.criterionGAN(pred_real, True).mean()

        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D.backward()

    def get_lambda_pair(self):
        # if self.opt.use_lambda_pair:
        #     total_epochs = self.opt.n_epochs + self.opt.n_epochs_decay
        #     t = (self.epoch - 1) / float(total_epochs)
        #     lambda_pair = 1.0 - math.sin(t * math.pi / 2)
        # else:
        #     lambda_pair = 0.0
        return 0.5

    def backward_G(self):
        # 使用双分支生成器的内容编码器提取中间特征供 InfoNCE 损失计算
        feat_real_A = self.netG(self.real_A, None, encode_only=True, layers=self.nce_layers)
        feat_fake_B = self.netG(self.fake_B, None, encode_only=True, layers=self.nce_layers)
        feat_real_B = self.netG(self.real_B, None, encode_only=True, layers=self.nce_layers)
        feat_idt_B = self.netG(self.idt_B, None, encode_only=True, layers=self.nce_layers)
        lambda_pair = self.get_lambda_pair()

        fake = self.registered

        pred_fake_reg = self.netD(fake)
        pred_fake_unreg = self.netD(self.fake_B) if self.opt.use_lambda_pair else None
        self.loss_G_GAN = ((1 - lambda_pair) * self.criterionGAN(pred_fake_reg, True).mean() +
                           lambda_pair * self.criterionGAN(pred_fake_unreg,
                                                           True).mean()) * self.opt.lambda_GAN if self.opt.use_lambda_pair else self.criterionGAN(
            pred_fake_reg, True).mean()
        self.loss_G_GAN = self.loss_G_GAN * self.opt.lambda_GAN
        loss_pixel_L1_reg = self.criterionL1(self.registered, self.real_B)
        self.loss_pixel_L1 = loss_pixel_L1_reg * self.opt.lambda_pixel_L1
        if self.opt.lamba_MMR > 0.0:
            self.loss_MMR = (-torch.log(
                self.opt.w_barrier - self.criterionNMI(self.registered, self.real_B))) * self.opt.lamba_MMR
        else:
            self.loss_MMR = 0.0
        self.loss_smooth = smooothing_loss(self.deformation_field) * self.opt.lambda_smooth
        self.loss_pixel = (self.loss_pixel_L1 + self.loss_MMR + self.loss_smooth) * self.opt.lambda_pixel

        self.loss_NCE = self.calculate_NCE_loss(feat_real_A, feat_fake_B, self.netF)
        self.loss_NCE_Y = self.calculate_NCE_loss(feat_real_B, feat_idt_B, self.netF)
        self.loss_contrast = self.loss_NCE


        self.loss_G = self.loss_G_GAN + self.loss_contrast + self.loss_pixel
        return self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        # 更新判别器
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        # 更新生成器和配准网络
        self.optimizer_G.zero_grad()
        self.optimizer_R.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        self.optimizer_R.step()

        # 更新特征提取网络 F
        self.optimizer_F.step()

    def get_current_errors(self):
        return OrderedDict([('loss_G_GAN', self.loss_G_GAN.item()),
                            ('loss_D', self.loss_D.item()),
                            ('loss_contrast', self.loss_contrast.item()),
                            ('loss_pixel', self.loss_pixel.item()),
                            ('loss_smooth', self.loss_smooth.item()),
                            ('loss_MMR', self.loss_MMR.item()),
                            ('loss_pixel_L1', self.loss_pixel_L1.item())
                            ])

    def calculate_NCE_loss(self, feat_src, feat_tgt, netF):
        n_layers = len(feat_src) if isinstance(feat_src, list) else 1
        feat_q = feat_tgt
        feat_k = feat_src
        feat_k_pool, sample_ids = netF(feat_k, self.num_patches, None)
        feat_q_pool, _ = netF(feat_q, self.num_patches, sample_ids)
        total_nce_loss = 0.0
        for f_q, f_k in zip(feat_q_pool, feat_k_pool):
            loss = self.criterionNCE(f_q, f_k)
            total_nce_loss += loss.mean()
        return total_nce_loss / n_layers
