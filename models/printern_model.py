import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from collections import OrderedDict

import torchvision
from PIL import Image
from torchvision.transforms import Compose, CenterCrop, ToTensor, Normalize
import numpy as np
from skimage.filters import threshold_otsu
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks, utils
from .dn import init_dense_instance_norm, use_dense_instance_norm, DenseInstanceNorm
from .frequency_loss import Gauss_Pyramid_Conv
from .global_objective import SCLossCriterion
from .losses import NMI_Loss
from .patch_alignment_loss import PatchAlignmentLoss
from .content_loss import VGGLoss
from .patchnce import FocalNCELoss
import models.voxelmorph.torchvoxelmorph as vxm
from .registration_model import open_image_to_torch as reg_open_image_to_torch
from .voxelmorph.torchvoxelmorph.layers import SpatialTransformer
from .networks import ResnetBlock, init_net


class DifferentiableOtsuLoss(nn.Module):
    """
    Fully differentiable Otsu-like loss:
      1) learnable RGB->gray projection
      2) soft histogram via Gaussian kernel
      3) soft threshold from between-class variance with softmax
      4) soft mask via sigmoid around T
    """
    def __init__(self,
                 bins: int = 64,
                 hist_sigma: float = 0.02,     # Gaussian kernel width in value space [0,1]
                 softmax_temp: float = 0.05,   # temperature for soft-argmax over thresholds
                 mask_tau: float = 0.1,        # sigmoid temperature for mask
                 eps: float = 1e-6,
                 learn_rgb_weights: bool = True):
        super().__init__()
        self.bins = bins
        self.hist_sigma = hist_sigma
        self.softmax_temp = softmax_temp
        self.mask_tau = mask_tau
        self.eps = eps

        # learnable RGB->Gray weights (non-negative and sum to 1 via softmax)
        if learn_rgb_weights:
            w_init = torch.tensor([0.213, 0.715, 0.072])  # from your code
            self.rgb_logits = nn.Parameter(torch.log(w_init + 1e-8))
        else:
            self.register_parameter('rgb_logits', None)

        # optional learnable temperatures (comment out if you prefer fixed)
        # self.log_hist_sigma = nn.Parameter(torch.log(torch.tensor(self.hist_sigma)))
        # self.log_softmax_temp = nn.Parameter(torch.log(torch.tensor(self.softmax_temp)))
        # self.log_mask_tau = nn.Parameter(torch.log(torch.tensor(self.mask_tau)))

    def _to_gray(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,3,H,W], values assumed in [0,1]
        if self.rgb_logits is None:
            r, g, b = x[:, 0], x[:, 1], x[:, 2]
            gray = 0.213 * r + 0.715 * g + 0.072 * b
            return gray
        else:
            w = torch.softmax(self.rgb_logits, dim=0).to(x.device)  # [3], non-neg, sum=1
            gray = (x * w.view(1, 3, 1, 1)).sum(dim=1)  # [B,H,W]
            return gray

    @torch.no_grad()
    def _make_bin_centers(self, device, dtype):
        # centers in [0,1], shape [bins]
        return torch.linspace(0.0, 1.0, steps=self.bins, device=device, dtype=dtype)

    def forward(self, input_rgb: torch.Tensor, mask_gt: torch.Tensor) -> torch.Tensor:
        """
        input_rgb: [B,3,H,W], expected in [0,1]
        mask_gt:   [B,H,W],   {0,1}
        returns: scalar loss (MSE between soft mask and GT)
        """
        B, C, H, W = input_rgb.shape
        assert C == 3, "input_rgb should be 3-channel RGB in [0,1]"
        device = input_rgb.device
        dtype = input_rgb.dtype

        # if you enabled learnable temps:
        # hist_sigma = torch.exp(self.log_hist_sigma).clamp(min=1e-4)
        # softmax_temp = torch.exp(self.log_softmax_temp).clamp(min=1e-4)
        # mask_tau = torch.exp(self.log_mask_tau).clamp(min=1e-4)
        hist_sigma = self.hist_sigma
        softmax_temp = self.softmax_temp
        mask_tau = self.mask_tau

        # 1) RGB -> Gray (learnable)
        x = input_rgb.clamp(0, 1)
        gray = self._to_gray(x)  # [B,H,W]

        # 2) Soft histogram (Gaussian kernel assignment)
        centers = self._make_bin_centers(device, dtype)  # [K]
        # [B, H, W, 1] - [1,1,1,K] => [B,H,W,K]
        diff = gray.unsqueeze(-1) - centers.view(1, 1, 1, -1)
        # Gaussian kernel
        # note: scale by hist_sigma; bigger sigma => smoother histogram
        weights = torch.exp(-0.5 * (diff / (hist_sigma + self.eps)) ** 2)  # [B,H,W,K]
        weights = weights / (weights.sum(dim=-1, keepdim=True) + self.eps)

        # sum over pixels to get histogram per image
        hist = weights.sum(dim=(1, 2))  # [B,K]
        # normalize to probability distribution
        p = hist / (hist.sum(dim=-1, keepdim=True) + self.eps)  # [B,K]

        # 3) Compute between-class variance for every candidate threshold (prefix/suffix splits)
        # cumulative sums along bins
        cdf = torch.cumsum(p, dim=-1)                        # P0(t) ∈ [0,1], [B,K]
        cdf_clamped = cdf.clamp(self.eps, 1.0 - self.eps)
        P0 = cdf_clamped
        P1 = 1.0 - cdf_clamped

        # means per bin
        m = centers.view(1, -1)                              # [1,K]
        mp = p * m                                           # [B,K]
        csum_mp = torch.cumsum(mp, dim=-1)                   # sum_{<=t} c p
        # class means μ0, μ1 (safe division)
        mu0 = csum_mp / P0                                   # [B,K]
        total_mean = (p * m).sum(dim=-1, keepdim=True)       # [B,1]
        mu1_num = (total_mean - csum_mp)                     # sum_{>t} c p
        mu1 = mu1_num / P1                                   # [B,K]

        # between-class variance σ_b^2(t) = P0 * P1 * (μ0 - μ1)^2
        bc_var = P0 * P1 * (mu0 - mu1).pow(2)                # [B,K]

        # 4) Soft-argmax over thresholds to get differentiable T
        alpha = torch.softmax(bc_var / (softmax_temp + self.eps), dim=-1)  # [B,K]
        T = (alpha * m).sum(dim=-1)                           # [B]

        # 5) Soft mask with sigmoid around T (broadcast T to HxW)
        # foreground ≈ low gray → like classic Otsu on tissue-darker assumption
        # If你的数据相反（组织更亮），可改成 sigmoid((gray - T)/mask_tau)
        T_map = T.view(B, 1, 1)
        mask_soft = 1.0 - torch.sigmoid((gray - T_map) / (mask_tau + self.eps))  # [B,H,W]
        mask_gt = mask_gt.squeeze(1)
        # 6) Loss vs GT
        loss = F.mse_loss(mask_soft, mask_gt.to(dtype))

        return loss, mask_soft

###############################################
# 辅助函数及模块定义
###############################################
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



class SpatialAttention(nn.Module):
    """
    空间注意力模块：计算输入特征的注意力图并进行加权
    """

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv1(x_cat))
        return x * attention


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
###############################################
# 基于 ResNet 的编码器和解码器实现
###############################################

class ResnetEncoder(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect', no_antialias=False, no_antialias_up=False, opt=None):
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
        assert (n_blocks >= 0)
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
            if (no_antialias):
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True)]
            else:
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True),
                          Downsample()]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):  # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias)]

        self.model = nn.Sequential(*model)

    def forward(self, input, layers=[], encode_only=False):
        if -1 in layers:
            layers.append(len(self.model))
        if len(layers) > 0:

            feat = input
            feats = []
            for layer_id, layer in enumerate(self.model):
                # print(layer_id, layer)
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
            use_bias = norm_layer.func == nn.InstanceNorm2d or norm_layer.func == DenseInstanceNorm
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


def momentum_update(old_value, new_value, momentum):
    """
    动量更新：old_value * momentum + new_value * (1 - momentum)
    """
    return momentum * old_value + (1 - momentum) * new_value

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
    双分支生成器：采用基于 ResNet 的内容编码器与风格编码器，
    内容编码器支持返回中间层特征（通过传入 layers 与 encode_only 参数），
    风格编码器提取风格信息，解码器融合两者生成目标图像.

    forward() 接口：
      - 当 encode_only 为 False 时，要求传入 content_img 与 style_img，返回生成图像；
      - 当 encode_only 为 True 时，传入 layers 参数可返回内容编码器的中间特征。
    """

    def __init__(self, opt, gpu_ids):
        super(DualBranchGenerator, self).__init__()
        self.gpu_ids = gpu_ids
        self.n_layers_D = opt.n_layers_D

        self.content_encoder = ResnetEncoder(opt.input_nc, opt.output_nc, opt.ngf, norm_layer='instance', use_dropout=not opt.no_dropout,
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

    def _forward_with_anchor_layers(self, x, layers, y_anchor, x_anchor, padding, **kwargs):
        for layer in layers:
            if isinstance(layer, DenseInstanceNorm):
                print("DenseInstanceNorm")
                x = layer(x, y_anchor=y_anchor, x_anchor=x_anchor, padding=padding, **kwargs)
            elif isinstance(layer, ResnetBlock):
                residual = x
                for sub_layer in layer.conv_block:
                    if isinstance(sub_layer, DenseInstanceNorm):
                        x = sub_layer(x, y_anchor=y_anchor, x_anchor=x_anchor, padding=padding, **kwargs)
                    else:
                        x = sub_layer(x)
                x = x + residual
            elif isinstance(layer, nn.Sequential):
                x = self._forward_with_anchor_layers(x, layer, y_anchor, x_anchor, padding, **kwargs)
            else:
                x = layer(x)
        return x

    def forward_with_anchor(self, content_img, style_img=None, y_anchor=None, x_anchor=None, padding=None, **kwargs):
        # 编码内容
        # content_feat = self._forward_with_anchor_layers(content_img, self.content_encoder.model, y_anchor, x_anchor,
        #                                                 padding, **kwargs)
        content_feat = self.content_encoder(content_img)

        # 编码风格
        if style_img is not None:
            style_feat = self.style_encoder(style_img)
            _ = self.style_proto_learner(style_feat)

        # 聚合风格
        agg_style, proto_weights = self.proto_aggregator(content_feat, self.style_proto_learner.prototypes)
        params = self.decoder.style_fc(agg_style)
        content_channels = content_feat.size(1)
        gamma, beta = params[:, :content_channels], params[:, content_channels:]
        gamma = gamma.view(-1, content_channels, 1, 1)
        beta = beta.view(-1, content_channels, 1, 1)
        t = adaptive_instance_normalization(content_feat, gamma, beta)

        # 解码
        out = self._forward_with_anchor_layers(t, self.decoder.model, y_anchor, x_anchor, padding, **kwargs)
        # out = self.decoder(content_feat, agg_style)

        return out


###############################################
# PPTPLUSModel 定义（基于双分支生成器，其他部分基本保持不变）
###############################################

class PRINTERNModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """ Configures options specific for CUT model """
        parser.add_argument('--CUT_mode', type=str, default="CUT", choices='(CUT, cut, FastCUT, fastcut)')

        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--lambda_DINO', type=float, default=1.0, help='weight for DINO loss: DINO(G(X), Y)')
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

        parser.add_argument('--style_dim', type=int, default=8)
        parser.add_argument('--content_dim', type=int, default=256)
        parser.add_argument('--num_prototypes', type=int, default=150)

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
        self.loss_names = ['G', 'D', 'NCE', 'NCE_Y', 'NCE_Y_hat', 'tissue_seg']
        if self.isTrain:
            self.visual_names = ['real_A', 'real_A_mask', 'fake_B', 'mask_soft', 'real_B', 'real_B_mask', 'idt_B']
        else:
            self.visual_names = ['real_A', 'fake_B', 'real_B', 'idt_B']
        if opt.nce_idt and self.isTrain:
            self.loss_names += ['NCE_Y']
            self.visual_names += ['idt_B']

        if self.isTrain:
            self.model_names = ['G', 'F', 'D']
        else:
            self.model_names = ['G']

        self.epoch = 0
        self.DifferentiableOtsuLoss = DifferentiableOtsuLoss()

        # 使用双分支生成器（采用基于 ResNet 的编码器/解码器）
        self.netG = DualBranchGenerator(opt, opt.gpu_ids).to(self.device)


        self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout,
                                      opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

        self.netD = networks.define_D(opt.output_nc + 1, opt.ndf, opt.netD, opt.n_layers_D, opt.normD,
                                      opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

        nb_features = [
            [16, 32, 32, 64, 64, 64],  # encoder
            [64, 64, 64, 32, 32, 32, 16]  # decoder
        ]
        self.SCLossCriterion = SCLossCriterion(self.gpu_ids)
        self.DifferentiableOtsuLoss = DifferentiableOtsuLoss(
            bins=64,  # 32~128 都可以，64 较均衡
            hist_sigma=0.02,  # 直方图平滑；染色差异大时可增到 0.03~0.05
            softmax_temp=0.05,  # 阈值选择的“尖锐度”；越小越接近硬 argmax，但梯度更不稳定
            mask_tau=0.1,  # 掩码边界软化程度；越小边界越硬
        )

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)

            self.criterionGAN = networks.MaskedGANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = FocalNCELoss(opt)
            self.criterionL1 = torch.nn.L1Loss()
            self.schedulers = []
            self.optimizers = []

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

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
            self.real_A_mask = input['A_mask']
            self.real_B_mask = input['B_mask']
            if len(self.gpu_ids) > 0:
                self.real_A = input['A'].cuda(self.gpu_ids[0], non_blocking=True)
                self.real_B = input['B'].cuda(self.gpu_ids[0], non_blocking=True)
                self.real_A_mask = input['A_mask'].cuda(self.gpu_ids[0], non_blocking=True)
                self.real_B_mask = input['B_mask'].cuda(self.gpu_ids[0], non_blocking=True)
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
                self.real_A_mask = input['A_mask'].cuda(self.gpu_ids[0], non_blocking=True)
                # self.real_B_mask = input['B_mask'].cuda(self.gpu_ids[0], non_blocking=True)
            self.image_paths = input['A_paths']

    def forward(self):
        if self.isTrain:
            self.fake_B = self.netG(self.real_A, self.real_B, encode_only=False)
            # 身份映射：使用 real_B 本身作为内容与风格输入生成 idt_B
            self.idt_B = self.netG(self.real_B, self.real_B, encode_only=False)
        else:
            self.fake_B = self.netG(self.real_A, encode_only=False)
            self.idt_B = self.netG(self.real_B, encode_only=False)


    def get_image_paths(self):
        return self.image_paths


    def backward_D(self):
        fake_B = self.fake_B  # 不 detach，保证可微

        # 拼接 mask 作为额外通道
        # mask: 1 表示组织，0 表示背景
        # 对应 real_A_mask 是 fake_B 的条件，real_B_mask 是 real_B 的条件
        fake_with_mask = torch.cat([fake_B, self.real_A_mask], dim=1)  # [B, C+1, H, W]
        real_with_mask = torch.cat([self.real_B, self.real_B_mask], dim=1)  # [B, C+1, H, W]

        # 判别器预测
        pred_fake = self.netD(fake_with_mask)
        pred_real = self.netD(real_with_mask)

        # 单判别器的 GAN loss（条件输入就是 mask）
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        self.loss_D_real = self.criterionGAN(pred_real, True).mean()

        # 总损失
        self.loss_D = 0.5 * (self.loss_D_fake + self.loss_D_real)

        # 反向传播
        return self.loss_D.backward(retain_graph=True)

    def backward_G(self):
        """
        生成器训练：使用区域感知双鉴别器
        """
        # NCE 特征
        feat_real_A = self.netG(self.real_A, None, encode_only=True, layers=self.nce_layers)
        feat_fake_B = self.netG(self.fake_B, None, encode_only=True, layers=self.nce_layers)
        feat_real_B = self.netG(self.real_B, None, encode_only=True, layers=self.nce_layers)
        feat_idt_B = self.netG(self.idt_B, None, encode_only=True, layers=self.nce_layers)

        fake_B_with_mask = torch.cat([self.fake_B, self.real_A_mask], dim=1)
        pred_fake = self.netD(fake_B_with_mask)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean()

        # NCE 损失
        self.loss_NCE = self.calculate_NCE_loss(feat_real_A, feat_fake_B, self.netF)
        self.loss_NCE_Y = self.calculate_NCE_loss(feat_real_B, feat_idt_B, self.netF)
        self.loss_NCE_Y_hat = self.calculate_NCE_loss(feat_idt_B, feat_real_B, self.netF)
        self.loss_contrast = self.loss_NCE + self.loss_NCE_Y + self.loss_NCE_Y_hat

        # 组织分割损失
        self.loss_tissue_seg,  self.mask_soft = self.DifferentiableOtsuLoss(self.fake_B, self.real_A_mask)
        self.mask_soft = self.mask_soft.unsqueeze(1)
        # 总生成器损失
        self.loss_G = self.loss_G_GAN + self.loss_contrast + self.loss_tissue_seg * 0
        return self.loss_G.backward(retain_graph=True)

    def optimize_parameters(self):
        self.forward()

        # 更新判别器
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        # 更新生成器和配准网络
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        # 更新特征提取网络 F
        self.optimizer_F.step()

    def get_current_errors(self):
        return OrderedDict([('loss_G_GAN', self.loss_G_GAN.item()),
                            ('loss_D', self.loss_D.item()),
                            ('loss_contrast', self.loss_contrast.item()),
                            ('loss_NCE', self.loss_NCE.item()),
                            ('loss_NCE_Y', self.loss_NCE_Y.item()),
                            ('loss_NCE_Y_hat', self.loss_NCE_Y_hat.item()),
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
