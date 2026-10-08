import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np

from models.networks import init_net




class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'hinge':
            self.loss = nn.ReLU()
        elif gan_mode in ['wgangp', 'nonsaturating']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def calculate_loss(self, prediction, target_is_real, is_dis=False):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        else:
            if is_dis:
                if target_is_real:
                    prediction = -prediction
                if self.gan_mode == 'wgangp':
                    loss = prediction.mean()
                elif self.gan_mode == 'nonsaturating':
                    loss = F.softplus(prediction).mean()
                elif self.gan_mode == 'hinge':
                    loss = self.loss(1+prediction).mean()
            else:
                if self.gan_mode == 'nonsaturating':
                   loss = F.softplus(-prediction).mean()
                else:
                    loss = -prediction.mean()
        return loss

    def __call__(self, predictions, target_is_real, is_dis=False):
        """Calculate loss for multi-scales gan"""
        if isinstance(predictions, list):
            losses = []
            for prediction in predictions:
                losses.append(self.calculate_loss(prediction, target_is_real, is_dis))
            loss = sum(losses)
        else:
            loss = self.calculate_loss(predictions, target_is_real, is_dis)

        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( ||gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        if isinstance(disc_interpolates, list):
            gradients = 0
            for disc_interpolate in disc_interpolates:
                gradients += torch.autograd.grad(outputs=disc_interpolate, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolate.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]
        else:
            gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                            grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                            create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


class StyleLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self):
        super(StyleLoss, self).__init__()
        self.add_module('vgg', VGG16())
        self.criterion = nn.L1Loss()

    def compute_gram(self, x):
        b, ch, h, w = x.size()
        f = x.view(b, ch, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (b * h * w * ch)

        return G

    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        # Compute loss
        style_loss = 0.0
        style_loss += self.criterion(self.compute_gram(x_vgg['relu1_2']), self.compute_gram(y_vgg['relu1_2']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu2_2']), self.compute_gram(y_vgg['relu2_2']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu3_3']), self.compute_gram(y_vgg['relu3_3']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu4_3']), self.compute_gram(y_vgg['relu4_3']))

        return style_loss


class PerceptualLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self, weights=[0.0, 0.0, 1.0, 0.0, 0.0]):
        super(PerceptualLoss, self).__init__()
        self.add_module('vgg', VGG16())
        self.criterion = nn.L1Loss()
        self.weights = weights

    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        content_loss = 0.0
        content_loss += self.weights[0] * self.criterion(x_vgg['relu1_2'], y_vgg['relu1_2']) if self.weights[0] > 0 else 0
        content_loss += self.weights[1] * self.criterion(x_vgg['relu2_2'], y_vgg['relu2_2']) if self.weights[1] > 0 else 0
        content_loss += self.weights[2] * self.criterion(x_vgg['relu3_3'], y_vgg['relu3_3']) if self.weights[2] > 0 else 0
        content_loss += self.weights[3] * self.criterion(x_vgg['relu4_3'], y_vgg['relu4_3']) if self.weights[3] > 0 else 0
        content_loss += self.weights[4] * self.criterion(x_vgg['relu5_3'], y_vgg['relu5_3']) if self.weights[4] > 0 else 0

        return content_loss


class PatchSim(nn.Module):
    """Calculate the similarity in selected patches"""
    def __init__(self, patch_nums=256, patch_size=None, norm=True):
        super(PatchSim, self).__init__()
        self.patch_nums = patch_nums
        self.patch_size = patch_size
        self.use_norm = norm

    def forward(self, feat, patch_ids=None):
        """
        Calculate the similarity for selected patches
        """
        B, C, W, H = feat.size()
        feat = feat - feat.mean(dim=[-2, -1], keepdim=True)
        feat = F.normalize(feat, dim=1) if self.use_norm else feat / np.sqrt(C)
        query, key, patch_ids = self.select_patch(feat, patch_ids=patch_ids)
        patch_sim = query.bmm(key) if self.use_norm else torch.tanh(query.bmm(key)/10)
        if patch_ids is not None:
            patch_sim = patch_sim.view(B, len(patch_ids), -1)

        return patch_sim, patch_ids

    def select_patch(self, feat, patch_ids=None):
        """
        Select the patches
        """
        B, C, W, H = feat.size()
        pw, ph = self.patch_size, self.patch_size
        feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2) # B*N*C
        if self.patch_nums > 0:
            if patch_ids is None:
                patch_ids = torch.randperm(feat_reshape.size(1), device=feat.device)
                patch_ids = patch_ids[:int(min(self.patch_nums, patch_ids.size(0)))]
            feat_query = feat_reshape[:, patch_ids, :]       # B*Num*C
            feat_key = []
            Num = feat_query.size(1)
            if pw < W and ph < H:
                pos_x, pos_y = patch_ids // W, patch_ids % W
                # patch should in the feature
                left, top = pos_x - int(pw / 2), pos_y - int(ph / 2)
                left, top = torch.where(left > 0, left, torch.zeros_like(left)), torch.where(top > 0, top, torch.zeros_like(top))
                start_x = torch.where(left > (W - pw), (W - pw) * torch.ones_like(left), left)
                start_y = torch.where(top > (H - ph), (H - ph) * torch.ones_like(top), top)
                for i in range(Num):
                    feat_key.append(feat[:, :, start_x[i]:start_x[i]+pw, start_y[i]:start_y[i]+ph]) # B*C*patch_w*patch_h
                feat_key = torch.stack(feat_key, dim=0).permute(1, 0, 2, 3, 4) # B*Num*C*patch_w*patch_h
                feat_key = feat_key.reshape(B * Num, C, pw * ph)  # Num * C * N
                feat_query = feat_query.reshape(B * Num, 1, C)  # Num * 1 * C
            else: # if patch larger than features size, use B * C * N (H * W)
                feat_key = feat.reshape(B, C, W*H)
        else:
            feat_query = feat.reshape(B, C, H*W).permute(0, 2, 1) # B * N (H * W) * C
            feat_key = feat.reshape(B, C, H*W)  # B * C * N (H * W)

        return feat_query, feat_key, patch_ids

class _Loss(nn.Module):
    def __init__(self, name = None):
        super().__init__()
        self.name = name

class NMI_Loss(_Loss):

    # Eugenio: what is vol_size needed for?
    def __init__(self, bin_centers, # vol_size,
                 device='cpu', sigma_ratio=0.5, max_clip=1,
                 crop_background=False, patch_size=1, name='nmi'):
        """
        Adapted from:

        Mutual information loss for image-image pairs.
        Author: Courtney Guo
        If you use this loss function, please cite the following:
        Guo, Courtney K. Multi-modal image registration with unsupervised deep learning. MEng. Thesis
        Unsupervised Learning of Probabilistic Diffeomorphic Registration for Images and Surfaces
        Adrian V. Dalca, Guha Balakrishnan, John Guttag, Mert R. Sabuncu
        MedIA: Medial Image Analysis. 2019. eprint arXiv:1903.03545
        """
        super().__init__(name=name)
        # self.vol_size = vol_size
        self.max_clip = max_clip
        self.patch_size = patch_size
        self.crop_background = crop_background
        self.mi = self.global_mi
        self.vol_bin_centers = torch.tensor(bin_centers, requires_grad=True, device=device, dtype=torch.float32)
        self.num_bins = len(bin_centers)
        self.sigma = np.mean(np.diff(bin_centers)) * sigma_ratio
        self.preterm = 1 / (2 * np.square(self.sigma))

    def global_mi(self, y_true, y_pred, padding_size=15, mask=None):

        ndims = len(y_true.shape) - 2
        conv_fn = getattr(F, 'conv%dd' % ndims)

        if self.crop_background:
            # does not support variable batch size
            thresh = 0.0001
            if mask is None:
                if ndims == 2:
                    filt = torch.ones((1, 1, padding_size, padding_size))
                    stride = [1, 1, 1, 1]
                elif ndims == 3:
                    filt = torch.ones((1, 1, padding_size, padding_size, padding_size))
                    stride = [1, 1, 1, 1, 1]
                else:
                    raise ValueError('Not valid image size')
                smooth = conv_fn(y_true, filt, stride=stride, padding=int((padding_size-1)/2))
                mask = smooth > thresh
            else:
                mask = mask > thresh

            y_pred = torch.masked_select(y_pred, mask).view((1,1,-1))
            y_true = torch.masked_select(y_true, mask).view((1,1,-1))

        else:
            # reshape: flatten images into shape (batch_size, 1, heightxwidthxdepthxchan)
            y_true = y_true.view((1,1,-1))
            y_pred = y_pred.view((1,1,-1))

        nb_voxels = y_pred.shape[2]

        # reshape bin centers to be (1, 1, B)
        o = [1, self.vol_bin_centers.numel(), 1]
        vbc = torch.reshape(self.vol_bin_centers, o)

        # compute image terms
        I_a = torch.exp(- self.preterm * (y_true  - vbc)**2)
        I_a_norm = I_a / torch.sum(I_a, dim=1, keepdim=True)

        I_b = torch.exp(- self.preterm * (y_pred  - vbc)**2)
        I_b_norm = I_b / torch.sum(I_b, dim=1, keepdim=True)

        # compute probabilities
        I_a_norm_permute = torch.transpose(I_a_norm, dim0=1, dim1=2)
        pab = torch.bmm(I_b_norm, I_a_norm_permute)  # should be the right size now, nb_labels x nb_bins
        pab /= nb_voxels

        pa = torch.mean(I_a_norm, dim=-1, keepdim=True)
        pb = torch.mean(I_b_norm, dim=-1, keepdim=True)
        papb = torch.bmm(pb, torch.transpose(pa, dim0=1, dim1=2)) + 1e-5

        return torch.sum(torch.sum(pab * torch.log(pab/papb + 1e-5), dim=1), dim=1)

    def __call__(self, y_true, y_pred, **kwargs):
        y_pred = torch.clamp(y_pred, 0, self.max_clip)
        y_true = torch.clamp(y_true, 0, self.max_clip)
        return 1-self.mi(y_true, y_pred, **kwargs)

class NCC_Loss(_Loss):

    def __init__(self, device, kernel_var=None, name=None, kernel_type='mean', eps=1e-5, *args, **kwargs):
        if name is None:
            name = 'ncc'
        super().__init__(name=name)
        self.device = device
        self.kernel_var = kernel_var
        self.kernel_type = kernel_type
        self.eps = eps

        assert kernel_type in ['mean', 'gaussian', 'linear']

    def _get_kernel(self, kernel_type, kernel_sigma):

        if kernel_type == 'mean':
            kernel = torch.ones([1, 3, *kernel_sigma]).to(self.device)

        elif kernel_type == 'linear':
            raise NotImplementedError("Linear kernel for NCC still not implemented")

        elif kernel_type == 'gaussian':
            kernel_size = kernel_sigma[0] * 3
            kernel_size += np.mod(kernel_size + 1, 2)

            # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
            x_cord = torch.arange(kernel_size)
            x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
            y_grid = x_grid.t()
            xy_grid = torch.stack([x_grid, y_grid], dim=-1)

            mean = (kernel_size - 1) / 2.
            variance = kernel_sigma[0] ** 2.

            # Calculate the 2-dimensional gaussian kernel which is
            # the product of two gaussian distributions for two different
            # variables (in this case called x and y)
            # 2.506628274631 = sqrt(2 * pi)

            kernel = (1. / (2.506628274631 * kernel_sigma[0])) * \
                     torch.exp(-torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))

            # Make sure sum of values in gaussian kernel equals 1.
            # gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

            # Reshape to 2d depthwise convolutional weight
            kernel = kernel.view(1, 3, kernel_size, kernel_size)
            kernel = kernel.to(self.device)

        return kernel

    def _compute_local_sums(self, I, J, filt, stride, padding):

        ndims = len(list(I.size())) - 2

        I2 = I * I
        J2 = J * J
        IJ = I * J

        conv_fn = getattr(F, 'conv%dd' % ndims)

        I_sum = conv_fn(I, filt, stride=stride, padding=padding)
        J_sum = conv_fn(J, filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, filt, stride=stride, padding=padding)

        win_size = torch.sum(filt)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        return I_var, J_var, cross

    def ncc(self, prediction, target):
        """
            calculate the normalize cross correlation between I and J
            assumes I, J are sized [batch_size, nb_feats, *vol_shape]
            """

        ndims = len(list(prediction.size())) - 2

        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        if self.kernel_var is None:
            if self.kernel_type == 'gaussian':
                kernel_var = [3] * ndims  # sigma=3, radius = 9
            else:
                kernel_var = [9] * ndims  # sigma=radius=9 for mean and linear filter

        else:
            kernel_var = self.kernel_var

        sum_filt = self._get_kernel(self.kernel_type, kernel_var)
        radius = sum_filt.shape[-1]
        pad_no = int(np.floor(radius / 2))

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # Eugenio: bug fixed where cross was not squared when computing cc
        I_var, J_var, cross = self._compute_local_sums(prediction, target, sum_filt, stride, padding)
        cc = cross * cross / (I_var * J_var + self.eps)

        return cc

    def forward(self, prediction, target, mask=None, *args, **kwargs):

        # if mask is not None:
        #     prediction = prediction * mask
        #     target = target * mask

        cc = self.ncc(prediction, target)


        if mask is None:
            score = 1.0 * torch.sqrt(torch.mean(cc))
        elif torch.sum(mask) == 0:
            score = torch.tensor(0)
        else:
            norm_factor = 1 / (torch.sum(mask))
            score = 1.0 * torch.sqrt(norm_factor * torch.sum(cc * mask))
        return [score, cc]
class SpatialCorrelativeLoss(nn.Module):
    """
    learnable patch-based spatially-correlative loss with contrastive learning
    """
    def __init__(self, loss_mode='cos', patch_nums=256, patch_size=32, norm=True, use_conv=True,
                 init_type='normal', init_gain=0.02, gpu_ids=[], T=0.1):
        super(SpatialCorrelativeLoss, self).__init__()
        self.patch_sim = PatchSim(patch_nums=patch_nums, patch_size=patch_size, norm=norm)
        self.patch_size = patch_size
        self.patch_nums = patch_nums
        self.norm = norm
        self.use_conv = use_conv
        self.conv_init = False
        self.init_type = init_type
        self.init_gain = init_gain
        self.gpu_ids = gpu_ids
        self.loss_mode = loss_mode
        self.T = T
        self.criterion = nn.L1Loss() if norm else nn.SmoothL1Loss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def update_init_(self):
        self.conv_init = True

    def create_conv(self, feat, layer):
        """
        create the 1*1 conv filter to select the features for a specific task
        :param feat: extracted features from a pretrained VGG or encoder for the similarity and dissimilarity map
        :param layer: different layers use different filter
        :return:
        """
        input_nc = feat.size(1)
        output_nc = max(32, input_nc // 4)
        conv = nn.Sequential(*[nn.Conv2d(input_nc, output_nc, kernel_size=1),
                               nn.ReLU(),
                               nn.Conv2d(output_nc, output_nc, kernel_size=1)])
        conv.to(feat.device)
        setattr(self, 'conv_%d' % layer, conv)
        init_net(conv, self.init_type, self.init_gain, self.gpu_ids)

    def cal_sim(self, f_src, f_tgt, f_other=None, layer=0, patch_ids=None):
        """
        calculate the similarity map using the fixed/learned query and key
        :param f_src: feature map from source domain
        :param f_tgt: feature map from target domain
        :param f_other: feature map from other image (only used for contrastive learning for spatial network)
        :return:
        """
        if self.use_conv:
            if not self.conv_init:
                self.create_conv(f_src, layer)
            conv = getattr(self, 'conv_%d' % layer)
            f_src, f_tgt = conv(f_src), conv(f_tgt)
            f_other = conv(f_other) if f_other is not None else None
        sim_src, patch_ids = self.patch_sim(f_src, patch_ids)
        sim_tgt, patch_ids = self.patch_sim(f_tgt, patch_ids)
        if f_other is not None:
            sim_other, _ = self.patch_sim(f_other, patch_ids)
        else:
            sim_other = None

        return sim_src, sim_tgt, sim_other

    def compare_sim(self, sim_src, sim_tgt, sim_other):
        """
        measure the shape distance between the same shape and different inputs
        :param sim_src: the shape similarity map from source input image
        :param sim_tgt: the shape similarity map from target output image
        :param sim_other: the shape similarity map from other input image
        :return:
        """
        B, Num, N = sim_src.size()
        if self.loss_mode == 'info' or sim_other is not None:
            sim_src = F.normalize(sim_src, dim=-1)
            sim_tgt = F.normalize(sim_tgt, dim=-1)
            sim_other = F.normalize(sim_other, dim=-1)
            sam_neg1 = (sim_src.bmm(sim_other.permute(0, 2, 1))).view(-1, Num) / self.T
            sam_neg2 = (sim_tgt.bmm(sim_other.permute(0, 2, 1))).view(-1, Num) / self.T
            sam_self = (sim_src.bmm(sim_tgt.permute(0, 2, 1))).view(-1, Num) / self.T
            sam_self = torch.cat([sam_self, sam_neg1, sam_neg2], dim=-1)
            loss = self.cross_entropy_loss(sam_self, torch.arange(0, sam_self.size(0), dtype=torch.long, device=sim_src.device) % (Num))
        else:
            tgt_sorted, _ = sim_tgt.sort(dim=-1, descending=True)
            num = int(N / 4)
            src = torch.where(sim_tgt < tgt_sorted[:, :, num:num + 1], 0 * sim_src, sim_src)
            tgt = torch.where(sim_tgt < tgt_sorted[:, :, num:num + 1], 0 * sim_tgt, sim_tgt)
            if self.loss_mode == 'l1':
                loss = self.criterion((N / num) * src, (N / num) * tgt)
            elif self.loss_mode == 'cos':
                sim_pos = F.cosine_similarity(src, tgt, dim=-1)
                loss = self.criterion(torch.ones_like(sim_pos), sim_pos)
            else:
                raise NotImplementedError('padding [%s] is not implemented' % self.loss_mode)

        return loss

    def loss(self, f_src, f_tgt, f_other=None, layer=0):
        """
        calculate the spatial similarity and dissimilarity loss for given features from source and target domain
        :param f_src: source domain features
        :param f_tgt: target domain features
        :param f_other: other random sampled features
        :param layer:
        :return:
        """
        sim_src, sim_tgt, sim_other = self.cal_sim(f_src, f_tgt, f_other, layer)
        # calculate the spatial similarity for source and target domain
        loss = self.compare_sim(sim_src, sim_tgt, sim_other)
        return loss


class Normalization(nn.Module):
    def __init__(self, device):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).to(device)
        self.mean = mean.view(-1, 1, 1)
        self.std = std.view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        features = models.vgg16(pretrained=True).features
        self.relu1_1 = torch.nn.Sequential()
        self.relu1_2 = torch.nn.Sequential()

        self.relu2_1 = torch.nn.Sequential()
        self.relu2_2 = torch.nn.Sequential()

        self.relu3_1 = torch.nn.Sequential()
        self.relu3_2 = torch.nn.Sequential()
        self.relu3_3 = torch.nn.Sequential()

        self.relu4_1 = torch.nn.Sequential()
        self.relu4_2 = torch.nn.Sequential()
        self.relu4_3 = torch.nn.Sequential()

        self.relu5_1 = torch.nn.Sequential()
        self.relu5_2 = torch.nn.Sequential()
        self.relu5_3 = torch.nn.Sequential()

        for x in range(2):
            self.relu1_1.add_module(str(x), features[x])

        for x in range(2, 4):
            self.relu1_2.add_module(str(x), features[x])

        for x in range(4, 7):
            self.relu2_1.add_module(str(x), features[x])

        for x in range(7, 9):
            self.relu2_2.add_module(str(x), features[x])

        for x in range(9, 12):
            self.relu3_1.add_module(str(x), features[x])

        for x in range(12, 14):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(14, 16):
            self.relu3_3.add_module(str(x), features[x])

        for x in range(16, 18):
            self.relu4_1.add_module(str(x), features[x])

        for x in range(18, 21):
            self.relu4_2.add_module(str(x), features[x])

        for x in range(21, 23):
            self.relu4_3.add_module(str(x), features[x])

        for x in range(23, 26):
            self.relu5_1.add_module(str(x), features[x])

        for x in range(26, 28):
            self.relu5_2.add_module(str(x), features[x])

        for x in range(28, 30):
            self.relu5_3.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        #for param in self.parameters():
        #    param.requires_grad = False

    def forward(self, x, layers=None, encode_only=False, resize=False):
        relu1_1 = self.relu1_1(x)
        relu1_2 = self.relu1_2(relu1_1)

        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)

        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)

        relu4_1 = self.relu4_1(relu3_3)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)

        relu5_1 = self.relu5_1(relu4_3)
        relu5_2 = self.relu5_2(relu5_1)
        relu5_3 = self.relu5_3(relu5_2)

        out = {
            'relu1_1': relu1_1,
            'relu1_2': relu1_2,

            'relu2_1': relu2_1,
            'relu2_2': relu2_2,

            'relu3_1': relu3_1,
            'relu3_2': relu3_2,
            'relu3_3': relu3_3,

            'relu4_1': relu4_1,
            'relu4_2': relu4_2,
            'relu4_3': relu4_3,

            'relu5_1': relu5_1,
            'relu5_2': relu5_2,
            'relu5_3': relu5_3,
        }
        if encode_only:
            if len(layers) > 0:
                feats = []
                for layer, key in enumerate(out):
                    if layer in layers:
                        feats.append(out[key])
                return feats
            else:
                return out['relu3_1']
        return out