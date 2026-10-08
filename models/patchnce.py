import math

from packaging import version
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.models import ResNet

from .vgg import VGG


class PatchNCELoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool

    def forward(self, feat_q, feat_k):
        batchSize = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()

        # pos logit
        l_pos = torch.bmm(feat_q.view(batchSize, 1, -1), feat_k.view(batchSize, -1, 1))
        l_pos = l_pos.view(batchSize, 1)

        # neg logit

        # Should the negatives from the other samples of a minibatch be utilized?
        # In CUT and FastCUT, we found that it's best to only include negatives
        # from the same image. Therefore, we set
        # --nce_includes_all_negatives_from_minibatch as False
        # However, for single-image translation, the minibatch consists of
        # crops from the "same" high-resolution image.
        # Therefore, we will include the negatives from the entire minibatch.
        if self.opt.nce_includes_all_negatives_from_minibatch:
            # reshape features as if they are all negatives of minibatch of size 1.
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = self.opt.batch_size

        # reshape features to batch size
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / self.opt.nce_T

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))

        return loss


class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.
            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])
        The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """

    def __init__(self, alpha=0.25, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs, dim=1)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)

        probs = (P * class_mask).sum(1).view(-1, 1)

        log_p = probs.log()

        batch_loss = -self.alpha * (torch.pow((1 - probs), self.gamma)) * log_p

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


# reference: https://github.com/taesungp/contrastive-unpaired-translation/tree/master

class FocalNCELoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.opt.nce_T = 0.07
        self.focal_loss = FocalLoss()
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool
        self.opt.choose_patch = 100

    def forward(self, feat_q, feat_k):
        num_patches = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()

        # pos logit
        l_pos = torch.bmm(
            feat_q.view(num_patches, 1, -1), feat_k.view(num_patches, -1, 1))
        l_pos = l_pos.view(num_patches, 1)

        # neg logit
        batch_dim_for_bmm = self.opt.batch_size

        # reshape features to batch size
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        # l_neg, _ = l_neg.sort(dim=0, descending=True)
        # print("score",l_neg[:50, :])
        # l_neg = l_neg[:50, :]

        out = torch.cat((l_pos, l_neg), dim=1) / self.opt.nce_T

        loss = self.focal_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                device=feat_q.device))

        return loss

class FDL_loss(torch.nn.Module):
    def __init__(
        self, patch_size=5, stride=1, num_proj=256, model="VGG", phase_weight=1.0
    ):
        """
        patch_size, stride, num_proj: SWD slice parameters
        model: feature extractor, support VGG, ResNet, Inception, EffNet
        phase_weight: weight for phase branch
        """

        super(FDL_loss, self).__init__()
        if model == "VGG":
            self.model = VGG()
        else:
            assert "Invalid model type! Valid models: VGG, Inception, EffNet, ResNet"

        self.phase_weight = phase_weight
        self.stride = stride
        for i in range(len(self.model.chns)):
            rand = torch.randn(num_proj, self.model.chns[i], patch_size, patch_size)
            rand = rand / rand.view(rand.shape[0], -1).norm(dim=1).unsqueeze(
                1
            ).unsqueeze(2).unsqueeze(3)
            self.register_buffer("rand_{}".format(i), rand)

        # print all the parameters

    def patch_selection(self, feats, num_patches=32):
        return_feats = []
        for feat_id, feat in enumerate(feats):
            B, C, H, W = feat.shape[0], feat.shape[1], feat.shape[2], feat.shape[3]
            # 展开成16，16大小的patch
            patch_height = math.ceil(H / num_patches)
            patch_width = math.ceil(W / num_patches)
            feat_reshape = feat.unfold(2, patch_height, patch_height).unfold(3, patch_width, patch_width)
            feat_reshape = feat_reshape.contiguous().view(-1, C, patch_height, patch_width)

            return_feats.append(feat_reshape)
        return return_feats


    def forward_once(self, x, y, idx):
        """
        x, y: input image tensors with the shape of (N, C, H, W)
        """
        rand = self.__getattr__("rand_{}".format(idx))
        projx = F.conv2d(x, rand, stride=self.stride)
        projx = projx.reshape(projx.shape[0], projx.shape[1], -1)
        projy = F.conv2d(y, rand, stride=self.stride)
        projy = projy.reshape(projy.shape[0], projy.shape[1], -1)

        # sort the convolved input
        projx, _ = torch.sort(projx, dim=-1)
        projy, _ = torch.sort(projy, dim=-1)

        # compute the mean of the sorted convolved input
        s = torch.abs(projx - projy).mean([1, 2])

        return s

    def forward(self, x, y, num_patches=64):
        x = self.model(x)
        y = self.model(y)


        score = []
        # 选择patch
        feats_x_pool = self.patch_selection(x, num_patches)
        feats_y_pool = self.patch_selection(y, num_patches)
        for i, (feat_x, feat_y) in enumerate(zip(feats_x_pool, feats_y_pool)):
            # Transform to Fourier Space
            fft_x = torch.fft.fftn(feat_x, dim=(-2, -1))
            fft_y = torch.fft.fftn(feat_y, dim=(-2, -1))

            # get the magnitude and phase of the extracted features
            x_mag = torch.abs(fft_x)
            x_phase = torch.angle(fft_x)
            y_mag = torch.abs(fft_y)
            y_phase = torch.angle(fft_y)

            s_amplitude = self.forward_once(x_mag, y_mag, i)
            s_phase = self.forward_once(x_phase, y_phase, i)

            score.append(s_amplitude + s_phase * self.phase_weight)

        score = sum(score)  # sumup between different layers
        score = score.mean()  # mean within batch
        return score  # the bigger the score, the bigger the difference between the two images
