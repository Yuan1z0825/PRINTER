import torch
from .base_model import BaseModel
from . import networks
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import kornia


class PyraMIDModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=25.0, help='weight for L1 loss')
            parser.add_argument('--which_perceptual', type=str, default='4_2', help='relu5_2 or relu4_2')
            parser.add_argument('--weight_perceptual', type=float, default=0.00003)
            parser.add_argument('--vgg_normal_correct', action='store_true',
                                help='if true, correct vgg normalization and replace vgg FM model with ctx model')
            parser.add_argument('--use_22ctx', action='store_true', help='if true, also use 2-2 in ctx loss')
            parser.add_argument('--lambda_vgg', type=float, default=2, help='weight for vgg loss')
            parser.add_argument('--ctx_w', type=float, default=1.0, help='ctx loss weight')
            parser.add_argument('--PONO', action='store_true', help='use positional normalization ')
            parser.add_argument('--weight_conv', type=float, default=100)
            parser.add_argument('--weight_sobel', type=float, default=50)
            parser.add_argument('--weight_L2', type=float, default=25)
            parser.add_argument('--weight_L3', type=float, default=25)
            parser.add_argument('--weight_L4', type=float, default=25)

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        if 'noGAN' in opt.pattern:
            self.loss_names = []
        else:
            if self.opt.netD == 'conv':
                self.loss_names = ['G_GAN', 'D']
                self.loss_D = 0
            else:
                self.loss_names = ['G_GAN', 'D_real', 'D_fake']
        if 'L1' in opt.pattern:
            self.loss_names += ['G_L1']
        if 'L2' in opt.pattern:
            self.loss_names += ['G_L2']
        if 'L3' in opt.pattern:
            self.loss_names += ['G_L3']
        if 'L4' in opt.pattern:
            self.loss_names += ['G_L4']
        if 'fft' in opt.pattern:
            self.loss_names += ['G_fft']
        if 'sobel' in opt.pattern:
            self.loss_names += ['G_sobel']
        if 'conv' in opt.pattern:
            self.loss_names += ['G_conv']
        if self.isTrain and ('perc' in opt.pattern or 'contextual' in opt.pattern):
            if 'perc' in opt.pattern:
                self.loss_names += ['G_perc']
                if self.opt.which_perceptual == '5_2':
                    self.perceptual_layer = -1
                elif self.opt.which_perceptual == '4_2':
                    self.perceptual_layer = -2
            if 'contextual' in opt.pattern:
                self.loss_names += ['G_contextual']
            self.vggnet_fix = CoCosNetworks.correspondence.VGG19_feature_color_torchversion(
                vgg_normal_correct=self.opt.vgg_normal_correct)
            self.vggnet_fix.load_state_dict(torch.load('models/vgg19_conv.pth'))
            self.vggnet_fix.eval()
            for param in self.vggnet_fix.parameters():
                param.requires_grad = False

            self.vggnet_fix.to(self.opt.gpu_ids[0])
            self.contextual_forward_loss = CoCosNetworks.ContextualLoss_forward(self.opt)

        self.visual_names = ['real_A', 'fake_B', 'real_B']
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:
            self.model_names = ['G']

        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt=opt)

        if self.isTrain:
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, self.gpu_ids)
            if opt.netD == 'conv':
                prenet = torch.load('models/vgg19_conv.pth')
                netdict = {}
                netdict['module.conv1_1.weight'] = prenet['conv1_1.weight']
                netdict['module.conv1_1.bias'] = prenet['conv1_1.bias']
                self.netD.load_state_dict(netdict)

        if self.isTrain:
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.
        """
        AtoB = self.opt.direction == 'AtoB'
        # Separate the images: `real_A` and `real_B` instead of concatenating
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

        if 'mask' in self.opt.pattern:
            self.mask = input['mask'].to(self.device)
            self.mask = 0.5 * self.mask + 1.5

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)

    def get_ctx_loss(self, source, target):
        contextual_style5_1 = torch.mean(self.contextual_forward_loss(source[-1], target[-1].detach())) * 8
        contextual_style4_1 = torch.mean(self.contextual_forward_loss(source[-2], target[-2].detach())) * 4
        contextual_style3_1 = torch.mean(
            self.contextual_forward_loss(F.avg_pool2d(source[-3], 2), F.avg_pool2d(target[-3].detach(), 2))) * 2
        if self.opt.use_22ctx:
            contextual_style2_1 = torch.mean(
                self.contextual_forward_loss(F.avg_pool2d(source[-4], 4), F.avg_pool2d(target[-4].detach(), 4))) * 1
            return contextual_style5_1 + contextual_style4_1 + contextual_style3_1 + contextual_style2_1
        return contextual_style5_1 + contextual_style4_1 + contextual_style3_1

    def sobel_conv(self, input):
        conv_op = nn.Conv2d(3, 1, 3, bias=False)
        sobel_kernel = torch.Tensor([[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]],
                                     [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]],
                                     [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]])
        sobel_kernel = sobel_kernel.expand((1, 3, 3, 3))
        conv_op.weight.data = sobel_kernel
        for param in conv_op.parameters():
            param.requires_grad = False
        conv_op.to(self.opt.gpu_ids[0])
        edge_detect = conv_op(input)

        conv_hor = nn.Conv2d(3, 1, 3, bias=False)
        hor_kernel = torch.Tensor([[[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]],
                                   [[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]],
                                   [[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]]])
        hor_kernel = hor_kernel.expand((1, 3, 3, 3))
        conv_hor.weight.data = hor_kernel
        for param in conv_hor.parameters():
            param.requires_grad = False
        conv_hor.to(self.opt.gpu_ids[0])
        hor_detect = conv_hor(input)

        conv_ver = nn.Conv2d(3, 1, 3, bias=False)
        ver_kernel = torch.Tensor([[[-3, -10, -3], [0, 0, 0], [3, 10, 3]],
                                   [[-3, -10, -3], [0, 0, 0], [3, 10, 3]],
                                   [[-3, -10, -3], [0, 0, 0], [3, 10, 3]]])
        ver_kernel = ver_kernel.expand((1, 3, 3, 3))
        conv_ver.weight.data = ver_kernel
        for param in conv_ver.parameters():
            param.requires_grad = False
        conv_ver.to(self.opt.gpu_ids[0])
        ver_detect = conv_ver(input)

        return [edge_detect, hor_detect, ver_detect]

    def frequency_division(self, src_img):
        fft_src = torch.rfft(src_img.to(self.opt.gpu_ids[0]), signal_ndim=2, onesided=False)
        fft_amp = (fft_src[:, :, :, :, 0] ** 2) + (fft_src[:, :, :, :, 1] ** 2)
        fft_amp = torch.sqrt(fft_amp)
        fft_pha = torch.atan2(fft_src[:, :, :, :, 1], fft_src[:, :, :, :, 0])

        _, _, h, w = fft_amp.size()
        amp_low = torch.zeros(fft_amp.size(), dtype=torch.float).to(self.opt.gpu_ids[0])
        b = np.floor(np.amin((h, w)) * 0.1).astype(int)
        amp_low[:, :, 0:b, 0:b] = fft_amp[:, :, 0:b, 0:b]
        amp_low[:, :, 0:b, w - b:w] = fft_amp[:, :, 0:b, w - b:w]
        amp_low[:, :, h - b:h, 0:b] = fft_amp[:, :, h - b:h, 0:b]
        amp_low[:, :, h - b:h, w - b:w] = fft_amp[:, :, h - b:h, w - b:w]
        amp_high = fft_amp - amp_low

        fft_low = torch.zeros(fft_src.size(), dtype=torch.float).to(self.opt.gpu_ids[0])
        fft_high = torch.zeros(fft_src.size(), dtype=torch.float).to(self.opt.gpu_ids[0])
        fft_low[:, :, :, :, 0] = torch.cos(fft_pha) * amp_low
        fft_low[:, :, :, :, 1] = torch.sin(fft_pha) * amp_low
        fft_high[:, :, :, :, 0] = torch.cos(fft_pha) * amp_high
        fft_high[:, :, :, :, 1] = torch.sin(fft_pha) * amp_high

        _, _, imgH, imgW = src_img.size()
        image_low = torch.irfft(fft_low, signal_ndim=2, onesided=False, signal_sizes=[imgH, imgW])
        image_high = torch.irfft(fft_high, signal_ndim=2, onesided=False, signal_sizes=[imgH, imgW])
        return image_low, image_high

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        if self.opt.netD == 'conv':
            fake_feature = self.netD(self.fake_B)
            real_feature = self.netD(self.real_B)
            self.loss_D = - self.criterionL1(fake_feature.detach(), real_feature) * self.opt.weight_conv
        else:
            fake_AB = torch.cat((self.real_A, self.fake_B), 1)
            pred_fake = self.netD(fake_AB.detach())
            self.loss_D_fake = self.criterionGAN(pred_fake, False)
            real_AB = torch.cat((self.real_A, self.real_B), 1)
            pred_real = self.netD(real_AB)
            self.loss_D_real = self.criterionGAN(pred_real, True)
            self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        self.loss_G = 0
        if 'noGAN' not in self.opt.pattern:
            if self.opt.netD == 'conv':
                fake_feature = self.netD(self.fake_B)
                real_feature = self.netD(self.real_B)
                self.loss_G_GAN = self.criterionL1(fake_feature, real_feature) * self.opt.weight_conv
            else:
                fake_AB = torch.cat((self.real_A, self.fake_B), 1)
                pred_fake = self.netD(fake_AB)
                self.loss_G_GAN = self.criterionGAN(pred_fake, True)
            self.loss_G += self.loss_G_GAN

        if 'L1' in self.opt.pattern:
            self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
            self.loss_G += self.loss_G_L1
            if 'L2' in self.opt.pattern:
                octave1_layer2_fake = kornia.filters.gaussian_blur2d(self.fake_B, (3, 3), (1, 1))
                octave1_layer3_fake = kornia.filters.gaussian_blur2d(octave1_layer2_fake, (3, 3), (1, 1))
                octave1_layer4_fake = kornia.filters.gaussian_blur2d(octave1_layer3_fake, (3, 3), (1, 1))
                octave1_layer5_fake = kornia.filters.gaussian_blur2d(octave1_layer4_fake, (3, 3), (1, 1))
                octave2_layer1_fake = kornia.filters.blur_pool2d(octave1_layer5_fake, 1, stride=2)
                octave1_layer2_real = kornia.filters.gaussian_blur2d(self.real_B, (3, 3), (1, 1))
                octave1_layer3_real = kornia.filters.gaussian_blur2d(octave1_layer2_real, (3, 3), (1, 1))
                octave1_layer4_real = kornia.filters.gaussian_blur2d(octave1_layer3_real, (3, 3), (1, 1))
                octave1_layer5_real = kornia.filters.gaussian_blur2d(octave1_layer4_real, (3, 3), (1, 1))
                octave2_layer1_real = kornia.filters.blur_pool2d(octave1_layer5_real, 1, stride=2)
                self.loss_G_L2 = self.criterionL1(octave2_layer1_fake, octave2_layer1_real) * self.opt.weight_L2
                self.loss_G += self.loss_G_L2
                if 'L3' in self.opt.pattern:
                    octave2_layer2_fake = kornia.filters.gaussian_blur2d(octave2_layer1_fake, (3, 3), (1, 1))
                    octave2_layer3_fake = kornia.filters.gaussian_blur2d(octave2_layer2_fake, (3, 3), (1, 1))
                    octave2_layer4_fake = kornia.filters.gaussian_blur2d(octave2_layer3_fake, (3, 3), (1, 1))
                    octave2_layer5_fake = kornia.filters.gaussian_blur2d(octave2_layer4_fake, (3, 3), (1, 1))
                    octave3_layer1_fake = kornia.filters.blur_pool2d(octave2_layer5_fake, 1, stride=2)
                    octave2_layer2_real = kornia.filters.gaussian_blur2d(octave2_layer1_real, (3, 3), (1, 1))
                    octave2_layer3_real = kornia.filters.gaussian_blur2d(octave2_layer2_real, (3, 3), (1, 1))
                    octave2_layer4_real = kornia.filters.gaussian_blur2d(octave2_layer3_real, (3, 3), (1, 1))
                    octave2_layer5_real = kornia.filters.gaussian_blur2d(octave2_layer4_real, (3, 3), (1, 1))
                    octave3_layer1_real = kornia.filters.blur_pool2d(octave2_layer5_real, 1, stride=2)
                    self.loss_G_L3 = self.criterionL1(octave3_layer1_fake, octave3_layer1_real) * self.opt.weight_L3
                    self.loss_G += self.loss_G_L3
                    if 'L4' in self.opt.pattern:
                        octave3_layer2_fake = kornia.filters.gaussian_blur2d(octave3_layer1_fake, (3, 3), (1, 1))
                        octave3_layer3_fake = kornia.filters.gaussian_blur2d(octave3_layer2_fake, (3, 3), (1, 1))
                        octave3_layer4_fake = kornia.filters.gaussian_blur2d(octave3_layer3_fake, (3, 3), (1, 1))
                        octave3_layer5_fake = kornia.filters.gaussian_blur2d(octave3_layer4_fake, (3, 3), (1, 1))
                        octave4_layer1_fake = kornia.filters.blur_pool2d(octave3_layer5_fake, 1, stride=2)
                        octave3_layer2_real = kornia.filters.gaussian_blur2d(octave3_layer1_real, (3, 3), (1, 1))
                        octave3_layer3_real = kornia.filters.gaussian_blur2d(octave3_layer2_real, (3, 3), (1, 1))
                        octave3_layer4_real = kornia.filters.gaussian_blur2d(octave3_layer3_real, (3, 3), (1, 1))
                        octave3_layer5_real = kornia.filters.gaussian_blur2d(octave3_layer4_real, (3, 3), (1, 1))
                        octave4_layer1_real = kornia.filters.blur_pool2d(octave3_layer5_real, 1, stride=2)
                        self.loss_G_L4 = self.criterionL1(octave4_layer1_fake, octave4_layer1_real) * self.opt.weight_L4
                        self.loss_G += self.loss_G_L4
            if 'mask' in self.opt.pattern:
                self.loss_G_L1 = self.criterionL1(self.fake_B * self.mask, self.real_B * self.mask) * self.opt.lambda_L1

        if 'fft' in self.opt.pattern:
            prenet = torch.load('models/vgg19_conv.pth')
            self.weight1_1 = prenet['conv1_1.weight'].type(torch.FloatTensor).cuda()
            # self.weight1_1 = prenet['conv1_1.weight'].type(torch.FloatTensor)
            fake_low, fake_high = self.frequency_division(self.fake_B)
            real_low, real_high = self.frequency_division(self.real_B)
            self.loss_G_fft = self.criterionL1(fake_low.to(self.opt.gpu_ids[0]), real_low.to(
                self.opt.gpu_ids[0])) * self.opt.weight_low_L1 + self.criterionL1(fake_high.to(self.opt.gpu_ids[0]),
                                                                                  real_high.to(self.opt.gpu_ids[
                                                                                                   0])) * self.opt.weight_high_L1
            self.loss_G += self.loss_G_fft

        if 'perc' in self.opt.pattern or 'contextual' in self.opt.pattern:
            real_features = self.vggnet_fix(self.real_B, ['r12', 'r22', 'r32', 'r42', 'r52'], preprocess=True)
            fake_features = self.vggnet_fix(self.fake_B, ['r12', 'r22', 'r32', 'r42', 'r52'], preprocess=True)
            if 'perc' in self.opt.pattern:
                feat_loss = torch.nn.MSELoss()(fake_features[self.perceptual_layer],
                                               real_features[self.perceptual_layer].detach())
                self.loss_G_perc = feat_loss * self.opt.weight_perceptual
                self.loss_G += self.loss_G_perc
            if 'contextual' in self.opt.pattern:
                self.loss_G_contextual = self.get_ctx_loss(fake_features,
                                                           real_features) * self.opt.lambda_vgg * self.opt.ctx_w
                self.loss_G += self.loss_G_contextual

        if 'conv' in self.opt.pattern:
            # real_conv = self.vggnet_fix(self.real_B, ['r11'], preprocess=True)
            # fake_conv = self.vggnet_fix(self.fake_B, ['r11'], preprocess=True)
            # conv_loss = 0
            # for i in range(len(fake_conv)):
            #     conv_loss += self.criterionL1(fake_conv[i], real_conv[i].detach())
            prenet = torch.load('models/vgg19_conv.pth')
            self.weight1_1 = prenet['conv1_1.weight'].type(torch.FloatTensor).cuda()
            fake_feature = F.conv2d(self.fake_B, self.weight1_1, padding=1)
            real_feature = F.conv2d(self.real_B, self.weight1_1, padding=1)
            conv_loss = self.criterionL1(fake_feature, real_feature)
            self.loss_G_conv = conv_loss * self.opt.weight_conv
            self.loss_G += self.loss_G_conv

        if 'sobel' in self.opt.pattern:
            real_sobels = self.sobel_conv(self.real_B)
            fake_sobels = self.sobel_conv(self.fake_B)
            sobel_loss = 0
            for i in range(len(fake_sobels)):
                sobel_loss += self.criterionL1(fake_sobels[i], real_sobels[i].detach())
            self.loss_G_sobel = sobel_loss * self.opt.weight_sobel
            self.loss_G += self.loss_G_sobel

        self.loss_G.backward()

    def optimize_parameters(self, fixD=False):
        self.forward()  # compute fake images: G(A)
        if 'noGAN' not in self.opt.pattern and not fixD:
            self.set_requires_grad(self.netD, True)  # enable backprop for D
            self.optimizer_D.zero_grad()  # set D's gradients to zero
            self.backward_D()  # calculate gradients for D
            self.optimizer_D.step()  # update D's weights

        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate gradients for G
        self.optimizer_G.step()  # update G's weights
