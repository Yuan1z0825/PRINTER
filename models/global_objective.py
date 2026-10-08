import torch
import torch.nn as nn

from models.vgg import Vgg16, Vgg19, Vgg16Experimental


def gram_matrix(x, normalize=True):
    '''
    Generate gram matrices of the representations of content and style images.
    '''
    (b, ch, h, w) = x.size()
    features = x.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t)
    if normalize:
        gram /= ch * h * w
    return gram


class SCLossCriterion(nn.Module):
    def __init__(self, device):
        super(SCLossCriterion, self).__init__()

        self.device = f'cuda:{device[0]}' if torch.cuda.is_available() else 'cpu'
        self._prepare_model()

        content_layer_names = ''
        for x in self.content_layer_names:
            content_layer_names = content_layer_names + x + ','
        style_layer_names = ''
        for x in self.style_layer_names:
            style_layer_names = style_layer_names + x + ','

    def _prepare_model(self):
        # we are not tuning model weights -> requires_grad=False
        model = Vgg16(requires_grad=False)

        self.content_layer_names = model.content_layer_names
        self.style_layer_names = model.style_layer_names
        self.model = model.to(self.device).eval()

    def _content_loss(self, real_content, fake_content):
        real_content = real_content.detach()
        return nn.MSELoss(reduction='mean')(real_content, fake_content)

    def _style_loss(self, real_style, fake_style, weighted=True):
        real_style = real_style.detach()     # we dont need the gradient of the target
        size = real_style.size()

        if not weighted:
            weights = torch.ones(size=real_style.shape[0])
        else:
            # https://arxiv.org/pdf/2104.10064.pdf
            Nl = size[1] * size[2]  # C x C = C^2
            real_style_norm = torch.linalg.norm(real_style, dim=(1, 2))
            fake_style_norm = torch.linalg.norm(fake_style, dim=(1, 2))
            normalize_term = torch.square(real_style_norm) + torch.square(fake_style_norm)
            weights = Nl / normalize_term

        se = (real_style.view(size[0], -1) - fake_style.view(size[0], -1)) ** 2
        return (se.mean(dim=1) * weights).mean()

    def forward(self, content_img, style_img, fake_img):
        content_img_feature_maps = self.model(content_img)
        style_img_feature_maps = self.model(style_img)
        fake_img_feature_maps = self.model(fake_img)

        real_content_representation = [x for cnt, x in enumerate(content_img_feature_maps) if cnt in self.model.content_feature_maps_indices]
        real_style_representation = [gram_matrix(x) for cnt, x in enumerate(style_img_feature_maps) if cnt in self.model.style_feature_maps_indices]

        fake_content_representation = [x for cnt, x in enumerate(fake_img_feature_maps) if cnt in self.model.content_feature_maps_indices]
        fake_style_representation = [gram_matrix(x) for cnt, x in enumerate(fake_img_feature_maps) if cnt in self.model.style_feature_maps_indices]

        # content loss
        content_loss = 0
        for i, layer in enumerate(self.content_layer_names):
            content_loss += self._content_loss(real_content_representation[i], fake_content_representation[i])

        # style loss
        style_loss = 0
        for i, layer in enumerate(self.style_layer_names):
            style_loss += self._style_loss(real_style_representation[i], fake_style_representation[i])

        return content_loss + style_loss











