import torch
import torch.nn as nn
import torch.nn.functional as F
from losses.contrastive_loss import ContrastiveLoss

from base_network import ResNet, EfficientNet
from gazetr_model import GazeTR

class BaseSimCLRException(Exception):
    """Base exception"""


class InvalidBackboneError(BaseSimCLRException):
    """Raised when the choice of backbone Convnet is invalid."""


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class GazeCLR(nn.Module):

    def __init__(self, config):
        super(GazeCLR, self).__init__()

        base_model = config.model.baseline
        out_dim = config.out_dim
        self.substring = 'face'

        self.backbone = self._get_basemodel(base_model, config)
        dim_mlp = self.backbone.fc.in_features
        self.backbone.fc = Identity()

        self.dim_mlp = dim_mlp

        # Equivariance projector p_2():  z_i = g(h_i) = W(2)σ(W(1)h_i)
        self.projector = nn.Sequential(
            nn.Linear(self.dim_mlp, config.projection_dim, bias=False),
            nn.ReLU(),
            nn.Linear(config.projection_dim, out_dim, bias=False),
        )

    def _get_basemodel(self, model_name, config):
        if config.model.transformer:
            model = GazeTR( config.model.baseline, 
                            config.model.maps, 
                            config.model.nhead, 
                            config.model.dim_feature, 
                            config.model.dim_feedforward, 
                            config.model.dropout, 
                            config.model.num_layers, 
                            config.model.mlp_hidden_size)
        elif "resnet" in model_name:
            model = ResNet(name=model_name, projection_head={"mlp_hidden_size": config.model.mlp_hidden_size, "projection_size": config.model.maps})
        elif "efficientnet" in model_name:
            model = EfficientNet(name=model_name, projection_head={"mlp_hidden_size": config.model.mlp_hidden_size, "projection_size": config.model.maps})
        else:
            raise InvalidBackboneError(f"Model {model_name} not available.")
        return model

    def get_embeddings(self, x, last_layer=False):
        if last_layer is True:
            return self.projector(self.backbone(x))
        else:
            return self.backbone(x)

    def forward_pass(self, imgs, rot1, rot2):
        batch_size = imgs.size(0)
        h_i = self.backbone(imgs)
        z_i = self.projector(h_i)
        z_i = z_i.view(batch_size, 3, -1)
        z_i = torch.bmm(rot1, z_i)
        z_i = torch.bmm(rot2, z_i)
        z_i = z_i.view(z_i.size()[0], -1)

        return h_i, z_i

    def forward(self, data_dict):

        output_dict = {}
        positive_images = data_dict['img_a']
        R_inv_gaze = data_dict['{}_R'.format(self.substring)]
        R_relative_cam = data_dict['camera_transformation'][:, :, :3, :3]
        assert len(positive_images.size()) == 5

        b, n, c, h, w = positive_images.size()
        assert R_inv_gaze.shape == (b, n, 3, 3)
        assert R_relative_cam.shape == (b, n, 3, 3)

        R_inv_gaze = torch.transpose(R_inv_gaze.view(b*n, 3, 3), 1, 2)
        R_relative_cam = torch.transpose(R_relative_cam.view(b*n, 3, 3), 1, 2)

        h_,  z_ = self.forward_pass(positive_images.view(b*n, c, h, w), R_inv_gaze, R_relative_cam)

        h_ = h_.view(b, n, -1)
        z_ = z_.view(b, n, -1)

        output_dict['feat'] = h_
        output_dict['feat_proj'] = z_

        return output_dict


class TrainGazeCLR(object):

    def __init__(self, config, device):

        self.device = device
        self.n_views = config.num_positives + 1
        self.batch_size = config.batch_size

        self.args = config

        self.model = GazeCLR(config).to(device)
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)

        self.criterion = ContrastiveLoss(device, temperature=config.temperature)

    def compute_losses(self, input_dict):

        loss_dict = {}

        output_dict = self.model(input_dict)

        features = output_dict['feat_proj']
        assert features.size()[1] == self.n_views

        contrastive_loss = 0
        count = 0
        for v1 in range(self.n_views):
            for v2 in range(v1+1, self.n_views):

                _loss = self.criterion(features[:, v1], features[:, v2])
                contrastive_loss += _loss
                count += 1
        contrastive_loss /= count

        loss_dict['contrastive_loss'] = contrastive_loss
        loss_dict['total_loss'] = contrastive_loss

        return loss_dict, output_dict
