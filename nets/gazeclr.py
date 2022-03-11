import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


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

        base_model = config.arch
        out_dim = config.out_dim
        self.substring = 'face'

        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False, num_classes=out_dim),
                            "resnet50": models.resnet50(pretrained=False, num_classes=out_dim)}

        self.backbone = self._get_basemodel(base_model)
        dim_mlp = self.backbone.fc.in_features
        self.backbone.fc = Identity()

        self.dim_mlp = dim_mlp

        # Equivariance projector p_2():  z_i = g(h_i) = W(2)σ(W(1)h_i)
        self.projector = nn.Sequential(
            nn.Linear(self.dim_mlp, config.projection_dim, bias=False),
            nn.ReLU(),
            nn.Linear(config.projection_dim, out_dim, bias=False),
        )

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise InvalidBackboneError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50")
        else:
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

        self.criterion = torch.nn.CrossEntropyLoss().to(device)

    def info_nce_loss(self, features, views, bsize):

        labels = torch.cat([torch.arange(bsize) for i in range(views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (views * b, views * b)
        assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        logits = logits / self.args.temperature

        loss = self.criterion(logits, labels)
        return loss

    def compute_losses(self, input_dict):

        loss_dict = {}

        output_dict = self.model(input_dict)

        features = output_dict['feat_proj']
        assert features.size()[1] == self.n_views

        b, n, d = features.shape

        contrastive_loss = self.info_nce_loss(features.view(b*n, d), views=n, bsize=b)

        loss_dict['contrastive_loss'] = contrastive_loss
        loss_dict['total_loss'] = contrastive_loss

        return loss_dict, output_dict
