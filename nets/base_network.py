from torchvision import models
from torch import nn

class MLPHead(nn.Module):
    """
    A neural network module that consists of a multi-layer perceptron (MLP) with one hidden layer.

    Args:
        in_channels (int): The number of input features.
        mlp_hidden_size (int): The number of neurons in the hidden layer.
        projection_size (int): The number of output features.

    Attributes:
        net (nn.Sequential): A sequential container of the MLP layers, including a linear layer, batch normalization, ReLU activation, and another linear layer.

    Methods:
        forward(x):
            Defines the forward pass of the MLPHead.
            Args:
                x (torch.Tensor): Input tensor.
            Returns:
                torch.Tensor: Output tensor after passing through the MLP.
    """
    def __init__(self, in_channels, mlp_hidden_size, projection_size):
        super(MLPHead, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_channels, mlp_hidden_size),
            nn.BatchNorm1d(mlp_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_size, projection_size)
        )

    def forward(self, x):
        return self.net(x)

class ResNet(nn.Module):
    """
    A ResNet-based neural network model for feature extraction and projection.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
            name (str): The name of the ResNet model to use ('resnet18' or 'resnet50').
            projection_head (dict): Keyword arguments for the MLPHead projection layer.

    Raises:
        ValueError: If the specified ResNet model name is not available.

    Attributes:
        encoder (nn.Sequential): The ResNet encoder excluding the final fully connected layer.
        projection (MLPHead): The projection head for the extracted features.

    Methods:
        forward(x):
            Forward pass through the network.

            Args:
                x (torch.Tensor): Input tensor.

            Returns:
                torch.Tensor: The projected features.
    """
    def __init__(self, *args, **kwargs):
        super(ResNet, self).__init__()
        if kwargs['name'] == 'resnet18':
            resnet = models.resnet18(pretrained=False)
        elif kwargs['name'] == 'resnet50':
            resnet = models.resnet50(pretrained=False)
        else:
            raise ValueError(f"Model {kwargs['name']} not available.")

        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.projection = MLPHead(in_channels=resnet.fc.in_features, **kwargs['projection_head'])

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.shape[0], h.shape[1])
        return self.projection(h)

class EfficientNet(nn.Module):
    """
    EfficientNet model with a custom projection head.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
            - name (str): Name of the EfficientNet model to use. Must be one of 
              'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 
              'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', or 'efficientnet_b7'.
            - projection_head (dict): Dictionary containing parameters for the MLPHead.

    Raises:
        ValueError: If the specified model name is not available.

    Attributes:
        encoder (nn.Sequential): Sequential model containing all layers of the EfficientNet 
            model except the final classification layer.
        projection (MLPHead): Custom projection head applied to the output of the encoder.

    Methods:
        forward(x):
            Forward pass through the EfficientNet encoder and the custom projection head.

            Args:
                x (torch.Tensor): Input tensor.

            Returns:
                torch.Tensor: Output tensor after passing through the encoder and projection head.
    """
    def __init__(self, *args, **kwargs):
        super(EfficientNet, self).__init__()
        if kwargs['name'] == 'efficientnet_b0':
            efficientnet = models.efficientnet_b0(weights=None)
        elif kwargs['name'] == 'efficientnet_b1':
            efficientnet = models.efficientnet_b1(weights=None)
        elif kwargs['name'] == 'efficientnet_b2':
            efficientnet = models.efficientnet_b2(weights=None)
        elif kwargs['name'] == 'efficientnet_b3':
            efficientnet = models.efficientnet_b3(weights=None)
        elif kwargs['name'] == 'efficientnet_b4':
            efficientnet = models.efficientnet_b4(weights=None)
        elif kwargs['name'] == 'efficientnet_b5':
            efficientnet = models.efficientnet_b5(weights=None)
        elif kwargs['name'] == 'efficientnet_b6':
            efficientnet = models.efficientnet_b6(weights=None)
        elif kwargs['name'] == 'efficientnet_b7':
            efficientnet = models.efficientnet_b7(weights=None)
        else:
            raise ValueError(f"Model {kwargs['name']} not available.")

        self.encoder = nn.Sequential(*list(efficientnet.children())[:-1])
        self.projection = MLPHead(in_channels=efficientnet.classifier[1].in_features, **kwargs['projection_head'])

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.shape[0], h.shape[1])
        return self.projection(h)
