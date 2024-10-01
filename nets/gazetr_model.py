import torch
import torch.nn as nn
from base_network import ResNet, EfficientNet
import numpy as np
import math
import copy

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class TransformerEncoder(nn.Module):
    """
    TransformerEncoder is a stack of N encoder layers.

    Args:
        encoder_layer (nn.Module): An instance of the encoder layer to be cloned.
        num_layers (int): The number of encoder layers.
        norm (nn.Module, optional): The normalization layer to be applied after the encoder layers. Default is None.

    Methods:
        forward(src, pos):
            Passes the input through the encoder layers and applies normalization if specified.

            Args:
                src (Tensor): The input tensor.
                pos (Tensor): The positional encoding tensor.

            Returns:
                Tensor: The output tensor after passing through the encoder layers and normalization.
    """
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, pos):
        output = src
        for layer in self.layers:
            output = layer(output, pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(nn.Module):
    """
    Transformer Encoder Layer.
    Args:
        d_model (int): The number of expected features in the input (required).
        nhead (int): The number of heads in the multiheadattention models (required).
        dim_feedforward (int, optional): The dimension of the feedforward network model (default=512).
        dropout (float, optional): The dropout value (default=0.1).
    Methods:
        pos_embed(src, pos):
            Adds positional embeddings to the input tensor.
            Args:
                src (Tensor): The input tensor.
                pos (Tensor): The positional embeddings tensor.
            Returns:
                Tensor: The input tensor with positional embeddings added.
        forward(src, pos):
            Passes the input through the transformer encoder layer.
            Args:
                src (Tensor): The input tensor.
                pos (Tensor): The positional embeddings tensor.
            Returns:
                Tensor: The output tensor after passing through the transformer encoder layer.
    """
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU(inplace=True)

    def pos_embed(self, src, pos):
        batch_pos = pos.unsqueeze(1).repeat(1, src.size(1), 1)
        return src + batch_pos  

    def forward(self, src, pos):
                # src_mask: Optional[Tensor] = None,
                # src_key_padding_mask: Optional[Tensor] = None):
                # pos: Optional[Tensor] = None):

        q = k = self.pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class GazeTR(nn.Module):
    """
    A neural network model for gaze estimation using a transformer encoder and a base model 
    (ResNet or EfficientNet) for feature extraction.
    Args:
        maps (int): The number of feature maps.
        nhead (int): The number of heads in the multi-head attention mechanism.
        dim_feature (int): The dimension of the input features.
        dim_feedforward (int): The dimension of the feedforward network model.
        dropout (float): The dropout value.
        num_layers (int): The number of layers in the transformer encoder.
        model_name (str): The name of the base model to use ('resnet' family or 'efficientnet' family).
        gaze_dim (int): The dimension of the gaze vector. Default is 2.
    Attributes:
        base_model (nn.Module): The base model for feature extraction.
        encoder (nn.TransformerEncoder): The transformer encoder.
        cls_token (nn.Parameter): The class token parameter.
        pos_embedding (nn.Embedding): The positional embedding.
        feed (nn.Linear): The final linear layer for gaze estimation.
    Methods:
        forward(x_in):
            Forward pass of the model.
            Args:
                x_in (dict): Input dictionary containing the 'face' key with the input tensor.
            Returns:
                torch.Tensor: The estimated gaze.
    """
    def __init__(self, model_name, maps, nhead, dim_feature, dim_feedforward, dropout, num_layers, mlp_hidden_size, gaze_dim=2):
        super(GazeTR, self).__init__()

        # self.base_model
        if "resnet" in model_name:
            self.base_model = ResNet(name=model_name, projection_head={"mlp_hidden_size": mlp_hidden_size, "projection_size": maps})
        elif "efficientnet" in model_name:
            self.base_model = EfficientNet(name=model_name, projection_head={"mlp_hidden_size": mlp_hidden_size, "projection_size": maps})
        else:
            raise ValueError(f"Model {model_name} not available.")

        # d_model: dim of Q, K, V 
        # nhead: seq num
        # dim_feedforward: dim of hidden linear layers
        # dropout: prob

        encoder_layer = TransformerEncoderLayer(
                  maps, 
                  nhead, 
                  dim_feedforward, 
                  dropout)

        encoder_norm = nn.LayerNorm(maps) 
        # num_encoder_layer: deeps of layers 

        self.encoder = TransformerEncoder(encoder_layer, num_layers, encoder_norm)

        self.cls_token = nn.Parameter(torch.randn(1, 1, maps))

        self.pos_embedding = nn.Embedding(dim_feature+1, maps)

        self.feed = nn.Linear(maps, gaze_dim)

    def forward(self, x_in):
        feature = self.base_model(x_in["face"])
        batch_size = feature.size(0)
        feature = feature.flatten(2)
        feature = feature.permute(2, 0, 1)
        
        cls = self.cls_token.repeat( (1, batch_size, 1))
        feature = torch.cat([cls, feature], 0)
        
        position = torch.from_numpy(np.arange(0, 50)).cuda()

        pos_feature = self.pos_embedding(position)

        # feature is [HW, batch, channel]
        feature = self.encoder(feature, pos_feature)
  
        feature = feature.permute(1, 2, 0)

        feature = feature[:,:,0]

        gaze = self.feed(feature)
        
        return gaze