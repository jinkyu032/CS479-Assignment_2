"""
Pytorch implementation of MLP used in NeRF (ECCV 2020).
"""

from typing import Tuple
from typeguard import typechecked

from jaxtyping import Float, jaxtyped
import torch
import torch.nn as nn


class NeRF(nn.Module):
    """
    A multi-layer perceptron (MLP) used for learning neural radiance fields.

    For architecture details, please refer to 'NeRF: Representing Scenes as
    Neural Radiance Fields for View Synthesis (ECCV 2020, Best paper honorable mention)'.

    Attributes:
        pos_dim (int): Dimensionality of coordinate vectors of sample points.
        view_dir_dim (int): Dimensionality of view direction vectors.
        feat_dim (int): Dimensionality of feature vector within forward propagation.
    """

    def __init__(
        self,
        pos_dim: int,
        view_dir_dim: int,
        feat_dim: int = 256,
    ) -> None:
        """
        Constructor of class 'NeRF'.
        """
        super().__init__()

        # TODO
        # Initialize the sequence of MLP layers, which has 256 dimensions for each hidden layer.
        MLP_firstinput_to_beforeskip = []
        MLP_firstinput_to_beforeskip.append(nn.Linear(pos_dim, feat_dim))
        MLP_firstinput_to_beforeskip.append(nn.ReLU())
        for _ in range(3):
            MLP_firstinput_to_beforeskip.append(nn.Linear(feat_dim, feat_dim))
            MLP_firstinput_to_beforeskip.append(nn.ReLU())
        self.MLP_firstinput_to_beforeskip = nn.Sequential(*MLP_firstinput_to_beforeskip)

        MLP_skip_to_beforeviewdir = []
        MLP_skip_to_beforeviewdir.append(nn.Linear(feat_dim + pos_dim, feat_dim))
        MLP_skip_to_beforeviewdir.append(nn.ReLU())
        for _ in range(3):
            MLP_skip_to_beforeviewdir.append(nn.Linear(feat_dim, feat_dim))
            MLP_skip_to_beforeviewdir.append(nn.ReLU())
        self.MLP_skip_to_beforeviewdir = nn.Sequential(*MLP_skip_to_beforeviewdir)


        self.view_layer = nn.Linear(feat_dim, feat_dim + 1)

        self.additional_layer = nn.Linear(feat_dim + view_dir_dim, int(feat_dim/2) )
        self.final_layer = nn.Linear(int(feat_dim/2), 3)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()


    @jaxtyped
    @typechecked
    def forward(
        self,
        pos: Float[torch.Tensor, "num_sample pos_dim"],
        view_dir: Float[torch.Tensor, "num_sample view_dir_dim"],
    ) -> Tuple[Float[torch.Tensor, "num_sample 1"], Float[torch.Tensor, "num_sample 3"]]:
        """
        Predicts color and density.

        Given sample point coordinates and view directions,
        predict the corresponding radiance (RGB) and density (sigma).

        Args:
            pos: The positional encodings of sample points coordinates on rays.
            view_dir: The positional encodings of ray directions.

        Returns:
            sigma: The density predictions evaluated at the given sample points.
            radiance: The radiance predictions evaluated at the given sample points.
        """
        feat = self.MLP_firstinput_to_beforeskip(pos)
        feat = self.MLP_skip_to_beforeviewdir(torch.cat([pos, feat], dim=-1))
        sigma_feat = self.view_layer(feat)
        sigma = torch.relu(sigma_feat[..., :1])
        feat = sigma_feat[..., 1:]
        feat = self.relu(self.additional_layer(torch.cat([feat, view_dir], dim=-1)))
        rgb = self.sigmoid(self.final_layer(feat))
        return sigma, rgb
