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
        #for _ in range(3):
        for _ in range(4):
            MLP_firstinput_to_beforeskip.append(nn.Linear(feat_dim, feat_dim))
            MLP_firstinput_to_beforeskip.append(nn.ReLU())
        self.MLP_firstinput_to_beforeskip = nn.Sequential(*MLP_firstinput_to_beforeskip)

        MLP_skip_to_beforeviewdir = []
        MLP_skip_to_beforeviewdir.append(nn.Linear(feat_dim + pos_dim, feat_dim))
        MLP_skip_to_beforeviewdir.append(nn.ReLU())
        #for _ in range(3):
        for _ in range(2):
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:#[Float[torch.Tensor, "num_sample 1"], Float[torch.Tensor, "num_sample 3"]]:
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
        #sigma = torch.relu(sigma_feat[..., :1])
        sigma = torch.relu(sigma_feat[..., 0])
        feat = sigma_feat[..., 1:]
        feat = self.relu(self.additional_layer(torch.cat([feat, view_dir], dim=-1)))
        rgb = self.sigmoid(self.final_layer(feat))
        return sigma, rgb

# class NeRF(nn.Module):
#     """
#     A multi-layer perceptron (MLP) used for learning neural radiance fields.

#     For architecture details, please refer to 'NeRF: Representing Scenes as
#     Neural Radiance Fields for View Synthesis (ECCV 2020, Best paper honorable mention)'.

#     Attributes:
#         pos_dim (int): Dimensionality of coordinate vectors of sample points.
#         view_dir_dim (int): Dimensionality of view direction vectors.
#         feat_dim (int): Dimensionality of feature vector within forward propagation.
#     """

#     def __init__(
#         self,
#         pos_dim: int,
#         view_dir_dim: int,
#         feat_dim: int = 256,
#     ):
#         """
#         Constructor of class 'NeRF'.

#         Args:
#             pos_dim (int): Dimensionality of coordinate vectors of sample points.
#             view_dir_dim (int): Dimensionality of view direction vectors.
#             feat_dim (int): Dimensionality of feature vector within forward propagation.
#                 Set to 256 by default following the paper.
#         """
#         super().__init__()

#         rgb_dim = 3
#         density_dim = 1

#         self._pos_dim = pos_dim
#         self._view_dir_dim = view_dir_dim
#         self._feat_dim = feat_dim

#         # fully-connected layers
#         self.fc_in = nn.Linear(self._pos_dim, self._feat_dim)
#         self.fc_1 = nn.Linear(self._feat_dim, self._feat_dim)
#         self.fc_2 = nn.Linear(self._feat_dim, self._feat_dim)
#         self.fc_3 = nn.Linear(self._feat_dim, self._feat_dim)
#         self.fc_4 = nn.Linear(self._feat_dim, self._feat_dim)
#         self.fc_5 = nn.Linear(self._feat_dim + self._pos_dim, self._feat_dim)
#         self.fc_6 = nn.Linear(self._feat_dim, self._feat_dim)
#         self.fc_7 = nn.Linear(self._feat_dim, self._feat_dim)
#         self.fc_8 = nn.Linear(self._feat_dim, self._feat_dim + density_dim)
#         self.fc_9 = nn.Linear(self._feat_dim + self._view_dir_dim, self._feat_dim // 2)
#         self.fc_out = nn.Linear(self._feat_dim // 2, rgb_dim)

#         # activation layer
#         self.relu_actvn = nn.ReLU()
#         self.sigmoid_actvn = nn.Sigmoid()

#     def forward(
#         self,
#         pos: torch.Tensor,
#         view_dir: torch.Tensor,
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         Predicts color and density.

#         Given sample point coordinates and view directions,
#         predict the corresponding radiance (RGB) and density (sigma).

#         Args:
#             pos (torch.Tensor): Tensor of shape (N, self.pos_dim).
#                 Coordinates of sample points along rays.
#             view_dir (torch.Tensor): Tensor of shape (N, self.dir_dim).
#                 View direction vectors.

#         Returns:
#             sigma (torch.Tensor): Tensor of shape (N, ).
#                 Density at the given sample points.
#             rgb (torch.Tensor): Tensor of shape (N, 3).
#                 Radiance at the given sample points.
#         """
#         # check input tensors
#         if (pos.ndim != 2) or (view_dir.ndim != 2):
#             raise ValueError(f"Expected 2D tensors. Got {pos.ndim}, {view_dir.ndim}-D tensors.")
#         if pos.shape[0] != view_dir.shape[0]:
#             raise ValueError(
#                 f"The number of samples must match. Got {pos.shape[0]} and {view_dir.shape[0]}."
#             )
#         if pos.shape[-1] != self._pos_dim:
#             raise ValueError(f"Expected {self._pos_dim}-D position vector. Got {pos.shape[-1]}.")
#         if view_dir.shape[-1] != self._view_dir_dim:
#             raise ValueError(
#                 f"Expected {self._view_dir_dim}-D view direction vector. Got {view_dir.shape[-1]}."
#             )

#         x = self.relu_actvn(self.fc_in(pos))
#         x = self.relu_actvn(self.fc_1(x))
#         x = self.relu_actvn(self.fc_2(x))
#         x = self.relu_actvn(self.fc_3(x))
#         x = self.relu_actvn(self.fc_4(x))

#         x = torch.cat([pos, x], dim=-1)

#         x = self.relu_actvn(self.fc_5(x))
#         x = self.relu_actvn(self.fc_6(x))
#         x = self.relu_actvn(self.fc_7(x))
#         x = self.fc_8(x)

#         sigma = self.relu_actvn(x[:, 0])
#         x = torch.cat([x[:, 1:], view_dir], dim=-1)

#         x = self.relu_actvn(self.fc_9(x))
#         rgb = self.sigmoid_actvn(self.fc_out(x))

#         return sigma, rgb