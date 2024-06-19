"""
Integrator implementing quadrature rule.
"""

from typing import Tuple
from typeguard import typechecked

from jaxtyping import Float, jaxtyped
import torch
from torch_nerf.src.renderer.integrators.integrator_base import IntegratorBase


class QuadratureIntegrator(IntegratorBase):
    """
    Numerical integrator which approximates integral using quadrature.
    """

    @jaxtyped
    @typechecked
    def integrate_along_rays(
        self,
        sigma: Float[torch.Tensor, "num_ray num_sample"],
        radiance: Float[torch.Tensor, "num_ray num_sample 3"],
        delta: Float[torch.Tensor, "num_ray num_sample"],
    ) -> Tuple[Float[torch.Tensor, "num_ray 3"], Float[torch.Tensor, "num_ray num_sample"]]:
        """
        Computes quadrature rule to approximate integral involving in volume rendering.
        Pixel colors are computed as weighted sums of radiance values collected along rays.

        For details on the quadrature rule, refer to 'Optical models for
        direct volume rendering (IEEE Transactions on Visualization and Computer Graphics 1995)'.

        Args:
            sigma: Density values sampled along rays.
            radiance: Radiance values sampled along rays.
            delta: Distance between adjacent samples along rays.

        Returns:
            rgbs: Pixel colors computed by evaluating the volume rendering equation.
            weights: Weights used to determine the contribution of each sample to the final pixel color.
                A weight at a sample point is defined as a product of transmittance and opacity,
                where opacity (alpha) is defined as 1 - exp(-sigma * delta).
        """
        # TODO
        # HINT: Look up the documentation of 'torch.cumsum'.
        sig_mul_del = sigma * delta
        opacity = 1 - torch.exp(-sig_mul_del)
        #wrong ver
        #transmittance = torch.cumsum(1 - opacity, dim=1)

        #ver1
        #transmittance = torch.exp(torch.cumsum(-sig_mul_del, dim=1))

        #ver2
        transmittance = torch.exp(torch.cat((torch.zeros((sig_mul_del.shape[0],1), device = sig_mul_del.device), torch.cumsum(-sig_mul_del[:,:-1], dim=1)), dim = 1))
        
        weight_origin = (transmittance * opacity)
        #weight = weight_origin[:,:,None].repeat(1, 1, 3)
        weight = weight_origin.unsqueeze(dim=-1)
        C_r = torch.sum(radiance * weight, dim=1)
        return C_r, weight_origin



        #solution code
        # sigma_delta = sigma * delta

        # # compute transmittance: T_{i}
        # transmittance = torch.exp(
        #     -torch.cumsum(
        #         torch.cat(
        #             [torch.zeros((sigma.shape[0], 1), device=sigma_delta.device), sigma_delta],
        #             dim=-1,
        #         ),
        #         dim=-1,
        #     )[..., :-1]
        # )

        # # compute alpha: (1 - exp (- sigma_{i} * delta_{i}))
        # alpha = 1.0 - torch.exp(-sigma_delta)

        # # compute weight - w_{i}
        # w_i = transmittance * alpha

        # # compute numerical integral to determine pixel colors
        # # C = sum_{i=1}^{S} T_{i} * alpha_{i} * c_{i}
        # rgb = torch.sum(
        #     w_i.unsqueeze(-1) * radiance,
        #     dim=1,
        # )

        # return rgb, w_i
