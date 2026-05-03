"""Gaussian splat model — stores and manages all per-gaussian parameters."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianModel(nn.Module):
    """Collection of 3D Gaussian splat parameters.

    Parameters stored in "raw" space and activated before use:
    - means: [N, 3] positions (no activation)
    - scales_raw: [N, 3] -> exp() -> positive scales
    - quats_raw: [N, 4] -> normalize() -> unit quaternions
    - opacities_raw: [N, 1] -> sigmoid() -> [0, 1] opacities
    - sh_coeffs: [N, K, 3] spherical harmonics (K=1 for degree 0 = RGB only)
    """

    def __init__(self, num_gaussians: int, sh_degree: int = 0,
                 device: torch.device = torch.device("cuda")):
        super().__init__()
        self.sh_degree = sh_degree
        self.num_sh_coeffs = (sh_degree + 1) ** 2

        self.means = nn.Parameter(torch.zeros(num_gaussians, 3, device=device))
        self.scales_raw = nn.Parameter(torch.zeros(num_gaussians, 3, device=device))
        self.quats_raw = nn.Parameter(torch.zeros(num_gaussians, 4, device=device))
        self.opacities_raw = nn.Parameter(torch.zeros(num_gaussians, 1, device=device))
        self.sh_coeffs = nn.Parameter(
            torch.zeros(num_gaussians, self.num_sh_coeffs, 3, device=device)
        )

        self._init_params()

    def _init_params(self):
        """Random initialization suitable for COLMAP-free training."""
        n = self.means.shape[0]
        device = self.means.device

        # Positions: random in [-1, 1]^3
        self.means.data.uniform_(-1.0, 1.0)

        # Scales: log-space, start small
        self.scales_raw.data.uniform_(-4.0, -2.0)  # exp(-4) ~ 0.018, exp(-2) ~ 0.135

        # Quaternions: near identity [1, 0, 0, 0] with noise
        self.quats_raw.data[:, 0] = 1.0
        self.quats_raw.data[:, 1:] = torch.randn(n, 3, device=device) * 0.1

        # Opacities: start moderately high (sigmoid(2) ~ 0.88)
        self.opacities_raw.data.fill_(2.0)

        # SH coefficients: random colors (degree 0 = DC component = base color)
        self.sh_coeffs.data[:, 0, :].uniform_(-0.5, 0.5)

    @property
    def num_gaussians(self) -> int:
        return self.means.shape[0]

    def get_activated(self) -> dict:
        """Return activated (usable) parameters."""
        return {
            "means": self.means,
            "scales": torch.exp(self.scales_raw),
            "quats": F.normalize(self.quats_raw, dim=-1),
            "opacities": torch.sigmoid(self.opacities_raw).squeeze(-1),
            "sh_coeffs": self.sh_coeffs,
        }

    def get_colors(self, viewdirs: torch.Tensor | None = None) -> torch.Tensor:
        """Evaluate SH to get per-gaussian RGB colors.

        For sh_degree=0, this is just the DC component (sigmoid applied for [0,1]).
        """
        if self.sh_degree == 0:
            # DC component only: sh_coeffs[:, 0, :] represents base color
            # Apply sigmoid to map to [0, 1]
            return torch.sigmoid(self.sh_coeffs[:, 0, :])
        else:
            raise NotImplementedError("Higher-order SH not yet implemented")

    def reset_opacities(self):
        """Reset all opacities to low value (prevents floaters)."""
        # sigmoid^{-1}(0.01) ~ -4.6
        self.opacities_raw.data.fill_(-4.6)

    def clone_gaussians(self, mask: torch.Tensor) -> dict:
        """Extract parameters for gaussians matching mask (for cloning)."""
        return {
            "means": self.means.data[mask].clone(),
            "scales_raw": self.scales_raw.data[mask].clone(),
            "quats_raw": self.quats_raw.data[mask].clone(),
            "opacities_raw": self.opacities_raw.data[mask].clone(),
            "sh_coeffs": self.sh_coeffs.data[mask].clone(),
        }

    def split_gaussians(self, mask: torch.Tensor) -> tuple[dict, dict]:
        """Split large gaussians into two smaller ones.

        Each gaussian is replaced by two, offset along the longest axis,
        with scale reduced by factor of 1.6.
        """
        params = self.clone_gaussians(mask)
        scales = torch.exp(params["scales_raw"])

        # Find longest axis per gaussian
        longest_axis = scales.argmax(dim=-1)  # [M]

        # Offset along longest axis
        offsets = torch.zeros_like(params["means"])
        for i in range(3):
            ax_mask = longest_axis == i
            offsets[ax_mask, i] = scales[ax_mask, i]

        split1 = {k: v.clone() for k, v in params.items()}
        split2 = {k: v.clone() for k, v in params.items()}

        split1["means"] = params["means"] + offsets * 0.5
        split2["means"] = params["means"] - offsets * 0.5

        # Reduce scale by 1.6
        scale_reduction = torch.log(torch.tensor(1.6))
        split1["scales_raw"] = params["scales_raw"] - scale_reduction
        split2["scales_raw"] = params["scales_raw"] - scale_reduction

        return split1, split2

    def add_gaussians(self, new_params: dict):
        """Append new gaussians to the model."""
        self.means = nn.Parameter(torch.cat([self.means.data, new_params["means"]]))
        self.scales_raw = nn.Parameter(torch.cat([self.scales_raw.data, new_params["scales_raw"]]))
        self.quats_raw = nn.Parameter(torch.cat([self.quats_raw.data, new_params["quats_raw"]]))
        self.opacities_raw = nn.Parameter(torch.cat([self.opacities_raw.data, new_params["opacities_raw"]]))
        self.sh_coeffs = nn.Parameter(torch.cat([self.sh_coeffs.data, new_params["sh_coeffs"]]))

    def remove_gaussians(self, keep_mask: torch.Tensor):
        """Remove gaussians where keep_mask is False."""
        self.means = nn.Parameter(self.means.data[keep_mask])
        self.scales_raw = nn.Parameter(self.scales_raw.data[keep_mask])
        self.quats_raw = nn.Parameter(self.quats_raw.data[keep_mask])
        self.opacities_raw = nn.Parameter(self.opacities_raw.data[keep_mask])
        self.sh_coeffs = nn.Parameter(self.sh_coeffs.data[keep_mask])
