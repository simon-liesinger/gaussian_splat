"""Adaptive density control: clone, split, and prune gaussians.

Runs periodically during training to grow the gaussian population where
the model is under-reconstructed and prune where it's wasted.
"""

import torch
from gaussian_model import GaussianModel


class DensificationController:
    """Tracks position gradients and performs densification."""

    def __init__(
        self,
        grad_threshold: float = 0.0002,
        min_opacity: float = 0.005,
        max_scale_ratio: float = 0.1,  # fraction of scene extent
        scene_extent: float = 3.0,
    ):
        self.grad_threshold = grad_threshold
        self.min_opacity = min_opacity
        self.max_scale = max_scale_ratio * scene_extent

        self.grad_accum: torch.Tensor | None = None
        self.grad_count: torch.Tensor | None = None

    def reset_accumulators(self, num_gaussians: int, device: torch.device):
        """Reset gradient accumulators (call after densification or model change)."""
        self.grad_accum = torch.zeros(num_gaussians, device=device)
        self.grad_count = torch.zeros(num_gaussians, device=device)

    def accumulate(self, means_grad: torch.Tensor):
        """Accumulate position gradient magnitudes."""
        if self.grad_accum is None:
            self.reset_accumulators(means_grad.shape[0], means_grad.device)

        grad_norm = means_grad.detach().norm(dim=-1)
        self.grad_accum[:grad_norm.shape[0]] += grad_norm
        self.grad_count[:grad_norm.shape[0]] += 1

    def densify(self, model: GaussianModel) -> dict:
        """Perform clone/split/prune. Returns stats dict."""
        if self.grad_accum is None:
            return {"cloned": 0, "split": 0, "pruned": 0, "total": model.num_gaussians}

        device = model.means.device
        avg_grad = self.grad_accum / torch.clamp(self.grad_count, min=1)
        scales = torch.exp(model.scales_raw.data)
        max_scale = scales.max(dim=-1).values
        opacities = torch.sigmoid(model.opacities_raw.data).squeeze(-1)

        # Identify gaussians needing densification (large gradients)
        needs_densify = avg_grad > self.grad_threshold

        # Clone: small gaussians with large gradients
        is_small = max_scale < self.max_scale * 0.5
        clone_mask = needs_densify & is_small

        # Split: large gaussians with large gradients
        is_large = max_scale >= self.max_scale * 0.5
        split_mask = needs_densify & is_large

        # Prune: low opacity or oversized
        prune_mask = (opacities < self.min_opacity) | (max_scale > self.max_scale)

        # Execute cloning
        cloned_params = None
        num_cloned = clone_mask.sum().item()
        if num_cloned > 0:
            cloned_params = model.clone_gaussians(clone_mask)

        # Execute splitting
        split1_params = None
        split2_params = None
        num_split = split_mask.sum().item()
        if num_split > 0:
            split1_params, split2_params = model.split_gaussians(split_mask)

        # Remove: pruned + split originals (split originals are replaced by split1+split2)
        remove_mask = prune_mask | split_mask
        keep_mask = ~remove_mask
        num_pruned = prune_mask.sum().item()

        model.remove_gaussians(keep_mask)

        # Add new gaussians
        to_add = []
        if cloned_params is not None:
            to_add.append(cloned_params)
        if split1_params is not None:
            to_add.append(split1_params)
            to_add.append(split2_params)

        if to_add:
            combined = {
                k: torch.cat([p[k] for p in to_add], dim=0)
                for k in to_add[0].keys()
            }
            model.add_gaussians(combined)

        # Reset accumulators for new population
        self.reset_accumulators(model.num_gaussians, device)

        return {
            "cloned": num_cloned,
            "split": num_split,
            "pruned": num_pruned,
            "total": model.num_gaussians,
        }
