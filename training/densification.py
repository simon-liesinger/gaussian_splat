"""Adaptive density control: clone, split, and prune gaussians.

Gated on pose stability — under cold start, early gradients are dominated
by pose error, not missing geometry. Densifying on that signal amplifies
garbage. We track camera xi movement and only enable densification once
poses stabilize.
"""

import torch
from gaussian_model import GaussianModel


class DensificationController:
    """Tracks position gradients and performs densification."""

    def __init__(
        self,
        grad_threshold: float = 0.0002,
        min_opacity: float = 0.005,
        max_scale_ratio: float = 0.1,
        scene_extent: float = 3.0,
        pose_stability_threshold: float = 0.01,
        pose_stability_window: int = 100,
    ):
        self.grad_threshold = grad_threshold
        self.min_opacity = min_opacity
        self.max_scale = max_scale_ratio * scene_extent

        # Pose stability tracking
        self.pose_stability_threshold = pose_stability_threshold
        self.xi_history: list[float] = []
        self.pose_stability_window = pose_stability_window

        self.grad_accum: torch.Tensor | None = None
        self.grad_count: torch.Tensor | None = None

    def reset_accumulators(self, num_gaussians: int, device: torch.device):
        """Reset gradient accumulators."""
        self.grad_accum = torch.zeros(num_gaussians, device=device)
        self.grad_count = torch.zeros(num_gaussians, device=device)

    def update_pose_stability(self, xi_norm: float):
        """Track mean camera xi norm over time."""
        self.xi_history.append(xi_norm)
        if len(self.xi_history) > self.pose_stability_window:
            self.xi_history = self.xi_history[-self.pose_stability_window:]

    def poses_stable(self) -> bool:
        """Check if poses have stabilized enough for densification."""
        if len(self.xi_history) < self.pose_stability_window:
            return False
        recent_mean = sum(self.xi_history[-self.pose_stability_window:]) / self.pose_stability_window
        return recent_mean < self.pose_stability_threshold

    def accumulate(self, means_grad: torch.Tensor):
        """Accumulate position gradient magnitudes."""
        if self.grad_accum is None:
            self.reset_accumulators(means_grad.shape[0], means_grad.device)

        n = min(means_grad.shape[0], self.grad_accum.shape[0])
        grad_norm = means_grad[:n].detach().norm(dim=-1)
        self.grad_accum[:n] += grad_norm
        self.grad_count[:n] += 1

    def densify(self, model: GaussianModel) -> dict:
        """Perform clone/split/prune. Returns stats dict."""
        if self.grad_accum is None:
            return {"cloned": 0, "split": 0, "pruned": 0, "total": model.num_gaussians}

        device = model.means.device
        n = model.num_gaussians
        avg_grad = self.grad_accum[:n] / torch.clamp(self.grad_count[:n], min=1)
        scales = torch.exp(model.scales_raw.data)
        max_scale = scales.max(dim=-1).values
        opacities = torch.sigmoid(model.opacities_raw.data).squeeze(-1)

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

        # Remove: pruned + split originals
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

        self.reset_accumulators(model.num_gaussians, device)

        return {
            "cloned": num_cloned,
            "split": num_split,
            "pruned": num_pruned,
            "total": model.num_gaussians,
        }
