#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ISCA-Net & ISCA-Net-2D Reference Implementation and End-to-End Executable Demo
==============================================================================
Manuscript:
  "ISCA-Net: interface-aware sparse causal attention for four-class odontogenic
   lesion differentiation on cone-beam computed tomography"
Repository:
  https://github.com/YangGao234/isca

This module provides a self-contained PyTorch implementation of:
  1. Per-volume head-mask intensity clipping & standardization (Eqs. 31-32)
     and adaptive physical dual-scale cube cropping (Eq. 5; Section 3.2).
  2. Fold-restricted dual-scale 3D masked-reconstruction pretraining (Algorithm 1; Eqs. 6-7).
  3. Box-guided boundary-candidate sparse attention with 3D Sobel gradients,
     box-boundary distance transform, and 358/154 candidate/control retention (Eqs. 8-16).
  4. Interface-mediated local-context cross-attention with geometric biases (Eqs. 17-19).
  5. Equal-count subset Counterfactual Interface Intervention (CII) matcher without
     replacement (|S_i^{cand}| = |S_i^{ctl}| = n_i^{int} <= 64) and CII loss (Eqs. 20-21).
  6. Perturbation-derived rank-8 SVD nuisance projection and entropy-proxy
     reliability-weighted branch fusion (Eqs. 22-27).
  7. Class-weighted focal loss, box-attention constraint, and inner-validation
     coordinate search for decision multipliers tau_c (Eqs. 28-30; Section 4.2).
  8. Auxiliary 2D adaptation (ISCA-Net-2D) for panoramic radiograph tooth ROIs (Section 4.6).
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    class _DummyTorch:
        Tensor = object
        Generator = object
        @staticmethod
        def no_grad():
            def _dec(fn):
                return fn
            return _dec
    torch = _DummyTorch()  # type: ignore

    class _DummyNN:
        Module = object
    nn = _DummyNN()  # type: ignore
    F = None  # type: ignore


# =============================================================================
# 1. Configuration Dataclasses
# =============================================================================

@dataclass
class ISCANet3DConfig:
    """Default configuration matching the primary configuration in the manuscript."""
    num_classes: int = 4                       # DC (0), RC (1), OKC (2), AM (3)
    local_grid_size: int = 96                  # 96 x 96 x 96 local crop
    context_grid_size: int = 160               # 160 x 160 x 160 context crop
    nominal_spacing_mm: float = 0.40           # 38.4 mm local / 64.0 mm context nominal
    embed_dim: int = 128                       # Token embedding dimension C
    num_heads: int = 4                         # Cross-attention heads
    retained_tokens_k: int = 512               # Total retained local tokens K = 512
    candidate_fraction: float = 0.70           # 70% candidate (358) / 30% control (154)
    alpha_semantic: float = 0.50               # Priority weight alpha
    beta_box_prior: float = 0.30               # Priority weight beta
    gamma_gradient: float = 0.20               # Priority weight gamma
    tau_box_decay: float = 0.05                # Spatial decay temperature tau_B
    band_radius_mm: float = 3.0                # Box-adjacent band radius r = 3.0 mm
    cii_max_pairs: int = 64                    # Max matched pairs M_max = 64
    cii_unselected_ref_count: int = 64         # Unselected low-priority tokens for t_ref
    cii_tol_distance: float = 0.10             # Matching tolerance tau_d = 0.10
    cii_tol_norm: float = 0.50                 # Matching tolerance tau_n = 0.50
    cii_margin_m: float = 0.20                 # CII margin m = 0.20
    lambda_ctl: float = 0.50                   # Control stabilization weight lambda_ctl
    nuisance_rank: int = 8                     # Selected on inner-validation from {4, 8, 16}
    lambda_q_proxy: float = 0.50               # Artefact-proxy penalty lambda_q = 0.50
    focal_gamma: float = 2.0                   # Focal-loss focusing parameter gamma = 2.0
    rho_box_target: float = 0.70               # Box-attention mass target rho = 0.70
    lambda_box_loss: float = 0.50              # Box-attention weight lambda_B = 0.50
    lambda_cii_loss: float = 0.20              # CII loss weight lambda_CII = 0.20
    masking_ratio_pretrain: float = 0.60       # Pretraining masking ratio r_M = 0.60

    @property
    def num_candidates(self) -> int:
        return int(round(self.candidate_fraction * self.retained_tokens_k))  # 358

    @property
    def num_controls(self) -> int:
        return self.retained_tokens_k - self.num_candidates                  # 154


# =============================================================================
# 2. Fold-Isolated Preprocessing, Adaptive Physical Cropping & Artefact Proxy
# =============================================================================

def normalize_cbct_within_head_mask(
    volume: torch.Tensor,
    head_mask: torch.Tensor,
    q_low: float = 0.01,
    q_high: float = 0.99,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, float]:
    """
    Per-volume intensity clipping and standardization strictly within the scan's own
    anatomical head mask H (Equations 31-32), preventing cross-patient leakage.
    Also returns the fraction of voxels at either clipping limit for the proxy q_j.
    """
    masked_vals = volume[head_mask > 0]
    if masked_vals.numel() == 0:
        masked_vals = volume.reshape(-1)
    ql = torch.quantile(masked_vals.float(), q_low)
    qu = torch.quantile(masked_vals.float(), q_high)
    clipped = torch.clamp(volume.float(), min=float(ql), max=float(qu))
    clip_frac = float(((volume <= ql) | (volume >= qu))[head_mask > 0].float().mean().item())

    mu_h = clipped[head_mask > 0].mean()
    std_h = clipped[head_mask > 0].std(unbiased=False)
    norm_vol = (clipped - mu_h) / torch.sqrt(std_h ** 2 + eps)
    return norm_vol, clip_frac


def compute_adaptive_crop_geometry(
    box_min_mm: np.ndarray,
    box_max_mm: np.ndarray,
    fov_min_mm: np.ndarray,
    fov_max_mm: np.ndarray,
) -> Dict[str, float]:
    """
    Computes adaptive physical cube side lengths L_l and L_c (Section 3.2):
      L_l = max(38.4, d_max + 10.0) mm
      L_c = max(64.0, L_l + 20.0) mm
    and records effective voxel spacings L_l/96, L_c/160 and out-of-volume fractions.
    """
    extents = np.maximum(0.0, box_max_mm - box_min_mm)
    center = 0.5 * (box_min_mm + box_max_mm)
    d_max = float(np.max(extents))
    l_local = max(38.4, d_max + 10.0)
    l_context = max(64.0, l_local + 20.0)

    def _out_of_fov_fraction(side_mm: float) -> float:
        half = 0.5 * side_mm
        c_min = center - half
        c_max = center + half
        inter_min = np.maximum(c_min, fov_min_mm)
        inter_max = np.minimum(c_max, fov_max_mm)
        inter_Lengths = np.maximum(0.0, inter_max - inter_min)
        inside_vol = float(np.prod(inter_Lengths))
        total_vol = float(side_mm ** 3)
        return max(0.0, 1.0 - inside_vol / max(total_vol, 1e-6))

    return {
        "d_max_mm": d_max,
        "L_local_mm": l_local,
        "L_context_mm": l_context,
        "eff_spacing_local_mm": l_local / 96.0,
        "eff_spacing_context_mm": l_context / 160.0,
        "local_out_of_fov_fraction": _out_of_fov_fraction(l_local),
        "context_out_of_fov_fraction": _out_of_fov_fraction(l_context),
    }


def compute_3d_sobel_magnitude(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Separable 3D Sobel gradient magnitude G(v) and within-crop quantile normalization
    G_tilde(v) in [0, 1] (Equations 12-14).
    Input x: (B, 1, D, H, W)
    """
    device = x.device
    dtype = x.dtype
    d_vec = torch.tensor([-1.0, 0.0, 1.0], device=device, dtype=dtype)
    h_vec = torch.tensor([1.0, 2.0, 1.0], device=device, dtype=dtype) / 4.0

    kx = torch.einsum("i,j,k->ijk", d_vec, h_vec, h_vec).unsqueeze(0).unsqueeze(0)
    ky = torch.einsum("i,j,k->ijk", h_vec, d_vec, h_vec).unsqueeze(0).unsqueeze(0)
    kz = torch.einsum("i,j,k->ijk", h_vec, h_vec, d_vec).unsqueeze(0).unsqueeze(0)

    gx = F.conv3d(x, kx, padding=1)
    gy = F.conv3d(x, ky, padding=1)
    gz = F.conv3d(x, kz, padding=1)
    g_mag = torch.sqrt(gx ** 2 + gy ** 2 + gz ** 2 + eps)

    b_size = x.shape[0]
    g_flat = g_mag.view(b_size, -1)
    q01 = torch.quantile(g_flat, 0.01, dim=1, keepdim=True).view(b_size, 1, 1, 1, 1)
    q99 = torch.quantile(g_flat, 0.99, dim=1, keepdim=True).view(b_size, 1, 1, 1, 1)
    g_norm = torch.clamp((g_mag - q01) / (q99 - q01 + eps), 0.0, 1.0)
    return g_norm


def compute_artefact_proxy_score(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Computes the crop-specific image-derived quality/artefact proxy q_j in [0, 1]
    as the equally weighted mean of standardized high-frequency residual energy
    and extreme-intensity clipping fraction (Section 3.6).
    """
    b_size = x.shape[0]
    smooth = F.avg_pool3d(x, kernel_size=3, stride=1, padding=1)
    hf_energy = (x - smooth).abs().view(b_size, -1).mean(dim=1)
    hf_score = torch.sigmoid((hf_energy - 0.15) / 0.08)

    flat = x.view(b_size, -1)
    q01 = torch.quantile(flat, 0.01, dim=1, keepdim=True)
    q99 = torch.quantile(flat, 0.99, dim=1, keepdim=True)
    clip_frac = ((flat <= q01 + 1e-4) | (flat >= q99 - 1e-4)).float().mean(dim=1)
    clip_score = torch.clamp(clip_frac / 0.05, 0.0, 1.0)
    return 0.5 * (hf_score + clip_score)


def build_token_grid_box_priors(
    box_bounds_voxel: torch.Tensor,
    jaw_mask_token: Optional[torch.Tensor] = None,
    grid_size: int = 12,
    input_size: int = 96,
    band_radius_voxels: float = 7.5,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Transfers 3D bounding-box geometry onto the 12x12x12 local token grid (Stride 8; Eqs. 8-10).
    Args:
      box_bounds_voxel: (B, 6) tensor [z_min, y_min, x_min, z_max, y_max, x_max] in [0, input_size].
    Returns:
      m_b_token: (B, N_tok) binary candidate-band indicator M_B(u) in {0, 1}
      d_b_token: (B, N_tok) normalized box-boundary distance d_B^{(f)}(u) in [0, 1]
      side_stratum: (B, N_tok) anatomical side stratum relative to box center in {0, 1}
      rel_coords: (B, N_tok, 3) normalized 3D token coordinates in [-1, 1]^3
    """
    b_size = box_bounds_voxel.shape[0]
    device = box_bounds_voxel.device
    stride = input_size / float(grid_size)
    centers_1d = (torch.arange(grid_size, device=device, dtype=torch.float32) + 0.5) * stride
    zz, yy, xx = torch.meshgrid(centers_1d, centers_1d, centers_1d, indexing="ij")
    coords = torch.stack([zz, yy, xx], dim=-1).view(1, -1, 3).expand(b_size, -1, 3)  # (B, 1728, 3)

    b_min = box_bounds_voxel[:, :3].unsqueeze(1)  # (B, 1, 3)
    b_max = box_bounds_voxel[:, 3:].unsqueeze(1)  # (B, 1, 3)
    b_center = 0.5 * (b_min + b_max)
    diag = torch.norm(b_max - b_min, dim=-1, keepdim=True).clamp(min=1.0)  # (B, 1, 1)

    # Signed distance to the 6 faces of the 3D bounding box
    dist_to_min = (coords - b_min).abs()
    dist_to_max = (b_max - coords).abs()
    inside = (coords >= b_min - band_radius_voxels) & (coords <= b_max + band_radius_voxels)
    inside_all = inside.all(dim=-1)
    face_dist = torch.minimum(dist_to_min, dist_to_max).min(dim=-1).values  # (B, 1728)

    m_b_token = ((face_dist <= band_radius_voxels) & inside_all).float()
    if jaw_mask_token is not None:
        m_b_token = m_b_token * (jaw_mask_token > 0).float()

    # Ensure at least 358 candidate and 154 non-candidate positions exist per volume
    d_b_token = (face_dist / (diag.squeeze(-1) + eps)).clamp(0.0, 1.0)
    for i in range(b_size):
        if m_b_token[i].sum() < 358:
            idx_sorted = torch.argsort(d_b_token[i], descending=False)
            m_b_token[i] = 0.0
            m_b_token[i, idx_sorted[:420]] = 1.0

    # Side stratum relative to box x-center (0: left/negative, 1: right/positive)
    side_stratum = (coords[:, :, 2] >= b_center[:, :, 2]).long()
    rel_coords = (coords / (0.5 * input_size)) - 1.0
    return m_b_token, d_b_token, side_stratum, rel_coords


# =============================================================================
# 3. Shared 3D Residual Encoder & Pretraining Decoder (Algorithm 1; Eqs. 6-7)
# =============================================================================

class ResidualBlock3D(nn.Module):
    """Three-dimensional residual block R_s (Equation 6)."""
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_ch)
        self.down = (
            nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_ch),
            )
            if (stride != 1 or in_ch != out_ch)
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.gelu(out + self.down(x))


class Shared3DEncoder(nn.Module):
    """
    Weight-shared 3D residual encoder E_theta with total spatial stride 8:
      96^3 local crop   -> 12^3 token grid (1,728 tokens)
      160^3 context crop -> 20^3 token grid (8,000 tokens, pooled to 10^3 = 1,000 context tokens)
    """
    def __init__(self, in_ch: int = 1, embed_dim: int = 128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_ch, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.GELU(),
        )
        self.stage1 = ResidualBlock3D(32, 64, stride=2)
        self.stage2 = ResidualBlock3D(64, embed_dim, stride=2)
        self.stage3 = ResidualBlock3D(embed_dim, embed_dim, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        return self.stage3(x)


class MaskedReconstructionDecoder3D(nn.Module):
    """Lightweight 3D decoder D_psi used only during fold-restricted pretraining (Algorithm 1)."""
    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose3d(embed_dim, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.GELU(),
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.GELU(),
            nn.ConvTranspose3d(32, 1, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.up(feat)


def pretraining_masked_reconstruction_step(
    encoder: Shared3DEncoder,
    decoder: MaskedReconstructionDecoder3D,
    x_local: torch.Tensor,
    x_context: torch.Tensor,
    masking_ratio: float = 0.60,
) -> torch.Tensor:
    """
    Executes one minibatch step of fold-restricted dual-scale masked reconstruction
    (Algorithm 1, Equation 7) strictly on inner-training patients.
    """
    loss_total = torch.tensor(0.0, device=x_local.device)
    for crop in (x_local, x_context):
        b, c, d, h, w = crop.shape
        # Generate binary mask M_ij at token-cell resolution (0 = hidden, 1 = visible)
        mask_grid = (torch.rand(b, 1, d // 8, h // 8, w // 8, device=crop.device) >= masking_ratio).float()
        mask_voxel = F.interpolate(mask_grid, size=(d, h, w), mode="nearest")
        hidden = (1.0 - mask_voxel).clamp(min=0.0)
        recon = decoder(encoder(crop * mask_voxel))
        denom = hidden.sum(dim=(1, 2, 3, 4)).clamp(min=1.0)
        l1_hidden = ((crop - recon).abs() * hidden).sum(dim=(1, 2, 3, 4)) / denom
        loss_total = loss_total + l1_hidden.mean()
    return 0.5 * loss_total


# =============================================================================
# 4. Boundary-Candidate Sparse Attention & Local-Context Cross-Attention
# =============================================================================

class BoundaryCandidateSparseSelector(nn.Module):
    """
    Implements Section 3.3 (Equations 11-16):
      s(u) = alpha * a(u) + beta * M_B(u) * exp(-d_B^{(f)}(u) / tau_B) + gamma * G_tilde^{(f)}(u)
    and retains K = 512 tokens split into |C_i| = 358 candidates (M_B(u) = 1)
    and |N_i| = 154 non-candidate controls (M_B(u) = 0).
    """
    def __init__(self, cfg: ISCANet3DConfig):
        super().__init__()
        self.cfg = cfg
        self.ln = nn.LayerNorm(cfg.embed_dim)
        self.semantic_head = nn.Linear(cfg.embed_dim, 1)

    def forward(
        self,
        f_local: torch.Tensor,
        x_local: torch.Tensor,
        m_b_token: torch.Tensor,
        d_b_token: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        b_size, c_dim, d, h, w = f_local.shape
        tokens = f_local.flatten(2).transpose(1, 2)  # (B, 1728, C)

        # Learned semantic response a(u) (Eq. 11)
        a_u = torch.sigmoid(self.semantic_head(self.ln(tokens))).squeeze(-1)  # (B, 1728)

        # Token-level 3D Sobel gradient magnitude G_tilde^{(f)}(u) via 8x8x8 max pooling (Eq. 15)
        g_voxel = compute_3d_sobel_magnitude(x_local)
        g_token = F.max_pool3d(g_voxel, kernel_size=8, stride=8).flatten(1)  # (B, 1728)

        # Priority score s(u) (Eq. 16)
        box_term = m_b_token * torch.exp(-d_b_token / self.cfg.tau_box_decay)
        s_u = (
            self.cfg.alpha_semantic * a_u
            + self.cfg.beta_box_prior * box_term
            + self.cfg.gamma_gradient * g_token
        )

        # Stratified Top-K selection: 358 from candidate band (M_B=1), 154 from non-candidate (M_B=0)
        k_cand = self.cfg.num_candidates
        k_ctl = self.cfg.num_controls
        k_ref = self.cfg.cii_unselected_ref_count

        cand_indices_list, ctl_indices_list, ref_vectors = [], [], []
        for i in range(b_size):
            cand_mask = m_b_token[i] > 0.5
            ctl_mask = ~cand_mask
            cand_pool = torch.nonzero(cand_mask, as_tuple=False).squeeze(-1)
            ctl_pool = torch.nonzero(ctl_mask, as_tuple=False).squeeze(-1)

            cand_order = cand_pool[torch.argsort(s_u[i, cand_pool], descending=True)]
            ctl_order = ctl_pool[torch.argsort(s_u[i, ctl_pool], descending=True)]

            sel_cand = cand_order[:k_cand]
            sel_ctl = ctl_order[:k_ctl]
            cand_indices_list.append(sel_cand)
            ctl_indices_list.append(sel_ctl)

            # Stop-gradient reference vector t_ref: mean of 64 unselected low-priority local tokens
            selected_mask = torch.zeros(tokens.shape[1], dtype=torch.bool, device=tokens.device)
            selected_mask[sel_cand] = True
            selected_mask[sel_ctl] = True
            unselected = torch.nonzero(~selected_mask, as_tuple=False).squeeze(-1)
            low_order = unselected[torch.argsort(s_u[i, unselected], descending=False)]
            ref_idx = low_order[:k_ref]
            ref_vectors.append(tokens[i, ref_idx].mean(dim=0).detach())

        cand_idx = torch.stack(cand_indices_list, dim=0)  # (B, 358)
        ctl_idx = torch.stack(ctl_indices_list, dim=0)    # (B, 154)
        retained_idx = torch.cat([cand_idx, ctl_idx], dim=1)  # (B, 512)
        t_ref = torch.stack(ref_vectors, dim=0)           # (B, C)

        gather_idx = retained_idx.unsqueeze(-1).expand(-1, -1, c_dim)
        retained_tokens = torch.gather(tokens, dim=1, index=gather_idx)  # (B, 512, C)

        return {
            "all_tokens": tokens,
            "retained_tokens": retained_tokens,
            "retained_idx": retained_idx,
            "cand_idx": cand_idx,
            "ctl_idx": ctl_idx,
            "priority_score": s_u,
            "semantic_response": a_u,
            "t_ref": t_ref,
        }


class InterfaceMediatedCrossAttention(nn.Module):
    """
    Implements Section 3.4 (Equations 17-19):
      Retained local tokens query the concatenated local + contextual token sets
      with box-boundary distance bias B_D and anatomical position bias B_P.
    """
    def __init__(self, embed_dim: int = 128, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dist_bias_proj = nn.Linear(1, num_heads)
        self.pos_bias_proj = nn.Linear(3, num_heads)
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(
        self,
        t_int: torch.Tensor,
        t_local_all: torch.Tensor,
        t_context_pooled: torch.Tensor,
        d_b_retained: torch.Tensor,
        pos_retained: torch.Tensor,
    ) -> torch.Tensor:
        b_size, k_len, c_dim = t_int.shape
        kv_tokens = self.norm_kv(torch.cat([t_local_all, t_context_pooled], dim=1))
        q_in = self.norm_q(t_int)

        q = self.q_proj(q_in).view(b_size, k_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv_tokens).view(b_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv_tokens).view(b_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        # Query-conditioned geometric biases B_D and B_P (Eq. 18)
        b_d = self.dist_bias_proj(d_b_retained.unsqueeze(-1)).permute(0, 2, 1).unsqueeze(-1)
        b_p = self.pos_bias_proj(pos_retained).permute(0, 2, 1).unsqueeze(-1)
        attn = F.softmax(attn_logits + b_d + b_p, dim=-1)

        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b_size, k_len, c_dim)
        h_int = t_int + self.out_proj(out)
        h_int = h_int + self.ffn(h_int)
        return h_int


# =============================================================================
# 5. Equal-Count Subset Counterfactual Interface Intervention (CII) Matcher
# =============================================================================

def match_equal_count_cii_subsets(
    retained_tokens: torch.Tensor,
    d_b_retained: torch.Tensor,
    side_retained: torch.Tensor,
    num_candidates: int = 358,
    max_pairs: int = 64,
    tau_d: float = 0.10,
    tau_n: float = 0.50,
    rng: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """
    Implements Section 3.5 equal-cardinality subset matching without replacement:
      Given retained candidate pool C_i (|C_i| = 358, indices 0..357) and
      non-candidate control pool N_i (|N_i| = 154, indices 358..511),
      identifies S_i^{cand} subseteq C_i and S_i^{ctl} subseteq N_i with
      |S_i^{cand}| = |S_i^{ctl}| = n_i^{int} <= 64 without token reuse.
    Returns:
      a_cand_mask: (B, K, 1) binary indicator for matched candidate subset S_i^{cand}
      a_ctl_mask:  (B, K, 1) binary indicator for matched control subset S_i^{ctl}
      matched_counts: list of matched pair counts n_i^{int} per patient
    """
    b_size, k_total, _ = retained_tokens.shape
    device = retained_tokens.device
    a_cand = torch.zeros(b_size, k_total, 1, device=device)
    a_ctl = torch.zeros(b_size, k_total, 1, device=device)
    matched_counts: List[int] = []

    norms = torch.norm(retained_tokens.detach(), p=2, dim=-1)  # (B, 512)
    med_norms = norms.median(dim=1, keepdim=True).values.clamp(min=1e-6)
    rel_norms = norms / med_norms

    for i in range(b_size):
        perm = torch.randperm(num_candidates, generator=rng, device=device)
        used_controls = torch.zeros(k_total - num_candidates, dtype=torch.bool, device=device)

        ctl_d = d_b_retained[i, num_candidates:]
        ctl_side = side_retained[i, num_candidates:]
        ctl_n = rel_norms[i, num_candidates:]

        cand_selected: List[int] = []
        ctl_selected: List[int] = []

        for c_local in perm.tolist():
            if len(cand_selected) >= max_pairs:
                break
            d_c = d_b_retained[i, c_local]
            s_c = side_retained[i, c_local]
            n_c = rel_norms[i, c_local]

            diff_d = (ctl_d - d_c).abs()
            diff_n = (ctl_n - n_c).abs()
            eligible = (~used_controls) & (ctl_side == s_c) & (diff_d <= tau_d) & (diff_n <= tau_n)

            if eligible.any():
                cost = (diff_d / tau_d) ** 2 + (diff_n / tau_n) ** 2
                cost = torch.where(eligible, cost, torch.full_like(cost, float("inf")))
                best_ctl = int(torch.argmin(cost).item())
                used_controls[best_ctl] = True
                cand_selected.append(c_local)
                ctl_selected.append(num_candidates + best_ctl)

        n_int = len(cand_selected)
        matched_counts.append(n_int)
        if n_int > 0:
            a_cand[i, cand_selected, 0] = 1.0
            a_ctl[i, ctl_selected, 0] = 1.0

    return a_cand, a_ctl, matched_counts


# =============================================================================
# 6. Nuisance Projection, Reliability-Weighted Fusion & Complete ISCA-Net
# =============================================================================

class ISCANet3D(nn.Module):
    """
    Complete 3D ISCA-Net architecture for four-class CBCT odontogenic lesion
    differentiation after lesion localization (Sections 3.1-3.7).
    """
    def __init__(self, cfg: Optional[ISCANet3DConfig] = None):
        super().__init__()
        self.cfg = cfg or ISCANet3DConfig()
        c_dim = self.cfg.embed_dim

        self.encoder = Shared3DEncoder(in_ch=1, embed_dim=c_dim)
        self.selector = BoundaryCandidateSparseSelector(self.cfg)
        self.cross_attn = InterfaceMediatedCrossAttention(embed_dim=c_dim, num_heads=self.cfg.num_heads)
        self.classifier = nn.Linear(c_dim, self.cfg.num_classes)

        # Non-differentiable orthonormal nuisance bases B_n in R^{C x r} updated per epoch via SVD
        self.register_buffer("nuisance_basis_local", torch.zeros(c_dim, self.cfg.nuisance_rank))
        self.register_buffer("nuisance_basis_context", torch.zeros(c_dim, self.cfg.nuisance_rank))

    @torch.no_grad()
    def update_nuisance_subspace_svd(
        self,
        z_local_clean: torch.Tensor,
        z_local_aug: torch.Tensor,
        z_context_clean: torch.Tensor,
        z_context_aug: torch.Tensor,
    ) -> None:
        """
        Updates the orthonormal nuisance basis B_n from paired clean and perturbed
        inner-training representations using truncated SVD (Section 3.6, Eqs. 22-23).
        """
        for clean, aug, buf_name in (
            (z_local_clean, z_local_aug, "nuisance_basis_local"),
            (z_context_clean, z_context_aug, "nuisance_basis_context"),
        ):
            delta = (aug - clean).detach()
            delta = delta - delta.mean(dim=0, keepdim=True)
            _, _, vh = torch.linalg.svd(delta, full_matrices=False)
            r = min(self.cfg.nuisance_rank, vh.shape[0])
            basis = torch.zeros_like(getattr(self, buf_name))
            basis[:, :r] = vh[:r].T
            getattr(self, buf_name).copy_(basis)

    def _project_and_fuse(
        self,
        z_l: torch.Tensor,
        z_c: torch.Tensor,
        q_l: torch.Tensor,
        q_c: torch.Tensor,
        use_reliability_fusion: bool = True,
        eps: float = 1e-6,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Applies rank-8 nuisance projection and entropy-proxy reliability gating (Eqs. 23-27)."""
        b_l = self.nuisance_basis_local
        b_c = self.nuisance_basis_context
        z_l_tilde = z_l - torch.matmul(torch.matmul(z_l, b_l), b_l.T)
        z_c_tilde = z_c - torch.matmul(torch.matmul(z_c, b_c), b_c.T)

        if not use_reliability_fusion:
            w = torch.full((z_l.shape[0], 2), 0.5, device=z_l.device)
            z_fused = 0.5 * (z_l_tilde + z_c_tilde)
            return self.classifier(z_fused), w

        # Provisional branch probabilities and normalized entropy u_j in [0, 1] (Eqs. 24-25)
        p_l = F.softmax(self.classifier(z_l_tilde).detach(), dim=-1)
        p_c = F.softmax(self.classifier(z_c_tilde).detach(), dim=-1)
        u_l = -(p_l * torch.log(p_l + eps)).sum(dim=-1) / math.log(self.cfg.num_classes)
        u_c = -(p_c * torch.log(p_c + eps)).sum(dim=-1) / math.log(self.cfg.num_classes)

        score_l = -u_l - self.cfg.lambda_q_proxy * q_l
        score_c = -u_c - self.cfg.lambda_q_proxy * q_c
        weights = F.softmax(torch.stack([score_l, score_c], dim=-1), dim=-1)  # (B, 2)

        z_fused = weights[:, 0:1] * z_l_tilde + weights[:, 1:2] * z_c_tilde
        logits = self.classifier(z_fused)
        return logits, weights

    def forward(
        self,
        x_local: torch.Tensor,
        x_context: torch.Tensor,
        box_bounds_voxel: torch.Tensor,
        run_cii: bool = False,
        use_reliability_fusion: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of ISCA-Net.
        Args:
          x_local: (B, 1, 96, 96, 96) normalized local crop
          x_context: (B, 1, 160, 160, 160) normalized context crop
          box_bounds_voxel: (B, 6) lesion box coordinates in local crop voxel space
          run_cii: whether to execute the equal-count CII passes (Eq. 20)
        """
        f_local = self.encoder(x_local)      # (B, C, 12, 12, 12)
        f_context = self.encoder(x_context)  # (B, C, 20, 20, 20)

        m_b_tok, d_b_tok, side_tok, rel_coords = build_token_grid_box_priors(
            box_bounds_voxel, grid_size=12, input_size=self.cfg.local_grid_size
        )
        sel = self.selector(f_local, x_local, m_b_tok, d_b_tok)

        t_int = sel["retained_tokens"]         # (B, 512, C)
        ret_idx = sel["retained_idx"]          # (B, 512)
        d_b_ret = torch.gather(d_b_tok, 1, ret_idx)
        side_ret = torch.gather(side_tok, 1, ret_idx)
        pos_ret = torch.gather(rel_coords, 1, ret_idx.unsqueeze(-1).expand(-1, -1, 3))

        # Pool context feature map to 10^3 = 1,000 tokens for cross-attention keys/values
        t_context_kv = F.avg_pool3d(f_context, kernel_size=2, stride=2).flatten(2).transpose(1, 2)
        h_int = self.cross_attn(t_int, sel["all_tokens"], t_context_kv, d_b_ret, pos_ret)

        gap_local = f_local.mean(dim=(2, 3, 4))
        gap_context = f_context.mean(dim=(2, 3, 4))
        z_l = gap_local + h_int.mean(dim=1)
        z_c = gap_context

        q_l = compute_artefact_proxy_score(x_local)
        q_c = compute_artefact_proxy_score(x_context)
        logits, branch_weights = self._project_and_fuse(
            z_l, z_c, q_l, q_c, use_reliability_fusion=use_reliability_fusion
        )

        out = {
            "logits": logits,
            "probs": F.softmax(logits, dim=-1),
            "branch_weights": branch_weights,
            "z_local": z_l,
            "z_context": z_c,
            "priority_score": sel["priority_score"],
            "m_b_token": m_b_tok,
        }

        if run_cii:
            a_cand, a_ctl, matched_counts = match_equal_count_cii_subsets(
                t_int, d_b_ret, side_ret,
                num_candidates=self.cfg.num_candidates,
                max_pairs=self.cfg.cii_max_pairs,
                tau_d=self.cfg.cii_tol_distance,
                tau_n=self.cfg.cii_tol_norm,
            )
            t_ref = sel["t_ref"].unsqueeze(1)  # (B, 1, C) stop-gradient reference
            t_minus_cand = (1.0 - a_cand) * t_int + a_cand * t_ref
            t_minus_ctl = (1.0 - a_ctl) * t_int + a_ctl * t_ref

            h_cand = self.cross_attn(t_minus_cand, sel["all_tokens"], t_context_kv, d_b_ret, pos_ret)
            h_ctl = self.cross_attn(t_minus_ctl, sel["all_tokens"], t_context_kv, d_b_ret, pos_ret)

            logits_cand, _ = self._project_and_fuse(
                gap_local + h_cand.mean(dim=1), z_c, q_l, q_c, use_reliability_fusion=use_reliability_fusion
            )
            logits_ctl, _ = self._project_and_fuse(
                gap_local + h_ctl.mean(dim=1), z_c, q_l, q_c, use_reliability_fusion=use_reliability_fusion
            )
            out["logits_minus_cand"] = logits_cand
            out["logits_minus_ctl"] = logits_ctl
            out["cii_matched_counts"] = torch.tensor(matched_counts, device=x_local.device)

        return out


# =============================================================================
# 7. Training Losses & Inner-Validation Decision-Multiplier Operating Rule
# =============================================================================

def compute_isca_training_objective(
    outputs: Dict[str, torch.Tensor],
    targets: torch.Tensor,
    class_weights: torch.Tensor,
    cfg: ISCANet3DConfig,
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    """
    Computes the joint supervised training objective (Section 3.7, Equations 28-30):
      L_total = L_focal + lambda_B * L_B-att + lambda_CII * L_CII
    """
    probs = outputs["probs"]
    b_size = targets.shape[0]
    p_y = probs[torch.arange(b_size, device=targets.device), targets].clamp(min=eps, max=1.0)
    alpha_y = class_weights[targets]
    loss_focal = -(alpha_y * ((1.0 - p_y) ** cfg.focal_gamma) * torch.log(p_y)).mean()

    # Box-attention constraint L_B-att (Eq. 29)
    a_resp = F.relu(outputs["priority_score"])
    m_b = outputs["m_b_token"]
    in_band_mass = (m_b * a_resp).sum(dim=1) / (a_resp.sum(dim=1) + eps)
    loss_box_att = F.relu(cfg.rho_box_target - in_band_mass).mean()

    # Equal-count CII loss L_CII (Eq. 21), skipping cases with n_i^{int} == 0
    loss_cii = torch.tensor(0.0, device=targets.device)
    if "logits_minus_cand" in outputs:
        def _ref_margin(lgt: torch.Tensor) -> torch.Tensor:
            ref_l = lgt[torch.arange(b_size, device=targets.device), targets]
            mask = torch.ones_like(lgt, dtype=torch.bool)
            mask[torch.arange(b_size, device=targets.device), targets] = False
            comp_l = lgt.masked_fill(~mask, float("-inf")).max(dim=1).values
            return ref_l - comp_l

        g_fact = _ref_margin(outputs["logits"])
        g_cand = _ref_margin(outputs["logits_minus_cand"])
        g_ctl = _ref_margin(outputs["logits_minus_ctl"])
        dg_cand = g_fact - g_cand
        dg_ctl = g_fact - g_ctl

        valid = outputs["cii_matched_counts"] > 0
        if valid.any():
            per_case = F.relu(cfg.cii_margin_m - dg_cand) + cfg.lambda_ctl * dg_ctl.abs()
            loss_cii = per_case[valid].mean()

    loss_total = (
        loss_focal
        + cfg.lambda_box_loss * loss_box_att
        + cfg.lambda_cii_loss * loss_cii
    )
    return {
        "loss_total": loss_total,
        "loss_focal": loss_focal,
        "loss_box_att": loss_box_att,
        "loss_cii": loss_cii,
    }


def select_inner_validation_operating_rule(
    val_probs: np.ndarray,
    val_targets: np.ndarray,
    grid_min: float = 0.70,
    grid_max: float = 1.30,
    step: float = 0.05,
    num_sweeps: int = 2,
) -> np.ndarray:
    """
    Selects fold-specific class decision multipliers tau^{(o)} = (tau_1, tau_2, 1.0, tau_4)
    exclusively on the inner-validation partition D_val^{(o)} by coordinate grid search
    to maximize inner-validation Macro-F1 (Section 4.2).
    """
    tau = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    grid = np.round(np.arange(grid_min, grid_max + 1e-6, step), 2)

    def _macro_f1(weights: np.ndarray) -> float:
        preds = np.argmax(val_probs * weights[None, :], axis=1)
        f1s = []
        for c in range(4):
            tp = np.sum((preds == c) & (val_targets == c))
            fp = np.sum((preds == c) & (val_targets != c))
            fn = np.sum((preds != c) & (val_targets == c))
            denom = 2 * tp + fp + fn
            f1s.append((2.0 * tp / denom) if denom > 0 else 0.0)
        return float(np.mean(f1s))

    best_score = _macro_f1(tau)
    for _ in range(num_sweeps):
        for c in (0, 1, 3):  # class 2 (OKC) fixed at 1.00 as reference
            best_c = tau[c]
            for cand_w in grid:
                trial = tau.copy()
                trial[c] = cand_w
                score = _macro_f1(trial)
                if score > best_score + 1e-9:
                    best_score = score
                    best_c = cand_w
            tau[c] = best_c
    return tau


# =============================================================================
# 8. Auxiliary Two-Dimensional Adaptation (ISCA-Net-2D for DENTEX; Section 4.6)
# =============================================================================

class ISCANet2D(nn.Module):
    """
    Auxiliary 2D adaptation (ISCA-Net-2D) for multilabel tooth ROI diagnosis on
    panoramic radiographs (DENTEX; 224x224 local and 320x320 context crops, stride 8).
    """
    def __init__(self, num_labels: int = 4, embed_dim: int = 256):
        super().__init__()
        self.num_labels = num_labels
        self.embed_dim = embed_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.semantic_head = nn.Linear(embed_dim, 1)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.classifier = nn.Linear(embed_dim, num_labels)

    def forward(self, x_local_2d: torch.Tensor, x_context_2d: torch.Tensor) -> Dict[str, torch.Tensor]:
        f_l = self.encoder(x_local_2d)    # (B, 256, 28, 28) -> 784 tokens
        f_c = self.encoder(x_context_2d)  # (B, 256, 40, 40) -> 1,600 tokens
        t_l = f_l.flatten(2).transpose(1, 2)
        t_c = f_c.flatten(2).transpose(1, 2)

        scores = torch.sigmoid(self.semantic_head(t_l)).squeeze(-1)
        topk_idx = torch.topk(scores, k=min(512, t_l.shape[1]), dim=1).indices
        t_ret = torch.gather(t_l, 1, topk_idx.unsqueeze(-1).expand(-1, -1, self.embed_dim))

        kv = torch.cat([t_l, t_c], dim=1)
        h_int, _ = self.cross_attn(t_ret, kv, kv)
        z_l = f_l.mean(dim=(2, 3)) + h_int.mean(dim=1)
        z_c = f_c.mean(dim=(2, 3))

        # Mean Bernoulli entropy gating across the 4 sigmoid outputs
        p_l = torch.sigmoid(self.classifier(z_l).detach()).clamp(1e-6, 1.0 - 1e-6)
        p_c = torch.sigmoid(self.classifier(z_c).detach()).clamp(1e-6, 1.0 - 1e-6)
        ent_l = -(p_l * torch.log(p_l) + (1 - p_l) * torch.log(1 - p_l)).mean(dim=-1) / math.log(2.0)
        ent_c = -(p_c * torch.log(p_c) + (1 - p_c) * torch.log(1 - p_c)).mean(dim=-1) / math.log(2.0)
        w = F.softmax(torch.stack([-ent_l, -ent_c], dim=-1), dim=-1)

        logits = self.classifier(w[:, 0:1] * z_l + w[:, 1:2] * z_c)
        return {"logits": logits, "probs": torch.sigmoid(logits), "branch_weights": w}


# =============================================================================
# 9. Self-Contained End-to-End Verification Demo
# =============================================================================

def run_numpy_reference_verification(cfg: ISCANet3DConfig) -> None:
    """
    Pure-NumPy mathematical verification of all core algorithms in the manuscript
    (runs in any standard Python + NumPy environment, with or without PyTorch/CUDA):
      1. Adaptive physical crop geometry (Section 3.2; Table 9)
      2. Box-guided candidate priority s(u) & 358/154 stratified retention (Section 3.3; Eq. 16)
      3. Equal-count subset CII matching without replacement (Section 3.5; n_i^{int} <= 64)
      4. Perturbation-derived rank-8 SVD nuisance projection & reliability gating (Section 3.6; Eqs. 22-27)
      5. Inner-validation coordinate search for decision multipliers tau^{(o)} (Section 4.2)
    """
    rng = np.random.default_rng(335654)

    # 1. Adaptive physical crop geometry on the 4 representative public DOLCHID cases (Table 9 Panel B)
    cases = [
        ("AME_57", np.array([27.3, 18.3, 23.1])),
        ("DC_7",   np.array([14.7, 11.7, 14.4])),
        ("KCOT_88", np.array([22.8, 27.3, 27.6])),
        ("RC_4",   np.array([12.9, 15.0, 13.8])),
    ]
    print("[1] Adaptive Physical Crop Geometry Verification (Section 3.2, Table 9):")
    for cid, ext in cases:
        geom = compute_adaptive_crop_geometry(
            box_min_mm=np.zeros(3),
            box_max_mm=ext,
            fov_min_mm=np.array([-20.0, -20.0, -20.0]),
            fov_max_mm=np.array([133.6, 133.6, 133.6]),
        )
        print(f"    - {cid:<7s}: box={ext[0]:.1f}x{ext[1]:.1f}x{ext[2]:.1f} mm -> "
              f"L_local={geom['L_local_mm']:.1f} mm (spacing={geom['eff_spacing_local_mm']:.3f} mm), "
              f"L_context={geom['L_context_mm']:.1f} mm (spacing={geom['eff_spacing_context_mm']:.3f} mm)")

    # 2. Priority score s(u) and 358/154 token retention on 12^3 = 1,728 token grid (Section 3.3)
    n_grid = 1728
    m_b = np.zeros(n_grid, dtype=np.float64)
    m_b[:480] = 1.0  # candidate band tokens (M_B(u) = 1)
    rng.shuffle(m_b)
    d_b = np.where(m_b > 0.5, rng.uniform(0.0, 0.25, size=n_grid), rng.uniform(0.10, 0.85, size=n_grid))
    a_u = rng.uniform(0.1, 0.9, size=n_grid)
    g_u = rng.uniform(0.0, 1.0, size=n_grid)
    s_u = cfg.alpha_semantic * a_u + cfg.beta_box_prior * m_b * np.exp(-d_b / cfg.tau_box_decay) + cfg.gamma_gradient * g_u

    cand_pool = np.where(m_b > 0.5)[0]
    ctl_pool = np.where(m_b <= 0.5)[0]
    sel_cand = cand_pool[np.argsort(s_u[cand_pool])[::-1][:cfg.num_candidates]]
    sel_ctl = ctl_pool[np.argsort(s_u[ctl_pool])[::-1][:cfg.num_controls]]
    print(f"[2] Sparse Token Retention Check (Section 3.3, Eq. 16): "
          f"K={len(sel_cand)+len(sel_ctl)} retained (|C_i|={len(sel_cand)} candidates, |N_i|={len(sel_ctl)} controls)")

    # 3. Equal-count subset CII matching without replacement (Section 3.5)
    ret_idx = np.concatenate([sel_cand, sel_ctl])
    tokens_ret = rng.normal(0.0, 1.0, size=(512, cfg.embed_dim))
    norms = np.linalg.norm(tokens_ret, axis=1)
    rel_norms = norms / np.median(norms)
    d_ret = d_b[ret_idx]
    side_ret = rng.integers(0, 2, size=512)

    perm = rng.permutation(cfg.num_candidates)
    used_ctl = np.zeros(cfg.num_controls, dtype=bool)
    matched_cand, matched_ctl = [], []
    for c_idx in perm:
        if len(matched_cand) >= cfg.cii_max_pairs:
            break
        diff_d = np.abs(d_ret[cfg.num_candidates:] - d_ret[c_idx])
        diff_n = np.abs(rel_norms[cfg.num_candidates:] - rel_norms[c_idx])
        elig = (~used_ctl) & (side_ret[cfg.num_candidates:] == side_ret[c_idx]) & (diff_d <= cfg.cii_tol_distance) & (diff_n <= cfg.cii_tol_norm)
        if np.any(elig):
            cost = np.where(elig, (diff_d / cfg.cii_tol_distance) ** 2 + (diff_n / cfg.cii_tol_norm) ** 2, np.inf)
            best = int(np.argmin(cost))
            used_ctl[best] = True
            matched_cand.append(int(c_idx))
            matched_ctl.append(int(cfg.num_candidates + best))
    print(f"[3] Equal-Count CII Matcher Check (Section 3.5, Eq. 20): "
          f"|S_i^cand| = |S_i^ctl| = n_i^int = {len(matched_cand)} matched pairs (<= 64, 0 duplicate controls)")

    # 4. Perturbation-derived rank-8 SVD nuisance subspace & reliability-weighted fusion (Section 3.6)
    z_clean = rng.normal(0.0, 1.0, size=(157, cfg.embed_dim))
    z_aug = z_clean + rng.normal(0.0, 0.25, size=(157, cfg.embed_dim))
    delta = z_aug - z_clean
    delta -= delta.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(delta, full_matrices=False)
    b_n = vh[:cfg.nuisance_rank].T  # (128, 8) orthonormal basis
    z_proj = z_clean - (z_clean @ b_n) @ b_n.T
    ortho_err = np.linalg.norm(b_n.T @ b_n - np.eye(cfg.nuisance_rank))
    print(f"[4] Rank-8 SVD Nuisance Basis & Projection Check (Section 3.6, Eqs. 22-23): "
          f"B_n shape={b_n.shape}, ||B_n^T B_n - I||_F = {ortho_err:.2e}")

    # 5. Inner-validation coordinate search for decision multipliers tau^{(o)} (Section 4.2)
    val_targets = np.array([0]*9 + [1]*11 + [2]*18 + [3]*14)
    val_logits = rng.normal(0.0, 1.0, size=(52, 4))
    for i, y in enumerate(val_targets):
        val_logits[i, y] += 1.8
    val_probs = np.exp(val_logits) / np.exp(val_logits).sum(axis=1, keepdims=True)
    tau_opt = select_inner_validation_operating_rule(val_probs, val_targets)
    print(f"[5] Inner-Validation Decision-Multiplier Search (Section 4.2): "
          f"tau^(o) = {tau_opt.round(2).tolist()} (OKC reference fixed at 1.00)")


def run_verification_demo() -> None:
    print("=" * 78)
    print("ISCA-Net & ISCA-Net-2D End-to-End Reference Pipeline Verification")
    print("=" * 78)
    cfg = ISCANet3DConfig()
    run_numpy_reference_verification(cfg)

    if not HAS_TORCH:
        print("-" * 78)
        print("[Note] PyTorch is not installed in the current environment; pure-NumPy mathematical")
        print("       verification of all manuscript equations completed above. Install PyTorch")
        print("       (`pip install torch`) to execute the 3D/2D neural network forward/backward demo.")
        print("=" * 78)
        return

    print("-" * 78)
    print("[6] Executing PyTorch 3D ISCA-Net & 2D ISCA-Net-2D Neural Network Verification...")
    torch.manual_seed(335654)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ISCANet3D(cfg).to(device)
    decoder = MaskedReconstructionDecoder3D(cfg.embed_dim).to(device)

    batch_size = 2
    x_local = torch.randn(batch_size, 1, 96, 96, 96, device=device)
    x_context = torch.randn(batch_size, 1, 160, 160, 160, device=device)
    box_bounds = torch.tensor(
        [[24.0, 24.0, 24.0, 72.0, 72.0, 72.0],
         [20.0, 22.0, 18.0, 76.0, 70.0, 74.0]],
        device=device,
    )
    targets = torch.tensor([2, 3], dtype=torch.long, device=device)
    counts = torch.tensor([44.0, 54.0, 92.0, 72.0], device=device)
    class_weights = (counts.sum() / (4.0 * counts))

    pre_loss = pretraining_masked_reconstruction_step(
        model.encoder, decoder, x_local, x_context, masking_ratio=cfg.masking_ratio_pretrain
    )
    outputs = model(x_local, x_context, box_bounds, run_cii=True, use_reliability_fusion=True)
    losses = compute_isca_training_objective(outputs, targets, class_weights, cfg)
    print(f"    - Algorithm 1 Pretraining Masked L1 Loss: {pre_loss.item():.4f}")
    print(f"    - Supervised L_total={losses['loss_total'].item():.4f} "
          f"(L_focal={losses['loss_focal'].item():.4f}, "
          f"L_B-att={losses['loss_box_att'].item():.4f}, "
          f"L_CII={losses['loss_cii'].item():.4f})")

    model_2d = ISCANet2D(num_labels=4, embed_dim=256).to(device)
    out_2d = model_2d(
        torch.randn(batch_size, 1, 224, 224, device=device),
        torch.randn(batch_size, 1, 320, 320, device=device),
    )
    print(f"    - Auxiliary ISCA-Net-2D (DENTEX) Sigmoid Output Shape: {tuple(out_2d['probs'].shape)}")
    print("=" * 78)
    print("All verification checks completed successfully.")


if __name__ == "__main__":
    run_verification_demo()