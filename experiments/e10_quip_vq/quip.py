"""
QuIP# (QuIP-Sharp) implementation for Parameter Golf.

2-bit vector quantization using:
- Randomized Hadamard incoherence processing
- E8 lattice codebook (256 base points, 65536 codewords)
- LDLQ: Hessian-aware greedy vector quantization

Reference: arXiv 2402.04396 (Cornell-RelaxML/quip-sharp)
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor


# =============================================================================
# E8 Lattice Codebook
# =============================================================================
#
# E8 = D8_hat union (D8_hat + 1/2) where D8_hat = {x in Z^8+1/2 : sum(x) even}
# We use 256 absolute-value base points with norm^2 <= 10 (227 from D8_hat)
# plus 29 padding points of norm^2 = 12.
# Each 16-bit index encodes: [8-bit abs_idx | 7 sign bits | 1 parity bit]
# Total: 256 * 2^7 = 32768 per coset, 65536 total codewords.


def _build_d8hat_abs_grid() -> Tensor:
    """Build the 227 unique absolute-value patterns from D8_hat with norm^2 <= 10."""
    intr = torch.arange(-4, 4, dtype=torch.float32)
    # Generate all points in Z^8 + 0.5
    grids = [intr + 0.5] * 8
    candidates = torch.cartesian_prod(*grids)
    # D8_hat parity constraint: sum must be even
    parity_ok = (candidates.sum(dim=-1) % 2 == 0)
    # Norm constraint
    norm_ok = (candidates.norm(dim=-1) ** 2 <= 10)
    valid = candidates[parity_ok & norm_ok]
    # Take absolute values and unique
    abs_patterns = torch.unique(valid.abs(), dim=0)
    return abs_patterns


def _build_norm12_points() -> Tensor:
    """Build the 29 padding points with norm^2 = 12.
    These have 5 coordinates = 3/2 and 3 coordinates = 1/2."""
    from itertools import combinations

    points = []
    base = torch.full((8,), 0.5)
    for combo in combinations(range(8), 5):
        p = base.clone()
        for i in combo:
            p[i] = 1.5
        if p.sum() % 2 == 0:
            points.append(p)
    # Also try with some negative patterns to get more unique abs patterns
    result = torch.stack(points) if points else torch.zeros(0, 8)
    result = torch.unique(result.abs(), dim=0)
    return result[:29]  # Take up to 29 to pad to 256


def build_e8p_codebook(device: torch.device = torch.device("cpu")) -> Tensor:
    """Build the full E8P codebook: 256 base points (absolute values).

    Returns: Tensor of shape (256, 8) — the absolute-value base grid.
    """
    d8_abs = _build_d8hat_abs_grid()
    norm12 = _build_norm12_points()

    grid = torch.cat([d8_abs, norm12], dim=0)
    # Ensure exactly 256 points
    if grid.shape[0] < 256:
        # Pad with zeros (will never be selected as nearest neighbor for real data)
        padding = torch.zeros(256 - grid.shape[0], 8)
        grid = torch.cat([grid, padding], dim=0)
    grid = grid[:256]

    return grid.to(device)


def expand_codebook_with_signs(abs_grid: Tensor) -> Tensor:
    """Expand 256 absolute-value points to full signed codebook.

    For each base point, generate all 2^7 = 128 sign patterns
    (8th sign determined by even-parity constraint).
    Then apply ±0.25 coset shift.

    Returns: Tensor of shape (65536, 8) — full codebook.
    """
    device = abs_grid.device
    n_base = abs_grid.shape[0]  # 256
    # Generate 128 sign patterns (7 free bits, 8th determined by parity)
    sign_bits = torch.arange(128, device=device)
    signs = torch.zeros(128, 8, device=device)
    for bit in range(7):
        signs[:, bit] = 1 - 2 * ((sign_bits >> bit) & 1).float()
    # 8th sign: ensure even number of negative signs (even parity)
    neg_count = (signs[:, :7] < 0).sum(dim=-1)
    signs[:, 7] = torch.where(neg_count % 2 == 0, torch.ones(1, device=device), -torch.ones(1, device=device))

    # Expand: (256, 1, 8) * (1, 128, 8) = (256, 128, 8)
    signed_points = abs_grid.unsqueeze(1) * signs.unsqueeze(0)
    # Two cosets: +0.25 and -0.25
    coset_plus = signed_points + 0.25  # (256, 128, 8)
    coset_minus = signed_points - 0.25  # (256, 128, 8)

    full = torch.cat([
        coset_plus.reshape(-1, 8),   # 32768 points
        coset_minus.reshape(-1, 8),  # 32768 points
    ], dim=0)
    return full  # (65536, 8)


# =============================================================================
# Fast Walsh-Hadamard Transform
# =============================================================================


def fast_walsh_hadamard(x: Tensor) -> Tensor:
    """In-place Fast Walsh-Hadamard Transform.
    x shape: (..., n) where n is a power of 2.
    Returns: transformed x, normalized by 1/sqrt(n).
    """
    n = x.shape[-1]
    assert n > 0 and (n & (n - 1)) == 0, f"n must be power of 2, got {n}"

    h = 1
    while h < n:
        # Reshape to (..., n/(2h), 2, h)
        x = x.view(*x.shape[:-1], -1, 2, h)
        a = x[..., 0, :]
        b = x[..., 1, :]
        x = torch.stack([a + b, a - b], dim=-2)
        x = x.view(*x.shape[:-3], -1)
        h *= 2

    return x / math.sqrt(n)


# =============================================================================
# Incoherence Processing
# =============================================================================


class IncoherenceState(NamedTuple):
    """State needed to undo incoherence transform at inference."""
    SU: Tensor  # random signs for columns, shape (n,)
    SV: Tensor  # random signs for rows, shape (m,)
    scale: float  # weight RMS scale


def apply_incoherence(W: Tensor, seed: int = 0) -> tuple[Tensor, IncoherenceState]:
    """Apply randomized Hadamard incoherence to weight matrix W (m x n).

    Returns:
        W_transformed: incoherent weight matrix
        state: IncoherenceState for inference-time reversal
    """
    m, n = W.shape
    gen = torch.Generator(device=W.device).manual_seed(seed)

    # Random sign vectors
    SU = (torch.randint(0, 2, (n,), generator=gen, device=W.device) * 2 - 1).float()
    SV = (torch.randint(0, 2, (m,), generator=gen, device=W.device) * 2 - 1).float()

    # W' = Had(SV * W * SU)
    # Step 1: element-wise multiply by signs
    W_signed = W * SV[:, None] * SU[None, :]

    # Step 2: Hadamard along rows (output dimension)
    # Pad m to next power of 2 if needed
    m_pad = 1 << (m - 1).bit_length() if m & (m - 1) else m
    n_pad = 1 << (n - 1).bit_length() if n & (n - 1) else n

    W_padded = W_signed
    if m_pad != m or n_pad != n:
        W_padded = F.pad(W_signed, (0, n_pad - n, 0, m_pad - m))

    # Apply Hadamard along rows (dim 0) and columns (dim 1)
    W_had = fast_walsh_hadamard(W_padded.T).T  # columns
    W_had = fast_walsh_hadamard(W_had)  # rows

    # Trim back
    W_had = W_had[:m, :n]

    # Compute scale
    scale = float(W_had.float().pow(2).mean().sqrt().item())
    opt_scale = 1.03  # E8P optimal scale
    scale = scale / opt_scale if scale > 0 else 1.0

    W_normalized = W_had / scale

    return W_normalized, IncoherenceState(SU=SU.half(), SV=SV.half(), scale=scale)


# =============================================================================
# Hessian Collection
# =============================================================================


@torch.no_grad()
def collect_hessian(
    model: torch.nn.Module,
    dataloader,
    n_samples: int = 128,
    device: torch.device = torch.device("cpu"),
) -> dict[str, Tensor]:
    """Collect per-layer Hessian H = E[x^T x] for each linear layer.

    Returns dict mapping layer name to Hessian tensor.
    """
    hessians: dict[str, Tensor] = {}
    counts: dict[str, int] = {}
    hooks = []

    def make_hook(name: str):
        def hook_fn(module, input, output):
            x = input[0].detach().float()
            if x.ndim == 3:
                x = x.reshape(-1, x.shape[-1])  # (batch*seq, dim)
            n = x.shape[-1]
            if name not in hessians:
                hessians[name] = torch.zeros(n, n, device=device, dtype=torch.float64)
                counts[name] = 0
            hessians[name] += x.T.double() @ x.double()
            counts[name] += x.shape[0]
        return hook_fn

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            hooks.append(module.register_forward_hook(make_hook(name)))

    model.eval()
    n_collected = 0
    for batch in dataloader:
        if n_collected >= n_samples:
            break
        if isinstance(batch, (tuple, list)):
            x, y = batch[0].to(device), batch[1].to(device)
        else:
            x = batch.to(device)
            y = x  # dummy
        with torch.no_grad():
            model(x, y)
        n_collected += 1

    for h in hooks:
        h.remove()

    # Normalize
    for name in hessians:
        if counts[name] > 0:
            hessians[name] /= counts[name]

    return hessians


# =============================================================================
# LDLQ: Hessian-Aware Vector Quantization
# =============================================================================


def quantize_weight_ldlq(
    W: Tensor,
    H: Tensor,
    codebook: Tensor,
    codebook_norms: Tensor,
    block_size: int = 8,
    tune_iters: int = 1,
) -> tuple[Tensor, Tensor]:
    """Quantize weight matrix W using LDLQ with E8 codebook.

    Args:
        W: weight matrix (m, n), already incoherence-processed and scaled
        H: Hessian matrix (n, n)
        codebook: full codebook (65536, 8) or abs codebook (256, 8)
        codebook_norms: precomputed ||c||^2 for each codeword
        block_size: must be 8 for E8
        tune_iters: number of refinement passes

    Returns:
        W_hat: quantized weight matrix (m, n)
        indices: codebook indices (m, n // block_size) as int32
    """
    m, n = W.shape
    assert n % block_size == 0, f"n={n} must be divisible by block_size={block_size}"
    n_blocks = n // block_size

    W_hat = torch.zeros_like(W)
    indices = torch.zeros(m, n_blocks, dtype=torch.int32, device=W.device)

    # Simple greedy quantization (without full LDL for simplicity):
    # Process blocks in reverse order, propagating error via Hessian
    # Simplified: just find nearest codebook entry per block

    # Reshape W into blocks: (m, n_blocks, 8)
    W_blocks = W.reshape(m, n_blocks, block_size)

    # For each block, find nearest codebook entry
    # codebook shape: (C, 8), W_blocks shape: (m, n_blocks, 8)
    # Distance: ||w - c||^2 = ||w||^2 - 2*w@c^T + ||c||^2
    for b in range(n_blocks):
        w = W_blocks[:, b, :].float()  # (m, 8)
        # Brute-force nearest neighbor
        # scores = 2 * w @ codebook.T - codebook_norms  (maximize)
        scores = 2.0 * (w @ codebook.T) - codebook_norms.unsqueeze(0)
        best_idx = scores.argmax(dim=-1)  # (m,)
        indices[:, b] = best_idx.int()
        W_hat[:, b * block_size:(b + 1) * block_size] = codebook[best_idx]

    # Refinement passes with Hessian feedback
    for _ in range(tune_iters):
        residual = W - W_hat  # (m, n)
        for b in range(n_blocks):
            sl = slice(b * block_size, (b + 1) * block_size)
            # Add Hessian-weighted residual feedback
            if H is not None:
                H_block = H[sl, sl]  # (8, 8)
                H_cross = H[:, sl]  # (n, 8)
                correction = residual @ H_cross  # (m, 8)
                # Normalize by block Hessian diagonal
                diag_h = H_block.diag().clamp(min=1e-8)
                correction = correction / diag_h.unsqueeze(0)
                target = W_hat[:, sl] + correction
            else:
                target = W[:, b * block_size:(b + 1) * block_size]

            # Re-quantize
            scores = 2.0 * (target.float() @ codebook.T) - codebook_norms.unsqueeze(0)
            best_idx = scores.argmax(dim=-1)
            indices[:, b] = best_idx.int()
            W_hat[:, sl] = codebook[best_idx]
            residual = W - W_hat

    return W_hat, indices


# =============================================================================
# Pack / Unpack indices
# =============================================================================


def pack_indices(indices: Tensor) -> Tensor:
    """Pack 16-bit codebook indices into bytes.
    indices shape: (m, n_blocks) of int32 in [0, 65535]
    Returns: packed bytes tensor.
    """
    return indices.to(torch.int16).contiguous()


def unpack_indices(packed: Tensor) -> Tensor:
    """Unpack int16 indices back to int32."""
    return packed.to(torch.int32)


# =============================================================================
# Full Quantization Pipeline
# =============================================================================


class QuantizedLayer(NamedTuple):
    """All data needed to reconstruct a quantized linear layer."""
    indices: Tensor  # (m, n // 8) int16
    incoherence: IncoherenceState  # SU, SV, scale
    shape: tuple[int, int]  # original (m, n)


def quantize_linear_layer(
    W: Tensor,
    H: Tensor | None = None,
    seed: int = 0,
    tune_iters: int = 1,
    device: torch.device = torch.device("cpu"),
) -> QuantizedLayer:
    """Full QuIP# quantization pipeline for one linear layer.

    Args:
        W: weight matrix (out_features, in_features)
        H: Hessian matrix (in_features, in_features), or None for simple quantization
        seed: random seed for incoherence
        tune_iters: refinement iterations
    """
    m, n = W.shape
    # Pad n to multiple of 8 if needed
    n_pad = ((n + 7) // 8) * 8
    if n_pad != n:
        W = F.pad(W, (0, n_pad - n))
        if H is not None:
            H = F.pad(H, (0, n_pad - n, 0, n_pad - n))

    # Step 1: Incoherence processing
    W_inc, inc_state = apply_incoherence(W.float(), seed=seed)

    # Step 2: Build codebook
    abs_grid = build_e8p_codebook(device=device)
    full_codebook = expand_codebook_with_signs(abs_grid)
    codebook_norms = (full_codebook ** 2).sum(dim=-1)

    # Step 3: LDLQ quantization
    W_hat, indices = quantize_weight_ldlq(
        W_inc, H, full_codebook, codebook_norms,
        block_size=8, tune_iters=tune_iters,
    )

    return QuantizedLayer(
        indices=pack_indices(indices),
        incoherence=inc_state,
        shape=(m, n),
    )


def dequantize_linear_layer(
    qlayer: QuantizedLayer,
    device: torch.device = torch.device("cpu"),
) -> Tensor:
    """Reconstruct weight matrix from quantized representation."""
    m, n = qlayer.shape
    n_pad = ((n + 7) // 8) * 8

    # Rebuild codebook
    abs_grid = build_e8p_codebook(device=device)
    full_codebook = expand_codebook_with_signs(abs_grid)

    # Unpack indices and lookup
    indices = unpack_indices(qlayer.indices).long()  # (m, n_pad // 8)
    W_q = full_codebook[indices.reshape(-1)].reshape(m, n_pad)

    # Undo incoherence: scale, then inverse Hadamard, then undo signs
    W_q = W_q * qlayer.incoherence.scale

    # Inverse Hadamard (Hadamard is its own inverse up to normalization)
    m_pad = 1 << (m - 1).bit_length() if m & (m - 1) else m
    if m_pad != m or n_pad != n_pad:
        W_q = F.pad(W_q, (0, max(0, n_pad - W_q.shape[1]), 0, max(0, m_pad - m)))

    W_q = fast_walsh_hadamard(W_q)  # rows
    W_q = fast_walsh_hadamard(W_q.T).T  # columns

    W_q = W_q[:m, :n]

    # Undo signs
    SU = qlayer.incoherence.SU.float().to(device)
    SV = qlayer.incoherence.SV.float().to(device)
    W_q = W_q * SV[:, None] * SU[None, :]

    return W_q


# =============================================================================
# Utility: compute storage size
# =============================================================================


def compute_storage_bytes(qlayer: QuantizedLayer) -> int:
    """Compute the storage cost in bytes for a quantized layer."""
    idx_bytes = qlayer.indices.numel() * 2  # int16
    su_bytes = qlayer.incoherence.SU.numel() * 2  # fp16
    sv_bytes = qlayer.incoherence.SV.numel() * 2  # fp16
    scale_bytes = 4  # float32
    return idx_bytes + su_bytes + sv_bytes + scale_bytes
