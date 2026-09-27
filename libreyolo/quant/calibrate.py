"""Histogram-based activation amax selection (mse / entropy calibration).

Both algorithms consume the same fixed-width histogram of absolute
activation values accumulated across calibration batches and return a
clipping threshold (amax):

- ``mse``: sweep candidate thresholds, simulate quantize/dequantize of every
  bin center at the activation format's resolution, and pick the threshold
  minimizing the histogram-weighted squared reconstruction error.
- ``entropy``: for each candidate threshold, compare the clipped reference
  distribution (out-of-range mass saturated into the top bin) against the
  quantized-then-expanded distribution and pick the threshold minimizing
  their KL divergence.

Everything runs on detached CPU tensors in double precision and the sweeps
are exhaustive over a fixed candidate grid, so results are deterministic.
"""

import torch

from .fake_quant import E4M3_MAX

HIST_BINS = 2048
# Candidate thresholds are histogram bin edges. The coarse pass visits every
# _SWEEP_STEP-th edge; mse then refines every edge around the coarse winner.
_SWEEP_STEP = 16
# Positive-side code counts used to simulate the deployed int8 activation
# format, which is 256-level affine over the intersected [lo, hi] window
# (see finalize_observation). Two-sided distributions get the standard
# symmetric 128-level stand-in for [-amax, amax]; one-sided (post-ReLU)
# distributions keep the full 256 codes on a single side, exactly matching
# what fake_quant_int8_affine does with a [0, amax] or [-amax, 0] window.
_INT8_LEVELS = 128
_INT8_LEVELS_ONE_SIDED = 256
_EPS = 1e-12

# histc chunk size: single-precision counters lose integer exactness past
# 2**24, so large activations are histogrammed in chunks and accumulated in
# float64 (also keeps CUDA atomic-add histograms bitwise reproducible).
_HISTC_CHUNK = 1 << 22


def accumulate_abs_histogram(hist, hist_amax: float, x: torch.Tensor, top: float = 0.0):
    """Fold one batch of activations into the running |x| histogram.

    Returns the updated ``(hist, hist_amax, top)`` triple; ``hist`` is a
    float64 CPU tensor of ``HIST_BINS`` counts over ``[0, hist_amax]`` and
    ``top`` is the count of samples sitting exactly on the current top edge.
    When a batch exceeds the current range, the range is doubled
    (pair-merging existing bins) until it fits, so earlier counts land
    exactly on the new grid and no second pass over the data is needed.

    The accumulated histogram is bit-identical to a direct ``torch.histc``
    of the concatenated data on the final grid, for any batch split:

    - ``torch.histc`` bins are left-inclusive except the last, which also
      includes the top edge. Under a doubling, the old top edge becomes an
      interior boundary of the new grid, so samples exactly on it move from
      the merged lower bin to the bin the new grid starts there; ``top``
      tracks that count so the merge can relocate it.
    - All-zero batches carry no range information but their mass still
      belongs in bin 0 on every grid, so it is recorded even before any
      nonzero sample has established a range (``hist_amax`` stays 0.0 until
      one does).
    """
    flat = x.detach().reshape(-1).float().abs()
    if not flat.numel():
        return hist, hist_amax, top
    amax = float(flat.max())
    if hist is None:
        hist = torch.zeros(HIST_BINS, dtype=torch.float64)
        hist_amax = 0.0
    if hist_amax == 0.0:
        # Only zeros so far (all of them in bin 0): any range works, so the
        # first nonzero batch establishes it directly.
        hist_amax = amax
    while hist_amax < amax:
        merged = hist.reshape(-1, 2).sum(dim=1)
        hist = torch.cat([merged, torch.zeros_like(merged)])
        # Samples exactly on the old top edge belong to the new grid's
        # left-inclusive bin starting there, not the merged lower bin.
        hist[HIST_BINS // 2 - 1] -= top
        hist[HIST_BINS // 2] += top
        top = 0.0
        hist_amax *= 2.0
    if hist_amax == 0.0:
        # This batch is all zeros too; zeros land in bin 0 on every grid.
        hist[0] += float(flat.numel())
        return hist, hist_amax, top
    for chunk in flat.split(_HISTC_CHUNK):
        hist += torch.histc(chunk, bins=HIST_BINS, min=0.0, max=hist_amax).double().cpu()
    top += float((flat == hist_amax).sum())
    return hist, hist_amax, top


def _int8_qmax(one_sided: bool) -> int:
    return (_INT8_LEVELS_ONE_SIDED if one_sided else _INT8_LEVELS) - 1


def _candidate_indices(start: int) -> torch.Tensor:
    idx = list(range(start, HIST_BINS, _SWEEP_STEP))
    idx.append(HIST_BINS)  # the full observed range (minmax) is always a candidate
    return torch.tensor(idx, dtype=torch.long)


def _reconstruct(
    centers: torch.Tensor, amaxes: torch.Tensor, aformat: str, one_sided: bool
) -> torch.Tensor:
    """Dequantized value of each bin center under each candidate amax.

    Returns [candidates, bins]. Saturation is part of the codebook clamp, so
    clipping error and rounding error come out of the same formula. The int8
    branch mirrors ``fake_quant_int8_affine`` on the window the calibrated
    module will actually use: round to a uniform grid of ``qmax`` steps over
    [0, amax], clamp at ``qmax``.
    """
    if aformat == "fp8":
        scale = (amaxes / E4M3_MAX).clamp_min(_EPS).unsqueeze(1)
        scaled = (centers.unsqueeze(0) / scale).clamp(max=E4M3_MAX).float()
        return scaled.to(torch.float8_e4m3fn).double() * scale
    qmax = _int8_qmax(one_sided)
    scale = (amaxes / qmax).clamp_min(_EPS).unsqueeze(1)
    codes = torch.round(centers.unsqueeze(0) / scale).clamp(max=qmax)
    return codes * scale


def mse_amax(
    hist: torch.Tensor,
    hist_amax: float,
    aformat: str = "int8",
    one_sided: bool = False,
) -> float:
    """Clipping threshold minimizing histogram-weighted reconstruction MSE.

    ``one_sided`` marks distributions whose observed signed range never
    crosses zero (post-ReLU activations): the deployed affine window then
    spends all 256 int8 codes on one side, doubling the resolution the
    sweep must simulate. Ignored for fp8, whose codebook is symmetric.
    """
    hist = hist.double()
    if float(hist.sum()) == 0.0:
        return hist_amax
    bin_width = hist_amax / HIST_BINS
    centers = (torch.arange(HIST_BINS, dtype=torch.float64) + 0.5) * bin_width

    def sweep(indices: torch.Tensor) -> int:
        amaxes = indices.double() * bin_width
        recon = _reconstruct(centers, amaxes, aformat, one_sided)
        losses = ((centers.unsqueeze(0) - recon) ** 2 * hist.unsqueeze(0)).sum(dim=1)
        return int(indices[int(torch.argmin(losses))])

    best = sweep(_candidate_indices(start=_SWEEP_STEP))
    fine = torch.arange(
        max(1, best - _SWEEP_STEP + 1), min(HIST_BINS, best + _SWEEP_STEP - 1) + 1
    )
    return sweep(fine) * bin_width


def _quantized_codes(i: int, bin_width: float, aformat: str, levels: int) -> torch.Tensor:
    """Group id of each of the first ``i`` bins under a clip at edge ``i``."""
    if aformat == "fp8":
        centers = (torch.arange(i, dtype=torch.float64) + 0.5) * bin_width
        scale = max(i * bin_width / E4M3_MAX, _EPS)
        snapped = (centers / scale).clamp(max=E4M3_MAX).float().to(torch.float8_e4m3fn)
        _, codes = torch.unique(snapped.float(), sorted=True, return_inverse=True)
        return codes
    # Uniform merge of i bins into the int8 level groups.
    return torch.arange(i) * levels // i


def entropy_amax(
    hist: torch.Tensor,
    hist_amax: float,
    aformat: str = "int8",
    one_sided: bool = False,
) -> float:
    """Clipping threshold minimizing KL(reference || quantized) divergence.

    The reference keeps the clipped tail (saturated into its top bin); the
    quantized distribution is built from the in-range mass only, grouped by
    quantization code and spread back uniformly over that code's occupied
    bins. The asymmetry is what makes aggressive clipping cost divergence.
    ``one_sided`` has the same meaning as in :func:`mse_amax`: the deployed
    one-sided int8 window carries 256 codes, so the grouping and the sweep
    floor use 256 levels instead of the symmetric 128-level stand-in.
    """
    hist = hist.double()
    if float(hist.sum()) == 0.0:
        return hist_amax
    bin_width = hist_amax / HIST_BINS
    levels = _int8_qmax(one_sided) + 1 if aformat != "fp8" else _INT8_LEVELS
    best_i, best_kl = HIST_BINS, float("inf")
    for i in _candidate_indices(start=levels).tolist():
        inside = hist[:i]
        if float(inside.sum()) == 0.0:
            continue
        reference = inside.clone()
        reference[-1] += hist[i:].sum()
        reference = reference / reference.sum()

        codes = _quantized_codes(i, bin_width, aformat, levels)
        ncodes = int(codes[-1]) + 1
        grouped = torch.zeros(ncodes, dtype=torch.float64).scatter_add_(0, codes, inside)
        occupied = torch.zeros(ncodes, dtype=torch.float64).scatter_add_(
            0, codes, (inside > 0).double()
        )
        mask = inside > 0
        quantized = torch.zeros(i, dtype=torch.float64)
        quantized[mask] = (grouped / occupied.clamp_min(1.0))[codes][mask]
        quantized = quantized / quantized.sum().clamp_min(_EPS)

        # KL over the reference's support: the saturated tail may sit in a
        # bin the in-range histogram left empty, and the eps floor on the
        # quantized side is what prices that clipped mass.
        support = reference > 0
        p = reference[support]
        q = quantized[support].clamp_min(_EPS)
        kl = float((p * torch.log(p / q)).sum())
        if kl < best_kl:
            best_kl, best_i = kl, i
    return best_i * bin_width
