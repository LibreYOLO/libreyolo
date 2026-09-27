"""Quantized module variants.

Each class subclasses its float counterpart so the module keeps the same
state_dict key layout (``weight`` / ``bias`` stay where checkpoints expect
them) and remains a drop-in inside existing architectures, trainers, and
optimizers. Quantization state lives in extra ``_q_*`` buffers so it
round-trips through checkpoints.

Weights are fake-quantized from the fp32 master copy on every forward, which
keeps the modules correct for QAT (gradients flow to the masters through the
straight-through estimator). Computation runs in an fp32 island even under
autocast so the simulated arithmetic is exactly the declared scheme.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import kernels as K
from .fake_quant import (
    E4M3_MAX,
    autocast_off,
    fp8_weight_qparams,
    int8_weight_qparams,
    int_group_qparams,
)
from .packing import (
    pack_fp8_weight,
    pack_int8_weight,
    pack_int_grouped_weight,
    pack_mxfp4_weight,
    pack_nvfp4_weight,
    unpack_fp8_weight,
    unpack_int8_weight,
    unpack_int_grouped_weight,
    unpack_mxfp4_weight,
    unpack_nvfp4_weight,
)


class _ActObserverMixin:
    """Shared INT8 activation observer / fake-quant state."""

    def _init_act_state(self, out_channels: int):
        # Buffers must live on the weight's device from the start: modules are
        # built with device=weight.device by from_float, and a CPU-born buffer
        # would survive until the post-quantize .to() call, crashing
        # multi-batch calibration on GPU.
        device = self.weight.device
        self.register_buffer("_q_act_lo", torch.zeros(1, device=device))
        self.register_buffer("_q_act_hi", torch.zeros(1, device=device))
        self.register_buffer(
            "_q_calibrated", torch.zeros(1, dtype=torch.uint8, device=device)
        )
        self.register_buffer("_q_w_scale", torch.zeros(out_channels, device=device))
        self._q_observing = False
        self._q_export_mode = False

    def freeze_weight_qparams(self):
        """Bake the current per-channel weight scales into a buffer so ONNX
        tracing emits them as constants (export mode)."""
        with torch.no_grad():
            if getattr(self, "_q_wformat", "int8") == "fp8":
                self._q_w_scale.copy_(self._fp8_weight_qparams())
            else:
                self._q_w_scale.copy_(int8_weight_qparams(self.weight.float()))

    def _fp8_weight_qparams(self) -> torch.Tensor:
        """Return the configured FP8 weight scales in the stable row layout."""
        weight = self.weight.float()
        if getattr(self, "_q_fp8_weight_scaling", "per_channel") == "tensorwise":
            scale = (weight.detach().abs().amax() / E4M3_MAX).clamp_min(1e-12)
            return scale.reshape(1).expand(self._q_w_scale.shape)
        return fp8_weight_qparams(weight)

    def _weight_scale(self):
        return self._q_w_scale if self._q_export_mode else None

    @property
    def is_finalized(self) -> bool:
        return "weight_packed" in self._buffers

    def make_finalized(self):
        """Crystallize: replace the fp32 master weight with packed low bits.

        Unpacking reproduces the simulation's weight values bit for bit, so
        finalized inference matches prepared inference exactly.
        """
        if self.is_finalized:
            return
        self.freeze_weight_qparams()
        with torch.no_grad():
            if getattr(self, "_q_wformat", "int8") == "fp8":
                packed = pack_fp8_weight(self.weight.detach(), self._q_w_scale)
            else:
                packed = pack_int8_weight(self.weight.detach(), self._q_w_scale)
        del self._parameters["weight"]
        self.register_buffer("weight_packed", packed)
        self._q_export_mode = False

    def make_prepared(self):
        """Reconstruct fp32 masters from packed weights (QAT-from-PTQ)."""
        if not self.is_finalized:
            return
        self.__dict__.pop("_q_unpacked_cache", None)
        self.__dict__.pop("_q_calibrated_host", None)
        with torch.no_grad():
            if getattr(self, "_q_wformat", "int8") == "fp8":
                weight = unpack_fp8_weight(self.weight_packed, self._q_w_scale)
            else:
                weight = unpack_int8_weight(self.weight_packed, self._q_w_scale)
        del self._buffers["weight_packed"]
        self.weight = nn.Parameter(weight)

    def _effective_weight(self) -> torch.Tensor:
        fp8 = getattr(self, "_q_wformat", "int8") == "fp8"
        if self.is_finalized:
            # Unpacking is deterministic while finalized; on CUDA, cache the
            # dequantized tensor so repeated forwards stop paying a full
            # weight-sized unpack per call (dropped by make_prepared, and on
            # any device move).
            cached = self.__dict__.get("_q_unpacked_cache")
            if cached is not None and cached.device == self.weight_packed.device:
                return cached
            if fp8:
                weight = unpack_fp8_weight(self.weight_packed, self._q_w_scale)
            else:
                weight = unpack_int8_weight(self.weight_packed, self._q_w_scale)
            if weight.is_cuda:
                self._q_unpacked_cache = weight
            return weight
        if fp8:
            return K.fake_quant_fp8(
                self.weight.float(),
                self._fp8_weight_qparams().reshape(
                    -1, *([1] * (self.weight.dim() - 1))
                ),
            )
        return K.fake_quant_int8_per_channel(
            self.weight.float(), scale=self._weight_scale()
        )

    @property
    def q_calibrated(self) -> bool:
        return bool(self._q_calibrated.item())

    _Q_PERCENTILE = 0.999
    _Q_OBS_SAMPLE_CAP = 1_000_000

    def _observe(self, x: torch.Tensor):
        with torch.no_grad():
            method = getattr(self, "_q_obs_method", "minmax")
            if method in ("mse", "entropy"):
                self._observe_histogram(x)
                return
            if method == "percentile":
                # Mean of per-batch (0.1%, 99.9%) quantiles: robust to the
                # activation outliers that make pure min/max ranges crush
                # int8 resolution (worst on the smallest models). Strided
                # subsampling keeps torch.quantile within its size limits.
                flat = x.detach().reshape(-1).float()
                if flat.numel() > self._Q_OBS_SAMPLE_CAP:
                    step = (flat.numel() + self._Q_OBS_SAMPLE_CAP - 1) // self._Q_OBS_SAMPLE_CAP
                    flat = flat[::step]
                qs = torch.quantile(
                    flat,
                    torch.tensor(
                        [1.0 - self._Q_PERCENTILE, self._Q_PERCENTILE],
                        device=flat.device,
                    ),
                )
                self._q_obs_lo_sum = getattr(self, "_q_obs_lo_sum", 0.0) + float(qs[0])
                self._q_obs_hi_sum = getattr(self, "_q_obs_hi_sum", 0.0) + float(qs[1])
                self._q_obs_n = getattr(self, "_q_obs_n", 0) + 1
                return
            lo = x.amin().float().reshape(1).to(self._q_act_lo.device)
            hi = x.amax().float().reshape(1).to(self._q_act_hi.device)
            if self.q_calibrated:
                torch.minimum(self._q_act_lo, lo, out=self._q_act_lo)
                torch.maximum(self._q_act_hi, hi, out=self._q_act_hi)
            else:
                self._q_act_lo.copy_(lo)
                self._q_act_hi.copy_(hi)
                self._q_calibrated.fill_(1)

    def _observe_histogram(self, x: torch.Tensor):
        """Accumulate the |x| histogram the mse/entropy sweeps consume."""
        from .calibrate import accumulate_abs_histogram

        x_lo = float(x.detach().amin())
        x_hi = float(x.detach().amax())
        lo_run = getattr(self, "_q_obs_lo_run", None)
        if lo_run is None:
            self._q_obs_lo_run, self._q_obs_hi_run = x_lo, x_hi
        else:
            self._q_obs_lo_run = min(lo_run, x_lo)
            self._q_obs_hi_run = max(self._q_obs_hi_run, x_hi)
        (
            self._q_obs_hist,
            self._q_obs_hist_amax,
            self._q_obs_hist_top,
        ) = accumulate_abs_histogram(
            getattr(self, "_q_obs_hist", None),
            getattr(self, "_q_obs_hist_amax", 0.0),
            x,
            getattr(self, "_q_obs_hist_top", 0.0),
        )

    def finalize_observation(self):
        """Write the collected statistics into the calibrated range buffers."""
        if hasattr(self, "_q_obs_lo_run"):
            from .calibrate import entropy_amax, mse_amax

            hist = getattr(self, "_q_obs_hist", None)
            if hist is None or getattr(self, "_q_obs_hist_amax", 0.0) == 0.0:
                # Every calibration sample was zero: mass sits in bin 0 and
                # no range was ever established.
                lo, hi = 0.0, 0.0
            else:
                select = (
                    mse_amax
                    if getattr(self, "_q_obs_method", "minmax") == "mse"
                    else entropy_amax
                )
                amax = select(
                    hist,
                    self._q_obs_hist_amax,
                    getattr(self, "_q_aformat", "int8"),
                    # One-sided (post-ReLU style) distributions deploy all
                    # 256 int8 codes on one side of zero after the window
                    # intersection below; the sweep must simulate that
                    # resolution or it systematically over-clips.
                    one_sided=self._q_obs_lo_run >= 0.0 or self._q_obs_hi_run <= 0.0,
                )
                # The sweep chose a symmetric clip; intersect it with the
                # observed signed range so one-sided activations (post-ReLU)
                # keep their one-sided affine window.
                lo = max(self._q_obs_lo_run, -amax)
                hi = min(self._q_obs_hi_run, amax)
                if lo > hi:
                    # A near-constant one-sided distribution can put all its
                    # mass inside the last histogram bin, letting the sweep
                    # return an amax a fraction of a bin below max|x| and
                    # inverting the intersection. Fall back to the observed
                    # window rather than emit a degenerate range.
                    lo, hi = self._q_obs_lo_run, self._q_obs_hi_run
            with torch.no_grad():
                self._q_act_lo.fill_(lo)
                self._q_act_hi.fill_(hi)
                self._q_calibrated.fill_(1)
        n = getattr(self, "_q_obs_n", 0)
        if n:
            with torch.no_grad():
                self._q_act_lo.fill_(self._q_obs_lo_sum / n)
                self._q_act_hi.fill_(self._q_obs_hi_sum / n)
                self._q_calibrated.fill_(1)
        for attr in (
            "_q_obs_lo_sum",
            "_q_obs_hi_sum",
            "_q_obs_n",
            "_q_obs_lo_run",
            "_q_obs_hi_run",
            "_q_obs_hist",
            "_q_obs_hist_amax",
            "_q_obs_hist_top",
        ):
            if hasattr(self, attr):
                delattr(self, attr)

    def _native_act_scale(self, device) -> torch.Tensor:
        """Calibrated static E4M3 activation scale, cached while finalized."""
        act_scale = self.__dict__.get("_q_native_act_scale")
        if act_scale is None or act_scale.device != device:
            amax = torch.maximum(self._q_act_lo.abs(), self._q_act_hi.abs())
            act_scale = (amax.float() / E4M3_MAX).reshape(()).to(device)
            self._q_native_act_scale = act_scale
        return act_scale

    def _native_eligible(self, x: torch.Tensor) -> bool:
        if (
            not self.is_finalized
            or getattr(self, "_q_wformat", "int8") != "fp8"
            or getattr(self, "_q_aformat", "int8") != "fp8"
            or not x.is_cuda
            or self._q_observing
            or getattr(self, "_q_export_mode", False)
        ):
            return False
        # q_calibrated does a device-to-host .item() sync, which would
        # invalidate CUDA graph capture if evaluated on the hot path; the
        # flag is immutable while finalized, so read it once and cache the
        # host bool (make_prepared drops the cache with the state).
        calibrated = self.__dict__.get("_q_calibrated_host")
        if calibrated is None:
            calibrated = self.q_calibrated
            self.__dict__["_q_calibrated_host"] = calibrated
        return calibrated

    def _maybe_quant_input(self, x: torch.Tensor) -> torch.Tensor:
        if self._q_observing:
            self._observe(x)
            return x
        if not self.q_calibrated:
            return x
        if getattr(self, "_q_aformat", "int8") == "fp8":
            amax = torch.maximum(self._q_act_lo.abs(), self._q_act_hi.abs())
            return K.fake_quant_fp8(x, (amax / E4M3_MAX).reshape(1))
        return K.fake_quant_int8_affine(
            x,
            self._q_act_lo,
            self._q_act_hi,
            scalars=getattr(self, "_q_export_mode", False),
        )


class QuantConv2d(nn.Conv2d, _ActObserverMixin):
    """INT8 W8A8 simulated convolution (per-channel weights, affine input).

    Finalized fp8 modules on CUDA take a half-precision native tier: the
    input convolves in fp16 against a cached weight dequantized from the
    packed E4M3 codes. This is the standard fp8-deployment convention
    (fp8-derived weights, half-precision conv activations); the E4M3
    activation snap is simulation-only for convs, so the native tier's conv
    activations carry fp16 precision instead. The prepared/simulated path
    (``LIBREYOLO_QUANT_KERNELS=off``) remains the strict W8A8 oracle.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_act_state(self.out_channels)

    def make_prepared(self):
        self.__dict__.pop("_q_native_act_scale", None)
        self.__dict__.pop("_q_native_w16", None)
        super().make_prepared()

    def _native_fp8_forward(self, x: torch.Tensor):
        """Half-precision native tier, or None to take the simulated path."""
        if not self._native_eligible(x) or K.resolve("fp8_gemm") is None:
            return None  # fp8_gemm availability doubles as the "native ok" gate
        cached = self.__dict__.get("_q_native_w16")
        if cached is None or cached[0].device != x.device:
            w16 = unpack_fp8_weight(self.weight_packed, self._q_w_scale).to(torch.float16)
            b16 = self.bias.to(torch.float16) if self.bias is not None else None
            cached = (w16, b16)
            self._q_native_w16 = cached
        in_dtype = x.dtype
        x16 = x if x.dtype == torch.float16 else x.to(torch.float16)
        out = self._conv_forward(x16, cached[0], cached[1])
        return out if in_dtype == torch.float16 else out.to(in_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._native_fp8_forward(x)
        if out is not None:
            return out
        in_dtype = x.dtype
        with autocast_off(x.device.type):
            x = x.float()
            x = self._maybe_quant_input(x)
            weight = self._effective_weight()
            bias = self.bias.float() if self.bias is not None else None
            out = self._conv_forward(x, weight, bias)
        return out.to(in_dtype) if in_dtype != out.dtype else out

    @classmethod
    def from_float(cls, conv: nn.Conv2d) -> "QuantConv2d":
        mod = cls(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=conv.bias is not None,
            padding_mode=conv.padding_mode,
            device=conv.weight.device,
            dtype=conv.weight.dtype,
        )
        mod.weight = conv.weight
        mod.bias = conv.bias
        return mod


class QuantLinear(nn.Linear, _ActObserverMixin):
    """INT8 W8A8 simulated linear (per-channel weights, affine input).

    Finalized fp8 modules on an fp8-capable GPU (Ada/Hopper/Blackwell) run
    natively on the fp8 tensor cores via the ``fp8_gemm`` registry kernel
    with the same calibrated static scales the simulation uses; everything
    else takes the simulated fp32-island path.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_act_state(self.out_features)

    @classmethod
    def from_float(cls, linear: nn.Linear) -> "QuantLinear":
        mod = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        mod.weight = linear.weight
        mod.bias = linear.bias
        return mod

    def _native_fp8_forward(self, x: torch.Tensor):
        """Native tensor-core path, or None to take the simulated path."""
        if not self._native_eligible(x):
            return None
        tensorwise = (
            getattr(self, "_q_fp8_weight_scaling", "per_channel") == "tensorwise"
        )
        impl = K.resolve("fp8_gemm_tensorwise" if tensorwise else "fp8_gemm")
        if impl is None:
            return None
        aux = self.__dict__.get("_q_native_aux")
        if aux is None or aux[0].device != x.device:
            from ..kernels.quant.execute.scaled_mm_fp8 import make_aux

            aux = make_aux(
                self._native_act_scale(x.device),
                self._q_w_scale,
                self.bias,
                x.device,
                tensorwise=tensorwise,
            )
            self._q_native_aux = aux
        return impl(x, self.weight_packed, aux)

    def make_prepared(self):
        self.__dict__.pop("_q_native_act_scale", None)
        self.__dict__.pop("_q_native_aux", None)
        super().make_prepared()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._native_fp8_forward(x)
        if out is not None:
            return out
        in_dtype = x.dtype
        with autocast_off(x.device.type):
            x = x.float()
            x = self._maybe_quant_input(x)
            weight = self._effective_weight()
            bias = self.bias.float() if self.bias is not None else None
            out = F.linear(x, weight, bias)
        return out.to(in_dtype) if in_dtype != out.dtype else out


class NVFP4Linear(nn.Linear):
    """NVFP4 W4A4 simulated linear.

    Weights: E2M1 in 16-element blocks with E4M3 block scales and a fixed
    fp32 per-tensor scale captured at quantize time. Activations: dynamically
    scaled per forward (two-level scaling), so no calibration is required.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("_q_w_amax", torch.zeros(1, device=self.weight.device))
        self._q_observing = False  # accepted for API symmetry; unused

    @classmethod
    def from_float(cls, linear: nn.Linear) -> "NVFP4Linear":
        mod = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        mod.weight = linear.weight
        mod.bias = linear.bias
        with torch.no_grad():
            mod._q_w_amax.copy_(linear.weight.detach().abs().amax().float().reshape(1))
        return mod

    @property
    def q_calibrated(self) -> bool:
        return bool((self._q_w_amax > 0).item())

    @property
    def is_finalized(self) -> bool:
        return "weight_packed" in self._buffers

    def make_finalized(self):
        """Crystallize: replace fp32 masters with packed E2M1 + E4M3 scales."""
        if self.is_finalized:
            return
        with torch.no_grad():
            payload, block_scale = pack_nvfp4_weight(
                self.weight.detach(), self._q_w_amax
            )
        del self._parameters["weight"]
        self.register_buffer("weight_packed", payload)
        self.register_buffer("weight_block_scale", block_scale)

    def make_prepared(self):
        """Reconstruct fp32 masters from packed weights (QAT-from-PTQ)."""
        if not self.is_finalized:
            return
        with torch.no_grad():
            weight = unpack_nvfp4_weight(
                self.weight_packed,
                self.weight_block_scale,
                self._q_w_amax,
                self.in_features,
            )
        del self._buffers["weight_packed"]
        del self._buffers["weight_block_scale"]
        self.weight = nn.Parameter(weight)

    def _effective_weight(self) -> torch.Tensor:
        if self.is_finalized:
            # Registry slot: accelerated unpack when available, packing
            # reference otherwise (identical values either way).
            return K.unpack_nvfp4(
                self.weight_packed,
                self.weight_block_scale,
                self._q_w_amax,
                self.in_features,
            )
        return K.fake_quant_nvfp4_weight(self.weight.float(), self._q_w_amax)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        with autocast_off(x.device.type):
            x = K.fake_quant_nvfp4_dynamic(x.float())
            weight = self._effective_weight()
            bias = self.bias.float() if self.bias is not None else None
            out = F.linear(x, weight, bias)
        return out.to(in_dtype) if in_dtype != out.dtype else out


class GroupQuantLinear(nn.Linear, _ActObserverMixin):
    """Grouped symmetric integer weight quantization (W4/W2 style linears).

    Weights: per-group symmetric int codes (``_q_bits`` in {2, 4}, group size
    along in_features). Activations: optional calibrated INT8 (W4A8/W2A8) or
    untouched float (W4A16 style).
    """

    def __init__(self, *args, bits: int = 4, group_size: int = 128, act_int8: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self._q_bits = int(bits)
        self._q_group = int(group_size)
        self._q_act_enabled = bool(act_int8)
        self._init_act_state(self.out_features)
        ngroups = (self.in_features + self._q_group - 1) // self._q_group
        self.register_buffer(
            "_q_w_gscale",
            torch.zeros(self.out_features, ngroups, device=self.weight.device),
        )

    @classmethod
    def from_float(
        cls,
        linear: nn.Linear,
        bits: int = 4,
        group_size: int = 128,
        act_int8: bool = False,
    ) -> "GroupQuantLinear":
        mod = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
            bits=bits,
            group_size=group_size,
            act_int8=act_int8,
        )
        mod.weight = linear.weight
        mod.bias = linear.bias
        return mod

    def freeze_weight_qparams(self):
        with torch.no_grad():
            self._q_w_gscale.copy_(
                int_group_qparams(self.weight.float(), self._q_bits, self._q_group)
            )

    def make_finalized(self):
        if self.is_finalized:
            return
        self.freeze_weight_qparams()
        with torch.no_grad():
            payload, _ = pack_int_grouped_weight(
                self.weight.detach().float(),
                self._q_bits,
                self._q_group,
                scale=self._q_w_gscale,
            )
        del self._parameters["weight"]
        self.register_buffer("weight_packed", payload)

    def make_prepared(self):
        if not self.is_finalized:
            return
        with torch.no_grad():
            weight = unpack_int_grouped_weight(
                self.weight_packed,
                self._q_w_gscale,
                self._q_bits,
                self._q_group,
                self.in_features,
            )
        del self._buffers["weight_packed"]
        self.weight = nn.Parameter(weight)

    def _effective_weight(self) -> torch.Tensor:
        if self.is_finalized:
            # Registry slot: accelerated unpack when available, packing
            # reference otherwise (identical values either way).
            return K.unpack_int_grouped(
                self.weight_packed,
                self._q_w_gscale,
                self._q_bits,
                self._q_group,
                self.in_features,
            )
        return K.fake_quant_int_grouped(
            self.weight.float(), self._q_bits, self._q_group
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        with autocast_off(x.device.type):
            x = x.float()
            if self._q_act_enabled or self._q_observing:
                x = self._maybe_quant_input(x)
            weight = self._effective_weight()
            bias = self.bias.float() if self.bias is not None else None
            out = F.linear(x, weight, bias)
        return out.to(in_dtype) if in_dtype != out.dtype else out


class MXFP4Linear(nn.Linear):
    """OCP MXFP4 W4A4 simulated linear: E2M1 elements, 32-element blocks,
    power-of-two (E8M0) block scales, dynamic activation scaling."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._q_observing = False  # API symmetry; unused

    @classmethod
    def from_float(cls, linear: nn.Linear) -> "MXFP4Linear":
        mod = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        mod.weight = linear.weight
        mod.bias = linear.bias
        return mod

    @property
    def q_calibrated(self) -> bool:
        return True  # dynamic activation scaling: nothing to calibrate

    @property
    def is_finalized(self) -> bool:
        return "weight_packed" in self._buffers

    def make_finalized(self):
        if self.is_finalized:
            return
        with torch.no_grad():
            payload, exponents = pack_mxfp4_weight(self.weight.detach().float())
        del self._parameters["weight"]
        self.register_buffer("weight_packed", payload)
        self.register_buffer("weight_block_exp", exponents)

    def make_prepared(self):
        if not self.is_finalized:
            return
        with torch.no_grad():
            weight = unpack_mxfp4_weight(
                self.weight_packed, self.weight_block_exp, self.in_features
            )
        del self._buffers["weight_packed"]
        del self._buffers["weight_block_exp"]
        self.weight = nn.Parameter(weight)

    def _effective_weight(self) -> torch.Tensor:
        if self.is_finalized:
            return unpack_mxfp4_weight(
                self.weight_packed, self.weight_block_exp, self.in_features
            )
        return K.fake_quant_mxfp4_weight(self.weight.float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        with autocast_off(x.device.type):
            x = K.fake_quant_mxfp4_dynamic(x.float())
            weight = self._effective_weight()
            bias = self.bias.float() if self.bias is not None else None
            out = F.linear(x, weight, bias)
        return out.to(in_dtype) if in_dtype != out.dtype else out


QUANT_MODULE_TYPES = (
    QuantConv2d,
    QuantLinear,
    NVFP4Linear,
    MXFP4Linear,
    GroupQuantLinear,
)
