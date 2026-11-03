"""Input padder taken from ptlflow.utils.external.raft and ptlflow.utils.utils.

ptlflow adapted it from https://github.com/princeton-vl/RAFT/blob/master/core/utils/utils.py.

Original license below

BSD 3-Clause License

Copyright (c) 2020, princeton-vl
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import torch.nn.functional as F


class InputPadder:
    """Pads images such that dimensions are divisible by stride."""

    def __init__(
        self,
        dims,
        stride=8,
        two_side_pad=True,
        pad_mode="replicate",
        pad_value=0.0,
    ):
        self.pad_mode = pad_mode
        self.pad_value = pad_value
        ht, wd = dims[-2:]
        pad_ht = (((ht // stride) + 1) * stride - ht) % stride
        pad_wd = (((wd // stride) + 1) * stride - wd) % stride
        if two_side_pad:
            self._pad = [
                pad_wd // 2,
                pad_wd - pad_wd // 2,
                pad_ht // 2,
                pad_ht - pad_ht // 2,
            ]
        else:
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]
        self.tgt_size = (ht + pad_ht, wd + pad_wd)

    def fill(self, x):
        in_shape = x.shape
        if len(in_shape) > 4:
            x = x.view(-1, *in_shape[-3:])
        x = F.pad(x, self._pad, mode=self.pad_mode, value=self.pad_value)
        if len(in_shape) > 4:
            x = x.view(*in_shape[:-2], *x.shape[-2:])
        return x

    def unfill(self, x):
        if x.shape[-2] == self.tgt_size[0] and x.shape[-1] == self.tgt_size[1]:
            ht, wd = x.shape[-2:]
            c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
            x = x[..., c[0] : c[1], c[2] : c[3]]
        return x
