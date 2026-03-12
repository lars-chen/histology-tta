"""
D4-Equivariant Wide ResNet for histology classification.

This model is architecturally invariant to the D4 group (4 rotations × 2
reflections = 8 elements), meaning it produces identical outputs for any D4
transformation of the input. This makes it a control experiment for TTA:
geometric TTA should provide zero benefit since the model already handles
those symmetries internally.

Architecture adapted from escnn's e2wrn.py (Weiler & Cesa, 2019).
Sized as WRN-16-4 with D4 equivariance to match ResNet-18 (~11M params).

References:
  Weiler & Cesa, 2019. General E(2)-Equivariant Steerable CNNs.
  https://arxiv.org/abs/1911.08251

Usage:
    from models.equivariant import get_equivariant_model
    model = get_equivariant_model("d4wrn", num_classes=31)
"""

from typing import Tuple

import math
import torch
import torch.nn.functional as F

import escnn.nn as enn
from escnn.nn import init
from escnn import gspaces


# ---------------------------------------------------------------------------
# Convolution helpers
# ---------------------------------------------------------------------------

def _conv5x5(in_type: enn.FieldType, out_type: enn.FieldType, stride=1,
             padding=2, bias=False):
    return enn.R2Conv(in_type, out_type, 5, stride=stride, padding=padding,
                      bias=bias, sigma=None, frequencies_cutoff=lambda r: 3*r)


def _conv3x3(in_type: enn.FieldType, out_type: enn.FieldType, stride=1,
             padding=1, bias=False):
    return enn.R2Conv(in_type, out_type, 3, stride=stride, padding=padding,
                      bias=bias, sigma=None, frequencies_cutoff=lambda r: 3*r)


def _conv1x1(in_type: enn.FieldType, out_type: enn.FieldType, stride=1,
             padding=0, bias=False):
    return enn.R2Conv(in_type, out_type, 1, stride=stride, padding=padding,
                      bias=bias, sigma=None, frequencies_cutoff=lambda r: 3*r)


# ---------------------------------------------------------------------------
# Field type builders
# ---------------------------------------------------------------------------

def _regular_feature_type(gspace, planes: int, fixparams: bool = True):
    """Build a regular feature map with the specified number of channels."""
    N = gspace.fibergroup.order()
    if fixparams:
        planes = int(planes * math.sqrt(N) / N)
    else:
        planes = int(planes / N)
    return enn.FieldType(gspace, [gspace.regular_repr] * planes)


def _trivial_feature_type(gspace, planes: int, fixparams: bool = True):
    """Build a trivial (invariant) feature map."""
    if fixparams:
        planes = int(planes * math.sqrt(gspace.fibergroup.order()))
    return enn.FieldType(gspace, [gspace.trivial_repr] * planes)


_FIELD_TYPE = {
    "regular": _regular_feature_type,
    "trivial": _trivial_feature_type,
}


# ---------------------------------------------------------------------------
# Equivariant residual block
# ---------------------------------------------------------------------------

class _WideBasic(enn.EquivariantModule):
    """Pre-activation equivariant residual block."""

    def __init__(self, in_type: enn.FieldType, inner_type: enn.FieldType,
                 dropout_rate: float, stride: int = 1,
                 out_type: enn.FieldType = None):
        super().__init__()
        if out_type is None:
            out_type = in_type

        self.in_type = in_type
        self.out_type = out_type

        # Use 3x3 for D4 (rotations_order=4), 5x5 otherwise
        rotations = in_type.gspace.rotations_order
        conv = _conv3x3 if rotations in [0, 2, 4] else _conv5x5

        self.bn1 = enn.InnerBatchNorm(self.in_type)
        self.relu1 = enn.ReLU(self.in_type, inplace=True)
        self.conv1 = conv(self.in_type, inner_type)

        self.bn2 = enn.InnerBatchNorm(inner_type)
        self.relu2 = enn.ReLU(inner_type, inplace=True)
        self.dropout = enn.PointwiseDropout(inner_type, p=dropout_rate)
        self.conv2 = conv(inner_type, self.out_type, stride=stride)

        self.shortcut = None
        if stride != 1 or self.in_type != self.out_type:
            self.shortcut = _conv1x1(self.in_type, self.out_type, stride=stride,
                                     bias=False)

    def forward(self, x):
        x_n = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(x_n)))
        out = self.dropout(out)
        out = self.conv2(out)
        if self.shortcut is not None:
            out += self.shortcut(x_n)
        else:
            out += x
        return out

    def evaluate_output_shape(self, input_shape: Tuple):
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        if self.shortcut is not None:
            return self.shortcut.evaluate_output_shape(input_shape)
        return input_shape


# ---------------------------------------------------------------------------
# D4 Equivariant Wide ResNet
# ---------------------------------------------------------------------------

class D4WideResNet(torch.nn.Module):
    """
    D4-equivariant Wide ResNet.

    The model is equivariant to the dihedral group D4 (4 rotations × 2
    reflections) throughout. The final layer maps to trivial (invariant)
    features before the linear classifier, ensuring the output is D4-invariant.

    Args:
        depth: network depth (must satisfy (depth-4) % 6 == 0)
        widen_factor: channel multiplier
        dropout_rate: dropout in residual blocks
        num_classes: number of output classes
        initial_stride: stride of first residual stage (2 for large images)
    """

    def __init__(self, depth: int = 16, widen_factor: int = 4,
                 dropout_rate: float = 0.3, num_classes: int = 31,
                 initial_stride: int = 2):
        super().__init__()
        assert (depth - 4) % 6 == 0, "Wide-ResNet depth should be 6n+4"
        n = (depth - 4) // 6
        k = widen_factor

        nStages = [16, 16 * k, 32 * k, 64 * k]

        # D4 group: 4 rotations × 2 reflections = 8 elements
        self.gspace = gspaces.flipRot2dOnR2(N=4)

        # Input: 3 RGB channels as trivial representations
        r1 = enn.FieldType(self.gspace, [self.gspace.trivial_repr] * 3)
        self.in_type = r1

        # First conv scales up channels
        r2 = _FIELD_TYPE["regular"](self.gspace, nStages[0], fixparams=True)
        self._in_type = r2

        self.conv1 = _conv5x5(r1, r2)

        # Three residual stages (last maps to trivial features for invariance)
        self._layer = 0
        self.layer1 = self._wide_layer(nStages[1], n, dropout_rate,
                                       stride=initial_stride)
        self.layer2 = self._wide_layer(nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(nStages[3], n, dropout_rate, stride=2,
                                       totrivial=True)

        self.bn = enn.InnerBatchNorm(self.layer3[-1].out_type, momentum=0.9)
        self.relu = enn.ReLU(self.bn.out_type, inplace=True)

        self._feature_dim = self.bn.out_type.size
        self.linear = torch.nn.Linear(self._feature_dim, num_classes)

        # Initialize weights
        with torch.no_grad():
            for name, module in self.named_modules():
                if isinstance(module, enn.R2Conv):
                    init.deltaorthonormal_init(module.weights, module.basisexpansion)
                elif isinstance(module, torch.nn.BatchNorm2d):
                    module.weight.data.fill_(1)
                    module.bias.data.zero_()
                elif isinstance(module, torch.nn.Linear):
                    module.bias.data.zero_()

    def _wide_layer(self, planes: int, num_blocks: int, dropout_rate: float,
                    stride: int, totrivial: bool = False) -> enn.SequentialModule:
        self._layer += 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        main_type = _FIELD_TYPE["regular"](self.gspace, planes, fixparams=True)
        inner_type = _FIELD_TYPE["regular"](self.gspace, planes, fixparams=True)
        if totrivial:
            out_type = _FIELD_TYPE["trivial"](self.gspace, planes, fixparams=True)
        else:
            out_type = _FIELD_TYPE["regular"](self.gspace, planes, fixparams=True)

        for b, s in enumerate(strides):
            out_f = out_type if b == num_blocks - 1 else main_type
            layers.append(_WideBasic(self._in_type, inner_type, dropout_rate,
                                     s, out_type=out_f))
            self._in_type = out_f

        return enn.SequentialModule(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = enn.GeometricTensor(x, self.in_type)
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.bn(out)
        out = self.relu(out)

        # Extract tensor from GeometricTensor for standard pooling
        out = out.tensor
        b, c, w, h = out.shape
        out = F.avg_pool2d(out, (w, h))
        out = out.view(b, -1)
        out = self.linear(out)
        return out

    def param_groups(self, backbone_lr: float = 1e-4, head_lr: float = 1e-3):
        """Differential learning rates for optimizer."""
        backbone_params = []
        head_params = list(self.linear.parameters())
        for name, p in self.named_parameters():
            if not name.startswith("linear."):
                backbone_params.append(p)
        return [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": head_params, "lr": head_lr},
        ]

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"D4WideResNet("
            f"classes={self.linear.out_features}, "
            f"feature_dim={self._feature_dim}, "
            f"trainable_params={self.count_trainable_params():,})"
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_EQUIVARIANT_MODELS = {"d4wrn"}


def get_equivariant_model(
    name: str,
    num_classes: int,
    dropout: float = 0.3,
    freeze_backbone: bool = False,
    pretrained: bool = True,
) -> D4WideResNet:
    """
    Instantiate a D4-equivariant model.

    Args:
        name: model key ("d4wrn")
        num_classes: number of output classes
        dropout: dropout rate in residual blocks
        freeze_backbone: ignored (no pretrained weights available)
        pretrained: ignored (no pretrained weights available)

    Returns:
        D4WideResNet instance
    """
    if name not in _EQUIVARIANT_MODELS:
        raise ValueError(
            f"Unknown equivariant model '{name}'. "
            f"Choose from: {list(_EQUIVARIANT_MODELS)}"
        )

    if freeze_backbone:
        print("  WARNING: freeze_backbone ignored for D4WideResNet (no pretrained weights).")
    if pretrained:
        # pretrained=True is the default — only warn if user might be confused
        pass

    # WRN-28-6 with D4 equivariance, initial_stride=2 for 224×224 input
    # ~11M params — comparable to ResNet-18 (11M)
    model = D4WideResNet(
        depth=28,
        widen_factor=6,
        dropout_rate=dropout,
        num_classes=num_classes,
        initial_stride=2,
    )
    print(f"Loaded {model}")
    return model
