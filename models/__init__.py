"""
Model factory for all backbone + classifier head models.

Dispatches to torchvision CNN backbones, DINOv2 ViT backbones, or
ConvNeXt V2 backbones based on name.

Torchvision models:  resnet18, resnet50, vgg16, efficientnet_b7, mobilenet_v3
DINOv2 models:       dinov2_s, dinov2_b, dinov2_l, dinov2_g
ConvNeXt V2 models:  convnextv2_tiny, convnextv2_small, convnextv2_base,
                     convnextv2_large, convnextv2_huge
                     (short aliases: convnextv2_t/s/b/l/h)
"""

from .torchvision_models import get_torchvision_model, _MODEL_REGISTRY
from .dinov2 import get_dinov2_model
from .convnextv2 import get_convnextv2_model, _ALIASES as _CONVNEXTV2_ALIASES

_DINOV2_MODELS = {"dinov2_s", "dinov2_b", "dinov2_l", "dinov2_g"}
_CONVNEXTV2_MODELS = set(_CONVNEXTV2_ALIASES.keys())


def get_model(
    name: str,
    num_classes: int,
    dropout: float = 0.2,
    freeze_backbone: bool = False,
):
    """
    Instantiate any supported model by name.

    Args:
        name: model key — one of the torchvision, DINOv2, or ConvNeXt V2 model names
        num_classes: number of output classes
        dropout: head dropout probability
        freeze_backbone: if True, backbone weights are frozen (linear probe)

    Returns:
        HistoBaseModel instance ready for training
    """
    if name in _DINOV2_MODELS:
        return get_dinov2_model(
            name,
            num_classes=num_classes,
            dropout=dropout,
            freeze_backbone=freeze_backbone,
        )
    if name in _CONVNEXTV2_MODELS:
        return get_convnextv2_model(
            name,
            num_classes=num_classes,
            dropout=dropout,
            freeze_backbone=freeze_backbone,
        )
    return get_torchvision_model(
        name,
        num_classes=num_classes,
        dropout=dropout,
        freeze_backbone=freeze_backbone,
    )
