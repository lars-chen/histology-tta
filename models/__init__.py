"""
Model factory for all backbone + classifier head models.

Dispatches to torchvision CNN backbones, DINOv2 ViT backbones,
ConvNeXt V2 backbones, or histology foundation models based on name.

Torchvision models:  resnet18, resnet50, vgg16, efficientnet_b7, mobilenet_v3
DINOv2 models:       dinov2_s, dinov2_b, dinov2_l, dinov2_g
ConvNeXt V2 models:  convnextv2_tiny, convnextv2_small, convnextv2_base,
                     convnextv2_large, convnextv2_huge
                     (short aliases: convnextv2_t/s/b/l/h)
Foundation models:   gigapath, hoptimus, uni, uni2, phikon, phikon2, virchow, virchow2
Equivariant models:  d4wrn (D4 Wide ResNet — trained from scratch)
"""

from .torchvision_models import get_torchvision_model, _MODEL_REGISTRY
from .dinov2 import get_dinov2_model
from .convnextv2 import get_convnextv2_model, _ALIASES as _CONVNEXTV2_ALIASES
from .foundation import get_foundation_model, _FOUNDATION_MODELS
from .equivariant import get_equivariant_model, _EQUIVARIANT_MODELS

_DINOV2_MODELS = {"dinov2_s", "dinov2_b", "dinov2_l", "dinov2_g"}
_CONVNEXTV2_MODELS = set(_CONVNEXTV2_ALIASES.keys())


def get_model(
    name: str,
    num_classes: int,
    dropout: float = 0.2,
    freeze_backbone: bool = False,
    pretrained: bool = True,
    mlp_hidden: int = None,
):
    """
    Instantiate any supported model by name.

    Args:
        name: model key — one of the torchvision, DINOv2, ConvNeXt V2,
              or foundation model names
        num_classes: number of output classes
        dropout: head dropout probability
        freeze_backbone: if True, backbone weights are frozen (linear probe)
        pretrained: if True, load pretrained weights (default). False = random init.
        mlp_hidden: if set, use a 2-layer MLP head (Linear→BN→ReLU→Dropout→Linear)
                    with this many hidden units instead of a single linear layer.

    Returns:
        HistoBaseModel instance ready for training
    """
    if name in _EQUIVARIANT_MODELS:
        return get_equivariant_model(
            name,
            num_classes=num_classes,
            dropout=dropout,
            freeze_backbone=freeze_backbone,
            pretrained=pretrained,
        )
    if name in _DINOV2_MODELS:
        return get_dinov2_model(
            name,
            num_classes=num_classes,
            dropout=dropout,
            freeze_backbone=freeze_backbone,
            pretrained=pretrained,
            mlp_hidden=mlp_hidden,
        )
    if name in _CONVNEXTV2_MODELS:
        return get_convnextv2_model(
            name,
            num_classes=num_classes,
            dropout=dropout,
            freeze_backbone=freeze_backbone,
            pretrained=pretrained,
            mlp_hidden=mlp_hidden,
        )
    if name in _FOUNDATION_MODELS:
        return get_foundation_model(
            name,
            num_classes=num_classes,
            dropout=dropout,
            freeze_backbone=freeze_backbone,
            pretrained=pretrained,
            mlp_hidden=mlp_hidden,
        )
    return get_torchvision_model(
        name,
        num_classes=num_classes,
        dropout=dropout,
        freeze_backbone=freeze_backbone,
        pretrained=pretrained,
        mlp_hidden=mlp_hidden,
    )
