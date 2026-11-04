"""Inference-only extraction of EgoSmith load_hawor; project-owned MANO paths."""
import os
import torch
from lib.models.hawor import HAWOR

def load_hawor(checkpoint_path):
    from pathlib import Path
    from hawor.configs import get_config
    model_cfg = str(Path(checkpoint_path).parent.parent / 'model_config.yaml')
    model_cfg = get_config(model_cfg, update_cachedir=True)

    model_cfg.defrost()
    mano_root = Path(os.environ['QN_EGO_MANO_ROOT'])
    model_cfg.MANO.DATA_DIR = str(mano_root / 'data')
    model_cfg.MANO.MODEL_PATH = str(mano_root / 'data/mano')
    model_cfg.MANO.MEAN_PARAMS = str(mano_root / 'data/mano_mean_params.npz')
    model_cfg.freeze()

    # Override some config values, to crop bbox correctly
    if (model_cfg.MODEL.BACKBONE.TYPE == 'vit') and ('BBOX_SHAPE' not in model_cfg.MODEL):
        model_cfg.defrost()
        assert model_cfg.MODEL.IMAGE_SIZE == 256, f"MODEL.IMAGE_SIZE ({model_cfg.MODEL.IMAGE_SIZE}) should be 256 for ViT backbone"
        model_cfg.MODEL.BBOX_SHAPE = [192,256]
        model_cfg.freeze()

    # Keep the original inference implementation, without torch.compile startup.
    # The upstream source is an actual local copy in this standalone project.
    if "TORCH_COMPILE" in model_cfg.MODEL.BACKBONE:
        model_cfg.defrost()
        model_cfg.MODEL.BACKBONE.TORCH_COMPILE = 0
        model_cfg.freeze()

    # torch >= 2.6 defaults torch.load to weights_only=True, which rejects the Lightning checkpoint
    # (same trusted-checkpoint load as lib.pipeline.stages.hawor_runtime.load_hawor; not imported
    # from there because lib.pipeline.stages pulls in every pipeline stage)
    original_load = torch.load
    # Lightning 2.6 passes weights_only=True explicitly.  This trusted local
    # checkpoint is a full Lightning object, so the compatibility override
    # must win over caller kwargs rather than being overwritten by them.
    torch.load = lambda *a, **k: original_load(*a, **{**k, "weights_only": False})
    try:
        model = HAWOR.load_from_checkpoint(checkpoint_path, strict=False, cfg=model_cfg)
    finally:
        torch.load = original_load
    return model, model_cfg
