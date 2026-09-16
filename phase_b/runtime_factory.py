"""Perception backend: RF-DETR detector + monocular depth head."""
import os
from pathlib import Path

from phase_b.perception_defaults import perception_defaults


def build_runtime(config=None):
    options = {**perception_defaults(), **(config or {})}
    checkpoint = options['perception_ckpt']
    if not checkpoint or not Path(checkpoint).expanduser().is_file():
        raise FileNotFoundError(f'Monocular depth checkpoint not found: {checkpoint}')
    backend = options['detector_backend']
    if backend == 'rfdetr_small_depth':
        if os.environ.get('LIGHTWM_ZOOM', '0') != '0':
            raise ValueError('The validated RF-DETR configuration uses full-frame inference (LIGHTWM_ZOOM=0)')
        import torch
        from phase_b.rfdetr_depth_runtime import RFDETRDepthRuntime
        return RFDETRDepthRuntime(str(Path(checkpoint).expanduser()),
            options['detector_path'], options['obj_thr'],
            device=options.get('device') or ('cuda' if torch.cuda.is_available() else 'cpu'))
    raise ValueError(f'Unknown perception backend: {backend}')
