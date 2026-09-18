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
        from phase_b.rfdetr_depth_runtime import (RFDETRDA2Runtime,
                                                  RFDETRDepthRuntime)
        device = options.get('device') or ('cuda' if torch.cuda.is_available() else 'cpu')
        source = (options.get('depth_source')
                  or os.environ.get('LIGHTWM_DEPTH_SOURCE', 'head')).lower()
        if source == 'da2':
            # the public metric-indoor model; measured 0.444 m vs the head's
            # 0.989 m on 250 held-out frames (tools/eval_depth_calibration.py)
            return RFDETRDA2Runtime(options['detector_path'], options['obj_thr'],
                                    device=device,
                                    da2_name=os.environ.get('LIGHTWM_DA2_NAME',
                                        'depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf'))
        return RFDETRDepthRuntime(str(Path(checkpoint).expanduser()),
            options['detector_path'], options['obj_thr'], device=device)
    raise ValueError(f'Unknown perception backend: {backend}')
