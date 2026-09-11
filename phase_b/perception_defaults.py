"""Validated perception selection; datasets and weights remain on NFS."""
import os
from pathlib import Path

DEFAULT_STORAGE_ROOT = '/nfs-stor/junchi.yao/2027ICLR/wm_for_spatialworld'


def perception_defaults():
    root = Path(os.environ.get('LIGHTWM_STORAGE_ROOT', DEFAULT_STORAGE_ROOT)).expanduser()
    backend = os.environ.get('LIGHTWM_DETECTOR', 'rfdetr_small_depth')
    return {
        'detector_backend': backend,
        'detector_path': os.environ.get('LIGHTWM_DETECTOR_PATH', str(
            root / 'checkpoints/rfdetr_small_228094/checkpoint_best_total.pth')),
        'perception_ckpt': os.environ.get('PERCEPTION_CKPT', str(
            root / 'checkpoints/small_objects_20260910/dense_depth_best.pt')),
        'obj_thr': float(os.environ.get('LIGHTWM_OBJ_THR',
                                      '0.4' if backend == 'rfdetr_small_depth' else '0.35')),
    }
