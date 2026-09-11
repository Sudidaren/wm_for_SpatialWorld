"""Verify selected NFS weights and optionally run one RGB frame, without GPT."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from phase_b.perception_defaults import perception_defaults


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=Path)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    manifest = json.loads((ROOT/'configs/rfdetr_small.json').read_text())
    config = perception_defaults()
    if config['detector_backend'] != manifest['backend'] or config['obj_thr'] != manifest['threshold']:
        raise ValueError('This verifier checks the selected RF-DETR model at threshold 0.40')
    paths = {'detector': Path(config['detector_path']), 'depth': Path(config['perception_ckpt']),
             'depth_metadata': Path(config['perception_ckpt'] + '.json')}
    for name, item in manifest['weights'].items():
        path = paths[name].expanduser()
        if not path.is_file():
            raise FileNotFoundError(f'{name}: {path}; mount/copy the NFS weights described in docs/perception_rfdetr.md')
        if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError(f'Weight hash mismatch: {path}')
    output = {'status': 'weights_verified', 'backend': manifest['backend'],
              'parameters': manifest['total_parameters'], 'threshold': config['obj_thr']}
    if args.image:
        import numpy as np
        from PIL import Image
        import torch
        from phase_b.runtime_factory import build_runtime
        torch.set_num_threads(4)
        torch.set_float32_matmul_precision('highest')
        runtime = build_runtime({**config, 'device': args.device})
        if torch.get_float32_matmul_precision() != 'highest':
            raise RuntimeError('Runtime changed the requested numerical policy')
        with Image.open(args.image) as im:
            rgb = np.array(im.convert('RGB'))
        result = runtime(rgb)
        assert result['depth'].shape == rgb.shape[:2] and np.isfinite(result['depth']).all()
        output.update(status='inference_passed', detections=len(result['detections']),
                      depth_shape=list(result['depth'].shape), device=args.device)
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
