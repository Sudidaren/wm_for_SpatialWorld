"""Compare the deployed runtime with completed CPU evaluation predictions.

Run explicitly with --validation <full_val directory>; no simulator or GPT call.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'eval_pipeline'))
sys.path.insert(0, str(ROOT/'runtime_overlay'))
from config_builder import _apply_wingman_block
from phase_b.runtime_factory import build_runtime
from mllm_base_agent.agent.world_model import WorldModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--validation', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report_path = args.validation/'comparison.json'
    report = json.loads(report_path.read_text())
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    assert (args.validation/'EVALUATION_COMPLETE').read_text().strip() == sha(report_path)
    assert report['split'] == 'val' and not report['smoke']
    cache = args.validation/'rfdetr_full_predictions.jsonl'
    assert sha(cache) == report['prediction_cache_sha256']
    rows = [json.loads(line) for line in cache.read_text().splitlines()]
    config = {}
    _apply_wingman_block(config, SimpleNamespace(target_types=[]))
    options = config['memory_probe']['world_model']
    assert sha(options['detector_path']) == report['checkpoint_sha256']
    assert options['obj_thr'] == report['models']['rfdetr_full']['best_f1']['threshold']
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    runtime = build_runtime({**options, 'device': 'cpu'})
    assert torch.get_float32_matmul_precision() == 'highest'
    results = []
    for row in rows[:6]:
        with Image.open(row['rgb']) as image:
            rgb = np.array(image.convert('RGB'))
        output = runtime(rgb)
        expected = [d for d in row['detections'] if d['score'] >= options['obj_thr']]
        assert len(output['detections']) == len(expected)
        for actual, previous in zip(output['detections'], expected):
            assert actual['bbox'] == previous['bbox'] and actual['type'] == previous['type']
            assert abs(actual['score']-previous['score']) < 1e-7
        assert output['depth'].shape == rgb.shape[:2] and np.isfinite(output['depth']).all()
        world = WorldModel(width=rgb.shape[1], height=rgb.shape[0], pose_from_action_log=True)
        world.attach_perception(lambda _: output)
        world.observe(SimpleNamespace(metadata={}), env=SimpleNamespace(
            controller=SimpleNamespace(last_event=SimpleNamespace(frame=rgb))))
        results.append({'rgb': row['rgb'], 'detections': len(expected),
                        'anchors': len(world._slots)+len(world._furniture)})
    assert any(item['anchors'] for item in results)
    result = {'status': 'passed', 'parameters': runtime.parameters, 'frames': results,
              'checkpoint_sha256': report['checkpoint_sha256'],
              'threshold': options['obj_thr'], 'precision_preserved': True,
              'scope': 'Generated configuration to RF-DETR/depth to WorldModel; cached prediction parity; no task success claim.'}
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
