"""Configuration/runner contract without simulator, GPU or model downloads."""
import importlib
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'eval_pipeline'))
from phase_b.perception_defaults import perception_defaults


class PerceptionConfigTests(unittest.TestCase):
    def test_default_and_relocated_weights(self):
        with patch.dict(os.environ, {'LIGHTWM_STORAGE_ROOT': '/nfs/test'}, clear=True):
            options = perception_defaults()
        self.assertEqual(options['detector_backend'], 'rfdetr_small_depth')
        self.assertEqual(options['obj_thr'], .4)
        self.assertTrue(options['detector_path'].startswith('/nfs/test/checkpoints/'))

    def test_explicit_legacy_selection(self):
        with patch.dict(os.environ, {'LIGHTWM_DETECTOR': 'dino', 'PERCEPTION_CKPT': '/tmp/old.pt'}, clear=True):
            options = perception_defaults()
        self.assertEqual(options['detector_backend'], 'dino')
        self.assertEqual(options['obj_thr'], .35)
        self.assertEqual(options['perception_ckpt'], '/tmp/old.pt')

    def test_generated_world_model_config(self):
        from types import SimpleNamespace
        with patch.dict(os.environ, {}, clear=True):
            cfg = importlib.import_module('eval_config')
            importlib.reload(cfg)
            builder = importlib.import_module('config_builder')
            data = {}
            builder._apply_wingman_block(data, SimpleNamespace(target_types=['Potato']))
        wm = data['memory_probe']['world_model']
        self.assertEqual(wm['detector_backend'], 'rfdetr_small_depth')
        self.assertEqual(wm['obj_thr'], .4)
        self.assertEqual(wm['detector_path'], cfg.WINGMAN_OPTIONS['detector_path'])
        self.assertEqual(wm['perception_runtime_root'], str(ROOT))
        self.assertFalse(data['env']['render_depth'])
        self.assertFalse(data['env']['render_instance_segmentation'])

    def test_missing_weights_do_not_fall_back_to_dino(self):
        from phase_b.runtime_factory import build_runtime
        with self.assertRaises(FileNotFoundError):
            build_runtime({'perception_ckpt': '/missing/rfdetr-depth.pt'})


if __name__ == '__main__':
    unittest.main()
