"""Integrity checks must not leave partial or overwrite different checkpoints."""
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location('download_rfdetr',
    Path(__file__).resolve().parents[1] / 'scripts/download_rfdetr.py')
downloader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(downloader)


class WeightDownloadTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = self.root / 'source.bin'
        self.source.write_bytes(b'fixture checkpoint bytes')
        self.item = {'relative_path': 'checkpoints/model.pt', 'url': self.source.as_uri(),
                     'size_bytes': self.source.stat().st_size,
                     'sha256': hashlib.sha256(self.source.read_bytes()).hexdigest()}

    def test_download_then_skip_verified_file_without_network(self):
        target = downloader.download_weight(self.item, self.root)
        self.assertEqual(target.read_bytes(), self.source.read_bytes())
        self.source.unlink()
        self.assertEqual(downloader.download_weight(self.item, self.root), target)

    def test_corrupt_download_leaves_no_checkpoint_or_partial(self):
        self.source.write_bytes(b'corrupt')
        with self.assertRaises(ValueError):
            downloader.download_weight(self.item, self.root)
        self.assertEqual(list((self.root / 'checkpoints').iterdir()), [])

    def test_same_size_bad_hash_is_rejected(self):
        self.source.write_bytes(b'x' * self.item['size_bytes'])
        with self.assertRaises(ValueError):
            downloader.download_weight(self.item, self.root)
        self.assertEqual(list((self.root / 'checkpoints').iterdir()), [])

    def test_existing_different_checkpoint_is_preserved(self):
        target = self.root / self.item['relative_path']
        target.parent.mkdir()
        target.write_bytes(b'other experiment')
        with self.assertRaises(ValueError):
            downloader.download_weight(self.item, self.root)
        self.assertEqual(target.read_bytes(), b'other experiment')

    def test_destination_cannot_escape_storage_root(self):
        for path in ['../outside.pt', '/outside.pt']:
            with self.subTest(path=path), self.assertRaises(ValueError):
                downloader.download_weight({**self.item, 'relative_path': path}, self.root)


if __name__ == '__main__':
    unittest.main()
