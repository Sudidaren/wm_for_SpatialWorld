"""Download the published perception weights using only Python's standard library."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def download_weight(item, storage_root):
    relative = Path(item['relative_path'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Weight destination must be relative to the storage root')
    destination = Path(storage_root).expanduser() / relative
    if destination.exists():
        if sha256(destination) != item['sha256']:
            raise ValueError(f'Existing file has a different SHA256; move it aside before retrying: {destination}')
        print(f'Already verified: {destination}', flush=True)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        request = Request(item['url'], headers={'User-Agent': 'WingmanWM-weight-downloader'})
        with urlopen(request, timeout=120) as response, tempfile.NamedTemporaryFile(
                dir=destination.parent, prefix=destination.name + '.', suffix='.part', delete=False) as out:
            temporary = Path(out.name)
            digest, size = hashlib.sha256(), 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                digest.update(chunk)
                size += len(chunk)
        if size != item['size_bytes'] or digest.hexdigest() != item['sha256']:
            raise ValueError(f'Download size or SHA256 mismatch: {destination}')
        # Publish only a fully verified file; a failed download never becomes a checkpoint.
        os.replace(temporary, destination)
        print(f'Downloaded and verified: {destination} ({size:,} bytes)', flush=True)
        return destination
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--storage-root', type=Path,
                        default=Path(os.environ.get('LIGHTWM_STORAGE_ROOT', str(ROOT))))
    args = parser.parse_args()
    storage = args.storage_root.expanduser().resolve()
    manifest = json.loads((ROOT / 'configs/rfdetr_small.json').read_text())
    for item in manifest['weights'].values():
        download_weight(item, storage)
    print(f'All perception weights verified. Set LIGHTWM_STORAGE_ROOT to {storage} before sourcing '
          'scripts/selected_perception_env.sh.')


if __name__ == '__main__':
    main()
