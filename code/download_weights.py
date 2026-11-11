"""Download the three pinned fold-1 checkpoints and verify their SHA256."""
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request

package = Path(__file__).resolve().parents[1]
repository = json.loads((package / 'models/github_release.json').read_text(encoding='utf-8'))
checksums = json.loads((package / 'models/checksums.json').read_text(encoding='utf-8'))

def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

for relative, expected in checksums.items():
    destination = package / relative
    if destination.exists():
        if sha256(destination) != expected['sha256']:
            raise SystemExit(f'Existing file has a different SHA256: {destination}')
        print(f'{relative}: verified existing checkpoint')
        continue
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = repository['assets'][relative]
    partial = destination.with_suffix('.pt.part')
    print(f'Downloading {relative}', flush=True)
    with urllib.request.urlopen(url, timeout=120) as response, partial.open('wb') as handle:
        shutil.copyfileobj(response, handle, length=8 * 1024 * 1024)
    if sha256(partial) != expected['sha256']:
        raise SystemExit(f'Download SHA256 mismatch: {partial}')
    partial.replace(destination)
    print(f'{relative}: downloaded and verified')
