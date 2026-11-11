"""Download only the two pinned ESL caches required by this submission."""
import hashlib
import urllib.request
from pathlib import Path

output = Path(__file__).resolve().parents[1] / 'data_info/cache'
output.mkdir(parents=True, exist_ok=True)
commit = 'd89b0267051183da52d59ac5bb8d2b3fa775c8eb'
hashes = {'train': '728c00c43f74c34a95883984c02e7a78d553be5186f9e83df8c954ed71a18999',
          'dev': 'c128c11485793ba8095d1cf2b56abb665cb57c7cc60c0a0305b1672940eb057f'}
for split, expected in hashes.items():
    name = f'EventStoryLine_bert_intra_and_inter_{split}_features.pkl'
    path = output / name
    if not path.exists():
        url = f'https://raw.githubusercontent.com/LchengC/iLIF/{commit}/data/database/exist/{name}'
        urllib.request.urlretrieve(url, path)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f'Cache hash mismatch for {name}. Preserve the file and check the source.')
    print(name, actual)
