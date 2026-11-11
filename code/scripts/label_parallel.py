"""Generate disjoint MAVEN document shards, then verify and export one cache."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from prepare_data import load_features

root = Path(__file__).resolve().parents[1]
output = root / 'labels/llama8b/maven/full'
run = root / 'runs-cgl-fixed/llama8b_maven_labels_full'
plan_path = root / 'data/maven_label_parallel_plan.json'
plan = json.loads(plan_path.read_text())
cache = output / 'responses.jsonl'
serial = root / 'label-acceleration-20261106/responses.serial.jsonl'
if not serial.exists():
    os.link(cache, serial)

started = time.time()
workers, logs = [], []
try:
    for gpu in [0, 1]:
        shard = output.parent / f'parallel_shard_{gpu}'
        command = [sys.executable, '-u', str(root / 'scripts/label_local.py'),
                   '--model', str(root / 'models/llama3-8b-instruct'), '--name', 'llama8b',
                   '--dataset', 'maven', '--plan', str(plan_path), '--shard', str(gpu),
                   '--base_cache', str(cache), '--output_dir', str(shard)]
        if gpu == 0:
            command += ['--memory_fraction', '0.77']
        environment = os.environ.copy()
        environment['CUDA_VISIBLE_DEVICES'] = str(gpu)
        log = (run / f'shard_{gpu}.log').open('a')
        logs.append(log)
        workers.append(subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT))
    (run / 'workers.json').write_text(json.dumps([
        {'gpu': gpu, 'pid': worker.pid} for gpu, worker in enumerate(workers)], indent=2))
    print('Both label workers started', [worker.pid for worker in workers], flush=True)
    while any(worker.poll() is None for worker in workers):
        for gpu, worker in enumerate(workers):
            if worker.poll() not in (None, 0):
                raise RuntimeError(f'GPU {gpu} label worker failed; inspect shard_{gpu}.log')
        time.sleep(5)
    assert all(worker.returncode == 0 for worker in workers)
finally:
    for worker in workers:
        if worker.poll() is None:
            worker.terminate()
            try:
                worker.wait(timeout=30)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait()
    for log in logs:
        log.close()

records = {}
counts = {}
for path in [cache, output.parent / 'parallel_shard_0/responses.jsonl',
             output.parent / 'parallel_shard_1/responses.jsonl']:
    count = 0
    with path.open() as handle:
        for line in handle:
            record = json.loads(line)
            key = record['id']
            if key in records:
                assert records[key] == record, key
            records[key] = record
            count += 1
    counts[str(path)] = count
required = set()
for split in ['train', 'dev']:
    features = load_features(root / f'data/maven/MAVEN_bert_intra_and_inter_{split}_features.pkl')
    for feature in features:
        required.update(f'{split}/{feature.doc_id}/event/{i}' for i in range(len(feature.node_event)))
        required.update(f'{split}/{feature.doc_id}/pair/{i}' for i, target in enumerate(feature.target) if target)
missing = required - records.keys()
assert not missing, sorted(missing)[:10]
merged = output / 'responses.merged.jsonl'
with merged.open('w') as handle:
    for record in records.values():
        handle.write(json.dumps(record, ensure_ascii=False) + '\n')
os.replace(merged, cache)
audit = {'required_labels': len(required), 'missing_labels': len(missing), 'input_records': counts,
         'gpus': [0, 1], 'wall_seconds': time.time() - started,
         'sampling_note': plan['sampling_note'], 'original_records_preserved': True}
(output / 'parallel_audit.json').write_text(json.dumps(audit, indent=2))
print('Merged label coverage verified', audit, flush=True)
del records
environment = os.environ.copy()
environment['CUDA_VISIBLE_DEVICES'] = ''
subprocess.run([sys.executable, '-u', str(root / 'scripts/label_local.py'),
                '--model', str(root / 'models/llama3-8b-instruct'), '--name', 'llama8b',
                '--dataset', 'maven', '--export_only'], env=environment, check=True)
summary_path = output / 'summary.json'
summary = json.loads(summary_path.read_text())
summary['parallel_generation'] = audit
summary['max_gpu_memory_bytes'] = max(json.loads(
    (output.parent / f'parallel_shard_{gpu}/summary.json').read_text())['max_gpu_memory_bytes'] for gpu in [0, 1])
summary_path.write_text(json.dumps(summary, indent=2))
print('Parallel labels exported for MAVEN main training', flush=True)
