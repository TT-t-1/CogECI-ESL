"""CPU-only validation of archived metrics, split IDs, and selected cases."""
import csv
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
cases = json.loads((root / 'results/selected_cases.json').read_text(encoding='utf-8'))
splits = json.loads((root / 'results/E1/splits.json').read_text(encoding='utf-8'))
for split in splits:
    assert not set(split['train']) & set(split['dev'])
    assert not set(split['train']) & set(split['test'])
    assert not set(split['dev']) & set(split['test'])
assert len({doc for s in splits for doc in s['test']}) == 233
assert sum(len(s['test']) for s in splits) == 233
assert all(set(s['dev']) == set(splits[0]['dev']) for s in splits)
metrics = []
for name in ['E0', 'E1', 'E2']:
    directory = root / 'results' / name
    records = ([f['test'] for f in json.loads((directory / 'five_fold_results.json').read_text(encoding='utf-8'))['folds']]
               if name == 'E0' else
               [json.loads(line) for line in (directory / 'fold_metrics.jsonl').read_text(encoding='utf-8').splitlines()])
    folds = list(range(1, 6))
    for fold, record in zip(folds, records):
        docs = json.loads((directory / f'fold_{fold}_predictions.json').read_text(encoding='utf-8'))
        assert set(d['doc_id'] for d in docs) == set(splits[fold - 1]['test'])
        tp = fp = fn = 0
        for d in docs:
            assert len(d['gold']) == len(d['predicted']) == len(d['rel_type'])
            for g, p in zip(d['gold'], d['predicted']):
                tp += int(g == p == 1)
                fp += int(g == 0 and p == 1)
                fn += int(g == 1 and p == 0)
        p, r = tp / (tp + fp), tp / (tp + fn)
        f = 2 * p * r / (p + r)
        assert max(abs(record[k] - value) for k, value in [('p', p), ('r', r), ('f1', f)]) < 1e-9
        metrics.append((name, fold, f))
        for scope, kind in [('intra', 0), ('inter', 1)]:
            pairs = [(g, p) for d in docs for g, p, k in zip(d['gold'], d['predicted'], d['rel_type']) if k == kind]
            tp = sum(g == p == 1 for g, p in pairs)
            fp = sum(g == 0 and p == 1 for g, p in pairs)
            fn = sum(g == 1 and p == 0 for g, p in pairs)
            assert abs(record[scope]['f1'] - 2 * tp / (2 * tp + fp + fn)) < 1e-8
for case in cases:
    docs = json.loads((root / 'results/E1/fold_1_predictions.json').read_text(encoding='utf-8'))
    doc = next(d for d in docs if d['doc_id'] == case['doc_id'])
    i = case['pair_index']
    assert (doc['gold'][i], doc['predicted'][i]) == (case['gold'], case['predicted'])
print(json.dumps({'validated_completed_folds': len(metrics), 'selected_cases': len(cases),
                  'all_metric_recalculations_match': True}, ensure_ascii=False))
