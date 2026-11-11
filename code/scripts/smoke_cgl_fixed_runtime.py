"""A runtime fixture, not an experiment: real features with explicitly synthetic zero context labels."""
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
from prepare_data import load_features

parallel = '--parallel_bert' in sys.argv
dataset_name = 'maven' if '--maven' in sys.argv else 'esl'
prefix = 'MAVEN' if dataset_name == 'maven' else 'EventStoryLine'
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'CogECI-cgl-fixed'))
from parameter import parse_args
from model import bertCSRModel
from transformers import BertTokenizer

output = root / 'runs-cgl-fixed' / ('maven_runtime_smoke' if dataset_name == 'maven' else 'runtime_smoke')
if parallel:
    output = root / 'runs-cgl-fixed/maven_dual_gpu_runtime_smoke'
output.mkdir(parents=True, exist_ok=True)
dataset = root / 'data' / dataset_name
fixtures = root / 'data' / ('maven_runtime_fixture' if dataset_name == 'maven' else 'runtime_fixture')
csv_dir = fixtures / 'examples/llm'
csv_dir.mkdir(parents=True, exist_ok=True)
docs = {}
for split in (['train', 'dev', 'test'] if dataset_name == 'maven' else ['train', 'dev']):
    docs[split] = load_features(dataset / f'{prefix}_bert_intra_and_inter_{split}_features.pkl')
    for suffix in ['sen', 'sen_more']:
        with (csv_dir / f'llama3help_{split}_{suffix}.csv').open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['id', 'doc_id', 'text', 'sentences', 'event_pair', 'node_event_id', 'answer(id)'])
            for i, f in enumerate(docs[split]):
                writer.writerow([i, f.doc_id, '', '', '', [list(p) for p in f.event_pairs], ['None'] * len(f.target)])
sys.argv = ['smoke', '--run_dir', str(output), '--model_name', 'bert-base-uncased']
if parallel:
    sys.argv.append('--parallel_bert')
args = parse_args()
args.device = torch.device('cuda')
model = bertCSRModel(args).cuda()
f = min((f for f in docs['train'] if set(f.rel_type) == {0, 1}), key=lambda f: len(f.enc_text))
labels = torch.zeros(len(f.target), len(f.sentences), device='cuda')
context = []
hook = model.bce_loss.register_forward_hook(lambda module, inputs, result: context.append({
    'input_requires_grad': inputs[0].requires_grad, 'loss_requires_grad': result.requires_grad}))
model.train()
loss, predictions = model(f.enc_input_ids.cuda(), f.enc_mask_ids.cuda(), f.node_event,
                          [f.t1_pos], [f.t2_pos], [f.target], [f.rel_type], f.event_pairs,
                          labels, labels, 'train', f.doc_id,
                          BertTokenizer.from_pretrained('bert-base-uncased', local_files_only=True))
assert predictions.shape == (len(f.target), 2)
assert torch.isfinite(loss)
loss.backward()
assert context and all(x['loss_requires_grad'] for x in context)
assert model.pretrained_model.embeddings.word_embeddings.weight.grad is not None
optimized = ('pretrained_model.', 'mlp.', 'CGE.')
audit = {'fixture': f'{dataset_name} features with synthetic all-zero context labels; not benchmark results',
         'context_calls': context, 'classification_loss_requires_grad': loss.requires_grad,
         'prediction_shape': list(predictions.shape),
         'optimizer_policy': 'all model parameters, BERT separate learning rate',
         'parameters_omitted_by_original_optimizer_now_included': [n for n, p in model.named_parameters()
                                          if p.requires_grad and not n.startswith(optimized)]}
(output / 'gradient_audit.json').write_text(json.dumps(audit, indent=2))
hook.remove()
del model, loss, predictions
torch.cuda.empty_cache()
environment = os.environ.copy()
environment.update(COGECI_DATA_DIR=str(fixtures), HF_HUB_OFFLINE='1')
command = [str(root.parent / 'env/bin/python'), '-u', 'main.py', '--run_dir', str(output),
           '--dataset_type', prefix,
           '--model_name', 'bert-base-uncased', '--cache_path', str(dataset),
           '--seed', '209', '--epoch', '1', '--smoke', '--beta_intra', '.3', '--beta_inter', '.7']
if parallel:
    command.append('--parallel_bert')
subprocess.run(command, cwd=root / 'CogECI-cgl-fixed', env=environment, check=True)
print('Runtime smoke passed; corrected context loss has a gradient', flush=True)
