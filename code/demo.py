"""Reload one saved model and compare one document with archived predictions."""
import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path

import torch
from transformers import BertTokenizer

package = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(package / 'code/scripts'))
sys.path.insert(0, str(package / 'code/CogECI-cgl-fixed'))
from prepare_data import load_features
from model import bertCSRModel

parser = argparse.ArgumentParser()
parser.add_argument('--experiment', choices=['E0', 'E1', 'E2'], default='E1')
parser.add_argument('--doc-id', default='1_1')
parser.add_argument('--cache', type=Path, default=package / 'data_info/cache')
parser.add_argument('--output', type=Path)
args = parser.parse_args()
checkpoint = torch.load(package / f'models/{args.experiment}/fold_1_best.pt', map_location='cpu')
features = {f.doc_id: f for f in load_features(args.cache / 'EventStoryLine_bert_intra_and_inter_train_features.pkl')}
feature = features[args.doc_id]
if args.experiment == 'E0':
    from train_e0 import BertEventPair
    model = BertEventPair().cuda()
    model.load_state_dict(checkpoint['model'])
    model.eval()
    with torch.no_grad():
        logits = model(feature)
else:
    model_args = Namespace(**checkpoint['args'])
    model_args.device = torch.device('cuda')
    model_args.model_name = 'bert-base-uncased'
    model_args.parallel_bert = False
    model = bertCSRModel(model_args).cuda()
    model.load_state_dict(checkpoint['model'])
    model.eval()
    labels = torch.zeros(len(feature.target), len(feature.sentences), device='cuda')
    with torch.no_grad():
        _, logits = model(feature.enc_input_ids.cuda(), feature.enc_mask_ids.cuda(),
                          feature.node_event, [feature.t1_pos], [feature.t2_pos],
                          [feature.target], [feature.rel_type], feature.event_pairs,
                          labels, labels, 'test', feature.doc_id,
                          BertTokenizer.from_pretrained('bert-base-uncased'))
predicted = logits.argmax(-1).cpu().tolist()
archived = json.loads((package / f'results/{args.experiment}/fold_1_predictions.json').read_text(encoding='utf-8'))
reference = next(d for d in archived if d['doc_id'] == args.doc_id)
assert predicted == reference['predicted'], 'Prediction differs from archived result.'
result = {'doc_id': args.doc_id, 'experiment': args.experiment,
          'predicted': predicted, 'gold': [int(t) for t in feature.target],
          'checkpoint_reload_matches_archive': True}
if args.output:
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(result, ensure_ascii=False))
