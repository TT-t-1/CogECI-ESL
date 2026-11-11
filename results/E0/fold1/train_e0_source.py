"""Supervised BERT event-pair baseline on the existing ESL fold-1 split."""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch
from torch import nn
from transformers import AdamW, BertModel, get_linear_schedule_with_warmup

from prepare_data import load_features

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'CogECI-cgl-fixed'))
from model.model import focal_loss
from utils import calculate, compute_f1, setup_seed


class BertEventPair(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased', local_files_only=True)
        self.classifier = nn.Sequential(nn.Linear(2304, 2304), nn.ReLU(),
                                        nn.Dropout(0.4), nn.Linear(2304, 2))

    def forward(self, feature):
        inputs = (feature.enc_input_ids.cuda(), feature.enc_mask_ids.cuda())
        if feature.enc_input_ids.shape[0] == 1:
            hidden = self.bert(*inputs, return_dict=False)[0]
        else:
            hidden = nn.parallel.data_parallel(
                self.bert, inputs, device_ids=[0, 1], module_kwargs={'return_dict': False})[0]
        # Each event span is [start, end) in its sentence's BERT token sequence.
        events = torch.stack([hidden[int(sentence), int(span[0]):int(span[1])].mean(0)
                              for sentence, span in feature.node_event])
        pairs = torch.as_tensor(feature.event_pairs, dtype=torch.long, device=hidden.device)
        first, second = events[pairs[:, 0]], events[pairs[:, 1]]
        return self.classifier(torch.cat([first, second, first - second], dim=-1))


def evaluate(model, features, epoch):
    model.eval()
    gold, predicted, relations, documents = [], [], [], []
    with torch.no_grad():
        for feature in features:
            prediction = model(feature).argmax(-1).cpu().tolist()
            targets = [int(value) for value in feature.target]
            kinds = [int(value) for value in feature.rel_type]
            gold.extend(targets)
            predicted.extend(prediction)
            relations.extend(kinds)
            documents.append({'doc_id': feature.doc_id, 'predicted': prediction,
                              'gold': targets, 'rel_type': kinds})
    p, r, f1 = compute_f1(gold, predicted, logging.getLogger(__name__))
    _, _, _, intra, inter = calculate(gold, predicted, relations, epoch, lambda message: None)
    return {'p': p, 'r': r, 'f1': f1, 'intra': intra, 'inter': inter}, documents


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', required=True, type=Path)
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    output = args.run_dir
    output.mkdir(parents=True, exist_ok=True)
    assert not (output / 'args.json').exists(), 'Use a fresh run directory.'
    assert torch.cuda.device_count() == 2
    setup_seed(209)
    source = root / 'runs-cgl-fixed/llama8b_esl_main'
    split = json.loads((source / 'splits.json').read_text())[0]
    qwen_split = json.loads((root / 'runs-cgl-fixed/qwen7b_esl_main/splits.json').read_text())[0]
    assert split == qwen_split
    all_features = {}
    for partition in ['train', 'dev']:
        for feature in load_features(root / f'data/esl/EventStoryLine_bert_intra_and_inter_{partition}_features.pkl'):
            all_features[feature.doc_id] = feature
    documents = {partition: [all_features[doc_id] for doc_id in ids] for partition, ids in split.items()}
    if args.smoke:
        train = documents['train']
        documents = {partition: features[:2] for partition, features in documents.items()}
        documents['train'] = [min(train, key=lambda f: f.enc_input_ids.shape[0]),
                              max(train, key=lambda f: f.enc_input_ids.shape[0])]
    assert not set(split['train']) & set(split['test'])
    assert not set(split['train']) & set(split['dev'])
    assert not set(split['test']) & set(split['dev'])
    metadata = {'experiment': 'E0_BERT_event_pair', 'protocol': 'single_fixed_split',
                'reference_fold': 1, 'reference_splits': str(source / 'splits.json'),
                'seed': 209, 'epoch': args.epoch, 'smoke': args.smoke, 'batch_size': 1,
                'bert_lr': 2e-5, 'classifier_lr': 1e-4, 'weight_decay': 0.01,
                'warm_up_ratio': 0.1, 'max_grad_norm': 1.0,
                'loss': 'repository focal_loss alpha=0.75 gamma=2',
                'architecture': 'BERT, mean event-span pooling, [e1,e2,e1-e2], MLP 2304->2304->2 with dropout 0.4',
                'parallelism': 'BERT sentence encoding across two GPUs',
                'supervision': 'event-pair classification only',
                'document_counts': {key: len(value) for key, value in documents.items()}}
    (output / 'args.json').write_text(json.dumps(metadata, indent=2))
    (output / 'splits.json').write_text(json.dumps([
        {partition: [feature.doc_id for feature in features] for partition, features in documents.items()}], indent=2))
    model = BertEventPair().cuda()
    groups = []
    for prefix, learning_rate in [('bert.', 2e-5), ('classifier.', 1e-4)]:
        for decay in [True, False]:
            parameters = [parameter for name, parameter in model.named_parameters()
                          if name.startswith(prefix)
                          and (not any(part in name for part in ['bias', 'LayerNorm.weight'])) == decay]
            groups.append({'params': parameters, 'lr': learning_rate, 'weight_decay': 0.01 if decay else 0.0})
    optimizer = AdamW(groups, eps=1e-8)
    steps = len(documents['train']) * args.epoch
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=steps * 0.1,
                                               num_training_steps=steps)
    loss_function = focal_loss(alpha=0.75, gamma=2, num_classes=2)
    best_dev = -1.0
    started = time.time()
    for epoch in range(1, args.epoch + 1):
        epoch_started = time.time()
        model.train()
        losses = []
        for index in torch.randperm(len(documents['train'])).tolist():
            feature = documents['train'][index]
            optimizer.zero_grad(set_to_none=True)
            logits = model(feature)
            targets = torch.as_tensor(feature.target, dtype=torch.long, device='cuda')
            assert logits.shape == (len(feature.target), 2)
            loss = loss_function(logits, targets)
            assert torch.isfinite(loss)
            loss.backward()
            if args.smoke:
                assert model.bert.embeddings.word_embeddings.weight.grad is not None
                assert model.classifier[0].weight.grad.abs().sum() > 0
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
        dev, _ = evaluate(model, documents['dev'], epoch)
        if dev['f1'] >= best_dev:
            best_dev = dev['f1']
            test, predictions = evaluate(model, documents['test'], epoch)
            selected = {'selected_epoch': epoch, 'dev': dev, 'test': test}
            torch.save({'model': model.state_dict(), 'args': metadata, **selected}, output / 'fold_1_best.pt')
            (output / 'fold_1_predictions.json').write_text(json.dumps(predictions))
        record = {'epoch': epoch, 'train_loss': sum(losses) / len(losses), 'dev': dev,
                  'epoch_seconds': time.time() - epoch_started, 'elapsed_seconds': time.time() - started}
        with (output / 'epoch_metrics.jsonl').open('a') as handle:
            handle.write(json.dumps(record) + '\n')
        print(json.dumps(record), flush=True)
    if args.smoke:
        checkpoint = torch.load(output / 'fold_1_best.pt', map_location='cuda')
        model.load_state_dict(checkpoint['model'])
        _, predictions = evaluate(model, documents['test'], selected['selected_epoch'])
        assert predictions == json.loads((output / 'fold_1_predictions.json').read_text())
        print('SMOKE train/dev/test and checkpoint reload passed', flush=True)
    e1 = json.loads((source / 'fold_metrics.jsonl').read_text().splitlines()[0])
    result = {'protocol': 'single_fixed_split', 'reference_fold': 1, 'smoke': args.smoke,
              **selected, 'e1_llama8b_same_fold': e1, 'wall_seconds': time.time() - started}
    (output / 'results.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
