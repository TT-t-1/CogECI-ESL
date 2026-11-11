"""Audit ESL caches and adapt official MAVEN-ERE mentions to CogECI's feature contract."""
import hashlib
import itertools
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from transformers import BertTokenizer

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'CogECI-cgl-fixed'))
from data.process_esc_csr import ESC_features

class FeatureUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'ESC_features':
            return ESC_features
        return super().find_class(module, name)

def load_features(path):
    with Path(path).open('rb') as handle:
        return FeatureUnpickler(handle).load()

def stats(features):
    return {'documents': len(features), 'events': sum(len(f.node_event) for f in features),
            'pairs': sum(len(f.target) for f in features),
            'positive': sum(sum(t != 0 for t in f.target) for f in features),
            'positive_by_type': dict(Counter('inter' if r else 'intra' for f in features
                                            for t, r in zip(f.target, f.rel_type) if t)),
            'max_sentences': max(len(f.sentences) for f in features),
            'max_bert_length': max(f.enc_input_ids.shape[1] for f in features)}

def maven_feature(doc, tokenizer):
    # MAVEN annotates event chains. Expand each annotated chain relation to its mention pairs.
    mentions = [(event['id'], mention) for event in doc['events'] for mention in event['mention']]
    mentions.sort(key=lambda item: (item[1]['sent_id'], item[1]['offset'], item[1]['id']))
    chains = [item[0] for item in mentions]
    relation = {tuple(pair) for kind in ['CAUSE', 'PRECONDITION']
                for pair in doc['causal_relations'][kind]}
    # Preserve full sentences up to BERT's native 512-token limit; never silently drop triggers.
    encoded = tokenizer(doc['tokens'], is_split_into_words=True, padding=True,
                        truncation=False, return_tensors='pt')
    if encoded['input_ids'].shape[1] > 512:
        raise ValueError(f"MAVEN sentence exceeds BERT's 512-token limit: {doc['id']}")
    nodes = []
    for _, mention in mentions:
        sid = mention['sent_id']
        a, b = mention['offset']
        word_ids = encoded.word_ids(sid)
        positions = [i for i, word in enumerate(word_ids) if word is not None and a <= word < b]
        assert positions, (doc['id'], mention['id'])
        nodes.append([sid, [positions[0], positions[-1] + 1]])
    pairs = list(itertools.combinations(range(len(nodes)), 2))
    targets = [int((chains[a], chains[b]) in relation or (chains[b], chains[a]) in relation)
               for a, b in pairs]
    feature = ESC_features('MAVEN', doc['id'], ' '.join(doc['sentences']), [], doc['tokens'],
                           encoded['input_ids'], encoded['attention_mask'], nodes,
                           [a for a, b in pairs], [b for a, b in pairs], targets,
                           [int(nodes[a][0] != nodes[b][0]) for a, b in pairs], pairs)
    feature.event_mentions = [m for _, m in mentions]
    return feature

def main():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', local_files_only=True)
    out = root / 'data'
    out.mkdir(exist_ok=True)
    report = {'seed': 209, 'sources': {}, 'ESL': {}, 'MAVEN': {}}
    esl = out / 'esl'
    esl.mkdir(exist_ok=True)
    for split in ['train', 'dev']:
        source = root.parent / 'iLIF/data/database/exist' / f'EventStoryLine_bert_intra_and_inter_{split}_features.pkl'
        features = load_features(source)
        assert all(len(f.target) == len(f.event_pairs) == len(f.rel_type) for f in features)
        assert all(0 <= int(a) < int(b) <= f.enc_input_ids.shape[1]
                   for f in features for _, (a, b) in f.node_event)
        with (esl / f'EventStoryLine_bert_intra_and_inter_{split}_features.pkl').open('wb') as h:
            pickle.dump(features, h)
        report['ESL'][split] = stats(features)
        report['sources'][str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
    maven = out / 'maven'
    maven.mkdir(exist_ok=True)
    # Fast tokenizer supplies exact word -> WordPiece offsets, preserving mention spans.
    from transformers import BertTokenizerFast
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', local_files_only=True)
    documents = {}
    for original in ['train', 'valid']:
        source = root / 'sources/MAVEN_ERE' / f'{original}.jsonl'
        report['sources'][str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
        documents[original] = [json.loads(line) for line in source.read_text().splitlines()]
    indices = np.random.RandomState(209).permutation(len(documents['train']))
    dev_count = int(np.ceil(len(indices) * 0.1))
    dev_indices = set(indices[:dev_count].tolist())
    split_docs = {'train': [d for i, d in enumerate(documents['train']) if i not in dev_indices],
                  'dev': [d for i, d in enumerate(documents['train']) if i in dev_indices],
                  'test': documents['valid']}
    (maven / 'split_ids.json').write_text(json.dumps({s: [d['id'] for d in ds] for s, ds in split_docs.items()}, indent=2))
    all_features = {}
    for split, docs in split_docs.items():
        features = [maven_feature(d, tokenizer) for d in docs]
        for feature in features:
            feature.data_split = split
        # A document with fewer than two mentions contributes no relation examples.
        no_pairs = [f.doc_id for f in features if not f.event_pairs]
        features = [f for f in features if f.event_pairs]
        all_features[split] = features
        with (maven / f'MAVEN_bert_intra_and_inter_{split}_features.pkl').open('wb') as h:
            pickle.dump(features, h)
        report['MAVEN'][split] = stats(features)
        report['MAVEN'][split]['zero_pair_documents'] = no_pairs
        print('MAVEN', split, report['MAVEN'][split], flush=True)
    ids = [set(d['id'] for d in ds) for ds in split_docs.values()]
    assert all(not (a & b) for a, b in itertools.combinations(ids, 2))
    report['choices'] = {
        'ESL': 'iLIF binary exist cache; repository document KFold; published positive count differs',
        'MAVEN_split': 'paper: original valid used as test, 10% train dev; our choice: seed209, ceil rounding, fixed ids',
        'MAVEN_pairs': 'our adapter: all unordered mention pairs, chain relations expanded to mentions, binary existence',
        'MAVEN_tokens': 'our adapter: dynamic full-sentence padding, no 105-token truncation; native BERT max512',
    }
    (out / 'audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)

if __name__ == '__main__':
    main()
