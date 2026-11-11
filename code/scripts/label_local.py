"""Two-stage local generation, preserving author prompts and explicit cached pair order."""
import argparse
import ast
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BertTokenizer

from prepare_data import load_features

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'CogECI-cgl-fixed'))
from llm.run_llm_processing import _prompt_select_sentences, _parse_last_bracket_list

def event_prompt(numbered, trigger, sentence):
    return [
        {'role': 'system', 'content': 'You are an expert with few words and will only answer questions accurately'},
        {'role': 'user', 'content': (
            f'I have a document and an event word who is in sentence ({sentence}) in the document, '
            f'please fully understand this event and generat a subject-verb-object short sentence. '
            f'It is possible to describe this event. Output only this sentence. '
            f'document:{list(numbered)} event:{trigger}')},
    ]

def parse_indices(answer):
    bracket = _parse_last_bracket_list(answer)
    try:
        indices = ast.literal_eval(bracket) if bracket else None
    except (ValueError, SyntaxError):
        return None
    if not isinstance(indices, list) or not all(type(x) is int for x in indices):
        return None
    return sorted(set(indices))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--name', choices=['qwen7b', 'llama8b'], required=True)
    parser.add_argument('--dataset', choices=['esl', 'maven'], required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--shard', type=int, choices=[0, 1])
    parser.add_argument('--base_cache', type=Path)
    parser.add_argument('--output_dir', type=Path)
    parser.add_argument('--memory_fraction', type=float)
    parser.add_argument('--export_only', action='store_true')
    args = parser.parse_args()
    torch.manual_seed(209)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    bert = BertTokenizer.from_pretrained('bert-base-uncased', local_files_only=True)
    data = root.parent / 'data_info/cache' if args.dataset == 'esl' else root / 'data' / args.dataset
    prefix = 'EventStoryLine' if args.dataset == 'esl' else 'MAVEN'
    features = {s: load_features(data / f'{prefix}_bert_intra_and_inter_{s}_features.pkl')
                for s in ['train', 'dev']}
    if args.plan:
        selected = json.loads(args.plan.read_text())['shards'][args.shard]
        for split in features:
            ids = {item['doc_id'] for item in selected if item['split'] == split}
            features[split] = [f for f in features[split] if f.doc_id in ids]
    if args.smoke:
        # Exercise real positive/negative, intra/inter documents; include the longest prompt.
        for split in features:
            candidates = [f for f in features[split] if any(f.target) and set(f.rel_type) == {0, 1}]
            shortest = min(candidates, key=lambda f: len(f.enc_text))
            longest = max(candidates, key=lambda f: len(tokenizer.encode(f.enc_text)))
            features[split] = [shortest, longest] if shortest.doc_id != longest.doc_id else [shortest]
    out = root / 'labels' / args.name / args.dataset / ('smoke' if args.smoke else 'full')
    if args.output_dir:
        out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    records = {}
    cache = out / 'responses.jsonl'
    for cached in [args.base_cache, cache]:
        if cached and cached.exists():
            with cached.open() as handle:
                for line in handle:
                    if not line.endswith('\n'):
                        continue  # A snapshot may end during the other worker's final write.
                    record = json.loads(line)
                    if record['id'] in records:
                        assert records[record['id']] == record, record['id']
                    records[record['id']] = record
    max_length = 0
    longest_messages = None
    for docs in features.values():
        for f in docs:
            numbered = [f'{i}. {" ".join(s)}' for i, s in enumerate(f.sentences)]
            message = event_prompt(numbered, 'event', 0)
            count = len(tokenizer.apply_chat_template(message, tokenize=True, add_generation_prompt=True))
            if count > max_length:
                max_length, longest_messages = count, message
    if not args.export_only:
        if args.memory_fraction:
            torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                    device_map={'': 0}, local_files_only=True)
        model.eval()
        context_limit = model.config.max_position_embeddings
    terminators = [tokenizer.eos_token_id]
    if args.name == 'llama8b':
        terminators.append(tokenizer.convert_tokens_to_ids('<|eot_id|>'))

    def generate(messages, key, extra):
        digest = hashlib.sha256(json.dumps(messages, ensure_ascii=False).encode()).hexdigest()
        if key in records:
            assert records[key]['prompt_sha256'] == digest, key
            return records[key]
        if args.export_only:
            raise ValueError(f'Missing label during cache-only export: {key}')
        inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                               return_tensors='pt', return_dict=True).to('cuda')
        n = inputs['input_ids'].shape[1]
        if n + 256 > context_limit:
            raise ValueError(f'{key}: {n}+256 > context limit {context_limit}; no truncation applied')
        start = time.monotonic()
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=256, do_sample=True,
                                    temperature=0.6, top_p=0.9, eos_token_id=terminators,
                                    pad_token_id=tokenizer.eos_token_id)
        answer = tokenizer.decode(output[0, n:], skip_special_tokens=True).strip()
        record = {'id': key, 'model': args.name, 'prompt_sha256': digest, 'messages': messages,
                  'answer': answer, 'input_tokens': n, 'output_tokens': output.shape[1] - n,
                  'seconds': time.monotonic() - start, **extra}
        with cache.open('a') as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
        records[key] = record
        if len(records) % 25 == 0:
            print('completed', len(records), 'last', key, flush=True)
        return record

    # Run the longest actual prompt before thousands of calls. It is not a label and is marked.
    if not args.export_only:
        probe = 'memory_probe' if not args.plan else f'memory_probe_shard_{args.shard}'
        generate(longest_messages, probe, {'stage': 'probe', 'max_document_prompt_tokens': max_length})
    summary = {'model': args.name, 'dataset': args.dataset, 'smoke': args.smoke,
               'max_document_prompt_tokens': max_length, 'invalid': [],
               'out_of_range_contexts': [], 'documents': {},
               'sampling': {'seed': 209, 'temperature': 0.6, 'top_p': 0.9, 'max_new_tokens': 256},
               'choices': ['both stages use the named local model',
                           'one shared context list exported for both sen and sen_more; upstream distinction unavailable',
                           'iterate actual cached event_pairs, rather than reconstruct combinations',
                           'decode only completion tokens; malformed answers become None as in the original generator, with an audit record',
                           'out-of-range integers are audited and ignored, matching original loader range loop']}
    for split, docs in features.items():
        rows = []
        for f in docs:
            numbered = [f'{i}. {" ".join(s)}' for i, s in enumerate(f.sentences)]
            triggers = [bert.decode(f.enc_input_ids[int(sid), int(a):int(b)], clean_up_tokenization_spaces=True)
                        for sid, (a, b) in f.node_event]
            descriptions = {}
            event_ids = range(min(2, len(triggers))) if args.smoke else range(len(triggers))
            # Smoke additionally describes every event used by the selected positive pairs.
            pair_indices = [i for i, t in enumerate(f.target) if t]
            if args.smoke:
                pair_indices = [next(i for i in pair_indices if f.rel_type[i] == r)
                                for r in (0, 1) if any(f.rel_type[i] == r for i in pair_indices)]
                event_ids = sorted(set(event_ids) | {e for i in pair_indices for e in f.event_pairs[i]})
            for event in event_ids:
                record = generate(event_prompt(numbered, triggers[event], int(f.node_event[event][0])),
                                  f'{split}/{f.doc_id}/event/{event}', {'stage': 'event'})
                descriptions[event] = record['answer']
            answers = ['None'] * len(f.target)
            for i in pair_indices:
                a, b = f.event_pairs[i]
                messages = _prompt_select_sentences(numbered, triggers[a], triggers[b], int(f.target[i]),
                                                    int(f.rel_type[i]), int(f.node_event[a][0]),
                                                    int(f.node_event[b][0]), descriptions[a], descriptions[b])
                record = generate(messages, f'{split}/{f.doc_id}/pair/{i}',
                                  {'stage': 'context', 'pair': [int(a), int(b)], 'pair_index': i})
                indices = parse_indices(record['answer'])
                if indices is None:
                    summary['invalid'].append(record['id'])
                    print('repository parse semantics: None for malformed answer', record['id'], flush=True)
                    continue
                ignored = [x for x in indices if not 0 <= x < len(f.sentences)]
                if ignored:
                    summary['out_of_range_contexts'].append({'id': record['id'], 'ignored': ignored,
                                                              'sentence_count': len(f.sentences)})
                    print('repository range semantics: ignored', ignored, 'for', record['id'], flush=True)
                indices = [x for x in indices if 0 <= x < len(f.sentences)]
                answers[i] = ','.join(map(str, sorted(set(indices)))) if indices else 'None'
            rows.append([len(rows), f.doc_id, f.enc_text, f.sentences, '',
                         [list(p) for p in f.event_pairs], answers])
        csv_dir = out / 'examples/llm'
        csv_dir.mkdir(parents=True, exist_ok=True)
        for suffix in ['sen', 'sen_more']:
            with (csv_dir / f'llama3help_{split}_{suffix}.csv').open('w', newline='') as handle:
                writer = csv.writer(handle)
                writer.writerow(['id', 'doc_id', 'text', 'sentences', 'event_pair', 'node_event_id', 'answer(id)'])
                writer.writerows(rows)
        summary['documents'][split] = [f.doc_id for f in docs]
    summary['records'] = len(records)
    if args.dataset == 'maven' and not args.smoke:
        # Public-valid test labels never enter LLM generation or training.
        test = load_features(data / 'MAVEN_bert_intra_and_inter_test_features.pkl')
        for suffix in ['sen', 'sen_more']:
            with (out / f'examples/llm/llama3help_test_{suffix}.csv').open('w', newline='') as handle:
                writer = csv.writer(handle)
                writer.writerow(['id', 'doc_id', 'text', 'sentences', 'event_pair', 'node_event_id', 'answer(id)'])
                for i, f in enumerate(test):
                    writer.writerow([i, f.doc_id, '', '', '', [list(p) for p in f.event_pairs], ['None'] * len(f.target)])
        summary['choices'].append('MAVEN test context arrays are zero placeholders; no test-gold LLM calls')
    summary['seconds'] = sum(r['seconds'] for r in records.values())
    summary['max_gpu_memory_bytes'] = 0 if args.export_only else torch.cuda.max_memory_allocated()
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)

if __name__ == '__main__':
    main()
