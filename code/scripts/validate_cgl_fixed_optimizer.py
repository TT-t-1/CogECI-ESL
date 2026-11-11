"""Minimum real-document forwards/backwards for newly added experiment paths."""
import json
import sys
from pathlib import Path
import torch
from transformers import BertTokenizer
from prepare_data import load_features

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'CogECI-cgl-fixed'))
from parameter import parse_args
from model import bertCSRModel
from utils import setup_seed

sys.argv = ['validate', '--run_dir', str(root / 'runs-cgl-fixed/maven_optimizer_smoke'), '--model_name', 'bert-base-uncased']
args = parse_args()
args.device = torch.device('cuda')
args.beta_intra, args.beta_inter = 0.3, 0.7
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', local_files_only=True)
esl = load_features(root / 'data/esl/EventStoryLine_bert_intra_and_inter_train_features.pkl')
maven = load_features(root / 'data/maven/MAVEN_bert_intra_and_inter_train_features.pkl')
small = min((f for f in esl if set(f.rel_type) == {0, 1}), key=lambda f: len(f.enc_text))
big = max(maven, key=lambda f: len(f.target) * len(f.sentences))
report = {'paths': [], 'maven_largest_pair_sentence_product_doc': big.doc_id}
reference = None
for variant in ['maven']:
    args.context_beta = 0.0 if variant == 'without_cgl' else 6.0
    args.no_sgm = variant == 'without_sgm'
    args.shared_dscrm = variant == 'shared_dscrm'
    args.sgm_layers = int(variant[-1]) if variant.startswith('layers') else 2
    args.graph_encoder = variant if variant in ['gcn', 'hypergraph', 'transformer'] else 'gat'
    setup_seed(209)
    model = bertCSRModel(args).cuda()
    f = big if variant == 'maven' else small
    model.train()
    # Use the real optimizer groups and one warm-up document to allocate Adam states.
    import ast
    from transformers import AdamW
    source = ast.parse((root / 'CogECI-cgl-fixed/main.py').read_text())
    names = {'no_decay', 'other_parameters', 'optimizer_grouped_parameters'}
    assignments = [n for n in ast.walk(source) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id in names for t in n.targets)]
    namespace = {'model': model, 'args': args}
    exec(compile(ast.Module(body=sorted(assignments, key=lambda n: n.lineno), type_ignores=[]), 'main.py', 'exec'), namespace)
    optimizer = AdamW(namespace['optimizer_grouped_parameters'], lr=args.lr, eps=args.adam_epsilon)
    warm_labels = torch.zeros(len(small.target), len(small.sentences), device='cuda')
    warm_loss, warm_prediction = model(small.enc_input_ids.cuda(), small.enc_mask_ids.cuda(), small.node_event,
                                      [small.t1_pos], [small.t2_pos], [small.target], [small.rel_type], small.event_pairs,
                                      warm_labels, warm_labels, 'train', small.doc_id, tokenizer)
    warm_loss.backward()
    optimizer.step()
    model.zero_grad()
    del warm_loss, warm_prediction, warm_labels

    torch.cuda.reset_peak_memory_stats()
    labels = torch.zeros(len(f.target), len(f.sentences), device='cuda')
    loss, prediction = model(f.enc_input_ids.cuda(), f.enc_mask_ids.cuda(), f.node_event,
                             [f.t1_pos], [f.t2_pos], [f.target], [f.rel_type], f.event_pairs,
                             labels, labels, 'train', f.doc_id, tokenizer)
    assert prediction.shape == (len(f.target), 2)
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    if variant in ['full', 'without_cgl']:
        gradients = {n: p.grad.detach().cpu().clone() for n, p in model.named_parameters() if p.grad is not None}
        if reference is None:
            reference = gradients
        else:
            assert gradients.keys() == reference.keys()
            assert all(torch.equal(gradients[n], reference[n]) for n in gradients)
            report['cgl_zero_vs_six_all_parameter_gradients_identical'] = True
            del reference, gradients
    report['paths'].append({'variant': variant, 'doc_id': f.doc_id, 'pairs': len(f.target),
                            'sentences': len(f.sentences), 'bert_length': f.enc_input_ids.shape[1],
                            'peak_memory_bytes': torch.cuda.max_memory_allocated()})
    print('passed', report['paths'][-1], flush=True)
    del optimizer, model, prediction, loss, labels
    torch.cuda.empty_cache()
directory = root / 'runs-cgl-fixed/maven_optimizer_smoke'
directory.mkdir(parents=True, exist_ok=True)
(directory / 'validation.json').write_text(json.dumps(report, indent=2))
