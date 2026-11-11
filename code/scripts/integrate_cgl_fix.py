"""Port the user-selected side-chat fix without overwriting the repository-priority copy."""
import ast
import difflib
import shutil
from pathlib import Path

root = Path(__file__).resolve().parents[1]
original = root / 'CogECI'
fixed = root / 'CogECI-cgl-fixed'
shutil.copytree(original, fixed, ignore=shutil.ignore_patterns('.git', 'logout', '__pycache__'))

def write(path, text):
    path.write_bytes(text.replace('\n', '\r\n').encode())

path = fixed / 'model/model.py'
text = path.read_text()
start = text.index('            # cosine_sim = F.softmax(cosine_sim,dim=1)')
end = text.index('            iteration += 1', start)
text = text[:start] + '''            # User-selected side-chat fix: continuous logits preserve context gradients.
            sen_loss = self.bce_loss(cosine_sim.float(), all_senlable.float())
            loss += (1 / (iteration + 1)) * w * sen_loss
''' + text[end:]
start = text.index('            event_expanded = event_pair.unsqueeze(1)')
end = text.index('            # User-selected side-chat fix:', start)
text = text[:start] + '''            # [pairs, dim] @ [dim, sentences] avoids expanded autograd buffers.
            normalized_pairs = F.normalize(event_pair, p=2, dim=1, eps=1e-8)
            normalized_sentences = F.normalize(sent_emb, p=2, dim=1, eps=1e-8)
            cosine_sim = normalized_pairs @ normalized_sentences.t()
''' + text[end:]
write(path, text)

path = fixed / 'main.py'
text = path.read_text()
start = text.index("        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']")
end = text.index('        optimizer = AdamW(', start)
text = text[:start] + '''        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        other_parameters = [(n, p) for n, p in model.named_parameters()
                            if not n.startswith('pretrained_model.')]
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.pretrained_model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay, 'lr': args.t_lr},
            {'params': [p for n, p in model.pretrained_model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0, 'lr': args.t_lr},
            {'params': [p for n, p in other_parameters if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in other_parameters if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]

''' + text[end:]
old = '    args = parse_args()'
assert text.count(old) == 1
text = text.replace(old, old + '''
    args.implementation = 'user_selected_cgl_gradient_fix'
    if args.context_threshold != '1/n':
        raise ValueError('The selected continuous CGL fix has no training threshold; this sweep is undefined.')''')
write(path, text)

# Isolate all new phase logs and classifier results. LLM raw-response caches remain reusable.
text = (root / 'scripts/pipeline.py').read_text()
text = text.replace("root / 'runs'", "root / 'runs-cgl-fixed'")
text = text.replace("root / 'CogECI'", "root / 'CogECI-cgl-fixed'")
text = text.replace("f'{name}_pipeline_status.json'", "f'{name}_cgl_fixed_pipeline_status.json'")
text = text.replace("'scripts/write_report.py'", "'scripts/write_report_cgl_fixed.py'")
start = text.index("        for cutoff in ['0.1', '0.5']:")
end = text.index("        for operator", start)
text = text[:start] + text[end:]
(root / 'scripts/pipeline_cgl_fixed.py').write_text(text)

text = (root / 'scripts/smoke.py').read_text()
text = text.replace("root / 'CogECI'", "root / 'CogECI-cgl-fixed'")
text = text.replace("root / 'runs'", "root / 'runs-cgl-fixed'")
text = text.replace("all(not x['loss_requires_grad'] for x in context)", "all(x['loss_requires_grad'] for x in context)")
text = text.replace("'optimizer_prefixes_from_original_main': list(optimized)", "'optimizer_policy': 'all model parameters, BERT separate learning rate'")
text = text.replace("'omitted_trainable_parameters':", "'parameters_omitted_by_original_optimizer_now_included':")
text = text.replace('original context loss has no gradient', 'corrected context loss has a gradient')
(root / 'scripts/smoke_cgl_fixed_runtime.py').write_text(text)

patch = []
for name in ['model/model.py', 'main.py']:
    patch.extend(difflib.unified_diff((original / name).read_text().splitlines(True),
                                    (fixed / name).read_text().splitlines(True),
                                    fromfile='a/' + name, tofile='b/' + name))
(root / 'cgl_gradient_fix_integrated.patch').write_text(''.join(patch))
for path in [fixed / 'model/model.py', fixed / 'main.py', root / 'scripts/pipeline_cgl_fixed.py']:
    ast.parse(path.read_text())
print('User-selected CGL fix integrated in', fixed, flush=True)
