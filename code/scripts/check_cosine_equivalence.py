"""Check actual score/gradient equivalence for the memory adaptation necessitated by 3090 OOM."""
import json
from pathlib import Path
import torch
from torch.nn import functional as F

torch.manual_seed(209)
checks = []
for dtype in [torch.float32, torch.float64]:
    pairs = torch.randn(7, 16, dtype=dtype, requires_grad=True)
    sentences = torch.randn(5, 16, dtype=dtype, requires_grad=True)
    # Include the zero-vector edge case without changing live differentiable inputs.
    with torch.no_grad():
        pairs[0].zero_()
    original = F.cosine_similarity(pairs[:, None, :], sentences[None, :, :], dim=2)
    adapted = F.normalize(pairs, dim=1, eps=1e-8) @ F.normalize(sentences, dim=1, eps=1e-8).t()
    weights = torch.randn_like(original)
    old_grad = torch.autograd.grad((original * weights).sum(), (pairs, sentences), retain_graph=True)
    new_grad = torch.autograd.grad((adapted * weights).sum(), (pairs, sentences))
    assert torch.allclose(original, adapted, atol=1e-6, rtol=1e-5)
    assert all(torch.allclose(a, b, atol=1e-6, rtol=1e-5) for a, b in zip(old_grad, new_grad))
    checks.append({'dtype': str(dtype), 'scores_match': True, 'gradients_match': True,
                   'max_score_error': (original - adapted).abs().max().item()})
root = Path(__file__).resolve().parents[1]
(root / 'runs-cgl-fixed/cosine_equivalence.json').write_text(json.dumps(checks, indent=2))
print(json.dumps(checks, indent=2))
