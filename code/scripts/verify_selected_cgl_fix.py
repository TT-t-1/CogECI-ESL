"""Replay the side-chat proof against the actual integrated source files."""
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sources = {
    'original_model': subprocess.check_output(['git', 'show', 'HEAD:model/model.py'], cwd=root / 'CogECI', text=True),
    'fixed_model': (root / 'CogECI-cgl-fixed/model/model.py').read_text(),
    'fixed_main': (root / 'CogECI-cgl-fixed/main.py').read_text(),
}
proof = (root / 'scripts/side_chat_smoke_cgl.py').read_text()
proof = proof.replace("self.device = torch.device('cpu')",
                      "self.device = torch.device('cpu')\n        self.context_beta = 6.0")
exec(compile(proof, 'side_chat_smoke_cgl.py', 'exec'), {'sources': sources})
