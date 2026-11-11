"""Portable ESL entry point. Run from the extracted submission directory."""
import argparse
import os
import subprocess
import sys
from pathlib import Path

package = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('experiment', choices=['E0', 'E1', 'E2'])
parser.add_argument('--fold', type=int, choices=range(1, 6), default=1)
parser.add_argument('--epoch', type=int, default=30)
parser.add_argument('--run-dir', type=Path, required=True)
parser.add_argument('--cache', type=Path, default=package / 'data_info/cache')
parser.add_argument('--gpus', default='0,1')
args = parser.parse_args()
args.run_dir = args.run_dir.resolve()
if args.run_dir.exists() and any(args.run_dir.iterdir()):
    raise SystemExit('Use a new empty run directory to preserve previous results.')
environment = os.environ.copy()
environment['COGECI_GPUS'] = args.gpus
if args.experiment == 'E0':
    command = [sys.executable, '-u', str(package / 'code/scripts/train_e0.py'),
               '--run_dir', str(args.run_dir), '--fold', str(args.fold), '--epoch', str(args.epoch),
               '--cache', str(args.cache.resolve())]
    cwd = package
else:
    model = {'E1': 'llama8b', 'E2': 'qwen7b'}[args.experiment]
    environment['COGECI_DATA_DIR'] = str(package / 'prompts' / model)
    command = [sys.executable, '-u', 'main.py', '--dataset_type', 'EventStoryLine',
               '--cache_path', str(args.cache.resolve()), '--model_name', 'bert-base-uncased',
               '--run_dir', str(args.run_dir), '--epoch', str(args.epoch), '--seed', '209',
               '--beta_intra', '.3', '--beta_inter', '.7', '--method', args.experiment]
    cwd = package / 'code/CogECI-cgl-fixed'
guard = package / 'code/scripts/guard.py'
subprocess.run([sys.executable, str(guard), str(args.run_dir), *command],
               cwd=cwd, env=environment, check=True)
