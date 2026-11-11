"""Run a task on the explicitly selected GPUs with continuous temperature monitoring."""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

run_dir = Path(sys.argv[1]).resolve()
command = sys.argv[2:]
run_dir.mkdir(parents=True, exist_ok=True)

gpus = [int(gpu) for gpu in os.environ.get('COGECI_GPUS', '1').split(',')]

def temperature():
    return {gpu: int(subprocess.check_output([
        'nvidia-smi', '-i', str(gpu), '--query-gpu=temperature.gpu',
        '--format=csv,noheader,nounits'], text=True).strip()) for gpu in gpus}

baseline = temperature()
if max(baseline.values()) >= 85:
    raise SystemExit(f'GPU at or above 85 C; task was not started: {baseline}')
environment = os.environ.copy()
environment.update(CUDA_VISIBLE_DEVICES=','.join(map(str, gpus)),
                   TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4')
(run_dir / 'command.json').write_text(json.dumps({'argv': command, 'gpus': gpus, 'baseline_c': baseline}, indent=2))
with (run_dir / 'task.log').open('a') as log, (run_dir / 'temperature.csv').open('a') as monitor:
    process = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT,
                               start_new_session=True)
    (run_dir / 'pid').write_text(str(process.pid))
    monitor.write('unix_time,gpu,temperature_c,task_pid\n')
    stopped = False
    try:
        while process.poll() is None:
            temp = temperature()
            for gpu, value in temp.items():
                monitor.write(f'{time.time()},{gpu},{value},{process.pid}\n')
            monitor.flush()
            if max(temp.values()) >= 85:
                stopped = True
                os.killpg(process.pid, signal.SIGTERM)
                break
            time.sleep(10)
    finally:
        if process.poll() is None:
            if not stopped:
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
        code = process.wait()
        (run_dir / 'status.json').write_text(json.dumps({'exit_code': code, 'thermal_stop': stopped}))
raise SystemExit(code if code >= 0 else 1)
