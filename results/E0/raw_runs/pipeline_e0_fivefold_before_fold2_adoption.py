"""Complete the remaining E0 folds and report an arithmetic five-fold mean."""
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

root = Path(__file__).resolve().parents[1]
run_root = root / 'runs-e0'
state_file = root / 'e0_esl_fivefold_status.json'
environment = os.environ.copy()
environment.update(COGECI_GPUS='1,0', HF_HUB_OFFLINE='1')
reference = json.loads((root / 'runs-cgl-fixed/llama8b_esl_main/splits.json').read_text())
e1 = [json.loads(line) for line in
      (root / 'runs-cgl-fixed/llama8b_esl_main/fold_metrics.jsonl').read_text().splitlines()]
started = time.time()


def report():
    folds = []
    for fold in range(1, 6):
        folder = run_root / f'esl_fold{fold}'
        if not (folder / 'status.json').exists():
            continue
        status = json.loads((folder / 'status.json').read_text())
        if status['exit_code'] or status['thermal_stop']:
            continue
        args = json.loads((folder / 'args.json').read_text())
        result = json.loads((folder / 'results.json').read_text())
        actual = json.loads((folder / 'splits.json').read_text())
        assert actual == [reference[fold - 1]], fold
        assert args['epoch'] == 30 and not args['smoke']
        assert args['reference_fold'] == result['reference_fold'] == fold
        epochs = [json.loads(line) for line in (folder / 'epoch_metrics.jsonl').read_text().splitlines()]
        assert [epoch['epoch'] for epoch in epochs] == list(range(1, 31))
        folds.append({'fold': fold, **result})
    result = {'protocol': 'five_fold_cross_validation', 'completed_folds': len(folds),
              'folds': folds, 'updated_unix': time.time(),
              'timing_scope': 'sum of epoch loops including training, validation, selected tests and checkpoint saves; model loading excluded',
              'completed_folds_seconds': sum(fold['wall_seconds'] for fold in folds)}
    lines = ['# E0 ESL 五折实验', '', f'已完成 {len(folds)}/5 折，每折30轮，双卡 BERT 句子编码并行。', '',
             '| 折 | P | R | F1 | 最佳epoch | 耗时（秒） |', '|---|---:|---:|---:|---:|---:|']
    for fold in folds:
        test = fold['test']
        lines.append(f'| {fold["reference_fold"]} | {test["p"]*100:.2f} | {test["r"]*100:.2f} | {test["f1"]*100:.2f} | {fold["selected_epoch"]} | {fold["wall_seconds"]:.2f} |')
    if len(folds) == 5:
        result['mean'] = {key: statistics.mean(fold['test'][key] for fold in folds) for key in ['p', 'r', 'f1']}
        result['f1_sample_std'] = statistics.stdev(fold['test']['f1'] for fold in folds)
        for relation in ['intra', 'inter']:
            result['mean'][relation] = {key: statistics.mean(fold['test'][relation][key] for fold in folds)
                                        for key in ['p', 'r', 'f1']}
        result['e1_llama8b_five_fold_mean'] = {key: statistics.mean(fold[key] for fold in e1)
                                             for key in ['p', 'r', 'f1']}
        mean = result['mean']
        comparator = result['e1_llama8b_five_fold_mean']
        lines += ['', f'五折算术平均 P {mean["p"]*100:.2f}，R {mean["r"]*100:.2f}，F1 {mean["f1"]*100:.2f}。',
                  f'逐折F1样本标准差 {result["f1_sample_std"]*100:.2f} 个百分点。',
                  f'句内F1 {mean["intra"]["f1"]*100:.2f}，跨句F1 {mean["inter"]["f1"]*100:.2f}。',
                  f'相同五折划分下，E1 Llama8B 总体F1为 {comparator["f1"]*100:.2f}。']
    lines += ['', f'已完成折次的训练与评估合计 {result["completed_folds_seconds"]:.2f} 秒。',
              '第1折复用此前已完成的实验，其余4折独立初始化，seed均为209。每折按验证F1选择模型。',
              'E0 为 BERT、事件跨度均值池化、[e1,e2,e1-e2] 与 MLP，仅使用事件对分类监督。',
              '数据划分取自既有 E1 五折，逐折耗时、预测、最佳模型和温度日志保存在各折目录。']
    (run_root / 'five_fold_results.json').write_text(json.dumps(result, indent=2))
    (run_root / 'FIVE_FOLD_RESULTS.md').write_text('\n'.join(lines) + '\n')
    return result


for fold in range(1, 6):
    folder = run_root / f'esl_fold{fold}'
    status_path = folder / 'status.json'
    if status_path.exists():
        status = json.loads(status_path.read_text())
        assert status['exit_code'] == 0 and not status['thermal_stop'], f'Inspect failed fold {fold} before retry'
        report()
        continue
    assert not (folder / 'args.json').exists(), f'Fold {fold} has an unfinished run'
    state_file.write_text(json.dumps({'phase': 'training', 'fold': fold, 'run_dir': str(folder),
                                      'updated_unix': time.time()}, indent=2))
    command = ['python3', str(root / 'scripts/guard.py'), str(folder),
               str(root.parent / 'env/bin/python'), '-u', str(root / 'scripts/train_e0.py'),
               '--run_dir', str(folder), '--epoch', '30', '--fold', str(fold)]
    code = subprocess.run(command, env=environment).returncode
    status = json.loads(status_path.read_text())
    if code or status['thermal_stop']:
        state_file.write_text(json.dumps({'phase': 'stopped', 'fold': fold, **status}, indent=2))
        raise SystemExit(1)
    report()

final = report()
assert final['completed_folds'] == 5
state_file.write_text(json.dumps({'phase': 'completed', 'completed_folds': 5,
                                  'additional_pipeline_wall_seconds': time.time() - started,
                                  'updated_unix': time.time()}, indent=2))
print(json.dumps(final), flush=True)
