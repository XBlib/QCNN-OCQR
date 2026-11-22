"""汇总各次实验的测试结果。"""

import argparse,csv,json,statistics
from pathlib import Path
ARCHITECTURE='cifar10_4class_five_functional_qkernels_ocqr_overlap_consistency_context_difference_v1'
PROTOCOL='train2560_unused_train_val400_test400_final_only'
DEFAULT_OUTPUT_DIR='results/4class_ocqr_generalization_v1_manual'
FIELDS=['run','architecture','protocol','epochs_trained','best_val_epoch','best_val_loss',
        'best_val_acc_percent','test_acc_percent','test_loss','test_count','optimizer',
        'weight_decay','run_seconds','checkpoint','model_selection',
        'train_eval_acc_percent','train_eval_loss','train_test_accuracy_gap_pp',
        'test_evaluated_only_after_model_selection']
def gather_completed(root):
    rows=[]
    for d in sorted(root.glob('run_*')):
        if not d.is_dir() or not d.name[4:].isdigit():continue
        p=d/'final_result.json'
        if not p.exists():continue
        r=json.loads(p.read_text(encoding='utf-8'))
        if r.get('architecture')!=ARCHITECTURE or r.get('protocol')!=PROTOCOL:
            raise ValueError(f'Incompatible experiment result in {p}')
        if r['run']!=int(d.name[4:]):raise ValueError(f'Run number mismatch: {p}')
        rows.append(r)
    return rows

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',default=DEFAULT_OUTPUT_DIR)
    a=p.parse_args();root=Path(a.results)
    rows=gather_completed(root)
    if not rows:raise FileNotFoundError(f'No completed generalization runs at {root}')
    acc=[r['test_acc_percent'] for r in rows]
    print(f'Completed runs: {len(rows)}')
    for r in rows: print(f"run_{r['run']:02d}: selected epoch {r['best_val_epoch']} | TEST_ACC={r['test_acc_percent']:.2f}% | TEST_LOSS={r['test_loss']:.4f}")
    print(f"Mean test ACC={statistics.mean(acc):.2f}%; sample SD={statistics.stdev(acc):.2f}%" if len(rows)>1 else f"Single-run TEST_ACC={acc[0]:.2f}%")
    with (root/'runs_summary.csv').open('w',newline='',encoding='utf-8') as file:
        w=csv.DictWriter(file,fieldnames=FIELDS);w.writeheader();w.writerows(rows)
if __name__=='__main__':main()
