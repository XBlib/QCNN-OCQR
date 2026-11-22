"""OCQR 五分类训练与评估。"""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader, Subset
import Tool
from generalization import build_validation_loader, evaluate, train_transform

ARCHITECTURE='cifar10_5class_five_functional_qkernels_ocqr_overlap_consistency_context_difference_v1'
PROTOCOL='train3200_unused_train_val500_test500_final_only'
DEFAULT_OUTPUT_DIR='results/5class_ocqr_generalization_v1_manual'

def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-id',type=int,default=None)
    p.add_argument('--epochs',type=int,default=50,help='Maximum epochs (original default 50)')
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--num-workers',type=int,default=8)
    p.add_argument('--torch-threads',type=int,default=16)
    p.add_argument('--data-root',default='../CIFAR-10/Train/')
    p.add_argument('--no-download',action='store_true')
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--output-dir',default=DEFAULT_OUTPUT_DIR)
    p.add_argument('--seed',type=int,default=2026,help='Torch/Python initial state (not original CIFAR subset indices)')
    p.add_argument('--val-seed',type=int,default=314159)
    p.add_argument('--val-per-class',type=int,default=100)
    p.add_argument('--augment',choices=['none','flip','flip-jitter'],default='flip')
    p.add_argument('--label-smoothing',type=float,default=0.05)
    p.add_argument('--weight-decay',type=float,default=1e-4)
    p.add_argument('--lr',type=float,default=1e-3)
    p.add_argument('--lr-patience',type=int,default=4,
                   help='Plateau epochs before LR reduction; 0 disables scheduler')
    p.add_argument('--lr-factor',type=float,default=0.5)
    p.add_argument('--min-lr',type=float,default=1e-5)
    p.add_argument('--early-stop-patience',type=int,default=15,
                   help='Epochs since best validation loss; 0 disables early stop')
    p.add_argument('--skip-train-eval',action='store_true',
                   help='Skip optional final TRAIN-set eval-mode diagnostic (saves inference time)')
    a=p.parse_args()
    if a.run_id is not None and a.run_id<1:p.error('--run-id >=1')
    if min(a.epochs,a.batch_size,a.torch_threads,a.val_per_class)<1:p.error('Positive numeric arguments required')
    if a.num_workers<0 or a.lr_patience<0 or a.early_stop_patience<0:p.error('Patience/workers must be >=0')
    if not (0 <= a.label_smoothing < 1):p.error('--label-smoothing in [0,1)')
    if a.weight_decay<0 or a.lr<=0 or a.min_lr<0:p.error('Invalid optimizer setting')
    if not (0<a.lr_factor<1):p.error('--lr-factor must be in (0,1)')
    return a

def allocate_run_dir(root,run_id=None):
    root.mkdir(parents=True,exist_ok=True)
    if run_id is not None:
        d=root/f'run_{run_id:02d}'; d.mkdir(parents=True,exist_ok=False)
        return run_id,d
    i=1
    while True:
        d=root/f'run_{i:02d}'
        try:d.mkdir(parents=True,exist_ok=False);return i,d
        except FileExistsError:i+=1

def make_model(num_classes=5):
    from OCQRConvolutionLayer import OCQR_Conv2d, OCQRClassifierHead
    return torch.nn.Sequential(
        OCQR_Conv2d(in_channels=3, out_channels=5, kernel_size=2, stride=1),
        OCQRClassifierHead(num_classes=num_classes),
    )

def main():
    a=parse_args()
    rid,folder=allocate_run_dir(Path(a.output_dir).resolve(),a.run_id)
    (folder/'Models_V1').mkdir()
    (folder/'acc_and_loss').mkdir()
    if a.device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    device=torch.device('cuda:0' if torch.cuda.is_available() and a.device!='cpu' else 'cpu')
    torch.set_num_threads(a.torch_threads)
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(a.seed)
    Tool.Path=a.data_root
    Tool.Download=not a.no_download
    tr,te,_=Tool.DataLoader_1280(batch_size=a.batch_size,num_workers=a.num_workers,
                                 input_size=10,Train_num=640,Test_num=100,category=5)
    selected_train=np.asarray(tr.dataset.indices,dtype=int)
    selected_test=np.asarray(te.dataset.indices,dtype=int)
    val,selected_val=build_validation_loader(
        data_root=a.data_root,download=not a.no_download,
        train_indices=selected_train,num_workers=a.num_workers,
        per_class=a.val_per_class,seed=a.val_seed,
        image_size=10,batch_size=max(1,a.val_per_class*5),num_classes=5)
    if len(tr.dataset)!=3200 or len(te.dataset)!=500:
        raise ValueError('Unexpected 5-class sample counts; expected 3200 train and 500 test images')
    if len(val.dataset)!=a.val_per_class*5:
        raise ValueError('Unexpected validation count; expected val_per_class × 10 classes')
    train_class_counts=np.bincount(np.asarray(tr.dataset.dataset.targets)[selected_train],minlength=5)
    val_class_counts=np.bincount(np.asarray(val.dataset.dataset.targets)[selected_val],minlength=5)
    test_class_counts=np.bincount(np.asarray(te.dataset.dataset.targets)[selected_test],minlength=5)
    if not np.array_equal(train_class_counts,np.full(5,640)):
        raise ValueError(f'Unexpected train class balance: {train_class_counts.tolist()}')
    if not np.array_equal(val_class_counts,np.full(5,a.val_per_class)):
        raise ValueError(f'Unexpected validation class balance: {val_class_counts.tolist()}')
    if not np.array_equal(test_class_counts,np.full(5,100)):
        raise ValueError(f'Unexpected test class balance: {test_class_counts.tolist()}')
    tr.dataset.dataset.transform=train_transform(10,a.augment)
    (folder/'sample_indices.json').write_text(json.dumps({
        'train_indices':selected_train.tolist(),
        'validation_indices':selected_val.tolist(),
        'test_indices':selected_test.tolist(),
        'num_classes':5,
        'selected_class_ids':list(range(5)),
        'sampler':'legacy_six_class_rng_seed0_keep_first_5',
        'train_per_class':640,'validation_per_class':a.val_per_class,'test_per_class':100,
        'validation_origin':'UNUSED CIFAR10 train=True; excluded all 3200 selected train indices',
        'test_origin':'selected CIFAR10 train=False 100/class',
        'train_class_counts':train_class_counts.tolist(),
        'validation_class_counts':val_class_counts.tolist(),
        'test_class_counts':test_class_counts.tolist(),
    },ensure_ascii=False),encoding='utf-8')
    model=make_model().to(device)
    print(device); print(model)
    train_ce=torch.nn.CrossEntropyLoss(label_smoothing=a.label_smoothing).to(device)
    optimizer=optim.Adam(model.parameters(),lr=a.lr,weight_decay=a.weight_decay)
    scheduler=(optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=a.lr_factor,
               patience=a.lr_patience,min_lr=a.min_lr) if a.lr_patience else None)
    cfg={
        'architecture':ARCHITECTURE,'protocol':PROTOCOL,'run':rid,
        'train_count':len(tr.dataset),'val_count':len(val.dataset),'test_count':len(te.dataset),
        'train_class_counts':train_class_counts.tolist(),
        'validation_class_counts':val_class_counts.tolist(),
        'test_class_counts':test_class_counts.tolist(),
        'data_root':a.data_root,'seed':a.seed,'sample_seed':0,'val_seed':a.val_seed,
        'selected_class_ids':list(range(5)),
        'sampler':'legacy_six_class_rng_seed0_keep_first_5',
        'epochs_max':a.epochs,'batch_size':a.batch_size,'num_workers':a.num_workers,
        'test_batch_size':te.batch_size,'head_dim':5,'device':str(device),
        'augment_train_only':a.augment,'label_smoothing_train_only':a.label_smoothing,
        'optimizer':type(optimizer).__name__,
        'initial_lr':optimizer.param_groups[0]['lr'],
        'weight_decay_actual':optimizer.param_groups[0]['weight_decay'],
        'lr_scheduler':'ReduceLROnPlateau' if scheduler else 'none',
        'lr_patience':a.lr_patience,'lr_factor':a.lr_factor,'min_lr':a.min_lr,
        'early_stop_patience':a.early_stop_patience,
        'model_selection':'best_validation_loss_no_test_selection',
        'test_evaluations_during_training':0,'independent_run_manually_started':True,
        'train_eval_diagnostic_at_end':not a.skip_train_eval,
        'architecture_fusion':'OCQR_overlap_consistent_and_context_difference',
        'qubit_positions':{'q0':'top_left','q1':'top_right','q2':'bottom_left','q3':'bottom_right'},
        'ocqr_quantum_patch_size':2,
        'ocqr_quantum_stride':1,
        'ocqr_quantum_qnodes_per_patch':13,
        'ocqr_single_z_features_per_separate_color':4,
        'ocqr_connected_zz_pairs_per_branch':[4,4,6,6,6],
        'ocqr_readout_dim_per_branch':8,
        'ocqr_aggregation':'same_pixel_overlap_mean_and_signed_RMS_context_difference',
        'ocqr_spatial_output_channels':40,
        'ocqr_spatial_output_size':[10,10],
        'ocqr_spatial_context':'depthwise_3x3_and_dilated_depthwise_3x3',
        'classical_head':'avg_and_max_2x2_to_64_to_logits',
        'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
        'class_names':['airplane', 'automobile', 'bird', 'cat', 'deer'],
    }
    (folder/'experiment_config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf-8')
    history=[]; best_loss=float('inf'); best_epoch=None; non_improve=0
    start=time.time()
    print(f'OCQR GENERALIZATION | manual run {rid:02d} | train {len(tr.dataset)} / val {len(val.dataset)} / held-out test {len(te.dataset)}')
    print('Test set is NOT evaluated per epoch. Checkpoint selection uses validation LOSS only.')
    print(f'Adam lr={a.lr} weight_decay={a.weight_decay} | augment={a.augment} | label_smoothing={a.label_smoothing}')
    for ep in range(1,a.epochs+1):
        model.train(); loss_running=0.0; train_sum=0.0; nb=0
        lr_used=optimizer.param_groups[0]['lr']
        for step,(inputs,labels) in enumerate(tr,1):
            inputs=inputs.to(device);labels=labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs=model(inputs)
            loss=train_ce(outputs,labels)
            loss.backward();optimizer.step()
            train_sum+=loss.item();nb+=1;loss_running+=loss.item()
            if step % 10==0:
                print('[epoch:%d, step:%5d] loss: %f' % (ep,step,loss_running),flush=True)
                loss_running=0.0
        val_scores=evaluate(model,val,device)
        if scheduler:scheduler.step(val_scores['loss'])
        improved=val_scores['loss'] < (best_loss-1e-4)
        if improved:
            best_loss=val_scores['loss'];best_epoch=ep;non_improve=0
            torch.save(model,folder/'Models_V1'/'model_best.pt')
            torch.save(model.state_dict(),folder/'Models_V1'/'state_dict_best.pt')
        else:non_improve+=1
        history.append({
            'epoch':ep,'train_loss_smoothed_CE':train_sum/max(nb,1),
            'val_acc_percent':val_scores['acc_percent'],'val_loss_hard_CE':val_scores['loss'],
            'lr_used':lr_used,'next_lr':optimizer.param_groups[0]['lr'],
            'best_epoch_so_far':best_epoch,'epochs_without_val_improvement':non_improve
        })
        print(f"Epoch {ep:02d} | VAL_ACC={val_scores['acc_percent']:.2f}% | VAL_LOSS={val_scores['loss']:.4f} | LR={lr_used:.6g} | best_val_epoch={best_epoch}",flush=True)
        model.train()
        torch.save(model,folder/'Models_V1'/f'model_{ep-1}.pt')
        if a.early_stop_patience and non_improve>=a.early_stop_patience:
            print(f'EARLY STOP at epoch {ep} (validation loss patience={a.early_stop_patience})',flush=True)
            break
    if best_epoch is None:raise RuntimeError('No best validation model saved')
    with (folder/'acc_and_loss'/'training_history.csv').open('w',encoding='utf-8',newline='') as file:
        wr=csv.DictWriter(file,fieldnames=list(history[0]));wr.writeheader();wr.writerows(history)
    (folder/'acc_and_loss'/'val_acc.txt').write_text(''.join(f"{r['val_acc_percent']}\n" for r in history),encoding='utf-8')
    (folder/'acc_and_loss'/'val_loss.txt').write_text(''.join(f"{r['val_loss_hard_CE']}\n" for r in history),encoding='utf-8')
    # 根据验证损失选定模型，随后评估测试集。
    best_model=torch.load(folder/'Models_V1'/'model_best.pt',map_location=device,weights_only=False)
    train_eval=None
    if not a.skip_train_eval:
        train_eval_loader=DataLoader(
            Subset(val.dataset.dataset,selected_train.tolist()),
            batch_size=max(1,a.val_per_class*5),shuffle=False,
            num_workers=a.num_workers,pin_memory=True)
        train_eval=evaluate(best_model,train_eval_loader,device)
    final_test=evaluate(best_model,te,device)
    result={
        'run':rid,'architecture':ARCHITECTURE,'protocol':PROTOCOL,
        'epochs_trained':len(history),'best_val_epoch':best_epoch,
        'best_val_loss':best_loss,'best_val_acc_percent':history[best_epoch-1]['val_acc_percent'],
        'test_acc_percent':final_test['acc_percent'],'test_loss':final_test['loss'],
        'test_count':final_test['count'],
        'train_eval_acc_percent':train_eval['acc_percent'] if train_eval else None,
        'train_eval_loss':train_eval['loss'] if train_eval else None,
        'train_test_accuracy_gap_pp':(train_eval['acc_percent']-final_test['acc_percent']) if train_eval else None,
        'optimizer':'Adam',
        'weight_decay':optimizer.param_groups[0]['weight_decay'],
        'run_seconds':time.time()-start,'checkpoint':'Models_V1/model_best.pt',
        'model_selection':'validation_loss', 'test_evaluated_only_after_model_selection':True,
    }
    (folder/'final_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    if train_eval:
        print(f"FINAL TRAIN (eval mode) ACC={train_eval['acc_percent']:.2f}% | LOSS={train_eval['loss']:.4f}",flush=True)
    print(f"FINAL (val-selected epoch {best_epoch}) | TEST_ACC={final_test['acc_percent']:.2f}% | TEST_LOSS={final_test['loss']:.4f}",flush=True)
    print(f'Saved to {folder}. Start manually again for another independent run.')

if __name__=='__main__': main()
