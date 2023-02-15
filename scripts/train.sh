#!/bin/bash

# Python arguments
random_state=1989
random_state_pre=1989
results_root=/net/projects/yuxinlab/sparsity/results

# Slurm arguments
node=f002
partition=general
mem=24G
jobname=train

# ################################################
# 0. Arguments
# ################################################
# ================================================
# (You can edit here) General Arguments
# ================================================
# Arguments for network
net_name=resnet
depth=32
widen_factor=4

# Arguments for [dense training]
lr_pre=0.1
lr_schedule_pre=stepwise
lr_milestones_pre=100-150
lr_gamma_pre=0.1
n_epochs_pre=200
batch_size_pre=128

# Arguments for [sparse training]
lr_finetune=0.1
lr_scratch=0.1
lr_schedule=stepwise
lr_milestones=100-150-200
lr_gamma=0.1
n_epochs_finetune=250
n_epochs_scratch=250
batch_size=128

# Arguments for dataset
loader_name=cifar100
filename=CIFAR100
out_dim=100


# Arguments for pruning
prune_method=l1
prune_last=0

# Other unimportant Arguments
optimizer_name=sgd
momentum=0.9
weight_decay=0.0002
device=cuda
init_method=kaiming
n_jobs_dataloader=4

# Arguments for saving things
save_best=1
save_snr=0
save_fisher=0

# ================================================
# (You can edit here) Arguments for Fisher
# ================================================
# The loader to use when evaluating fisher
use_loader=train

# Set some proper range to load
start_epoch=0
end_epoch=300

# Fisher metrics
fisher_method=nngeometry
fisher_metric=fim_monte_carlo
monte_carlo_samples=1000
monte_carlo_trials=5
batch_size_fisher=256


# ################################################
# 1. Train the dense model
# ################################################
# ================================================
# Automatically set the path
# ================================================
NET=${net_name}-${depth}-${widen_factor}
EPOCH=epochs_${n_epochs_pre}
BATCH=batch_${batch_size_pre}
OPTIM=${optimizer_name}_
OPTIM+=lr_${lr_pre}-${lr_schedule_pre}-${lr_milestones_pre}-${lr_gamma_pre}_
OPTIM+=mm_${momentum}_w_decay_${weight_decay}_init_${init_method}
SEED=seed_${random_state_pre}

PRE_PATH=${results_root}/${loader_name}/${NET}/dense/${EPOCH}_${BATCH}_${OPTIM}_${SEED}
PRE_PATH_MODEL=${results_root}/${loader_name}/${NET}/dense/${EPOCH}_${BATCH}_${OPTIM}_${SEED}/model.tar
echo ${PRE_PATH}

# ================================================
# Dense model training
srun -w ${node} --gres=gpu:1 -c 6 --mem ${mem} -p ${partition} --job-name=${jobname} python3 run.py -m 'dense' -lp 0 -ln ${loader_name} -fn ${filename} -rs ${random_state_pre} -nt ${net_name} -dp ${depth} -wf ${widen_factor} -ot ${out_dim} -im ${init_method} -on ${optimizer_name} -mm ${momentum} -lr ${lr_pre} -ls ${lr_schedule_pre} -lm ${lr_milestones_pre} -lg ${lr_gamma_pre} -ne ${n_epochs_pre} -bs ${batch_size_pre} -wt ${weight_decay} -dv ${device} -nj ${n_jobs_dataloader} -sb ${save_best} -ss ${save_snr} -sf ${save_fisher} -rr ${results_root}


# ################################################ #
# 2. Sparse training
# ################################################ #
# Set the path
OPTIM_SP=${optimizer_name}_
OPTIM_SP+=lr_${lr_finetune}-${lr_schedule}-${lr_milestones}-${lr_gamma}_
OPTIM_SP+=mm_${momentum}_w_decay_${weight_decay}_init_${init_method}

# ================================================
# Sparse finetuning
# ================================================
for prune_ratio in 0.8 0.82 0.84 0.86 0.88 0.9 0.92 0.94 0.95 0.96 0.98
do
    # [PATH] Set the path
    FINETUNE_PATH=${results_root}/${loader_name}/${NET}/sparse-finetune/method_${prune_method}_ratio_${prune_ratio}_epochs_${n_epochs_finetune}_batch_${batch_size}_${OPTIM_SP}_seed_${random_state}_epoch_pre_${n_epochs_pre}_batch_pre_${batch_size_pre}_lr_pre_${lr_pre}-${lr_schedule_pre}-${lr_milestones_pre}-${lr_gamma_pre}
    echo ${FINETUNE_PATH}

    # [TRAIN] Train the model
    srun -w ${node} --gres=gpu:1 -c 6 --mem ${mem} -p ${partition} --job-name=${jobname} python3 run.py -m 'sparse-finetune' -pm ${prune_method} -lp 1 -ln ${loader_name} -fn ${filename} -rs ${random_state} -nt ${net_name} -dp ${depth} -wf ${widen_factor} -ot ${out_dim} -im ${init_method} -on ${optimizer_name} -mm ${momentum} -lr ${lr_finetune} -ls ${lr_schedule} -lm ${lr_milestones} -lg ${lr_gamma} -nep ${n_epochs_pre} -ne ${n_epochs_finetune} -bs ${batch_size} -wt ${weight_decay} -dv ${device} -nj ${n_jobs_dataloader} -pp ${PRE_PATH_MODEL} -lrp ${lr_pre} -lsp ${lr_schedule_pre} -lmp ${lr_milestones_pre} -lgp ${lr_gamma_pre} -bsp ${batch_size_pre} -sb ${save_best} -ss ${save_snr} -sf ${save_fisher} -pr ${prune_ratio} -pl ${prune_last} -rr ${results_root}

    # [TRAIN] Reload and continue training (to combat the 4-hour cluster limit)
    srun -w ${node} --gres=gpu:1 -c 6 --mem ${mem} -p ${partition} --job-name=${jobname} python3 run.py -m 'sparse-finetune' -pm ${prune_method} -lp 1 -ln ${loader_name} -fn ${filename} -rs ${random_state} -nt ${net_name} -dp ${depth} -wf ${widen_factor} -ot ${out_dim} -im ${init_method} -on ${optimizer_name} -mm ${momentum} -lr ${lr_finetune} -ls ${lr_schedule} -lm ${lr_milestones} -lg ${lr_gamma} -nep ${n_epochs_pre} -ne ${n_epochs_finetune} -bs ${batch_size} -wt ${weight_decay} -dv ${device} -nj ${n_jobs_dataloader} -pp ${PRE_PATH_MODEL} -lrp ${lr_pre} -lsp ${lr_schedule_pre} -lmp ${lr_milestones_pre} -lgp ${lr_gamma_pre} -bsp ${batch_size_pre} -sb ${save_best} -ss ${save_snr} -sf ${save_fisher} -pr ${prune_ratio} -pl ${prune_last} -rr ${results_root} -ree 180
done


# ================================================
# Sparse scratch
# ================================================
for prune_ratio in 0.8 0.82 0.84 0.86 0.88 0.9 0.92 0.94 0.95 0.96 0.98
do
    # [PATH] Set the path
    SCRATCH_PATH=${results_root}/${loader_name}/${NET}/sparse-scratch/method_${prune_method}_ratio_${prune_ratio}_epochs_${n_epochs_scratch}_batch_${batch_size}_${OPTIM_SP}_seed_${random_state}_epoch_pre_${n_epochs_pre}_batch_pre_${batch_size_pre}_lr_pre_${lr_pre}-${lr_schedule_pre}-${lr_milestones_pre}-${lr_gamma_pre}
    echo ${SCRATCH_PATH}

    # [TRAIN] Train the model
    srun -w ${node} --gres=gpu:1 -c 6 --mem ${mem} -p ${partition} --job-name=${jobname} python3 run.py  -m 'sparse-scratch' -pm ${prune_method} -lp 1 -ln ${loader_name} -fn ${filename} -rs ${random_state} -nt ${net_name} -dp ${depth} -wf ${widen_factor} -ot ${out_dim} -im ${init_method} -on ${optimizer_name} -mm ${momentum} -lr ${lr_scratch} -ls ${lr_schedule} -lm ${lr_milestones} -lg ${lr_gamma} -nep ${n_epochs_pre} -ne ${n_epochs_scratch} -bs ${batch_size} -wt ${weight_decay} -dv ${device} -nj ${n_jobs_dataloader} -pp ${PRE_PATH_MODEL} -lrp ${lr_pre} -lsp ${lr_schedule_pre} -lmp ${lr_milestones_pre} -lgp ${lr_gamma_pre} -bsp ${batch_size_pre} -sb ${save_best} -ss ${save_snr} -sf ${save_fisher} -pr ${prune_ratio} -pl ${prune_last} -rr ${results_root}

    # Give the acces
    chmod 777 ${results_root}

    # [TRAIN] Reload and continue training (to combat the 4-hour cluster limit)
    srun -w ${node} --gres=gpu:1 -c 6 --mem ${mem} -p ${partition} --job-name=${jobname} python3 run.py  -m 'sparse-scratch' -pm ${prune_method} -lp 1 -ln ${loader_name} -fn ${filename} -rs ${random_state} -nt ${net_name} -dp ${depth} -wf ${widen_factor} -ot ${out_dim} -im ${init_method} -on ${optimizer_name} -mm ${momentum} -lr ${lr_scratch} -ls ${lr_schedule} -lm ${lr_milestones} -lg ${lr_gamma} -nep ${n_epochs_pre} -ne ${n_epochs_scratch} -bs ${batch_size} -wt ${weight_decay} -dv ${device} -nj ${n_jobs_dataloader} -pp ${PRE_PATH_MODEL} -lrp ${lr_pre} -lsp ${lr_schedule_pre} -lmp ${lr_milestones_pre} -lgp ${lr_gamma_pre} -bsp ${batch_size_pre} -sb ${save_best} -ss ${save_snr} -sf ${save_fisher} -pr ${prune_ratio} -pl ${prune_last} -rr ${results_root} -ree 180

    # Give the acces
    chmod 777 ${results_root}
done
