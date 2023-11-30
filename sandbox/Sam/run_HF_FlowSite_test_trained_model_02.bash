#!/bin/bash
#SBATCH --partition=gpu_quad
#SBATCH --gres=gpu:1,vram:16G
#SBATCH -c 8
#SBATCH --time 23:00:00
#SBATCH --mem=32G
#SBATCH -o /n/groups/marks/users/samhuang/projects/MLCB2023_Final_Project/sandbox/Sam/logs/run_HF_FlowSite_test_trained_model_02_%j.out                 # File to which STDOUT will be written, including job ID (%j)
#SBATCH -e /n/groups/marks/users/samhuang/projects/MLCB2023_Final_Project/sandbox/Sam/logs/run_HF_FlowSite_test_trained_model_02_%j.err                 # File to which STDERR will be written, including job ID (%j)
#SBATCH --mail-type=ALL                                                                                  # ALL email notification type
#SBATCH --mail-user=yininghuang@hms.harvard.edu                                                          # Email to which notifications will be sent

cd /n/groups/marks/users/samhuang/projects/MLCB2023_Final_Project/HF/FlowSite

CUDA_VISIBLE_DEVICES="0" python -m train --run_name test_FlowSite_pdbbind_seqSimSplit --wandb --run_test --checkpoint pocket_gen/89b8ojq8/checkpoints/best.ckpt --batch_size 16 --train_split_path index/pdbbind_mmseqs_30sim_train.txt --val_split_path index/pdbbind_mmseqs_30sim_val.txt --predict_split_path index/pdbbind_mmseqs_30sim_test.txt --pocket_residue_cutoff 12 --clamp_loss 10 --epochs 150 --num_inference 10 --layer_norm --gradient_clip_val 1 --save_inference --check_nan_grads --num_all_res_train_epochs 100000 --fake_constant_dur 100000 --fake_decay_dur 10 --fake_ratio_start 0.2 --fake_ratio_end 0.2 --residue_loss_weight 0.2 --use_tfn --use_inv --time_condition_tfn --correct_time_condition --time_condition_inv --time_condition_repeat --flow_matching --flow_matching_sigma 0.5 --prior_scale 1 --num_angle_pred 5 --ns 32 --nv 8 --batch_norm --self_condition_x --self_condition_inv --no_tfn_self_condition_inv --self_fancy_init
