#!/bin/bash

# output file contains time stamp
output_file="/home/samhuang/Documents/GitHub/MLCB2023_Final_Project/sandbox/Sam/desktop/logs/run_train_HarmonicFlow_timesplit_RadPock_$(date +"%Y%m%d_%H%M%S").out"
error_file="/home/samhuang/Documents/GitHub/MLCB2023_Final_Project/sandbox/Sam/desktop/logs/run_train_HarmonicFlow_timesplit_RadPock_$(date +"%Y%m%d_%H%M%S").err"

cd /home/samhuang/Documents/GitHub/MLCB2023_Final_Project/HF/FlowSite_Exp_Priors

CUDA_VISIBLE_DEVICES="0" python -m train --run_name train_HarmonicFlow_timesplit_RadPock_$(date +"%Y%m%d_%H%M%S") --wandb --lr 1e-3 --batch_size 4 --train_split_path index/timesplit_no_lig_overlap_train --val_split_path index/timesplit_no_lig_overlap_val --predict_split_path index/timesplit_test --clamp_loss 10 --epochs 150 --num_inference 10 --gradient_clip_val 1 --save_inference --check_nan_grads --num_all_res_train_epochs 100000 --fake_constant_dur 0 --fake_decay_dur 0 --fake_ratio_start 0 --fake_ratio_end 0 --residue_loss_weight 0 --use_tfn --time_condition_tfn --correct_time_condition --time_condition_inv --time_condition_repeat --flow_matching --flow_matching_sigma 0.5 --prior_scale 1 --layer_norm --tfn_detach --max_lig_size 200 --num_workers 4 --check_val_every_n_epoch 1 --cross_radius 50 --protein_radius 30 --lig_radius 50  --ns 32 --nv 8 --tfn_use_aa_identities --self_condition_x --pocket_residue_cutoff_sigma 0.5 --pocket_center_sigma 0.2 --pocket_residue_cutoff_sigma 0.5 --pocket_center_sigma 0.2 --pocket_type ca_distance --pocket_residue_cutoff 14  > "$output_file" 2> "$error_file"
