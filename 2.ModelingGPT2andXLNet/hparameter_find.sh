#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate w266-hugging

learning_rates='1e-4 5e-5 1e-5'
gradient_accumulation_steps='1 2 4'
weight_decays='-0.5 0 0.5'

for lr in $learning_rates
do

for gas in $gradient_accumulation_steps
do

for wd in $weight_decays
do

outpath="/home/tennisonyu/w266_project/2.ModelingGPT2andXLNet/processed_data/sentence_level_train/obama/lr_${lr}_gas_${gas}_wd_${wd}" 
mkdir $outpath

python run_lm_finetuning.py \
	--num_train_epochs 10 \
	--per_gpu_train_batch_size 2 \
	--per_gpu_eval_batch_size 2 \
	--overwrite_output_dir \
	--train_data_file "/home/tennisonyu/w266_project/2.ModelingGPT2andXLNet/processed_data/sentence_level_train/obama/train.txt" \
	--output_dir $outpath \
	--eval_data_file "/home/tennisonyu/w266_project/2.ModelingGPT2andXLNet/processed_data/sentence_level_train/obama/val.txt" \
	--model_type gpt2 \
	--model_name_or_path gpt2 \
	--evaluate_during_training \
	--do_train \
	--do_eval \
	--eval_all_checkpoints \
	--learning_rate $lr \
	--gradient_accumulation_steps $gas \
	--weight_decay $wd
done
done
done

echo All done
