python calculate.py --file step2_results/test_gpt2_results.txt --out 20191129_run/test_gpt2_metrics.txt

python calculate.py --file step2_results/test_gpt2tuned_results.txt --out 20191129_run/test_gpt2tuned_metrics.txt

python calculate.py --file step2_results/test_xlnet-base-cased_results.txt --out 20191129_run/test_xlnet-base-cased_metrics.txt

python calculate.py --file step2_results/LSTM-attention-result-new-dim2.txt --out 20191129_run/test_lstm-attention_metrics.txt 

python calculate.py --file step2_results/gpt2_BERT_result-new-dim.txt --out 20191129_run/test_bert_smash_metrics.txt 

#files="test_gpt2_results test_gpt2tuned_results test_xlnet-base-cased_results LSTM-attention-result-new-dim2 gpt2_BERT_result-new-dim blank"
#model_types="gpt2 xlnet"
#model_paths="gpt2 gpt2_tuned xlnet-base-cased" 

# for f in $files
# do

# for mt in $model_types
# do

# for mp in $model_paths
# do

# echo $f
# echo $mt
# echo $mp

# if [ "$f" == "test_gpt2_results" ]
# then

# python calculate_perplexity.py --generate False  --model_type ${mt} --model_name_or_path ${mp} --input_path step2_results/${f}.txt --out 20191129_run/original_${mp}_perplexity.txt

# fi

# python calculate_perplexity.py $extra --model_type ${mt} --model_name_or_path ${mp} --input_path step2_results/${f}.txt --out 20191129_run/${f}_${mp}_perplexity.txt

# done
# done
# done

# echo "All Done"
