
fopRoot=../../../../Desktop/AugRep_local/QueryPlus_RP/data/models/query-to-absAST-translation/javascript/

python finetune_translation_gen.py \
	--do_test \
	--model_name_or_path microsoft/unixcoder-base \
	--test_filename $fopRoot"test.jsonl.json" \
	--output_dir $fopRoot"saved_models" \
	--max_source_length 30 \
	--max_target_length 30 \
	--beam_size 3 \
	--train_batch_size 16 \
	--eval_batch_size 6144 \
	--learning_rate 5e-5 \
	--gradient_accumulation_steps 1 \
	--num_train_epochs 10