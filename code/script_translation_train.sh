fopRoot=../../../../Desktop/AugRep_local/QueryPlus_RP/data/models/query-to-absAST-translation/javascript/

python finetune_translation_train.py \
	--do_train \
	--do_eval \
	--model_name_or_path microsoft/unixcoder-base \
	--train_filename $fopRoot"/train.jsonl.json" \
	--dev_filename $fopRoot"/valid.jsonl.json" \
	--output_dir $fopRoot"/saved_models/" \
	--max_source_length 50 \
	--max_target_length 50 \
	--beam_size 3 \
	--train_batch_size 128 \
	--eval_batch_size 256 \
	--learning_rate 5e-5 \
	--gradient_accumulation_steps 1 \
	--num_train_epochs 30

