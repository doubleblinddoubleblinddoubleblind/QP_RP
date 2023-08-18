lang=javascript
pathTrainTestData="../../../../Desktop/AugRep_local/QueryPlus_RP/data/models/codesearch/ast/"
mkdir -p $pathTrainTestData/$lang/eval_uni/
python finetune_search_train.py \
    --output_dir $pathTrainTestData/$lang/eval_uni/ \
    --model_name_or_path microsoft/unixcoder-base  \
    --do_train \
    --train_data_file $pathTrainTestData/$lang/train.jsonl \
    --eval_data_file $pathTrainTestData/$lang/test.jsonl \
    --codebase_file $pathTrainTestData/$lang/codebase.jsonl \
    --num_train_epochs 10 \
    --code_length 256 \
    --nl_length 128 \
    --train_batch_size 48 \
    --eval_batch_size 48 \
    --learning_rate 2e-5 \
    --seed 123456
