folder=javascript
pathTrainTestData=../../../../Desktop/AugRep_local/QueryPlus_RP/data/models/codesearch/ast/$folder
mkdir -p $pathTrainTestData/eval_uni/
python finetune_search_gen.py \
    --output_dir $pathTrainTestData/eval_uni/ \
    --model_name_or_path microsoft/unixcoder-base  \
    --output_vector_model=$pathTrainTestData/../vectors/ \
    --do_test \
    --test_data_file $pathTrainTestData/test.jsonl \
    --codebase_file $pathTrainTestData/codebase.jsonl \
    --num_train_epochs 30 \
    --code_length 256 \
    --nl_length 128 \
    --train_batch_size 48 \
    --eval_batch_size 48 \
    --learning_rate 2e-5 \
    --seed 123456
