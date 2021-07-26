# ############# mbert cased ##########################
python train.py --bert_hidden 768 \
    --epochs 6 \
    --pretrained_model_name "bert-base-multilingual-cased" \
    --learning_rate 2e-5 \
    --max_len 128 \
    --dropout 0.3 \
    --train_batch_size 16 \
    --valid_batch_size 32 \
    --model_layer "pool_bert" \
    --model_specification "bert-base-multilingual-cased-pooled-lr-2e-5" \
    --output "six_bert_base_multilingual_cased_pool_pred_2e5.csv"

python train.py --bert_hidden 768 \
    --epochs 6 \
    --pretrained_model_name "bert-base-multilingual-cased" \
    --learning_rate 3e-5 \
    --max_len 128 \
    --dropout 0.3 \
    --train_batch_size 16 \
    --valid_batch_size 32 \
    --model_layer "pool_bert" \
    --model_specification "bert-base-multilingual-cased-pooled-lr-3e-5" \
    --output "six_bert_base_multilingual_cased_pool_pred_3e5.csv"

python train.py --bert_hidden 768 \
    --epochs 6 \
    --pretrained_model_name "bert-base-multilingual-cased" \
    --learning_rate 5e-5 \
    --max_len 128 \
    --dropout 0.3 \
    --train_batch_size 16 \
    --valid_batch_size 32 \
    --model_layer "pool_bert" \
    --model_specification "bert-base-multilingual-cased-pooled-lr-5e-5" \
    --output "six_bert_base_multilingual_cased_pool_pred_5e5.csv"


python train.py --bert_hidden 768 \
    --epochs 6 \
    --pretrained_model_name "bert-base-multilingual-cased" \
    --learning_rate 2e-5 \
    --max_len 128 \
    --dropout 0.3 \
    --train_batch_size 16 \
    --valid_batch_size 32 \
    --model_layer "last_two" \
    --model_specification "bert-base-multilingual-cased-last_two-lr-2e-5" \
    --output "six_bert_base_multilingual_cased_last_two_pred_2e5.csv"

python train.py --bert_hidden 768 \
    --epochs 6 \
    --pretrained_model_name "bert-base-multilingual-cased" \
    --learning_rate 3e-5 \
    --max_len 128 \
    --dropout 0.3 \
    --train_batch_size 16 \
    --valid_batch_size 32 \
    --model_layer "last_two" \
    --model_specification "bert-base-multilingual-cased-last_two-lr-3e-5" \
    --output "six_bert_base_multilingual_cased_last_two_pred_3e5.csv"

python train.py --bert_hidden 768 \
    --epochs 6 \
    --pretrained_model_name "bert-base-multilingual-cased" \
    --learning_rate 5e-5 \
    --max_len 128 \
    --dropout 0.3 \
    --train_batch_size 16 \
    --valid_batch_size 32 \
    --model_layer "last_two" \
    --model_specification "bert-base-multilingual-cased-last_two-lr-5e-5" \
    --output "six_bert_base_multilingual_cased_last_two_pred_5e5.csv"

python train.py --bert_hidden 768 \
    --epochs 6 \
    --pretrained_model_name "bert-base-multilingual-cased" \
    --learning_rate 2e-5 \
    --max_len 128 \
    --dropout 0.3 \
    --train_batch_size 16 \
    --valid_batch_size 32 \
    --model_layer "last_four" \
    --model_specification "bert-base-multilingual-cased-last_four-lr-2e-5" \
    --output "six_bert_base_multilingual_cased_last_four_pred_2e5.csv"

python train.py --bert_hidden 768 \
    --epochs 6 \
    --pretrained_model_name "bert-base-multilingual-cased" \
    --learning_rate 3e-5 \
    --max_len 128 \
    --dropout 0.3 \
    --train_batch_size 16 \
    --valid_batch_size 32 \
    --model_layer "last_four" \
    --model_specification "bert-base-multilingual-cased-last_four-lr-3e-5" \
    --output "six_bert_base_multilingual_cased_last_four_pred_3e5.csv"

python train.py --bert_hidden 768 \
    --epochs 6 \
    --pretrained_model_name "bert-base-multilingual-cased" \
    --learning_rate 5e-5 \
    --max_len 128 \
    --dropout 0.3 \
    --train_batch_size 16 \
    --valid_batch_size 32 \
    --model_layer "last_four" \
    --model_specification "bert-base-multilingual-cased-last_four-lr-5e-5" \
    --output "six_bert_base_multilingual_cased_last_four_pred_5e5.csv"


# # ############# mbert uncased ##########################
python train.py --bert_hidden 768 \
    --epochs 6 \
    --pretrained_model_name "bert-base-multilingual-uncased" \
    --learning_rate 2e-5 \
    --max_len 128 \
    --dropout 0.3 \
    --train_batch_size 16 \
    --valid_batch_size 32 \
    --model_layer "pool_bert" \
    --model_specification "bert-base-multilingual-uncased-pooled-lr-2e-5" \
    --output "six_bert_base_multilingual_uncased_pool_pred_2e5.csv"

python train.py --bert_hidden 768 \
    --epochs 6 \
    --pretrained_model_name "bert-base-multilingual-uncased" \
    --learning_rate 3e-5 \
    --max_len 128 \
    --dropout 0.3 \
    --train_batch_size 16 \
    --valid_batch_size 32 \
    --model_layer "pool_bert" \
    --model_specification "bert-base-multilingual-uncased-pooled-lr-3e-5" \
    --output "six_bert_base_multilingual_uncased_pool_pred_3e5.csv"

python train.py --bert_hidden 768 \
    --epochs 6 \
    --pretrained_model_name "bert-base-multilingual-uncased" \
    --learning_rate 5e-5 \
    --max_len 128 \
    --dropout 0.3 \
    --train_batch_size 16 \
    --valid_batch_size 32 \
    --model_layer "pool_bert" \
    --model_specification "bert-base-multilingual-uncased-pooled-lr-5e-5" \
    --output "six_bert_base_multilingual_uncased_pool_pred_5e5.csv"


python train.py --bert_hidden 768 \
    --epochs 6 \
    --pretrained_model_name "bert-base-multilingual-uncased" \
    --learning_rate 2e-5 \
    --max_len 128 \
    --dropout 0.3 \
    --train_batch_size 16 \
    --valid_batch_size 32 \
    --model_layer "last_two" \
    --model_specification "bert-base-multilingual-uncased-last_two-lr-2e-5" \
    --output "six_bert_base_multilingual_uncased_last_two_pred_2e5.csv"

python train.py --bert_hidden 768 \
    --epochs 6 \
    --pretrained_model_name "bert-base-multilingual-uncased" \
    --learning_rate 3e-5 \
    --max_len 128 \
    --dropout 0.3 \
    --train_batch_size 16 \
    --valid_batch_size 32 \
    --model_layer "last_two" \
    --model_specification "bert-base-multilingual-uncased-last_two-lr-3e-5" \
    --output "six_bert_base_multilingual_uncased_last_two_pred_3e5.csv"

python train.py --bert_hidden 768 \
    --epochs 6 \
    --pretrained_model_name "bert-base-multilingual-uncased" \
    --learning_rate 5e-5 \
    --max_len 128 \
    --dropout 0.3 \
    --train_batch_size 16 \
    --valid_batch_size 32 \
    --model_layer "last_two" \
    --model_specification "bert-base-multilingual-uncased-last_two-lr-5e-5" \
    --output "six_bert_base_multilingual_uncased_last_two_pred_5e5.csv"

python train.py --bert_hidden 768 \
    --epochs 6 \
    --pretrained_model_name "bert-base-multilingual-uncased" \
    --learning_rate 2e-5 \
    --max_len 128 \
    --dropout 0.3 \
    --train_batch_size 16 \
    --valid_batch_size 32 \
    --model_layer "last_four" \
    --model_specification "bert-base-multilingual-uncased-last_four-lr-2e-5" \
    --output "six_bert_base_multilingual_uncased_last_four_pred_2e5.csv"

python train.py --bert_hidden 768 \
    --epochs 6 \
    --pretrained_model_name "bert-base-multilingual-uncased" \
    --learning_rate 3e-5 \
    --max_len 128 \
    --dropout 0.3 \
    --train_batch_size 16 \
    --valid_batch_size 32 \
    --model_layer "last_four" \
    --model_specification "bert-base-multilingual-uncased-last_four-lr-3e-5" \
    --output "six_bert_base_multilingual_uncased_last_four_pred_3e5.csv"

python train.py --bert_hidden 768 \
    --epochs 6 \
    --pretrained_model_name "bert-base-multilingual-uncased" \
    --learning_rate 5e-5 \
    --max_len 128 \
    --dropout 0.3 \
    --train_batch_size 16 \
    --valid_batch_size 32 \
    --model_layer "last_four" \
    --model_specification "bert-base-multilingual-uncased-last_four-lr-5e-5" \
    --output "six_bert_base_multilingual_uncased_last_four_pred_5e5.csv"

############# bangla bert ##########################
python train.py --bert_hidden 768 \
    --epochs 6 \
    --pretrained_model_name "sagorsarker/bangla-bert-base" \
    --learning_rate 2e-5 \
    --max_len 128 \
    --dropout 0.3 \
    --train_batch_size 16 \
    --valid_batch_size 32 \
    --model_layer "pool_bert" \
    --model_specification "bangla-bert-pooled-lr-2e-5" \
    --output "six_bangla_bert_pool_pred_2e5.csv"

python train.py --bert_hidden 768 \
    --epochs 6 \
    --pretrained_model_name "sagorsarker/bangla-bert-base" \
    --learning_rate 3e-5 \
    --max_len 128 \
    --dropout 0.3 \
    --train_batch_size 16 \
    --valid_batch_size 32 \
    --model_layer "pool_bert" \
    --model_specification "bangla-bert-pooled-lr-3e-5" \
    --output "six_bangla_bert_pool_pred_3e5.csv"

python train.py --bert_hidden 768 \
    --epochs 6 \
    --pretrained_model_name "sagorsarker/bangla-bert-base" \
    --learning_rate 5e-5 \
    --max_len 128 \
    --dropout 0.3 \
    --train_batch_size 16 \
    --valid_batch_size 32 \
    --model_layer "pool_bert" \
    --model_specification "bangla-bert-pooled-lr-5e-5" \
    --output "six_bangla_bert_pool_pred_5e5.csv"


python train.py --bert_hidden 768 \
    --epochs 6 \
    --pretrained_model_name "sagorsarker/bangla-bert-base" \
    --learning_rate 2e-5 \
    --max_len 128 \
    --dropout 0.3 \
    --train_batch_size 16 \
    --valid_batch_size 32 \
    --model_layer "last_two" \
    --model_specification "bangla-bert-last_two-lr-2e-5" \
    --output "six_bangla_bert_last_two_pred_2e5.csv"

python train.py --bert_hidden 768 \
    --epochs 6 \
    --pretrained_model_name "sagorsarker/bangla-bert-base" \
    --learning_rate 3e-5 \
    --max_len 128 \
    --dropout 0.3 \
    --train_batch_size 16 \
    --valid_batch_size 32 \
    --model_layer "last_two" \
    --model_specification "bangla-bert-last_two-lr-3e-5" \
    --output "six_bangla_bert_last_two_pred_3e5.csv"

python train.py --bert_hidden 768 \
    --epochs 6 \
    --pretrained_model_name "sagorsarker/bangla-bert-base" \
    --learning_rate 5e-5 \
    --max_len 128 \
    --dropout 0.3 \
    --train_batch_size 16 \
    --valid_batch_size 32 \
    --model_layer "last_two" \
    --model_specification "bangla-bert-last_two-lr-5e-5" \
    --output "six_bangla_bert_last_two_pred_5e5.csv"

python train.py --bert_hidden 768 \
    --epochs 6 \
    --pretrained_model_name "sagorsarker/bangla-bert-base" \
    --learning_rate 2e-5 \
    --max_len 128 \
    --dropout 0.3 \
    --train_batch_size 16 \
    --valid_batch_size 32 \
    --model_layer "last_four" \
    --model_specification "bangla-bert-last_four-lr-2e-5" \
    --output "six_bangla_bert_last_four_pred_2e5.csv"

python train.py --bert_hidden 768 \
    --epochs 6 \
    --pretrained_model_name "sagorsarker/bangla-bert-base" \
    --learning_rate 3e-5 \
    --max_len 128 \
    --dropout 0.3 \
    --train_batch_size 16 \
    --valid_batch_size 32 \
    --model_layer "last_four" \
    --model_specification "bangla-bert-last_four-lr-3e-5" \
    --output "six_bangla_bert_last_four_pred_3e5.csv"

python train.py --bert_hidden 768 \
    --epochs 6 \
    --pretrained_model_name "sagorsarker/bangla-bert-base" \
    --learning_rate 5e-5 \
    --max_len 128 \
    --dropout 0.3 \
    --train_batch_size 16 \
    --valid_batch_size 32 \
    --model_layer "last_four" \
    --model_specification "bangla-bert-last_four-lr-5e-5" \
    --output "six_bangla_bert_last_four_pred_5e5.csv"
