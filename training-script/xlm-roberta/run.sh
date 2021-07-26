python train.py --roberta_hidden 1024 \
    --epochs 5 \
    --pretrained_model_name "xlm-roberta-large" \
    --learning_rate 2e-5 \
    --max_len 128 \
    --train_batch_size 16 \
    --valid_batch_size 32 \
    --model_layer "pool_output" \
    --model_specification "xlm-roberta-large-pooler-lr-2e-5" \
    --output "five_xlm_roberta_pool_pred_2e5.csv"

python train.py --roberta_hidden 1024 \
    --epochs 5 \
    --pretrained_model_name "xlm-roberta-large" \
    --learning_rate 3e-5 \
    --max_len 128 \
    --train_batch_size 16 \
    --valid_batch_size 32 \
    --model_layer "pool_output" \
    --model_specification "xlm-roberta-large-pooler-lr-3e-5" \
    --output "five_xlm_roberta_pool_pred_3e5.csv"

python train.py --roberta_hidden 1024 \
    --epochs 5 \
    --pretrained_model_name "xlm-roberta-large" \
    --learning_rate 2e-5 \
    --max_len 128 \
    --train_batch_size 16 \
    --valid_batch_size 32 \
    --model_layer "last_two" \
    --model_specification "xlm-roberta-large-last-two-hidden-lr-2e-5" \
    --output "five_xlm_roberta_last_two_pred_2e5.csv"

python train.py --roberta_hidden 1024 \
    --epochs 5 \
    --pretrained_model_name "xlm-roberta-large" \
    --learning_rate 3e-5 \
    --max_len 128 \
    --train_batch_size 16 \
    --valid_batch_size 32 \
    --model_layer "last_two" \
    --model_specification "xlm-roberta-large-last-two-hidden-lr-3e-5" \
    --output "five_xlm_roberta_last_two_pred_3e5.csv"
