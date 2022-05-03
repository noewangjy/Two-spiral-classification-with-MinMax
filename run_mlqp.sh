python src/single_model.py \
    --model_name "OneLayerMLQPwithLinearOut" \
    --data_path "data" \
    --activation "sigmoid" \
    --hidden_size 40 \
    --hidden_size2 8 \
    --learning_rate 1e-1 \
    --epochs 200000 \
    --do_train \
    --do_test \
    --visualize_db \
    --early_stop 

