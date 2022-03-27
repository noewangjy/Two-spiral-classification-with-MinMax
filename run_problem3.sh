python src/MinMax_model.py \
    --data_split_strategy "PK" \
    --hard_task 0,4,8 \
    --h_easy 5 \
    --h_hard 40 \
    --epochs 10000 \
    --learning_rate 1e-1 \
    --activation "sigmoid" \
    --do_train \
    --do_test \
    --visualize_data_subsets \
    --visualize_MinMax_db \
    --visualize_submodel_db \
    --multiprocessing

