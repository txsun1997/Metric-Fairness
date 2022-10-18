INPUT_PATH=./BERTScore/BERT-large/train.tsv

python train_BERTScore.py 
    --model_type bert-large-uncased \
    --adapter_name debiased-bertscore \
    --lr 5e-4 \
    --warmup 0.0 \
    --batch_size 16 \
    --n_epochs 4 \
    --seed 42 \
    --device cuda \
    --logging_steps 100 \
    --data_path ${INPUT_PATH}