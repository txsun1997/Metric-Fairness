wget http://data.statmt.org/prism/m39v1.tar
tar xf m39v1.tar

python metrics.py 
    --hyps_file hyps.txt \
    --refs_file refs.txt \
    --bert_score_model roberta-large \
    --bart_score_model facebook/bart-large-cnn \
    --mover_score_model distilbert-base-uncased \
    --frugal_score_model moussaKam/frugalscore_tiny_bert-base_bert-score \
    --bleurt_score_model Elron/bleurt-base-512 \
    --batch_size 4 \
    --device cuda \
    --output_file scores.csv 

