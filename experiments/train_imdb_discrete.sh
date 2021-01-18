#!/usr/bin/env bash

train_spec(){
  local attn_type=$1
  local scorer=$2
  local max_activation=$3
  python3 -m quati train \
      --seed 42 \
      --gpu-id 0  \
      --output-dir "runs/imdb-discrete-${attn_type}-${scorer}-${max_activation}/" \
      --save "data/saved-models/imdb-discrete-${attn_type}-${scorer}-${max_activation}/" \
      --save-attention "data/attentions/imdb-disc-${attn_type}-${scorer}-${max_activation}.txt" \
      --print-parameters-per-layer \
      --final-report \
      \
      --corpus imdb \
      --train-path "data/corpus/imdb/train/" \
      --dev-path "data/corpus/imdb/dev/" \
      --test-path "data/corpus/imdb/test/" \
      --max-length 9999999 \
      --min-length 0 \
      \
      --vocab-size 9999999 \
      --vocab-min-frequency 1 \
      --keep-rare-with-vectors \
      --add-embeddings-vocab \
      \
      --embeddings-format "text" \
      --embeddings-path "data/embs/glove/glove.840B.300d.small.raw.pickle.imdb" \
      --embeddings-binary \
      --embeddings-dropout 0.0 \
      --freeze-embeddings \
      \
      --model rnn_attn \
      \
      --rnn-type lstm \
      --hidden-size 128 \
      --bidirectional \
      --rnn-dropout 0.0 \
      \
      --attn-type "${attn_type}" \
      --attn-scorer "${scorer}" \
      --attn-max-activation "${max_activation}" \
      --attn-dropout 0.0 \
      --attn-hidden-size 128 \
      \
      --loss-weights "same" \
      --train-batch-size 16 \
      --dev-batch-size 16 \
      --epochs 10 \
      --optimizer "adamw" \
      --learning-rate 0.001 \
      --weight-decay 0.0001 \
      --save-best-only \
      --early-stopping-patience 5 \
      --restore-best-model
}

predict_spec(){
  local attn_type=$1
  local scorer=$2
  local max_activation=$3
  python3 -m quati predict \
      --gpu-id 0  \
      --prediction-type classes \
      --output-dir "data/predictions/imdb-discrete-${attn_type}-${scorer}-${max_activation}/" \
      --load "data/saved-models/imdb-discrete-${attn_type}-${scorer}-${max_activation}/" \
      --corpus imdb \
      --test-path "data/corpus/imdb/test/" \
      --dev-batch-size 4
}

evaluate_spec(){
  local attn_type=$1
  local scorer=$2
  local max_activation=$3
  python3 scripts/evaluate_predictions.py \
      --corpus imdb \
      --predictions-path "data/predictions/imdb-discrete-${attn_type}-${scorer}-${max_activation}/predictions.txt" \
      --corpus-path "data/corpus/imdb/test/" \
      --average "macro" > "data/evaluations/imdb-discrete-${attn_type}-${scorer}-${max_activation}.txt"
}


train_spec "regular" "self_add" "softmax"
predict_spec "regular" "self_add" "softmax"
evaluate_spec "regular" "self_add" "softmax"

train_spec "regular" "self_add" "sparsemax"
predict_spec "regular" "self_add" "sparsemax"
evaluate_spec "regular" "self_add" "sparsemax"

train_spec "regular" "self_add" "entmax15"
predict_spec "regular" "self_add" "entmax15"
evaluate_spec "regular" "self_add" "entmax15"
