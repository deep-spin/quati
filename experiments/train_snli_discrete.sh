#!/usr/bin/env bash

train_spec(){
  local attn_type=$1
  local scorer=$2
  local max_activation=$3
  python3 -m quati train \
      --seed 42 \
      --gpu-id 0  \
      --output-dir "runs/snli-discrete-${attn_type}-${scorer}-${max_activation}/" \
      --save "data/saved-models/snli-discrete-${attn_type}-${scorer}-${max_activation}/" \
      --print-parameters-per-layer \
      --final-report \
      \
      --corpus snli \
      --train-path "data/corpus/snli/snli_1.0_train.jsonl" \
      --dev-path "data/corpus/snli/snli_1.0_dev.jsonl" \
      --test-path "data/corpus/snli/snli_1.0_test.jsonl" \
      --max-length 9999999 \
      --min-length 0 \
      \
      --vocab-size 9999999 \
      --vocab-min-frequency 1 \
      --keep-rare-with-vectors \
      --add-embeddings-vocab \
      \
      --embeddings-format "text" \
      --embeddings-path "data/embs/glove/glove.840B.300d.small.raw.pickle.snli" \
      --embeddings-binary \
      --embeddings-dropout 0.0 \
      --freeze-embeddings \
      \
      --model rnn_attn_entailment \
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
      --train-batch-size 64 \
      --dev-batch-size 64 \
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
      --output-dir "data/predictions/snli-discrete-${attn_type}-${scorer}-${max_activation}/" \
      --load "data/saved-models/snli-discrete-${attn_type}-${scorer}-${max_activation}/" \
      --corpus snli \
      --test-path "data/corpus/snli/snli_1.0_test.jsonl" \
      --dev-batch-size 4
}

evaluate_spec(){
  local attn_type=$1
  local scorer=$2
  local max_activation=$3
  python3 scripts/evaluate_predictions.py \
      --corpus snli \
      --predictions-path "data/predictions/snli-discrete-${attn_type}-${scorer}-${max_activation}/predictions.txt" \
      --corpus-path "data/corpus/snli/snli_1.0_test.jsonl" \
      --average "macro" > "data/evaluations/snli-discrete-${attn_type}-${scorer}-${max_activation}.txt"
}


train_spec "regular" "add" "softmax"
predict_spec "regular" "add" "softmax"
evaluate_spec "regular" "add" "softmax"

train_spec "regular" "add" "sparsemax"
predict_spec "regular" "add" "sparsemax"
evaluate_spec "regular" "add" "sparsemax"

train_spec "regular" "add" "entmax15"
predict_spec "regular" "add" "entmax15"
evaluate_spec "regular" "add" "entmax15"
