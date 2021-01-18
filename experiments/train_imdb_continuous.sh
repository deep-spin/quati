#!/usr/bin/env bash

train_quati(){
    local attn_type=$1
    local encoder=$2
    local max_activation=$3
    local nb_waves=$4
    local pool=$5
    local supp=$6
    python3 -m quati train \
      --seed 42 \
      --gpu-id 0  \
      --output-dir "runs/imdb-continuous-${attn_type}-${encoder}-${max_activation}-${nb_waves}-${pool}-${supp}" \
      --save "data/saved-models/imdb-continuous-${attn_type}-${encoder}-${max_activation}-${nb_waves}-${pool}-${supp}" \
      --save-attention "data/attentions/imdb-continuous-${attn_type}-${encoder}-${max_activation}-${nb_waves}-${pool}-${supp}.txt" \
      --print-parameters-per-layer \
      --final-report \
      \
      --corpus imdb \
      --train-path "data/corpus/imdb/train/" \
      --dev-path "data/corpus/imdb/dev/" \
      --test-path "data/corpus/imdb/test/" \
      --max-length 9999999 \
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
      --attn-domain "continuous" \
      --attn-type "${attn_type}" \
      --attn-max-activation "${max_activation}" \
      --attn-cont-encoder "${encoder}" \
      --attn-cont-pool "${pool}" \
      --attn-cont-supp "${supp}" \
      --attn-nb-waves "${nb_waves}" \
      --attn-gaussian-basis \
      --attn-hidden-size 128 \
      --attn-dropout 0.0 \
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

predict_quati(){
    local attn_type=$1
    local encoder=$2
    local max_activation=$3
    local nb_waves=$4
    local pool=$5
    local supp=$6
    python3 -m quati predict \
      --gpu-id 0  \
      --prediction-type classes \
      --output-dir "data/predictions/imdb-continuous-${attn_type}-${encoder}-${max_activation}-${nb_waves}-${pool}-${supp}/" \
      --load "data/saved-models/imdb-continuous-${attn_type}-${encoder}-${max_activation}-${nb_waves}-${pool}-${supp}/" \
      --corpus imdb \
      --test-path "data/corpus/imdb/test/" \
      --dev-batch-size 1
}

evaluate_quati(){
    local attn_type=$1
    local encoder=$2
    local max_activation=$3
    local nb_waves=$4
    local pool=$5
    local supp=$6
    python3 scripts/evaluate_predictions.py \
      --corpus imdb \
      --predictions-path "data/predictions/imdb-continuous-${attn_type}-${encoder}-${max_activation}-${nb_waves}-${pool}-${supp}/predictions.txt" \
      --corpus-path "data/corpus/imdb/test/" \
      --average "macro" > "data/evaluations/imdb-continuous-${attn_type}-${encoder}-${max_activation}-${nb_waves}-${pool}-${supp}.txt"
}


# softmax
train_quati "regular" "conv" "softmax" 32 "max" "pred"
predict_quati "regular" "conv" "softmax" 32 "max" "pred"
evaluate_quati "regular" "conv" "softmax" 32 "max" "pred"

train_quati "regular" "conv" "softmax" 64 "max" "pred"
predict_quati "regular" "conv" "softmax" 64 "max" "pred"
evaluate_quati "regular" "conv" "softmax" 64 "max" "pred"

train_quati "regular" "conv" "softmax" 128 "max" "pred"
predict_quati "regular" "conv" "softmax" 128 "max" "pred"
evaluate_quati "regular" "conv" "softmax" 128 "max" "pred"

# sparsemax
train_quati "regular" "conv" "sparsemax" 32 "max" "pred"
predict_quati "regular" "conv" "sparsemax" 32 "max" "pred"
evaluate_quati "regular" "conv" "sparsemax" 32 "max" "pred"

train_quati "regular" "conv" "sparsemax" 64 "max" "pred"
predict_quati "regular" "conv" "sparsemax" 64 "max" "pred"
evaluate_quati "regular" "conv" "sparsemax" 64 "max" "pred"

train_quati "regular" "conv" "sparsemax" 128 "max" "pred"
predict_quati "regular" "conv" "sparsemax" 128 "max" "pred"
evaluate_quati "regular" "conv" "sparsemax" 128 "max" "pred"
