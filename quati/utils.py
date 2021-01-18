import math
from pathlib import Path

import torch

from quati.constants import PAD_ID
from quati.modules.attention import Attention
from quati.modules.continuous_attention import ContinuousAttention
from quati.modules.multi_headed_attention import MultiHeadedAttention


def save_word_ids(f, probas, word_ids, target_ids, pred_ids, w_vocab, t_vocab):
    for pvals, wids, tids, pids in zip(probas.tolist(),
                                       word_ids.tolist(),
                                       target_ids.tolist(),
                                       pred_ids.tolist()):
        ws_ps = ['{}:{:.6f}'.format(w_vocab.itos[w], p)
                 for w, p in zip(wids, pvals)]
        words = ' '.join(ws_ps)
        labels = ' '.join([t_vocab.itos[t] for t in tids])
        preds = ' '.join([t_vocab.itos[p] for p in pids])
        f.write('{}\t{}\t{}\n'.format(labels, preds, words))


def save_mu_and_sigma(f, mus, sigmas_sq, word_ids, target_ids, pred_ids,
                      w_vocab, t_vocab):
    for mu, sigma_sq, wids, tids, pids in zip(mus,
                                              sigmas_sq,
                                              word_ids.tolist(),
                                              target_ids.tolist(),
                                              pred_ids.tolist()):
        valid_seq_len = sum([int(w != PAD_ID) for w in wids])
        seq_len = len(wids)
        mean = mu.item() * seq_len
        shift = math.pow(1.5 * sigma_sq.item(), 1.0 / 3.0) * seq_len
        start = max(0, math.floor(mean - shift))
        end = min(valid_seq_len, math.ceil(mean + shift))
        words = ' '.join([w_vocab.itos[w] for w in wids[start:end]])
        labels = ' '.join([t_vocab.itos[t] for t in tids])
        preds = ' '.join([t_vocab.itos[p] for p in pids])
        f.write('{}\t{}\t{:.4f} {:.4f}\t{:.4f}/{} {:.4f}\t{}\n'.format(
            labels, preds,
            mu, sigma_sq,
            mean, valid_seq_len, shift,
            words
        ))


def save_attention(path, model, dataset_iterator):
    words_vocab = dataset_iterator.dataset.fields['words'].vocab
    target_vocab = dataset_iterator.dataset.fields['target'].vocab
    file_path = Path(path)
    model.eval()

    with torch.no_grad():
        with file_path.open('w', encoding='utf8') as f:
            for i, batch in enumerate(dataset_iterator, start=1):
                pred_ids = model.predict_classes(batch)
                target_ids = batch.target
                word_ids = batch.words
                if isinstance(model.attn, ContinuousAttention):
                    mu = model.attn.mu
                    sigma_sq = model.attn.sigma_sq
                    save_mu_and_sigma(f, mu, sigma_sq, word_ids, target_ids,
                                      pred_ids, words_vocab, target_vocab)

                elif isinstance(model.attn, Attention):
                    # _, word_ids = torch.topk(model.attn.alphas, k=5,
                    #                          dim=-1)
                    word_ids = batch.words
                    probas = model.attn.alphas.squeeze(1)
                    save_word_ids(f, probas, word_ids, target_ids, pred_ids,
                                  words_vocab, target_vocab)

                elif isinstance(model.attn, MultiHeadedAttention):
                    nbh = model.attn.nb_heads
                    pred_ids = pred_ids.repeat_interleave(nbh, dim=0)
                    target_ids = target_ids.repeat_interleave(nbh, dim=0)
                    word_ids = word_ids.repeat_interleave(nbh, dim=0)
                    if isinstance(model.attn.attention,
                                  ContinuousAttention):  # noqa
                        mu = model.attn.attention.mu
                        sigma_sq = model.attn.attention.sigma_sq
                        save_mu_and_sigma(f, mu, sigma_sq, word_ids, target_ids,
                                          pred_ids, words_vocab, target_vocab)
                    else:
                        # probas, word_ids = torch.topk(
                        #     model.attn.alphas.squeeze(1), k=5, dim=-1
                        # )
                        word_ids = batch.words
                        probas = model.attn.alphas.squeeze(1)
                        save_word_ids(f, probas, word_ids, target_ids, pred_ids,
                                      words_vocab, target_vocab)
