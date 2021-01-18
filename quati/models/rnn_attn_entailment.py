import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from quati import constants
from quati.initialization import init_xavier, init_kaiming
from quati.models.model import Model
from quati.modules.attention import Attention
from quati.modules.continuous_attention import ContinuousAttention
from quati.modules.continuous_encoders import (ConvEncoder, LSTMEncoder,
                                               DiscreteAttentionEncoder,
                                               LastHiddenStateEncoder)
from quati.modules.multi_headed_attention import MultiHeadedAttention
from quati.modules.scorer import (SelfAdditiveScorer, DotProductScorer,
                                  GeneralScorer, OperationScorer, MLPScorer,
                                  LSTMScorer, ConvScorer)


class RNNAttentionEntailment(Model):
    """Simple RNN with attention model for entailment classification."""

    def __init__(self, fields_tuples, options):
        super().__init__(fields_tuples)
        #
        # Embeddings
        #
        embeddings_weight = None
        if self.fields_dict['words'].vocab.vectors is not None:
            embeddings_weight = self.fields_dict['words'].vocab.vectors
            options.word_embeddings_size = embeddings_weight.size(1)

        self.word_emb = nn.Embedding(
            num_embeddings=len(self.fields_dict['words'].vocab),
            embedding_dim=options.word_embeddings_size,
            padding_idx=constants.PAD_ID,
            _weight=embeddings_weight,
        )
        self.dropout_emb = nn.Dropout(options.embeddings_dropout)
        if options.freeze_embeddings:
            self.word_emb.weight.requires_grad = False

        features_size = options.word_embeddings_size

        #
        # RNN
        #
        self.is_bidir = options.bidirectional
        self.sum_bidir = options.sum_bidir
        self.rnn_type = options.rnn_type

        rnn_map = {'gru': nn.GRU, 'lstm': nn.LSTM, 'rnn': nn.RNN}
        rnn_class = rnn_map[self.rnn_type]
        hidden_size = options.hidden_size[0]
        self.rnn_pre = rnn_class(features_size,
                                 hidden_size,
                                 bidirectional=self.is_bidir,
                                 batch_first=True)
        self.rnn_hyp = rnn_class(features_size,
                                 hidden_size,
                                 bidirectional=self.is_bidir,
                                 batch_first=True)
        self.dropout_rnn = nn.Dropout(options.rnn_dropout)

        n = 1 if not self.is_bidir or self.sum_bidir else 2
        features_size = n * hidden_size

        #
        # Attention
        #
        # define vector size for query and keys
        if options.attn_type == 'multihead':
            vector_size = options.attn_multihead_hidden_size // options.attn_nb_heads  # noqa
        else:
            vector_size = features_size

        # build scorer/encoder and discrete/continuous attention, respectively
        if options.attn_domain == 'discrete':
            scorer = None
            if options.attn_scorer == 'dot_product':
                scorer = DotProductScorer(
                    scaled=True
                )
            elif options.attn_scorer == 'self_add':
                scorer = SelfAdditiveScorer(
                    vector_size,
                    vector_size // 2,
                    scaled=False
                )
            elif options.attn_scorer == 'general':
                scorer = GeneralScorer(
                    vector_size,
                    vector_size
                )
            elif options.attn_scorer in ['add', 'concat']:
                scorer = OperationScorer(
                    vector_size,
                    vector_size,
                    options.attn_hidden_size,
                    op=options.attn_scorer
                )
            elif options.attn_scorer == 'mlp':
                scorer = MLPScorer(
                    vector_size,
                    vector_size,
                    layer_sizes=[options.attn_hidden_size]
                )
            elif options.attn_scorer == 'lstm':
                scorer = LSTMScorer(
                    vector_size,
                    options.attn_hidden_size,
                )
            elif options.attn_scorer == 'conv':
                scorer = ConvScorer(
                    vector_size,
                    options.attn_hidden_size,
                )
            self.attn = Attention(
                scorer,
                dropout=options.attn_dropout,
                max_activation=options.attn_max_activation
            )

        elif options.attn_domain == 'continuous':
            encoder = None
            if options.attn_cont_encoder == 'lstm':
                encoder = LSTMEncoder(
                    vector_size,
                    options.attn_hidden_size,
                    pool=options.attn_cont_pool,
                    supp_type=options.attn_cont_supp
                )
            elif options.attn_cont_encoder == 'last':
                encoder = LastHiddenStateEncoder(
                    vector_size,
                )
            elif options.attn_cont_encoder == 'conv':
                encoder = ConvEncoder(
                    vector_size,
                    options.attn_hidden_size,
                    pool=options.attn_cont_pool,
                    supp_type=options.attn_cont_supp
                )
            elif options.attn_cont_encoder == 'discrete_attn':
                encoder = DiscreteAttentionEncoder(
                    vector_size,
                    options.attn_hidden_size,
                    pool=options.attn_cont_pool,
                    supp_type=options.attn_cont_supp
                )
            self.attn = ContinuousAttention(
                encoder,
                dropout=options.attn_dropout,
                nb_waves=options.attn_nb_waves,
                wave_b=options.attn_wave_b,
                use_power_basis=options.attn_power_basis,
                use_wave_basis=options.attn_wave_basis,
                use_gaussian_basis=options.attn_gaussian_basis,
                dynamic_nb_basis=options.attn_dynamic_nb_basis,
                consider_pad=options.attn_consider_pad,
                max_activation=options.attn_max_activation,
                gpu_id=options.gpu_id
            )

        #
        # Multihead attention
        #
        if options.attn_type == 'multihead':
            self.attn = MultiHeadedAttention(
                self.attn,
                options.attn_nb_heads,
                features_size,
                features_size,
                features_size,
                options.attn_multihead_hidden_size
            )
            features_size = options.attn_multihead_hidden_size

        #
        # Linear
        #
        self.linear_pre = nn.Linear(features_size, features_size // 2)
        self.linear_hyp = nn.Linear(features_size, features_size // 2)
        self.linear_out = nn.Linear(features_size // 2, self.nb_classes)
        self.linear_activation = nn.Tanh()

        # stored variables
        self.embeddings_out_pre = None
        self.embeddings_out_hyp = None
        self.hidden_pre = None
        self.hidden_hyp = None
        self.attn_weights = None
        self.logits = None

        self.init_weights()
        self.is_built = True

    def init_weights(self):
        pass
        # init_xavier(self.rnn_pre, dist='uniform')
        # init_xavier(self.rnn_hyp, dist='uniform')
        # init_xavier(self.attn, dist='uniform')
        # init_xavier(self.linear_out, dist='uniform')

    def init_hidden(self, batch_size, hidden_size, device=None):
        # The axes semantics are (nb_layers, minibatch_size, hidden_dim)
        nb_layers = 2 if self.is_bidir else 1
        if self.rnn_type == 'lstm':
            return (torch.zeros(nb_layers, batch_size, hidden_size).to(device),
                    torch.zeros(nb_layers, batch_size, hidden_size).to(device))
        else:
            return torch.zeros(nb_layers, batch_size, hidden_size).to(device)

    def encode(self, rnn, h, mask):
        # initialize RNN hidden state
        hidden_state = self.init_hidden(
            h.shape[0], rnn.hidden_size, device=h.device
        )

        # (bs, ts, emb_dim) -> (bs, ts, hidden_size)
        lengths = mask.int().sum(dim=-1)
        h = pack(h, lengths, batch_first=True, enforce_sorted=False)
        h, hidden_state = rnn(h, hidden_state)
        h, _ = unpack(h, batch_first=True)

        # if you'd like to sum instead of concatenate:
        if self.sum_bidir:
            h = h[:, :, :rnn.hidden_size] + h[:, :, rnn.hidden_size:]

        # apply dropout
        h = self.dropout_rnn(h)

        return h, hidden_state

    def forward(self, batch):
        assert self.is_built
        assert self._loss is not None

        batch_size = batch.words.shape[0]
        h_pre = batch.words
        mask_pre = torch.ne(h_pre, constants.PAD_ID)
        h_hyp = batch.words_hyp
        mask_hyp = torch.ne(h_hyp, constants.PAD_ID)

        # (bs, ts) -> (bs, ts, emb_dim)
        self.embeddings_out_pre = self.word_emb(h_pre)
        h_pre = self.dropout_emb(self.embeddings_out_pre)
        h_enc_pre, self.hidden_pre = self.encode(self.rnn_pre, h_pre, mask_pre)

        self.embeddings_out_hyp = self.word_emb(h_hyp)
        h_hyp = self.dropout_emb(self.embeddings_out_hyp)
        h_enc_hyp, self.hidden_hyp = self.encode(self.rnn_hyp, h_hyp, mask_hyp)

        # (bs, ts, hidden_size)  -> (bs, 1, hidden_size)
        # get last valid outputs from the RNN for each batch
        # we can't get the last [-1] since it can be a pad position
        # last_valid_idx = mask_hyp.int().sum(dim=-1) - 1
        # arange_vector = torch.arange(h_hyp.shape[0]).to(h_hyp.device)
        # h_pooled_hyp = h_enc_hyp[arange_vector, last_valid_idx].unsqueeze(1)

        # recover hidden state from the correct rnn type
        if self.rnn_type == 'lstm':
            hidden_state_hyp = self.hidden_hyp[0]
        else:
            hidden_state_hyp = self.hidden_hyp

        # deal with bidirectionality
        if self.is_bidir:
            hidden_states = [hidden_state_hyp[0], hidden_state_hyp[1]]
        else:
            hidden_states = [hidden_state_hyp[0]]

        # concat the hidden states and create a time dim
        h_pooled_hyp = torch.cat(hidden_states, dim=-1).unsqueeze(1)

        # apply attention
        # (bs, ts, hidden_size)  -> (bs, 1, hidden_size)
        h_attn_pre, self.attn_weights = self.attn(
            h_pooled_hyp, h_enc_pre, values=h_enc_pre, mask=mask_pre
        )

        # (bs, 1, hidden_size) -> (bs, 1, hidden_size // 2)
        h_pre_out = self.linear_pre(h_attn_pre)
        h_hyp_out = self.linear_hyp(h_pooled_hyp)
        h = self.linear_activation(h_pre_out + h_hyp_out)

        # (bs, 1, hidden_size) -> (bs, 1, nb_classes)
        self.logits = self.linear_out(h)

        # (bs, 1, nb_classes) -> (bs, 1, nb_classes) in log simplex
        h = F.log_softmax(self.logits, dim=-1)

        return h
