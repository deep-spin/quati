import torch.nn as nn
import torch.nn.functional as F

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


class FNNAttention(Model):
    """Simple FeedForward net with attention model for text classification."""

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

        else:
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
        self.linear_out = nn.Linear(features_size, self.nb_classes)

        # stored variables
        self.embeddings_out = None
        self.hidden = None
        self.attn_weights = None
        self.logits = None

        self.init_weights()
        self.is_built = True

    def init_weights(self):
        pass
        # init_xavier(self.attn, dist='uniform')
        # init_xavier(self.linear_out, dist='uniform')

    def forward(self, batch):
        assert self.is_built
        assert self._loss is not None

        h = batch.words
        mask = h != constants.PAD_ID

        # (bs, ts) -> (bs, ts, emb_dim)
        self.embeddings_out = self.word_emb(h)
        h = self.dropout_emb(self.embeddings_out)

        # (bs, ts, emb_dim)  -> (bs, 1, emb_dim)
        h, self.attn_weights = self.attn(h, h, values=h, mask=mask)

        # (bs, 1, emb_dim) -> (bs, 1, nb_classes)
        self.logits = self.linear_out(h)

        # (bs, 1, nb_classes) -> (bs, 1, nb_classes) in log simplex
        h = F.log_softmax(self.logits, dim=-1)

        return h
