from pathlib import Path

from quati import constants
from quati import opts
from quati.models.fnn_attn import FNNAttention
from quati.models.linear_bow import LinearBoW
from quati.models.linear_tfidf import LinearTfIdf
from quati.models.rnn_attn import RNNAttention
from quati.models.rnn_attn_entailment import RNNAttentionEntailment
from quati.models.simple_mlp import SimpleMLP
from quati.models.simple_rnn import SimpleRNN

available_models = {
    'simple_rnn': SimpleRNN,
    'simple_mlp': SimpleMLP,
    'rnn_attn': RNNAttention,
    'fnn_attn': FNNAttention,
    'rnn_attn_entailment': RNNAttentionEntailment,
    'linear_bow': LinearBoW,
    'linear_tfidf': LinearTfIdf,
}


def build(options, fields_tuples, loss_weights):
    model_cls = available_models[options.model]
    model = model_cls(fields_tuples, options)
    model.build_loss(loss_weights)
    if options.gpu_id is not None:
        model = model.cuda(options.gpu_id)
    return model


def load_state(path, model):
    model_path = Path(path, constants.MODEL)
    model.load(model_path)


def load(path, fields_tuples, current_gpu_id):
    options = opts.load(path)

    # set gpu device to the current device
    options.gpu_id = current_gpu_id

    # hack: set dummy loss_weights (the correct values are going to be loaded)
    target_field = dict(fields_tuples)['target']
    loss_weights = None
    if options.loss_weights == 'balanced':
        loss_weights = [0] * (len(target_field.vocab) - 1)

    model = build(options, fields_tuples, loss_weights)
    load_state(path, model)
    return model


def save(path, model):
    model_path = Path(path, constants.MODEL)
    model.save(model_path)
