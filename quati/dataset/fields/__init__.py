from collections import defaultdict
from pathlib import Path

import torch


from quati import constants
from quati.dataset.vectors import load_vectors


def build_vocabs(fields_tuples, train_dataset, options):
    # transform fields_tuples to a dict in order to access fields easily
    dict_fields = defaultdict(lambda: None)
    dict_fields.update(dict(fields_tuples))

    # load word embeddings in case embeddings_format is not None
    vectors = load_vectors(
        options.embeddings_path,
        options.embeddings_format,
        binary=options.embeddings_binary
    )

    # build vocab for words based on the training set
    words_field = dict_fields['words']
    words_field.build_vocab(
        train_dataset,
        vectors=vectors,
        max_size=options.vocab_size,
        min_freq=options.vocab_min_frequency,
        keep_rare_with_vectors=options.keep_rare_with_vectors,
        add_vectors_vocab=options.add_embeddings_vocab
    )

    # build vocab for tags based on all datasets
    target_field = dict_fields['target']
    target_field.build_vocab(train_dataset, specials_first=False)

    # set global constants to their correct value
    constants.PAD_ID = dict_fields['words'].vocab.stoi[constants.PAD]

    # set target pad id (useful for seq classification)
    if constants.PAD in target_field.vocab.stoi:
        constants.TARGET_PAD_ID = target_field.vocab.stoi[constants.PAD]


def load_vocabs(path, fields_tuples):
    vocab_path = Path(path, constants.VOCAB)

    # load vocabs for each field and transform it to dict to access it easily
    vocabs = torch.load(str(vocab_path),
                        map_location=lambda storage, loc: storage)
    vocabs = dict(vocabs)

    # set field.vocab to its correct vocab object
    for name, field in fields_tuples:
        if field.use_vocab:
            if name == 'words_expl':
                field.vocab = vocabs['words']
                continue
            field.vocab = vocabs[name]

    # transform fields_tuples to a dict in order to access fields easily
    dict_fields = dict(fields_tuples)

    # ensure global constants to their correct value
    words_field = dict_fields['words']
    target_field = dict_fields['target']
    constants.PAD_ID = words_field.vocab.stoi[constants.PAD]
    if constants.PAD in target_field.vocab.stoi:
        constants.TARGET_PAD_ID = target_field.vocab.stoi[constants.PAD]


def save_vocabs(path, fields_tuples):
    # list of fields name and their vocab
    vocabs = []
    for name, field in fields_tuples:
        if field.use_vocab:
            vocabs.append((name, field.vocab))

    # save vectors in a temporary dict and save the vocabs
    vectors = {}
    for name, vocab in vocabs:
        vectors[name] = vocab.vectors
        vocab.vectors = None
    vocab_path = Path(path, constants.VOCAB)
    torch.save(vocabs, str(vocab_path))

    # restore vectors -> useful if we want to use fields later
    for name, vocab in vocabs:
        vocab.vectors = vectors[name]
