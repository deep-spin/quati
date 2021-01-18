from quati.dataset.corpora import available_corpora
from quati.dataset.corpora.text import TextCorpus
from quati.dataset.corpora.text_pair import TextPairCorpus
from quati.dataset.dataset import DocDataset, EntailmentDataset, POSDataset

available_datasets = {
    'doc': DocDataset,
    'entailment': EntailmentDataset,
    'pos': POSDataset
}


def filter_len_wrapper(options):
    def filter_example(ex):
        return options.min_length <= len(ex.words) <= options.max_length
    return filter_example


def build(path, fields_tuples, options):
    corpus_cls = available_corpora[options.corpus]
    corpus = corpus_cls(fields_tuples, lazy=options.lazy_loading)
    examples = corpus.read(path)
    dataset_cls = available_datasets[corpus.task]
    return dataset_cls(
        examples, fields_tuples, filter_pred=filter_len_wrapper(options)
    )


def build_texts(texts, fields_tuples, options):
    corpus = TextCorpus(fields_tuples, lazy=options.lazy_loading)
    examples = corpus.read(texts)
    return DocDataset(
        examples, fields_tuples, filter_pred=filter_len_wrapper(options)
    )


def build_pair_texts(texts_ab, fields_tuples, options):
    corpus = TextPairCorpus(fields_tuples, lazy=options.lazy_loading)
    examples = corpus.read(texts_ab)
    return EntailmentDataset(
        examples, fields_tuples, filter_pred=filter_len_wrapper(options)
    )

