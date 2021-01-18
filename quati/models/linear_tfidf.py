import torch

from quati.dataset.corpora import available_corpora
from quati.models import LinearBoW


class LinearTfIdf(LinearBoW):
    """
    Linear TF-IDF.
    """

    def __init__(self, fields_tuples, options):
        super().__init__(fields_tuples, options)

        self.idf = torch.zeros(self.words_vocab_size, device=options.gpu_id)
        self.nb_documents = 0
        self.read_corpus_and_populate_counts(options)
        div = torch.div(self.nb_documents, self.idf)
        assert(torch.all(div >= 1))
        self.idf = torch.log(div)
        self.idf[self.idf == float('inf')] = 1e-7  # smoothing
        self.idf = self.idf.unsqueeze(0)

    def read_corpus_and_populate_counts(self, options):
        dummy_corpus_cls = available_corpora[options.corpus]
        dummy_fields_tuples = dummy_corpus_cls.create_fields_tuples()
        dummy_corpus = dummy_corpus_cls(dummy_fields_tuples, lazy=True)
        for ex in dummy_corpus.read(options.train_path):
            for w in set(ex.words):
                w_id = self.fields_dict['words'].vocab.stoi[w]
                self.idf[w_id] += 1
        self.nb_documents = len(dummy_corpus)
        dummy_corpus.close()

    def get_bow(self, words):
        bow = super().get_bow(words)
        tfidf = bow * self.idf.unsqueeze(1)
        return tfidf
