from quati.dataset.corpora.agnews import AGNewsCorpus
from quati.dataset.corpora.imdb import IMDBCorpus
from quati.dataset.corpora.mnli import MNLICorpus
from quati.dataset.corpora.snli import SNLICorpus
from quati.dataset.corpora.sst import SSTCorpus
from quati.dataset.corpora.ttsbr import TTSBRCorpus
from quati.dataset.corpora.yelp import YelpCorpus

available_corpora = {
    'agnews': AGNewsCorpus,
    'imdb': IMDBCorpus,
    'snli': SNLICorpus,
    'mnli': MNLICorpus,
    'sst': SSTCorpus,
    'ttsbr': TTSBRCorpus,
    'yelp': YelpCorpus,
}
