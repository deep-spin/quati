from torchtext.data import Field

from quati import constants
from quati.dataset.vocabulary import Vocabulary


class WordsField(Field):
    """
    Defines a field for word tokens with default values from constant.py and
    with the vocabulary defined in vocabulary.py.
    """
    def __init__(self, **kwargs):
        super().__init__(unk_token=constants.UNK,
                         pad_token=constants.PAD,
                         batch_first=True,
                         **kwargs)
        self.vocab_cls = Vocabulary
