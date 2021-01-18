from torchtext.data import Field

from quati import constants
from quati.dataset.vocabulary import Vocabulary


class TagsField(Field):
    """
    Defines a field for text tags with default values from constant.py and
    with the vocabulary defined in vocabulary.py.
    """
    def __init__(self, **kwargs):
        super().__init__(unk_token=None,
                         pad_token=constants.PAD,
                         sequential=True,
                         is_target=True,
                         batch_first=True,
                         **kwargs)
        self.vocab_cls = Vocabulary
