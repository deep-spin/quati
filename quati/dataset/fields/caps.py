from torchtext.data import Field

from quati import constants
from quati.dataset.vocabulary import Vocabulary


class CapsField(Field):
    """
    Defines a field for capitalization (upper, lower, etc.) by setting only
    unk_token and pad_token to their default constant value.
    """
    def __init__(self, **kwargs):
        super().__init__(unk_token=constants.UNK,
                         pad_token=constants.PAD,
                         batch_first=True,
                         **kwargs)
        self.vocab_cls = Vocabulary
