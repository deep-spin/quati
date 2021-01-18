from torchtext.data import Field

from quati.dataset.vocabulary import Vocabulary


class LabelField(Field):
    """
    Defines a field for text labels. Equivalent to torchtext's LabelField but
    with my own vocabulary.
    """
    def __init__(self, **kwargs):
        super().__init__(unk_token=None,
                         pad_token=None,
                         sequential=False,
                         is_target=True,
                         batch_first=True,
                         **kwargs)
        self.vocab_cls = Vocabulary
