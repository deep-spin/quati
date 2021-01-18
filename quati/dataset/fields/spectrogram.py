import torchaudio
from torchtext.data import Field


class SpectrogramField(Field):
    """
    Defines a field for a spectrogram. The input should be a waveform.

    Args:
        pkwargs (dict): kwargs for torchaudio.transforms.Spectrogram
        fkwargs (dict): kwargs for torchtext.data.Field

    """
    def __init__(self, pkwargs=None, fkwargs=None):
        fkwargs = {} if fkwargs is None else fkwargs
        pkwargs = {} if pkwargs is None else pkwargs
        preprocessing = torchaudio.transforms.Spectrogram(**pkwargs)
        super().__init__(preprocessing=preprocessing,
                         unk_token=None,
                         pad_token=None,
                         batch_first=True,
                         use_vocab=False,
                         sequential=False,
                         **fkwargs)
