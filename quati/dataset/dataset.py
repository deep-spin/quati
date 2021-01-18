import warnings

from torchtext.data import interleave_keys
from quati.dataset.modules.lazy_dataset import LazyDataset


class BaseDataset(LazyDataset):
    """Defines a base dataset."""

    def __init__(self, examples, fields_tuples, filter_pred=None):
        """Create a dataset from a list of Examples and Fields.

        Args:
            examples: A list or a generator of examples. Usually, the output
                of corpus.read()
            filter_pred (callable or None): Use only examples for which
                filter_pred(example) is True, or use all examples if None.
                Default: None.
        """
        is_lazy = hasattr(examples, 'lazy') and examples.lazy is True
        super().__init__(examples, fields_tuples, filter_pred, not is_lazy)

    def __len__(self):
        try:
            return len(self.examples)
        except ValueError:
            warnings.warn("Corpus loaded in lazy mode and its length was not "
                          "determined yet. Returning 0 for now since in order "
                          "to calculate this number we'd have to go through "
                          "the entire dataset at least once, which can be very "
                          "expensive for large datasets.")
            return 0

    def get_loss_weights(self):
        """
        Get a weight for each class in order to have a balanced loss. This
        will be passed to the `weight` param in the loss function constructor.

        Returns:
            output of sklearn.utils.class_weight.compute_class_weight
        """
        from sklearn.utils.class_weight import compute_class_weight
        target_vocab = self.fields['target'].vocab.stoi
        y = [target_vocab[t] for ex in self.examples for t in ex.target]
        classes = list(set(y))
        return compute_class_weight('balanced', classes, y)

    @staticmethod
    def sort_key(ex):
        """
        key to use for sorting dataset examples for batching together
        examples with similar lengths to minimize padding. By default,
        it uses the the numbers of words (`ex.words`).

        Args:
            ex: torchtext's Example object.
        """
        raise NotImplementedError


class DocDataset(BaseDataset):
    @staticmethod
    def sort_key(ex):
        """Use the number of words as the criterion for sorting a batch."""
        return len(ex.words)


class EntailmentDataset(BaseDataset):
    @staticmethod
    def sort_key(ex):
        """Use the number of words in the premise and in the hypothesis as the
        criterion for sorting a batch. The example object must have a words_hyp
        attribute representing the words of the hypothesis. See SNLICorpus."""
        return interleave_keys(len(ex.words), len(ex.words_hyp))


class POSDataset(BaseDataset):
    @staticmethod
    def sort_key(ex):
        """Use the number of words as the criterion for sorting a batch."""
        return len(ex.words)
