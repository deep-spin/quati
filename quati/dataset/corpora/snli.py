import json

import nltk
import torchtext

from quati.dataset.fields.words import WordsField
from quati.dataset.fields.tags import TagsField
from quati.dataset.corpora.corpus import Corpus


class SNLICorpus(Corpus):
    task = 'entailment'

    @staticmethod
    def create_fields_tuples():
        # note that words and words_hyp share the same field, therefore when we
        # call words_field.build_vocab() we are creating a shared vocab.
        # Hence, words.vocab == words_hyp.vocab
        tokenizer = nltk.WordPunctTokenizer()
        words_field = WordsField(tokenize=tokenizer.tokenize)
        fields_tuples = [
            ('words', words_field),
            ('words_hyp', words_field),
            ('target', TagsField())
        ]
        return fields_tuples

    def _read(self, file):
        for line in file:
            data = json.loads(line)
            label = data['gold_label']
            premise = data['sentence1']
            hypothesis = data['sentence2']
            if label == '-':
                # These were cases where the annotators disagreed; we'll just
                # skip them. It's like 800 / 500k examples in the training data
                continue
            yield self.make_torchtext_example(premise, hypothesis, label)

    def make_torchtext_example(self, prem, hyp, label):
        ex = {'words': prem, 'words_hyp': hyp, 'target': label}
        if 'target' not in self.fields_dict.keys():
            del ex['target']
        assert ex.keys() == self.fields_dict.keys()
        return torchtext.data.Example.fromdict(ex, self.fields_dict)


if __name__ == '__main__':
    from quati.dataset.corpora.test_corpus import quick_test
    quick_test(
        SNLICorpus,
        '../../../data/corpus/snli/snli_1.0_test.jsonl',
        lazy=True,
    )
