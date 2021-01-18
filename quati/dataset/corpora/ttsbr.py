import torchtext

from quati.dataset.fields.words import WordsField
from quati.dataset.fields.tags import TagsField
from quati.dataset.corpora.corpus import Corpus


class TTSBRCorpus(Corpus):
    task = 'doc'

    @staticmethod
    def create_fields_tuples():
        fields_tuples = [
            ('words', WordsField()),
            ('target', TagsField())
        ]
        return fields_tuples

    def _read(self, file):
        for line in file:
            line = line.strip().split()
            label = line[0]
            text = ' '.join(line[2:])
            yield self.make_torchtext_example(text, label)

    def make_torchtext_example(self, text, label=None):
        ex = {'words': text, 'target': label}
        if 'target' not in self.fields_dict.keys():
            del ex['target']
        assert ex.keys() == self.fields_dict.keys()
        return torchtext.data.Example.fromdict(ex, self.fields_dict)


if __name__ == '__main__':
    from quati.dataset.corpora.test_corpus import quick_test
    quick_test(
        TTSBRCorpus,
        '../../../data/corpus/ttsbr/trainTT.txt',
        lazy=True,
    )
