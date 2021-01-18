import nltk
import torchtext

from quati.dataset.fields.words import WordsField
from quati.dataset.fields.tags import TagsField
from quati.dataset.corpora.corpus import Corpus
from xml.etree import ElementTree


class AGNewsCorpus(Corpus):
    task = 'doc'

    @staticmethod
    def create_fields_tuples():
        tokenizer = nltk.WordPunctTokenizer()
        fields_tuples = [
            ('words', WordsField(tokenize=tokenizer.tokenize)),
            ('target', TagsField())
        ]
        return fields_tuples

    def _read(self, file):
        root = ElementTree.parse(file).getroot()
        categories = [x.text for x in root.iter('category')]
        descriptions = [x.text for x in root.iter('description')]
        for text, label in zip(descriptions, categories):
            if text is None or label is None:
                continue
            # business vs world (binary classification)
            if label not in ['Business', 'World']:
                continue
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
        AGNewsCorpus,
        '../../../data/corpus/agnews/test.xml',
        lazy=False,
    )
