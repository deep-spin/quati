from itertools import chain
from pathlib import Path

import nltk
import torchtext

from quati.dataset.fields.words import WordsField
from quati.dataset.fields.tags import TagsField
from quati.dataset.corpora.corpus import Corpus


def create_single_file_for_pos_and_neg(corpus_path):
    new_file_path = Path(corpus_path, 'data.txt')
    # do not create this file again if it is already there
    if not new_file_path.exists():
        neg_files = sorted(Path(corpus_path, 'neg').glob('*.txt'))
        pos_files = sorted(Path(corpus_path, 'pos').glob('*.txt'))
        paths = chain(neg_files, pos_files)
        new_file = new_file_path.open('w', encoding='utf8')
        for file_path in paths:
            content = file_path.read_text().strip()
            content = content.replace('<br>', ' <br> ')
            content = content.replace('<br >', ' <br> ')
            content = content.replace('<br />', ' <br> ')
            content = content.replace('<br/>', ' <br> ')
            label = '1' if 'pos' in str(file_path) else '0'
            new_file.write(label + ' ' + content + '\n')
        new_file.seek(0)
        new_file.close()
    return new_file_path


class IMDBCorpus(Corpus):
    task = 'doc'

    @staticmethod
    def create_fields_tuples():
        # if you choose tokenizer='spacy', please install the en package:
        # python3 -m spacy download en
        tokenizer = nltk.WordPunctTokenizer()
        # tokenizer = nltk.TreebankWordTokenizer()
        fields_tuples = [
            ('words', WordsField(tokenize=tokenizer.tokenize)),
            ('target', TagsField())
        ]
        return fields_tuples

    def read(self, corpus_path):
        """
        First, read the positive and negative examples, which are located in
        different folders: `pos/` and `neg/`.

        Second, split the `<br>` tags from other tokens.

        Third, save a new file called `data.txt` in the root directory, with
        the following structure:
            label_0 text_0
            label_1 text_1
            ...
            label_M text_M

        Args:
            corpus_path: path to the root directory where `pos/` and `neg/`
                are located.
        """
        new_file_path = create_single_file_for_pos_and_neg(corpus_path)
        self.corpus_path = str(new_file_path)
        self.open(self.corpus_path)
        if self.lazy is True:
            return self
        else:
            return list(self)

    def _read(self, file):
        for line in file:
            line = line.strip().split()
            if line:
                label = line[0]
                text = ' '.join(line[1:])
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
        IMDBCorpus,
        '../../../data/corpus/imdb/test/',
        lazy=True,
    )
