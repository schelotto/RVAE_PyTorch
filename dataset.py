from torchtext.data import Dataset
from urllib.request import urlretrieve
from torchtext.data import Field
from torchtext.vocab import GloVe
from typing import Optional

import torchtext.data as data
import os

class dataset(Dataset):
    dirname = 'data/'

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self,
                 train='ptb.train.txt', test='ptb.test.txt', valid='ptb.valid.txt',
                 path=None, examples=None, **kwargs):

        self.text_field = Field(init_token='<sos>',
                                eos_token='<eos>',
                                batch_first=True,
                                tokenize=lambda x:x.split())

        fields = [('text', self.text_field)]

        path = self.dirname if path is None else path
        if not os.path.isdir(path):
            os.mkdir(path)

        url = 'https://raw.githubusercontent.com/tomsercu/lstm/master/data/'

        for file in [train, test, valid]:
            if not os.path.isfile(os.path.join(path, file)):
                print('Downloading {} data...'.format(file.split('.')[1]), end='')
                urlretrieve(url + file, os.path.join(path, file))
                print('Done')
            else:
                pass

        if examples is None:
            examples = []
            with open(path + '/' + train, errors='ignore') as f:
                examples += [data.Example.fromlist([line], fields) for line in f]
            with open(os.path.join(path, valid), errors='ignore') as f:
                examples += [data.Example.fromlist([line], fields) for line in f]
            with open(os.path.join(path, test), errors='ignore') as f:
                examples += [data.Example.fromlist([line], fields) for line in f]

        super(dataset, self).__init__(fields=fields, examples=examples, **kwargs)

    @classmethod
    def splits(cls,
               path=None,
               root='.data',
               **kwargs):

        examples = cls(**kwargs).examples
        train_index = 42068
        valid_index = 3370
        test_index = 3761

        return (cls(examples=examples[:train_index]),
                cls(examples=examples[train_index:(train_index + valid_index)]),
                cls(examples=examples[-test_index:]))

    @classmethod
    def ptb(cls,
            batch_size=32,
            device=-1,
            vector: Optional[str] = None,
            **kwargs):

        text_field = cls(**kwargs).text_field

        train, valid, test = cls.splits(**kwargs)
        train_iter, valid_iter, test_iter = data.BucketIterator.splits((train, valid, test),
                                                                       batch_sizes=(batch_size, batch_size, batch_size),
                                                                       device=device,
                                                                       **kwargs)
        if vector == 'glove_6B':
            vectors = GloVe('6B', dim=300)
        elif vector == 'glove_840B':
            vectors = GloVe('840B', dim=300)
        elif vector == 'glove_42B':
            vectors = GloVe('42B', dim=300)

        try:
            text_field.build_vocab(train, valid, test, vectors=vectors)
        except UnboundLocalError:
            text_field.build_vocab(train, valid, test)

        return (train_iter, valid_iter, test_iter), text_field

if __name__ == '__main__':
    TEXT = Field(init_token='<sos>',
                 eos_token='<eos>',
                 batch_first=True,
                 tokenize=lambda x:x.split())
    (train, valid, test), TEXT = dataset.ptb(TEXT)
    print(len(TEXT.vocab.stoi))
    print(len(train))