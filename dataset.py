from torchtext.data import Dataset
from urllib.request import urlretrieve

import torchtext.data as data
import os

class dataset(Dataset):
    dirname = 'data/'

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, text_field, path=None, examples=None, **kwargs):

        fields = [('text', text_field)]

        path = self.dirname if path is None else path
        if not os.path.isdir(path):
            os.mkdir(path)

        train_file = 'ptb.train.txt'
        test_file = 'ptb.test.txt'
        valid_file = 'ptb.valid.txt'
        url = 'https://raw.githubusercontent.com/tomsercu/lstm/master/data/'

        for file in [train_file, test_file, valid_file]:
            if not os.path.isfile(os.path.join(path, file)):
                print('Downloading {} data...'.format(file.split('.')[1]), end = '')
                urlretrieve(url + file, os.path.join(path, file))
                print('Done')
            else:
                pass

        if examples is None:
            examples = []
            with open(os.path.join(path, train_file), errors='ignore') as f:
                examples += [data.Example.fromlist(line, fields) for line in f]
                self.train_size = len(examples)
                print(self.train_size)
            with open(os.path.join(path, valid_file), errors='ignore') as f:
                examples += [data.Example.fromlist(line, fields) for line in f]
                self.valid_size = len(examples) - self.train_size
                print(self.valid_size)
            with open(os.path.join(path, test_file), errors='ignore') as f:
                examples += [data.Example.fromlist(line, fields) for line in f]
                self.test_size = len(examples) - self.train_size - self.valid_size
                print(self.test_size)

        super(dataset, self).__init__(fields=fields, examples=examples, **kwargs)

    @classmethod
    def splits(cls, text_field, path=None, root='.data', train=None, validation=None,
               test=None, **kwargs):
        examples = cls(text_field, **kwargs).examples
        train_index = 42068
        valid_index = 3370
        test_index = 3761

        return (cls(text_field, examples=examples[:train_index]),
                cls(text_field, examples=examples[train_index:(train_index + valid_index)]),
                cls(text_field, examples=examples[-test_index:]))