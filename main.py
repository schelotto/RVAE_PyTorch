import argparse
import numpy as np

from tqdm import tqdm
from model.VAE import RVAE
from dataset import load_data
from torchtext.data import Field

parser = argparse.ArgumentParser(description='Recurrent VAE')
parser.add_argument('-embed_dim', type=int, default=300, help='Dimension of word embeddings. [default: 300]')
parser.add_argument('-rnn_dim', type=int, default=200, help='Dimension of RNN output. [default: 200]')
parser.add_argument('-num_layer', type=int, default=1, help='Number of RNN layer(s). [default: 1]')
parser.add_argument('-z_dim', type=int, default=300, help='Dimension of hidden code. [default: 300]')
parser.add_argument('-p', type=float, default=0.3, help='Dropout probability. [default: 0.3]')
parser.add_argument('-bidirectional', default=False, action='store_true', help='Use bidirectional RNN.')
parser.add_argument('-word_emb', type=str, default='glove_840B',
                    help='Word embedding name. In glove_840B, glove_6B or glove_42B')
parser.add_argument('-save-file', type=str, default=None, help='File path/name for model to be saved.')
parser.add_argument('-log-interval', type=int, default=1000, help='Number of iterations to sample generated sentences')
args = parser.parse_args()

(train_iter, valid_iter, test_iter), text_field = load_data(args.word_emb)
args.vocab_size = len(text_field.vocab.stoi)
args.sos = text_field.vocab.stoi['<sos>']
args.eos = text_field.vocab.stoi['<eos>']
args.pad = text_field.vocab.stoi['<pad>']

rvae = RVAE(args)