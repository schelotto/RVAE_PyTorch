import argparse
import numpy as np
import torch
import torch.nn.functional as F

from tqdm import tqdm
from model.VAE import RVAE
from dataset import load_data
from torchtext.data import Field
from itertools import chain

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Recurrent VAE')
    parser.add_argument('-embed_dim', type=int, default=300, help='Dimension of word embeddings. [default: 300]')
    parser.add_argument('-rnn_dim', type=int, default=300, help='Dimension of RNN output. [default: 300]')
    parser.add_argument('-num_layer', type=int, default=1, help='Number of RNN layer(s). [default: 1]')
    parser.add_argument('-z_dim', type=int, default=300, help='Dimension of hidden code. [default: 300]')
    parser.add_argument('-p', type=float, default=0.3, help='Dropout probability. [default: 0.3]')
    parser.add_argument('-bidirectional', default=False, action='store_true', help='Use bidirectional RNN.')
    parser.add_argument('-word_emb', type=str, default='glove_840B',
                        help='Word embedding name. In glove_840B, glove_6B or glove_42B')
    parser.add_argument('-save-file', type=str, default=None, help='File path/name for model to be saved.')
    parser.add_argument('-log-interval', type=int, default=100, help='Number of iterations to sample generated sentences')
    parser.add_argument('-train_iter', type=int, default=40000, help='Number of iterations for training')
    args = parser.parse_args()

    (train_iter, _, _), text_field = load_data(args.word_emb)
    args.vocab_size = len(text_field.vocab.stoi)
    print('Vocabulary size: {}'.format(args.vocab_size))
    args.sos = text_field.vocab.stoi['<sos>']
    args.eos = text_field.vocab.stoi['<eos>']
    args.pad = text_field.vocab.stoi['<pad>']
    args.unk = text_field.vocab.stoi['<unk>']

    rvae = RVAE(args)

    kld_start_inc = 5000
    kld_max = 0.15
    kld_weight = 0.001
    kld_inc = (kld_max - kld_weight) / (args.train_iter - kld_start_inc)

    ###########################
    ## ASSIGN WORD EMBEDDING ##
    ###########################

    rvae.decoder.text_embedder.weight.data = text_field.vocab.vectors.data
    rvae.encoder.text_embedder.weight.data = text_field.vocab.vectors.data

    ###########################
    ## ASSIGN ADAM OPTIMIZER ##
    ###########################

    rvae.decoder.text_embedder.weight.requires_grad = False
    rvae.encoder.text_embedder.weight.requires_grad = False

    update_params = filter(lambda x:x.requires_grad,  chain(rvae.encoder.parameters(), rvae.decoder.parameters()))
    optim = torch.optim.Adam(update_params, lr = 1e-3)

    if torch.cuda.is_available():
        rvae = rvae.cuda()

    kld_weight = 0.001

    for it in range(args.train_iter):
        batch = next(train_iter)
        text = batch.text

        if torch.cuda.is_available():
            text = text.cuda()

        output, kld = rvae(text)

        if it > kld_start_inc and kld_weight < kld_max:
            kld_weight += kld_inc

        ce_loss = F.cross_entropy(output, text.view(-1), size_average=False)
        kl_loss = kld_weight * kld
        loss = ce_loss + kl_loss

        optim.zero_grad()
        loss.backward()
        optim.step()

        if (it + 1) % args.log_interval == 0:
            print('Iter {}/{} Recon Loss {:.4f} KL Loss {:.4f}'
                  .format(it + 1, args.train_iter, ce_loss.data.item(), kld.data.item()))

            z = torch.randn(1, args.z_dim)
            sampled_indices = rvae.sample_sentence(z)
            sampled_sentence = list(map(lambda x:text_field.itos[x], sampled_indices))
            print('Sample: {}'.format(' '.join(sampled_sentence)))