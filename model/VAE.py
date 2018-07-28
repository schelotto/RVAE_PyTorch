import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.binomial import Binomial

class WordDropout(nn.Module):
    def __init__(self,
                 unk_token = '<unk>',
                 p = 0.3):
        super(WordDropout, self).__init__()
        self.p = p
        self.unk_token = unk_token

    def forward(self, inputs, UNK_IDX = 1):
        if self.training:
            mask = Binomial(1, self.p * torch.ones_like(inputs)).sample().long()
            if torch.cuda.is_available():
                mask = mask.cuda()
            inputs[mask] = self.UNK_IDX
            return inputs
        else:
            return inputs

class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()

        self.vocab_size = args.vocab_size
        self.embed_dim = args.embed_dim
        self.rnn_dim = args.rnn_dim
        self.num_layer = args.num_layer
        self.bidirectional = args.bidirectional
        self.z_dim = args.z_dim
        self.pad = args.pad

        self.text_embedder = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=self.pad)
        self.word_dropout = WordDropout(p = args.p)

        self.rnn = nn.LSTM(self.embed_dim,
                           self.rnn_dim,
                           num_layers=self.num_layer,
                           bidirectional=self.bidirectional,
                           batch_first=True)

        self.hidden_dim = (2 if self.bidirectional else 1) * self.num_layer * self.rnn_dim
        self.proj_mu = nn.Linear(self.hidden_dim, self.z_dim)
        self.proj_logvar = nn.Linear(self.hidden_dim, self.z_dim)

    def forward(self,
                input: torch.Tensor):
        input = self.word_dropout(input)
        word_embed = self.text_embedder(input)
        _, (h_t, c_t) = self.rnn(word_embed)
        mu = self.proj_mu(h_t)
        logvar = self.proj_logvar(h_t)
        return h_t, (mu, logvar)

class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()

        self.rnn_dim = args.rnn_dim
        self.z_dim = args.z_dim
        self.vocab_size = args.vocab_size
        self.embed_dim = args.embed_dim
        self.pad = args.pad

        self.text_embedder = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=self.pad)
        self.input_dim = self.embed_dim + self.z_dim
        self.num_layer = args.num_layer
        self.bidirectional = args.bidirectional
        self.vocab_size = args.vocab_size

        self.dropout = nn.Dropout(p = args.p)
        self.rnn = nn.LSTM(self.input_dim,
                           self.rnn_dim,
                           num_layers=self.num_layer,
                           bidirectional=self.bidirectional,
                           batch_first=True)

        self.hidden_dim = (2 if self.bidirectional else 1) * self.num_layer * self.rnn_dim
        self.proj = nn.Linear(self.hidden_dim, self.vocab_size)

    def forward(self, input, z, init):
        input = self.dropout(input)
        batch_size, len, _ = input.size()

        z = torch.stack([z] * len, dim = 1)
        decoder_input = torch.cat([input, z], dim = -1)
        rnn_out, final_state = self.rnn(decoder_input, init)

        y = self.proj(rnn_out.contiguous().view(-1, self.hidden_dim))
        y = y.view(batch_size, len, self.vocab_size)

        return y

    def sample_(self, embed, z, h):
        input = torch.cat([embed, z], dim = -1)
        output, h = self.decoder(input, h)
        y = self.decoder.proj(output)
        return y, h

class RVAE(nn.Module):
    def __init__(self, args):
        super(RVAE, self).__init__()

        self.eos = args.eos
        self.sos = args.sos

        self.encoder = Encoder(args)
        self.decoder = Decoder(args)

        self.encoder.text_embedder.weight.requires_grad = False
        self.decoder.text_embedder.weight.requires_grad = False

    def kld_(self, mu, logvar):
        kld = (mu.pow(2) + logvar.exp() - logvar - 1).sum(1).mean()
        return kld

    def forward(self, input):
        h_t, (mu, logvar) = self.encoder(input)
        z_real = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        kld = self.kld_(mu, logvar)
        output, final_state = self.decoder(input, z_real, h_t)
        return output, kld, final_state

    def sample_sentence(self, z, temp = 1):
        """
        :param z: hidden code
        :param temp: temperature in softmax, default 1
        :return: sampled sequence of words
        """
        word = torch.LongTensor([self.sos])
        word = word.cuda() if torch.cuda.is_available() else word
        z = z.view(1, 1, -1)

