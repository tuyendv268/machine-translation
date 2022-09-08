from base64 import encode
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import torch

class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim,encode_units, batch_size):
        self.batch_size = batch_size
        self.encode_units = encode_units
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.gru = nn.GRU(self.embedding_dim, self.encode_units)
        
    def forward(self, x, lens, device):
        """
        Args:
            x (tensor): (batch_size, max_length, embedding_dim)
            lens (int): length of sentences
            device (device): cuda id
        """
        
        self.hidden = self.initialize_hidden_state(device)
        
        x = self.embedding(x)
        
        x = pack_padded_sequence(x, lens) # unpad
        output, self.hidden = self.gru(x, self.hidden) # gru returns hidden state of all timesteps as well as hidden state at last timestep
        
        output, _ = pad_packed_sequence(output)
        return output, self.hidden
    
    def initialize_hidden_state(self, device):
        return torch.zeros((1, self.batch_sz, self.enc_units)).to(device)
        