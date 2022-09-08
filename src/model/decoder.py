import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, decode_units, encode_units, batch_size):
        self.batch_size = batch_size
        self.decode_units = decode_units
        self.encode_units = encode_units
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.gru = nn.GRU(self.embedding_dim+self.encode_units, self.decode_units, batch_first=True)
        
        self.fc = nn.Linear(self.encode_units, self.vocab_size)
        
        # layers for attentions
        self.W1 = nn.Linear(self.encode_units, self.decode_units)
        self.W2 = nn.Linear(self.encode_units, self.decode_units)
        self.V = nn.Linear(self.encode_units, 1)
        
    def forward(self, x, hidden, encode_output):
        """param

        Args:
            x (tensor): _description_
            hidden (tensor): (batch_size, hidden_size)
            # encode_output (tensor): (max_length, batch_size, encode_units)
            encode_output (tensor): (batch_size, max_length, encode_units)
        """
        
        # encode_output = encode_output.permute(1, 0, 2)
        #  hidden with time axis == (batch_size, 1, hidden_size)
        hidden_with_time_axis = hidden.permute(1, 0, 2)
        
        score = torch.tanh(self.W1(encode_output) + self.W2(hidden_with_time_axis))
        
        attention_weights = torch.softmax(self.V(score), dim=1)
        
        context_vector = attention_weights * encode_output
        context_vector = torch.sum(context_vector, dim=1)
        
        # compute attention done
        x = self.embedding(x)
        x = torch.cat((context_vector.unsqueeze(1), x), -1)
        output, state = self.gru(x)
        
        output = output.view(-1, output.size(2))
        output = self.fc(output)
        
        return output, state, attention_weights
    
    def initialize_hidden_state(self):
        return torch.zeros((1, self.batch_sz, self.dec_units))
        