import torch
import torch.nn as nn

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, dropout=0.1, embedding_weights=None):
        super(EncoderRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        if embedding_weights is None:
            self.embedding = nn.Embedding(input_size, hidden_size)
        else:
            self.embedding = nn.Embedding(input_size, hidden_size)
            self.embedding.load_state_dict({'weight': embedding_weights})
            self.embedding.weight.requires_grad = True

        self.gru = nn.LSTM(hidden_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=True)

    def forward(self, input_seqs, input_lengths, hidden=None, cell=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        embedded = self.embedding(input_seqs)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        outputs, (hidden, cell) = self.gru(packed)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)  # unpack (back to padded)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]  # Sum bidirectional outputs
        return outputs, hidden, cell

class NMTEncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, dropout=0.1, embedding_weights=None):
        super(NMTEncoderRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        if embedding_weights is None:
            self.embedding = nn.Embedding(input_size, hidden_size)
        else:
            self.embedding = nn.Embedding(input_size, hidden_size)
            self.embedding.load_state_dict({'weight': embedding_weights})
            self.embedding.weight.requires_grad = True

        self.lstm1 = nn.LSTM(hidden_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=False)

    def forward(self, input_seqs, input_lengths, hidden=None, cell=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        embedded = self.embedding(input_seqs)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        outputs, (hidden, cell) = self.lstm1(packed)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)  # unpack (back to padded)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        outputs += outputs
        packed = torch.nn.utils.rnn.pack_padded_sequence(outputs, output_lengths)
        outputs, (hidden, cell) = self.lstm2(packed, (hidden, cell))
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)  # unpack (back to padded)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]  # Sum bidirectional outputs
        return outputs, hidden, cell