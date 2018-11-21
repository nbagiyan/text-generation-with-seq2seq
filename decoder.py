import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout, n_layers = 1, dropout_p = 0.1, embedding_weights = None):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        if embedding_weights is None:
            self.embedding = nn.Embedding(output_size, hidden_size)
        else:
            self.embedding = nn.Embedding(output_size, hidden_size)
            self.embedding.load_state_dict({'weight': embedding_weights})
            self.embedding.weight.requires_grad = True

        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.LSTM(hidden_size, hidden_size, n_layers, dropout=dropout_p)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden, cell):
        batch_size = input.size(0)
        output = self.embedding(input).view(1, 1, -1)
        output = self.embedding_dropout(output)
        output = output.view(1, batch_size, self.hidden_size)
        output = F.relu(output)
        output, (hidden, cell) = self.gru(output, hidden, cell)
        output = self.out(output[0])
        return output.squeeze(0), hidden, cell

    def initHidden(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.zeros(1, 1, self.hidden_size, device=device)



