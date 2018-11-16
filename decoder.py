import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout, embedding_layer):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding = embedding_layer
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        batch_size = input.size(0)
        output = self.embedding(input).view(1, 1, -1)
        output = self.embedding_dropout(output)
        output = output.view(1, batch_size, self.hidden_size)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output[0])
        return output.squeeze(0), hidden

    def initHidden(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.zeros(1, 1, self.hidden_size, device=device)



