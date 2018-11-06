import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout, embedding_weights = None):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        if embedding_weights is None:
            self.embedding = nn.Embedding(output_size, hidden_size)
        else:
            self.embedding = nn.Embedding(output_size, hidden_size)
            self.embedding.load_state_dict({'weight': embedding_weights})
            self.embedding.weight.requires_grad = True

        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = self.embedding_dropout(output)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output[0])
        return output.squeeze(0), hidden

    def initHidden(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.zeros(1, 1, self.hidden_size, device=device)



