import torch
from torch.utils.data import Dataset



class СustomDataset(Dataset):
    def __init__(self, df, lang, EOS_token, PAD_token, MAX_LENGTH):
        super(СustomDataset, self).__init__()
        self.df = df
        self.lang = lang
        self.EOS_token = EOS_token
        self.PAD_token = PAD_token
        self.MAX_LENGTH = MAX_LENGTH

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        x, y = self.df.iloc[idx, 0], self.df.iloc[idx, 1]
        source = [self.lang.word2index[token] for token in x.split(' ')]
        source, length = self.__pad_item(source)
        return {'input' : torch.LongTensor(source), 'length' : length}

    def __pad_item(self, x):
        if len(x) >= self.MAX_LENGTH:
            x = x[:self.MAX_LENGTH]
            x.append(self.EOS_token)
            return x, len(x)
        else:
            x.append(self.EOS_token)
            length = len(x)
            return x + [self.PAD_token] * (self.MAX_LENGTH - len(x) + 1), length

