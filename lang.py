import tqdm
import torch
import torchtext
from logger import logger

class Lang(object):
    def __init__(self):
        super(Lang, self).__init__()
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentences(self, sentences_lst):
        logger.info('started adding words')
        for sentence in tqdm.tqdm(sentences_lst):
            for word in sentence.split(' '):
                self.addWord(word)
        logger.info('Finished')
        self.create_embedding_matrix(torchtext.vocab.FastText(), 300)

    def addWord(self, word):

        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def create_embedding_matrix(self, pretrained_model, embedding_size):
        self.embedding_matrix = torch.zeros(self.n_words, embedding_size)
        logger.info('started creating embeddings matrix')
        for key, value in tqdm.tqdm(self.word2index.items()):
            if key == 'EOS':
                self.embedding_matrix[value] = pretrained_model['</s>']
            elif key == 'SOS':
                self.embedding_matrix[value] = torch.rand(1, 300)
            elif key == 'PAD':
                self.embedding_matrix[value] = torch.rand(1, 300)
            else:
                self.embedding_matrix[value] = pretrained_model[key]
        logger.info('Finished')