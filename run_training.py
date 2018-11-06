import argparse
import pandas as pd
import time
from logger import logger
from torch.utils.data import DataLoader
from lang import *
from dataset import *
from encoder import *
from decoder import *
from torch import optim
from train import *

USE_CUDA = 1
USE_PRETRAINED = False
PAD_token = 0
SOS_token = 1
EOS_token = 2
MAX_LENGTH = 15


if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data')
    parser.add_argument('--batch_size')
    parser.add_argument('--n_layers')
    parser.add_argument('--sample_ratio', default=0.15)
    parser.add_argument('--save_path_encoder')
    parser.add_argument('--save_path_decoder')
    parser.add_argument('--save_path_optimizer_encoder')
    parser.add_argument('--save_path_optimizer_decoder')
    args = vars(parser.parse_args())

    attn_model = 'dot'
    hidden_size = 300
    n_layers = int(args['n_layers'])
    dropout = 0.1
    batch_size = int(args['batch_size'])

    # Configure training/optimization
    clip = 50.0
    teacher_forcing_ratio = 0.5
    learning_rate = 0.001
    decoder_learning_ratio = 5.0
    n_epochs = 100

    logger.info('Reading data')
    df_all = pd.read_csv(args['input_data'])
    df_all.dropna(inplace = True)
    lang1 = Lang()
    logger.info('Creating embeddings')
    lang1.addSentences(df_all.sample(frac = float(args['sample_ratio']), random_state=123)['headline'].values.tolist())
    dataset = ClickBaitDataset(df_all.sample(frac = float(args['sample_ratio']), random_state=123), lang1, EOS_token,
                               PAD_token, MAX_LENGTH)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=16, drop_last=True)
    logger.info('Finished')
    if USE_PRETRAINED:
        encoder = EncoderRNN(lang1.n_words, hidden_size, n_layers, dropout, lang1.embedding_matrix)
        decoder = DecoderRNN(hidden_size, lang1.n_words, dropout, lang1.embedding_matrix)
    else:
        encoder = EncoderRNN(lang1.n_words, hidden_size, n_layers, dropout)
        decoder = DecoderRNN(hidden_size, lang1.n_words, dropout)

    # Initialize optimizers and criterion
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)

    try:
        encoder_state = torch.load(args['save_path_optimizer_encoder'], map_location="cpu")
        encoder.load_state_dict(encoder_state)

        decoder_state = torch.load(args['save_path_optimizer_decoder'], map_location="cpu")
        decoder.load_state_dict(decoder_state)

        encoder_optimizer_state = torch.load(args['save_path_optimizer_encoder'], map_location="cpu")
        encoder_optimizer.load_state_dict(encoder_optimizer_state)

        decoder_optimizer_state = torch.load(args['save_path_optimizer_decoder'], map_location="cpu")
        encoder_optimizer.load_state_dict(decoder_optimizer_state)
        from_scratch = False
        print('Continue training')
    except Exception as e:
        print(e)
        print('From scratch')
        pass

    if USE_CUDA:
        encoder = encoder.cuda()
        decoder = decoder.cuda()

    print_loss_total = 0
    save_every = 500
    batch_n = 0
    epoch = 0
    print_every = 10
    start = time.time()

    while epoch < n_epochs:
        epoch += 1
        for batch in tqdm.tqdm(dataloader):
            # Get training data for this cycle
            print(batch, batch[0], batch[1])
            input_batches, input_lengths = torch.cat(batch[0]), batch[1].numpy().tolist()
            input_batches = input_batches.transpose(0, 1)
            # Run the train function
            loss = train(
                input_batches, input_lengths, input_batches, input_lengths,
                encoder, decoder,
                encoder_optimizer, decoder_optimizer, batch_size, clip
            )

            # Keep track of loss
            print_loss_total += loss
            batch_n += 1

            if batch_n % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print_summary = 'Epoch:%d - Batch:%d - loss:%.4f' % (epoch, batch_n, print_loss_avg)
                print(print_summary)

            if batch_n % save_every == 0:
                torch.save(encoder.state_dict(), args['save_path_encoder'])
                torch.save(decoder.state_dict(), args['save_path_decoder'])
                torch.save(encoder_optimizer.state_dict(), args['save_path_optimizer_encoder'])
                torch.save(decoder_optimizer.state_dict(), args['save_path_optimizer_decoder'])
                print('Model saved on batch %d' % batch_n)

            torch.cuda.empty_cache()

