import argparse
import pandas as pd
import time
from evaluate import evaluate
from loader_state import create_correct_state_dict
from torch.utils.data import DataLoader
from lang import *
from dataset import *
from encoder import *
from decoder import *
from torch import optim
from train import *

USE_CUDA = 1
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
    parser.add_argument('--save_path_i2w')
    parser.add_argument('--save_path_w2i')
    parser.add_argument('--save_path_train')
    parser.add_argument('--save_path_val')
    parser.add_argument('--use_pretrained')

    args = vars(parser.parse_args())

    hidden_size = 300
    n_layers = int(args['n_layers'])
    dropout = 0.1
    batch_size = int(args['batch_size'])
    USE_PRETRAINED = int(args['use_pretrained'])

    # Configure training/optimization
    clip = 50.0
    teacher_forcing_ratio = 0.5
    learning_rate = 0.001
    decoder_learning_ratio = 5.0
    n_epochs = 100
    evaluate_every = 10

    logger.info('Reading data')
    df_all = pd.read_csv(args['input_data'])
    df_all.dropna(inplace = True)
    lang1 = Lang(args['save_path_w2i'], args['save_path_i2w'])
    logger.info('Creating embeddings')
    df_sample = df_all.sample(frac = float(args['sample_ratio']), random_state=123)

    df_train = df_sample.copy()
    df_train = df_train.iloc[:-100000, :].copy()
    df_val = df_sample.copy()
    df_val = df_val.iloc[-100000:, :].copy()

    df_train.to_csv(args['save_path_train'], index=False)
    df_val.to_csv(args['save_path_val'], index=False)

    lang1.addSentences(df_sample['headline'].values.tolist())

    dataset_train = ClickBaitDataset(df_train, lang1, EOS_token,PAD_token, MAX_LENGTH)
    dataset_val = ClickBaitDataset(df_val, lang1, EOS_token,PAD_token, MAX_LENGTH)

    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)

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
        logger.info('Trying to load model')
        encoder_state = torch.load(args['save_path_encoder'], map_location="cpu")
        encoder.load_state_dict(encoder_state)

        decoder_state = torch.load(args['save_path_decoder'], map_location="cpu")
        decoder.load_state_dict(decoder_state)

        if USE_CUDA:
            encoder = encoder.cuda()
            decoder = decoder.cuda()

        encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
        encoder_optimizer_state = torch.load(args['save_path_optimizer_encoder'], map_location="cpu")
        encoder_optimizer.load_state_dict(encoder_optimizer_state)
        encoder_optimizer = create_correct_state_dict(encoder_optimizer)

        decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
        decoder_optimizer_state = torch.load(args['save_path_optimizer_decoder'], map_location="cpu")
        decoder_optimizer.load_state_dict(decoder_optimizer_state)
        decoder_optimizer = create_correct_state_dict(decoder_optimizer)

        from_scratch = False
        logger.info('Continue training')
    except Exception as e:
        logger.exception(e)
        logger.info('From scratch')
        pass

    if USE_CUDA:
        encoder = encoder.cuda()
        decoder = decoder.cuda()

    logger.info('started training')

    print_loss_total = 0
    save_every = 500
    batch_n = 0
    epoch = 0
    print_every = 1
    start = time.time()

    while epoch < n_epochs:
        epoch += 1
        for batch in dataloader_train:
            # Get training data for this cycle
            logger.info('creating batches')
            input_batches, input_lengths = batch['input'], batch['length'].numpy().tolist()
            input_batches, input_lengths = zip(*sorted(zip(input_batches, input_lengths), key=lambda x: x[1], reverse=True))
            input_batches, input_lengths = torch.stack(input_batches), list(input_lengths)
            input_batches = input_batches[:, :max(input_lengths)]
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
                print_summary = 'Epoch:%d - Batch:%d - Train_loss:%.4f' % (epoch, batch_n, print_loss_avg)
                logger.info(print_summary)

            if batch_n % save_every == 0:
                torch.save(encoder.state_dict(), args['save_path_encoder'])
                torch.save(decoder.state_dict(), args['save_path_decoder'])
                torch.save(encoder_optimizer.state_dict(), args['save_path_optimizer_encoder'])
                torch.save(decoder_optimizer.state_dict(), args['save_path_optimizer_decoder'])
                logger.info('Model saved on batch %d' % batch_n)

            if batch_n % evaluate_every == 0:
                val_n = 0
                for batch in dataloader_val:
                    val_n += batch_size
                    input_batches, input_lengths = batch['input'], batch['length'].numpy().tolist()
                    input_batches, input_lengths = zip(
                        *sorted(zip(input_batches, input_lengths), key=lambda x: x[1], reverse=True))
                    input_batches, input_lengths = torch.stack(input_batches), list(input_lengths)
                    input_batches = input_batches[:, :max(input_lengths)]
                    input_batches = input_batches.transpose(0, 1)

                    val_loss, real, generated = evaluate(encoder, decoder, input_batches, input_lengths,
                                                         input_batches, input_lengths, batch_size, lang1)
                    print_loss_total += loss

                print_loss_avg = print_loss_total / val_n
                print_summary = '-- Epoch:%d - Batch:%d - Val_loss:%.4f' % (epoch, batch_n, print_loss_avg)
                logger.info(print_summary)
                print_loss_total = 0
                logger.info('-- Real sentence: {0}, Generated sentence {1}'.format(' '.join(real),
                                                                              ' '.join(generated))
                            )
            torch.cuda.empty_cache()

