from masked_cross_entropy import *
from logger import logger
import torch

PAD_token = 0
SOS_token = 1
EOS_token = 2
MAX_LENGTH = 15
USE_CUDA = 1


def evaluate(encoder, decoder, input_batches, input_lengths, target_batches, target_lengths, batch_size, lang1):
    encoder.train(False)
    decoder.train(False)
    if USE_CUDA:
        input_batches = input_batches.cuda()

    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)
    decoder_input = torch.LongTensor([SOS_token] * batch_size)
    decoder_hidden = encoder_hidden[:1]# Use last (forward) hidden state from encoder
    max_target_length = max(input_lengths)
    all_decoder_outputs = torch.zeros(max_target_length, batch_size, decoder.output_size)

    if USE_CUDA:
        decoder_input = decoder_input.cuda()
        all_decoder_outputs = all_decoder_outputs.cuda()

    decoded_words = []
    real_words = []

    for t in range(max_target_length):
        decoder_output, decoder_hidden = decoder(
            decoder_input, decoder_hidden
        )

        all_decoder_outputs[t] = decoder_output
        topv, topi = decoder_output.data.topk(1)
        decoder_input = topi
        if USE_CUDA:
            decoder_input = decoder_input.cuda()
        ni = topi[0][0]

        if ni != EOS_token:
            decoded_words.append(lang1.index2word[str(ni.detach().numpy().reshape((1,))[0])])
        elif decoded_words[-1] != '<EOS>':
            decoded_words.append('<EOS>')

        real_words.append(
            lang1.index2word[input_batches[t][0].detach().numpy().reshape((1,))[0]]
        )

    loss = masked_cross_entropy(
        all_decoder_outputs.transpose(0, 1).contiguous(),  # -> batch x seq
        input_batches.transpose(0, 1).contiguous(),  # -> batch x seq
        target_lengths
    )

    encoder.train(True)
    decoder.train(True)

    return loss.cpu().item(), real_words, decoded_words