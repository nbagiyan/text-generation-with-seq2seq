from masked_cross_entropy import *
from torch.distributions.normal import Normal
import random

PAD_token = 0
SOS_token = 1
EOS_token = 2
MAX_LENGTH = 15
USE_CUDA = 1

def train(input_batches, input_lengths, target_batches, target_lengths, encoder, decoder, encoder_optimizer,
          decoder_optimizer, batch_size, clip):
    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0  # Added onto for each word
    if USE_CUDA:
        input_batches = input_batches.cuda()
        target_batches = target_batches.cuda()

    # Run words through encoder
    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)

    # Prepare input and output variables
    decoder_input = torch.LongTensor([SOS_token] * batch_size)
    m = Normal(0, 0.01)
    decoder_hidden = encoder_hidden[:1]# Use last (forward) hidden state from encoder
    noise = m.sample(decoder_hidden.size())

    if USE_CUDA:
        noise = noise.cuda()

    decoder_hidden = decoder_hidden + noise
    max_target_length = max(target_lengths)
    all_decoder_outputs = torch.zeros(max_target_length, batch_size, decoder.output_size)

    # Move new Variables to CUDA
    if USE_CUDA:
        decoder_input = decoder_input.cuda()
        all_decoder_outputs = all_decoder_outputs.cuda()

    teacher_forcing_ratio = 0.5
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
    # Run through decoder one time step at a time
        for t in range(max_target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden
            )

            all_decoder_outputs[t] = decoder_output
            decoder_input = target_batches[t]  # Next input is current target

    else:
        # Run through decoder one time step at a time
        for t in range(max_target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden
            )

            all_decoder_outputs[t] = decoder_output
            topv, topi = decoder_output.data.topk(1)
            decoder_input = topi

    # Loss calculation and backpropagation

    loss = masked_cross_entropy(
        all_decoder_outputs.transpose(0, 1).contiguous(),  # -> batch x seq
        target_batches.transpose(0, 1).contiguous(),  # -> batch x seq
        target_lengths
    )

    loss.backward()

    # Clip gradient norms
    ec = torch.nn.utils.clip_grad_norm(encoder.parameters(), clip)
    dc = torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)

    # Update parameters with optimizers
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.cpu().item()