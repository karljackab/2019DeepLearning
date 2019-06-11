import torch
import dataloader as hand_DL
import model
import torch.utils.data as Data
import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
import torch.nn as nn
import string

hidden_size = 256
latent_hidden_size = 32
vocab_size = 28

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
SOS_token = 0
EOS_token = 1

alphabet_map = {0:'*', 1:'-'}
for idx, alpha in enumerate(string.ascii_lowercase):
    alphabet_map[idx + 2] = alpha # map number to alphabet

#compute BLEU-4 score
def compute_bleu(out_emb, truth_emb, alphabet_map = alphabet_map, show = False):
    cc = SmoothingFunction()
    output = ''
    reference = ''
    for i in out_emb:
        output += alphabet_map[i.item()]
    for j in truth_emb:
        reference += alphabet_map[j.item()]
    if show:
        print(f'truth: {reference}')
        print(f'predict: {output}')
        print('= = = = =')
    return sentence_bleu([reference], output,weights=(0.25, 0.25, 0.25, 0.25),smoothing_function=cc.method1)


def test(x, i_cond, y, t_cond, encoder, decoder, enc_last, criterion):
    encoder_hidden = encoder.initHidden()
    temp = torch.zeros(1, 1, 4).scatter_(2, torch.tensor(i_cond).reshape(1, 1, -1), 1)
    i_cond = temp.type(torch.float32).to(device)
    encoder_hidden = torch.cat([encoder_hidden, i_cond], dim = 2)

    x_len = x.size(0)
    for i in range(x_len):
        _, encoder_hidden = encoder(x[i], encoder_hidden)
    encoder_hidden, _, _ = enc_last(encoder_hidden)

    # decode
    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hidden = encoder_hidden
    temp = torch.zeros(1, 1, 4).scatter_(2, torch.tensor(t_cond).reshape(1, 1, -1), 1)
    t_cond = temp.type(torch.float32).to(device)
    decoder_hidden = torch.cat([decoder_hidden, t_cond], dim = 2)

    y_len = y.size(0)

    loss = 0
    output = []
    y = y.view(-1, 1)
    for di in range(y_len):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        _, topi = decoder_output.topk(1)
        output.append(int(torch.max(decoder_output, 1)[1]))
        decoder_input = topi.squeeze().detach()
        loss += criterion(decoder_output, y[di])
        if decoder_input.item() == EOS_token:
            break

    output = torch.tensor(output)
    y = y.view(-1)
    bleu = compute_bleu(output, y, show=True)

    return loss.item()/y_len, bleu


#####################################################
# Start main
#####################################################


folder = '32_a'
acc = '0.6626940598821827'
TEST = False
GAUS = True

enc_last = torch.load(f'{folder}/enc_last_{acc}.pkl')
encoder = torch.load(f'{folder}/encoder_{acc}.pkl')
decoder = torch.load(f'{folder}/decoder_{acc}.pkl')

if TEST:
    test_loader = hand_DL.TestLoader('test')
    print_loss_total = 0
    bleu_total = 0
    test_len = len(test_loader)
    criterion = nn.CrossEntropyLoss()
    for idx, data in enumerate(test_loader):
        x = torch.from_numpy(data[0]).to(device)
        y = torch.from_numpy(data[1]).to(device)

        loss, bleu = test(x, int(data[2]), y, int(data[3]), encoder, decoder, enc_last, criterion)
        print_loss_total += loss
        bleu_total += bleu
    print_loss_total /= test_len
    bleu_total /= test_len
    print(f'Test loss: {print_loss_total}, bleu: {bleu_total}')

if GAUS:
    embedding = torch.randn(1, 1, 256).to(device)
    for tense in range(4):
        decoder_input = torch.tensor([[SOS_token]], device=device)
        temp = torch.zeros(1, 1, 4).scatter_(2, torch.tensor(tense).reshape(1, 1, -1), 1)
        t_cond = temp.type(torch.float32).to(device)
        decoder_hidden = torch.cat([embedding, t_cond], dim = 2)

        output = ''
        MAX_ITER = 20
        for _ in range(MAX_ITER):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            _, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()
            if decoder_input.item() == EOS_token:
                break
            num = int(torch.max(decoder_output, 1)[1])
            char = alphabet_map[num]
            print(char, end='')
        print()