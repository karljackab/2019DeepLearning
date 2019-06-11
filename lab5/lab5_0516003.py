from io import open
import string
import random
import torch
import torch.nn as nn
from torch import optim
import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

import dataloader as hand_DL
import model
import torch.utils.data as Data

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
SOS_token = 0
EOS_token = 1
EOS_tensor = torch.tensor(EOS_token).to(device).view(1, 1)
#----------Hyper Parameters----------#
hidden_size = 256
latent_hidden_size = 32
vocab_size = 28
teacher_forcing_ratio = 0.6
highest_bleu = 0
KLD_weight = 0
LR = 0.05

alphabet_map = {0:'*', 1:'-'}
for idx, alpha in enumerate(string.ascii_lowercase):
    alphabet_map[idx + 2] = alpha # map number to alphabet

def KLD(weight, output, target):
    loss = weight * nn.KLDivLoss(torch.log(output), target)
    return loss

def gen_Gaussian(mean, var, shape):
    return np.random.normal(loc=mean, scale=var, size=shape)

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


def train(input_tensor, i_cond, target_tensor, t_cond, encoder, decoder, enc_last, encoder_optimizer, decoder_optimizer, enc_last_optimizer, criterion):
    encoder_hidden = encoder.initHidden()
    temp = torch.zeros(1, 1, 4).scatter_(2, i_cond.reshape(1, 1, -1), 1)
    i_cond = temp.type(torch.float32).to(device)
    encoder_hidden = torch.cat([encoder_hidden, i_cond], dim = 2)

    encoder_optimizer.zero_grad()
    enc_last_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_tensor = input_tensor.view(-1, 1)
    input_length = input_tensor.size(0)

    #----------sequence to sequence part for encoder----------#
    for in_iter in range(input_length):
        _, encoder_hidden = encoder(input_tensor[in_iter], encoder_hidden)
    encoder_hidden, mean, log_var = enc_last(encoder_hidden)
    
    #----------sequence to sequence part for decoder----------#
    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hidden = encoder_hidden
    temp = torch.zeros(1, 1, 4).scatter_(2, t_cond.reshape(1, 1, -1), 1)
    t_cond = temp.type(torch.float32).to(device)
    decoder_hidden = torch.cat([decoder_hidden, t_cond], dim = 2)

    KL_loss = 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    global KLD_weight
    loss = -KLD_weight * KL_loss

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    target_tensor = torch.cat([target_tensor, EOS_tensor], dim=1)
    target_tensor = target_tensor.view(-1, 1)
    target_length = target_tensor.size(0)
    output = []
    target_tensor = target_tensor.view(-1, 1)
    if use_teacher_forcing:
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            output.append(int(torch.max(decoder_output, 1)[1]))
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing
    else:
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            _, topi = decoder_output.topk(1)
            output.append(int(torch.max(decoder_output, 1)[1]))
            decoder_input = topi.squeeze().detach()  # detach from history as input
            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    enc_last_optimizer.step()
    decoder_optimizer.step()

    # calc bleu
    output = torch.tensor(output)
    target_tensor = target_tensor.view(-1)
    bleu = compute_bleu(output, target_tensor)

    return loss.item()/target_length, bleu, KL_loss

def trainIters(encoder, decoder, enc_last, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    train_loss_total = 0
    train_loss_list = []
    train_KL_total = 0
    train_KL_list = []
    test_bleu_list = []

    print_loss_total = 0  # Reset every print_every
    kl_total = 0
    bleu_total = 0

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    enc_last_optimizer = optim.SGD(enc_last.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    train_data = hand_DL.TrainLoader('train')
    train_loader = Data.DataLoader(dataset = train_data, batch_size = 1, shuffle = True, num_workers=2)
    data_len = len(train_loader)
    cnt = 0
    tot_cnt = 0
    highest_score = 0
    for iter in range(30, n_iters+1):
        print(f"Epoch: {iter}")
        
        global KLD_weight
        global highest_bleu
        global teacher_forcing_ratio

        KLD_weight = 0
        cnt = 0
        print_loss_total = 0
        bleu_total = 0
        kl_total = 0

        # ######################
        # slope = 0.01
        # KLD_weight = iter * slope
        # if KLD_weight > 1.0:
        #     KLD_weight = 1.0
        # ###################
        # slope = 0.01
        # teacher_forcing_ratio = 1.0 - (slope * iter)
        # if teacher_forcing_ratio <= 0.0:
        #     teacher_forcing_ratio = 0.0
        # ######################

        for idx, data in enumerate(train_loader):
            i_cond = data[0][0]
            t_cond = data[1][0]
            x = data[0][1].to(device)
            y = data[1][1].to(device)

            loss, bleu, kl_loss = train(x, i_cond, y, t_cond, encoder,
                        decoder, enc_last, encoder_optimizer, decoder_optimizer, enc_last_optimizer, criterion)
            kl_total += kl_loss
            print_loss_total += loss
            bleu_total += bleu
            KLD_constraint = 1
            if bleu > highest_bleu:
                highest_bleu = bleu
            if bleu > 0.8:
                teacher_forcing_ratio = 0
            elif bleu > 0.7:
                teacher_forcing_ratio = 0.1
            elif bleu > 0.6:
                teacher_forcing_ratio = 0.2
                KLD_constraint = 0.95
            elif bleu > 0.5:
                teacher_forcing_ratio = 0.3
                KLD_constraint = 0.9
            elif bleu > 0.4:
                teacher_forcing_ratio = 0.4
                KLD_constraint = 0.85
            elif bleu > 0.3:
                teacher_forcing_ratio = 0.5
                KLD_constraint = 0.8
            elif bleu > 0.2:
                teacher_forcing_ratio = 0.6
                KLD_constraint = 0.6
            else:
                teacher_forcing_ratio = 0.8
                KLD_constraint = 0.5

            KLD_weight += 0.0001
            if KLD_weight > KLD_constraint:
                KLD_weight = KLD_constraint

            cnt += 1
            tot_cnt += 1

            if idx % print_every == 0:
                train_KL_total += kl_total
                train_loss_total += print_loss_total
                print_loss_avg = print_loss_total/cnt
                bleu_avg = bleu_total/cnt
                print(f'Iter {idx}/{data_len} loss: {print_loss_avg}, kl_loss: {kl_total/cnt}, bleu: {bleu_avg}')
                cnt = 0
                print_loss_total = 0
                bleu_total = 0
                kl_total = 0

        test_loader = hand_DL.TestLoader('test')
        print_loss_total = 0
        bleu_total = 0
        test_len = len(test_loader)
        for idx, data in enumerate(test_loader):
            x = torch.from_numpy(data[0]).to(device)
            y = torch.from_numpy(data[1]).to(device)

            loss, bleu = test(x, int(data[2]), y, int(data[3]), encoder, decoder, enc_last, criterion)
            print_loss_total += loss
            bleu_total += bleu
        print_loss_total /= test_len
        bleu_total /= test_len
        print(f'Test loss: {print_loss_total}, bleu: {bleu_total}')
        
        with open(f'{latent_hidden_size}/train_loss', 'a') as f:
            f.write(f'{str(train_loss_total/tot_cnt)}\n')
        with open(f'{latent_hidden_size}/train_KL_loss', 'a') as f:
            f.write(f'{str(train_KL_total/tot_cnt)}\n')
        with open(f'{latent_hidden_size}/test_bleu', 'a') as f:
            f.write(f'{str(bleu_total)}\n')

        test_bleu_list.append(bleu_total)
        train_loss_list.append(train_loss_total/tot_cnt)
        train_KL_list.append(train_KL_total/tot_cnt)
        train_loss_total = 0
        train_KL_total = 0
        tot_cnt = 0

        if bleu_total > highest_score:
            highest_score = bleu_total
            torch.save(encoder, f'/home/karljackab/DL/lab5/{latent_hidden_size}/encoder_{str(bleu_total)}.pkl')
            torch.save(decoder, f'/home/karljackab/DL/lab5/{latent_hidden_size}/decoder_{str(bleu_total)}.pkl')
            torch.save(enc_last, f'/home/karljackab/DL/lab5/{latent_hidden_size}/enc_last_{str(bleu_total)}.pkl')
            print('save model')

enc_last = model.EncodeLast(hidden_size+4, latent_hidden_size, device).to(device)
encoder = model.EncoderRNN(vocab_size, hidden_size+4, device).to(device)
decoder = model.DecoderRNN(hidden_size+4, vocab_size, device).to(device)

trainIters(encoder, decoder, enc_last, 300, print_every=2000)