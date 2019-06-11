import torch.nn as nn
import torch
import torch.nn.functional as F

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, device):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.device = device

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size-4, device=self.device, dtype=torch.float32)

class EncodeLast(nn.Module):
    def __init__(self, size, output_size, device):
        super().__init__()
        self.fc_mu = nn.Linear(in_features=size, out_features=output_size)
        self.fc_logvr = nn.Linear(in_features=size, out_features=output_size)
        self.fc_emb = nn.Linear(in_features=output_size, out_features=size-4)
        self.device = device
    def forward(self, embedding):
        mu = self.fc_mu(embedding)
        log_var = self.fc_logvr(embedding)
        std = torch.exp(log_var/2)

        tmp = torch.randn_like(std)
        embedding = mu + tmp*std
        embedding = self.fc_emb(embedding)

        return embedding, mu, log_var

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, device):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.device = device

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output[0])
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size-4, device=self.device, dtype=torch.float32)
