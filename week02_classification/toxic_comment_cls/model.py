import pickle
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from utils import initialise_word_embedding

class TextCNN(nn.Module):

    def __init__(self, args):
        super(TextCNN, self).__init__()
        self.emb = nn.Embedding(args['vocab_size'], args['embedding_dim'])
        self.conv0 = nn.Conv1d(in_channels=args['embedding_dim'], out_channels=args['out_channels'], kernel_size=args['kernel_size'][0])
        self.conv0_bn = nn.BatchNorm1d(args['out_channels'])
        self.conv0_ln = nn.LayerNorm(args['out_channels'])

        self.conv1 = nn.Conv1d(in_channels=args['embedding_dim'], out_channels=args['out_channels'], kernel_size=args['kernel_size'][1])
        self.conv1_bn = nn.BatchNorm1d(args['out_channels'])
        self.conv1_ln = nn.LayerNorm(args['out_channels'])

        self.conv2 = nn.Conv1d(in_channels=args['embedding_dim'], out_channels=args['out_channels'], kernel_size=args['kernel_size'][2])
        self.conv2_bn = nn.BatchNorm1d(args['out_channels'])
        self.conv2_ln = nn.LayerNorm(args['out_channels'])

        self.dropout = nn.Dropout(args['dropout'])
        self.fc = nn.Linear(len(args['kernel_size']) * args['out_channels'], args['class_num'])
        # self.init_weights()

    def init_weights(self):
        init_range = 0.5
        self._init_emb_weights()
        # self.emb.weight.data.uniform_(-init_range, init_range)
        self.fc.weight.data.uniform_(-init_range, init_range)
        self.fc.bias.data.zero_()

    def _init_emb_weights(self):
        glove_weights = initialise_word_embedding()
        self.emb.weight.data.copy_(torch.tensor(glove_weights))
        # self.emb.load_state_dict({'weight': glove_weights})

    def forward(self, x):
        x = self.emb(x)
        x = x.permute(0, 2, 1)

        x0 = self.conv_and_pool(x, self.conv0, self.conv0_bn)
        x0 = self.conv0_ln(x0)
        x1 = self.conv_and_pool(x, self.conv1, self.conv1_bn)
        x1 = self.conv0_ln(x1)
        x2 = self.conv_and_pool(x, self.conv2, self.conv2_bn)
        x2 = self.conv0_ln(x2)
        x = torch.cat((x0, x1, x2), 1)
        x = self.dropout(x)
        # logits
        logit = F.log_softmax(self.fc(x), dim=1)
        return logit

    @staticmethod
    def conv_and_pool(x, conv, bn=None):
        x = conv(x)
        # x = bn(x)
        x = F.relu(x)
        x = F.max_pool1d(x, x.size(2))
        x = x.squeeze(2)
        return x


class TextLSTM(nn.Module):
    def __init__(self, args):
        super(TextLSTM, self).__init__()
        self.emb = nn.Embedding(args['vocab_size'], args['embedding_dim'])
        input_size = args['embedding_dim']
        self.hidden_dim = args['hidden_dim']
        self.num_lstms = args['num_lstms']
        self.bidirectional = args.get('bi', False)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=self.hidden_dim, num_layers=self.num_lstms, batch_first=True, bidirectional=self.bidirectional)
        self.lstm_out_size = 2 * self.hidden_dim if self.bidirectional else self.hidden_dim
        self.fc = nn.Linear(self.lstm_out_size, args['class_num'])

        # self._init_emb_weights()

    def _init_emb_weights(self):
        glove_weights = initialise_word_embedding()
        self.emb.weight.data.copy_(torch.tensor(glove_weights))

    def forward(self, x, x_len):
        x = self.emb(x)
        # print(x.shape)
        x_packed = pack_padded_sequence(x, x_len, batch_first=True, enforce_sorted=False)
        x, (h_n, c_n) = self.lstm(x_packed)
        # print(h_n.data.shape)
        # x_padded, x_lens = pad_packed_sequence(h_n[-1], batch_first=True)
        # print(x_padded.data.shape)
        out = self.fc(h_n[-1])
        # print(out.shape)
        logit = F.log_softmax(out, dim=1)
        return logit


if __name__ == '__main__':
    args = {'vocab_size': 100, 'embedding_dim': 100, 'in_channels': 100, 'out_channels': 64, 'kernel_size': [2, 3, 4]}
    model = TextCNN(args)
    x = [[1, 2, 3]]
    x = torch.tensor(x)
    print(model(x).shape)