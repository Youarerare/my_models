import torch
import torch.nn as nn
class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout, pad_idx):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.rnn = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           dropout=dropout)

        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        # self.fc = nn.Sequential(
        #     nn.Linear(hidden_dim*2,30),
        #     nn.Linear(30,output_dim)
        # )
        # self.fc = nn.Sequential(
        #     nn.BatchNorm1d(self.hidden_size * 8),
        #     nn.Linear(self.hidden_size * 8, args.linear_size),
        #     nn.ELU(inplace=True),
        #     nn.BatchNorm1d(args.linear_size),
        #     nn.Dropout(self.dropout),
        #     nn.Linear(args.linear_size, args.linear_size),
        #     nn.ELU(inplace=True),
        #     nn.BatchNorm1d(args.linear_size),
        #     nn.Dropout(self.dropout),
        #     nn.Linear(args.linear_size, 4)
        #     # nn.Softmax(dim=-1)
        # )

        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):
        # text = [sent len, batch size]
        embedded = self.dropout(self.embedding(text))
        # embedded = [sent len, batch size, emb dim]
        # pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)
        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        # unpack sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        # output = [sent len, batch size, hid dim * num directions]
        # output over padding tokens are zero tensors
        # hidden = [num layers * num directions, batch size, hid dim]
        # cell = [num layers * num directions, batch size, hid dim]
        # concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        # and apply dropout
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        # hidden = [batch size, hid dim * num directions]
        return self.fc(hidden)


INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 128
OUTPUT_DIM = 5
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

model = RNN(INPUT_DIM,
            EMBEDDING_DIM,
            HIDDEN_DIM,
            OUTPUT_DIM,
            N_LAYERS,
            BIDIRECTIONAL,
            DROPOUT,
            PAD_IDX)