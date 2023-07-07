import torch
import torch.nn as nn

class CNN_RNN_FFNN(nn.Module):
    def __init__(self, aa_window, n_input, hidden_lstm, hidden_linear_1,
                 padding_1=3, padding_2=6, out_channels=48, dropout=0.5, n_output=1):
        super().__init__()
        kernel_1 = padding_1*2 + 1
        self.cnn_1 = nn.Sequential(nn.Dropout(p=dropout),
                                   nn.Conv1d(in_channels=n_input, out_channels=out_channels,
                                             kernel_size=kernel_1, padding=padding_1),
                                   nn.ReLU())

        kernel_2 = padding_2*2 + 1
        self.cnn_2 = nn.Sequential(nn.Dropout(p=dropout),
                                   nn.Conv1d(in_channels=n_input, out_channels=out_channels,
                                             kernel_size=kernel_2, padding=padding_2),
                                   nn.ReLU())

        self.batchnorm = nn.BatchNorm1d(n_input + 2*out_channels)

        self.rnn = nn.LSTM(input_size=n_input + 2*out_channels, hidden_size=hidden_lstm,
                           batch_first=True, num_layers=2, bidirectional=True, dropout=dropout)

        self.aa_window = aa_window
        self.lnn = nn.Sequential(nn.Dropout(p=dropout),
                                 nn.Linear(hidden_lstm*2*(aa_window*2+1), hidden_linear_1),
                                 nn.ReLU(),
                                 nn.Linear(hidden_linear_1, n_output),
                                 nn.Sigmoid())
        
    def forward(self, x_0):
        x_0 = x_0.permute(0, 2, 1)
        x_1 = self.cnn_1(x_0)
        x_2 = self.cnn_2(x_0)
        x = torch.cat((x_0, x_1, x_2), dim=1)
        x = self.batchnorm(x)

        x = x.permute(0, 2, 1)
        x, _ = self.rnn(x)

        m = x.shape[1] // 2
        i, j = m - self.aa_window, m + self.aa_window + 1
        x = x[:, i:j].flatten(1)
        out = self.lnn(x)

        return out


if __name__ == '__main__':
    cnn_rnn_ffnn = CNN_RNN_FFNN(0, 1280, 256, 512)
    inp = torch.rand(3, 31, 1280)
    print('Running CNN_RNN_FFNN...')
    out = cnn_rnn_ffnn(inp)
    print(f'{out=}')