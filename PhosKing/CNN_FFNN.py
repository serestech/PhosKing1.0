import torch
import torch.nn as nn

class CNN_FFNN(nn.Module):
    def __init__(self, aa_window, n_input, n_hidden_1, n_hidden_2, padding_1=3, 
                 padding_2=6, out_channels=48, dropout=0.5, n_output=1):
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

        self.aa_window = aa_window
        self.lnn = nn.Sequential(nn.Dropout(p=dropout),
                                 nn.Linear(n_hidden_1, n_hidden_2),
                                 nn.ReLU(),
                                 nn.Linear(n_hidden_2, n_output),
                                 nn.Sigmoid())
        
    def forward(self, x_0):
        x_0 = x_0.permute(0, 2, 1)

        x_1 = self.cnn_1(x_0)
        x_2 = self.cnn_2(x_0)
        x = torch.cat((x_0, x_1, x_2), dim=1)
        x = self.batchnorm(x)

        x = x.flatten(1)

        out = self.lnn(x)

        return out


if __name__ == '__main__':
    cnn_rnn_ffnn = CNN_FFNN(6, 1280, 17888, 20000, 3, 6, 48, 0.5, 52)
    inp = torch.rand(2, 13, 1280)
    print('Running CNN_FFNN...')
    out = cnn_rnn_ffnn(inp)
    print(f'{out=}')