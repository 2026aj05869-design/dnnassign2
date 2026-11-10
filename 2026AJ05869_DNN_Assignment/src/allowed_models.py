"""Framework recurrent layers allowed for Problem 1 comparison experiments."""
from torch import nn

class RecurrentModel(nn.Module):
    def __init__(self, kind, input_size, hidden_size, output_size):
        super().__init__()
        choices = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}
        if kind not in choices:
            raise ValueError(f"Unsupported recurrent kind: {kind}")
        self.recurrent = choices[kind](input_size, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: [batch, time, features]
        sequence, state = self.recurrent(x)
        # sequence[:, -1]: final time-step representation [batch, hidden]
        return self.output(sequence[:, -1])
