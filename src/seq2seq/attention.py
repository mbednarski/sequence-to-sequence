import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, encoder_hidden_size: int, decoder_hidden_size: int):
        """An attention module.

        Takes list of hidden encoder states and produces values how much to attend for each of them.

        :param encoder_hidden_size: Encoder hidden size
        :type encoder_hidden_size: int
        :param decoder_hidden_size: Decoder hidden size
        :type decoder_hidden_size: int
        """
        super().__init__()

        self.attn = nn.Linear(
            (encoder_hidden_size + decoder_hidden_size), decoder_hidden_size
        )
        self.v = nn.Linear(decoder_hidden_size, 1, bias=False)

    def forward(
        self, hidden: torch.Tensor, encoder_outputs: torch.Tensor
    ) -> torch.Tensor:
        """Computes the attention values

        :param hidden: Last hidden state of the decoder (batch size x decoder hidden size)
        :type hidden: torch.Tensor
        :param encoder_outputs: All encoder hidden states (batch size x src seq len x encoder hidden size)
        :type encoder_outputs: torch.Tensor
        :return: An attention vector (batch size x src seq len)
        :rtype: torch.Tensor
        """
        src_len = encoder_outputs.shape[1]
        hidden = hidden.repeat(src_len, 1, 1)

        hidden = hidden.transpose(0, 1)

        energy = torch.cat((hidden, encoder_outputs), dim=2)
        energy = self.attn(energy)
        energy = torch.tanh(energy)

        attention = self.v(energy).squeeze(2)
        return torch.softmax(attention, dim=1)
