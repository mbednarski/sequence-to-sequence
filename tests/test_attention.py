import torch
from torch import bartlett_window

from seq2seq.attention import Attention


def test_attention():
    encoder_hidden_size = 20
    decoder_hidden_size = 30
    batch_size = 64
    src_seq_len = 50

    attn = Attention(encoder_hidden_size, decoder_hidden_size)

    decoder_hidden = torch.rand([batch_size, decoder_hidden_size])
    encoder_outputs = torch.rand([batch_size, src_seq_len, encoder_hidden_size])

    a = attn.forward(decoder_hidden, encoder_outputs)

    assert a.shape == (batch_size, src_seq_len)
    print(type(a.sum(dim=1)))
    assert torch.allclose(a.sum(dim=1), torch.ones(batch_size))
