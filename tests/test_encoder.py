import torch

from seq2seq.encoder import Encoder


def test_shapes():
    context_size = 60
    batch_size = 32
    embedding_size = 30
    vocab_size = 5
    padding_idx = 0
    seq_len = 25

    encoder = Encoder(context_size, embedding_size, vocab_size, padding_idx)

    source_sequence = torch.randint(0, vocab_size, (batch_size, seq_len))

    results = encoder.forward(source_sequence)
    outputs = results["outputs"]
    context = results["context"]

    assert outputs.shape == (batch_size, seq_len, context_size * 2)
    assert context.shape == (batch_size, context_size)


def test_shapes_for_batch_equal_one():
    context_size = 60
    batch_size = 1
    embedding_size = 30
    vocab_size = 5
    padding_idx = 0
    seq_len = 25

    encoder = Encoder(context_size, embedding_size, vocab_size, padding_idx)

    source_sequence = torch.randint(0, vocab_size, (batch_size, seq_len))

    results = encoder.forward(source_sequence)
    outputs = results["outputs"]
    context = results["context"]

    assert outputs.shape == (batch_size, seq_len, context_size * 2)
    assert context.shape == (batch_size, context_size)
