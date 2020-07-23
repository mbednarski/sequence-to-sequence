import pytest
import torch
from torch.autograd import backward

from seq2seq.attention import Attention
from seq2seq.decoder import Decoder
from seq2seq.encoder import Encoder


@pytest.mark.parametrize("batch_size", [(1), (64)])
def test_decoder_eval_without_attention(batch_size):
    embedding_size = 50
    target_vocab_size = 5
    context_size = 128
    max_target_seq_len = 30

    decoder = Decoder(
        embedding_size=embedding_size,
        target_vocab_size=target_vocab_size,
        context_size=context_size,
        start_token_idx=1,
    )
    decoder.eval()

    context_vector = torch.rand((batch_size, context_size))

    results = decoder.forward(context_vector, max_target_seq_len=max_target_seq_len)

    assert "outputs" in results.keys()
    assert "predictions" in results.keys()

    assert results["outputs"].shape == (
        batch_size,
        max_target_seq_len,
        target_vocab_size,
    )
    assert results["predictions"].shape == (batch_size, max_target_seq_len)


@pytest.mark.parametrize("batch_size", [(1), (64)])
def test_decoder_train_without_attention(batch_size):
    embedding_size = 50
    target_vocab_size = 5
    context_size = 128

    target_seq_len = 75

    decoder = Decoder(
        embedding_size=embedding_size,
        target_vocab_size=target_vocab_size,
        context_size=context_size,
        start_token_idx=1,
    )
    decoder.train()

    context_vector = torch.rand((batch_size, context_size))
    target_sequence = torch.randint(0, target_vocab_size, (batch_size, target_seq_len))

    results = decoder.forward(context_vector, target_sequence=target_sequence)

    assert "outputs" in results.keys()
    assert "predictions" in results.keys()

    assert results["outputs"].shape == (batch_size, target_seq_len, target_vocab_size,)
    assert results["predictions"].shape == (batch_size, target_seq_len)

    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(results["outputs"].transpose(1, 2), target_sequence)
    loss.backward()

    assert loss != 0
    for param in decoder.parameters():
        assert param.grad.max() != 0


@pytest.mark.parametrize("batch_size", [(1), (64)])
def test_decoder_eval_with_attention(batch_size):
    embedding_size = 50
    target_vocab_size = 5
    context_size = 128
    max_target_seq_len = 30
    input_seq_len = 58

    attn = Attention(context_size * 2, decoder_hidden_size=context_size)

    decoder = Decoder(
        embedding_size=embedding_size,
        target_vocab_size=target_vocab_size,
        context_size=context_size,
        start_token_idx=1,
        attention=attn,
    )
    decoder.eval()

    context_vector = torch.rand((batch_size, context_size))

    encoder_outputs = torch.rand(batch_size, input_seq_len, context_size * 2)

    results = decoder.forward(
        context_vector,
        max_target_seq_len=max_target_seq_len,
        encoder_outputs=encoder_outputs,
    )

    assert "outputs" in results.keys()
    assert "predictions" in results.keys()
    assert "attention" in results.keys()

    assert results["outputs"].shape == (
        batch_size,
        max_target_seq_len,
        target_vocab_size,
    )
    assert results["predictions"].shape == (batch_size, max_target_seq_len)
    assert results["attention"].shape == (batch_size, max_target_seq_len, input_seq_len)
    assert torch.allclose(
        results["attention"].sum(dim=2).squeeze(),
        torch.ones((batch_size, max_target_seq_len)),
    )
