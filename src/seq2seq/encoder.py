from typing import Dict

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(
        self,
        context_size: int,
        embedding_size: int,
        input_vocab_size: int,
        padding_idx: int,
        dropout: float = 0.0,
    ):
        """An GRU encoder for seq2seq model.

        Takens an input sequence and turns it into a fixed-size context vector

        :param context_size: Size of the context vector
        :type context_size: int
        :param embedding_size: Size of the encoder embedding
        :type embedding_size: int
        :param input_vocab_size: Vocabulary size for source sequence
        :type input_vocab_size: int
        :param padding_idx: Index of the padding token in source sequence
        :type padding_idx: int
        :param dropout: Dropout value, defaults to 0.0
        :type dropout: float, optional
        """
        super().__init__()
        self.hidden_size = context_size * 2

        self.embedding = nn.Embedding(
            num_embeddings=input_vocab_size,
            embedding_dim=embedding_size,
            padding_idx=padding_idx,
        )

        self.encoder_rnn = nn.GRU(
            input_size=embedding_size,
            hidden_size=context_size,
            batch_first=True,
            bidirectional=True,
        )

        self.fc = nn.Linear(2 * context_size, context_size)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout)

    def get_hidden_size(self):
        return self.hidden_size

    def forward(self, source_sequence: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forwards the encoder

        :param source_sequence: Tensor with encoded input sequence.
            Shape: batch size x src_seq_len x src_vocab_size
        :type source_sequence: torch.Tensor
        :return: Dictionary with encoder outputs (for attention) and context vector
            Shapes:
                outputs: batch size x src_seq_len x 2*context size
                context: batch size x context_size
        :rtype: dict[str, torch.Tensor]
        """
        embedded = self.embedding(source_sequence)
        outputs, encoded = self.encoder_rnn(embedded)

        encoded = torch.cat((encoded[0, :, :], encoded[1, :, :]), dim=1)

        encoded = self.fc(encoded)
        encoded = self.tanh(encoded)
        encoded = self.dropout(encoded)

        return {"outputs": outputs, "context": encoded}
