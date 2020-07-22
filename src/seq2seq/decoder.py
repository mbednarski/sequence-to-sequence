from typing import Dict, Optional

import torch
import torch.nn as nn

from seq2seq.attention import Attention


class Decoder(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        target_vocab_size: int,
        context_size: int,
        start_token_idx: int,
        teacher_force_ratio: float = 0.5,
        padding_idx: int = 0,
        attention: Optional[Attention] = None,
    ):
        """GRU-based decoder

        Decodes a context vector (with optional attention) to the target sequence.

        :param embedding_size: Decoder embedding size
        :type embedding_size: int
        :param target_vocab_size: Target vocab size
        :type target_vocab_size: int
        :param context_size: Size of the context vector
        :type context_size: int
        :param start_token_idx: Index of <START> token for target sequence
        :type start_token_idx: int
        :param teacher_force_ratio: Probability of using teacher forcing, defaults to 0.5
        :type teacher_force_ratio: float, optional
        :param padding_idx: Index of padding token, defaults to 0
        :type padding_idx: int, optional
        :param attention: Attentiom module to use, defaults to None
        :type padding_idx: Attention, optional
        """
        super().__init__()

        self.teacher_force_ratio = teacher_force_ratio
        self.start_idx = start_token_idx
        self.attention = attention
        self.embed = nn.Embedding(
            num_embeddings=target_vocab_size,
            embedding_dim=embedding_size,
            padding_idx=padding_idx,
        )

        if self.attention:
            self.decoder_rnn = nn.GRU(
                input_size=embedding_size + self.attention.get_encoder_hidden_size(),
                hidden_size=context_size,
                batch_first=True,
            )
        else:
            self.decoder_rnn = nn.GRU(
                input_size=embedding_size, hidden_size=context_size, batch_first=True,
            )

        self.fc = nn.Linear(context_size, target_vocab_size)

    def forward(
        self,
        context_vector: torch.Tensor,
        encoder_outputs: Optional[torch.Tensor] = None,
        target_sequence: Optional[torch.LongTensor] = None,
        max_target_seq_len: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """[summary]

        :param context_vector: Context vector to decode. Shape: batch size x context size
        :type context_vector: torch.Tensor
        :param encoder_outputs: If using attention, encoder outputs. Shape: batch size x source seq len x encoder hidden size
        :type encoder_outputs: Optional[torch.Tensor]
        :param target_sequence: Target sequence (ground truth). Not used in eval mode.
        :type target_sequence: Optional[torch.LongTensor]
        :param max_target_seq_len: Maximum lenght of decoded sequence during inference, defaults to None
        :type max_target_seq_len: int, optional
        :return: [description]
        :rtype: Dict[str, torch.Tensor]
        """

        batch_size = context_vector.shape[0]
        context_vector = context_vector.unsqueeze(0)

        if self.training:
            assert target_sequence
            max_seq_len = target_sequence.shape[1]
        else:
            assert max_target_seq_len
            max_seq_len = max_target_seq_len

        if self.attention:
            assert encoder_outputs
            attention_map = torch.zeros(
                batch_size, max_seq_len, encoder_outputs.shape[1],
            ).to(device=context_vector.device)

        # start with <start> token
        decoder_input = (
            torch.Tensor(batch_size * [self.start_idx])
            .unsqueeze(1)
            .long()
            .to(device=context_vector.device)
        )

        outputs = []
        predictions = []

        for t in range(0, max_seq_len):
            embedded_input = self.embed(decoder_input)

            if self.attention:
                a = self.attention(context_vector, encoder_outputs)
                attention_map[:, t, :] = a.detach()
                a = a.unsqueeze(1)
                weighted = torch.bmm(a, encoder_outputs)
                rnn_input = torch.cat((embedded_input, weighted), dim=2)
            else:
                rnn_input = embedded_input

            after_rnn_out, context_vector = self.decoder_rnn(rnn_input, context_vector)
            decoder_output = self.fc(after_rnn_out.squeeze(1))

            pred_idx = torch.argmax(decoder_output, dim=1).unsqueeze(1).detach()

            if self.training:
                if torch.rand(1) < self.teacher_force_ratio:
                    decoder_input = target_sequence[:, t].unsqueeze(1)
                else:
                    decoder_input = pred_idx
            else:
                decoder_input = pred_idx

            outputs.append(decoder_output)
            predictions.append(pred_idx)

        outputs = torch.stack(outputs).transpose(0, 1)
        predictions = torch.stack(predictions).squeeze(2).transpose(0, 1)

        return {"outputs": outputs, "predictions": predictions}
