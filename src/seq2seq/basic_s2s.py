import torch
import torch.nn as nn
import pytorch_lightning as pl
from seq2seq.dataset.math_dataset import MathDataset
from torch.utils.data.dataloader import DataLoader


class Encoder(pl.LightningModule):
    def __init__(self, context_size, embedding_size, input_vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=input_vocab_size, embedding_dim=embedding_size, padding_idx=0
        )

        self.encoder_rnn = nn.GRU(
            input_size=embedding_size, hidden_size=context_size, batch_first=True,
        )

    def forward(self, input_sequence):
        embedded = self.embedding(input_sequence)
        _, encoded = self.encoder_rnn(embedded)

        return encoded


class Decoder(pl.LightningModule):
    def __init__(self, embedding_size, target_vocab_size, context_size):
        super().__init__()
        self.embed = nn.Embedding(
            num_embeddings=target_vocab_size,
            embedding_dim=embedding_size,
            padding_idx=0,
        )

        # decoder
        self.decoder_rnn = nn.GRU(
            input_size=embedding_size, hidden_size=context_size, batch_first=True
        )
        self.fc = nn.Linear(context_size, target_vocab_size)

    def forward(self, context_vector, target_sequece=None):

        assert target_sequece is not None

        batch_size = target_sequece.shape[0]
        # start with <start> token
        decoder_input = torch.Tensor(batch_size * [1]).unsqueeze(1).long().cuda()

        loss = 0
        max_seq_len = target_sequece.shape[1]

        outputs = []
        predictions = []

        for i in range(1, max_seq_len):

            # embedded_input: Bx1xE
            embedded_input = self.embed(decoder_input)
            # print('embedded_input', embedded_input.shape)

            # after_rnn
            after_rnn_out, context_vector = self.decoder_rnn(
                embedded_input, context_vector
            )

            decoder_output = self.fc(after_rnn_out.squeeze())

            pred_idx = torch.argmax(decoder_output, dim=1).unsqueeze(1).detach()

            decoder_input = pred_idx

            outputs.append(decoder_output)
            predictions.append(pred_idx)

        return outputs, predictions


class SimpleSeq2Seq(pl.LightningModule):
    def __init__(self):
        super(SimpleSeq2Seq, self).__init__()
        context_size = 1024
        embedding_size = 64

        self.encoder = Encoder(
            context_size=context_size, embedding_size=64, input_vocab_size=17
        )

        self.decoder = Decoder(
            embedding_size=64, target_vocab_size=39, context_size=context_size
        )

        # loss
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

    def training_step(self, batch, batch_idx):
        x, y = batch

        # x: BxL
        # y: Bx(L-1)

        # print('x shape', x.shape)
        # print('y shape', y.shape)

        batch_size = x.shape[0]
        max_target_seq_len = y.shape[1]

        # print('batch_size', batch_size)
        # print('max_seq_len', max_seq_len)

        # encode
        encoded = self.encoder(x)

        outputs, predictions = self.decoder(encoded, y)

        loss = 0
        for i, o in enumerate(outputs):
            loss += self.criterion(o, y[:, i])

        loss /= max_target_seq_len

        if self.global_step % 100 == 0:
            print("Example 0 x   :", self.dset.number_tokenizer.decode_clean(x[0, :]))
            print("Example 0 y   :", self.dset.text_tokenizer.decode_clean(y[0, :]))
            predictions = torch.cat([x[0] for x in predictions])
            print(
                "Example 0 pred:", self.dset.text_tokenizer.decode_clean(predictions),
            )

        return {"loss": loss}

        # # context_vector: BxC ! LSTM 2xC
        # # print('context_vector shape', context_vector.shape)

        # # decoder_input: Bx1
        # decoder_input = torch.Tensor(batch_size * [1]).unsqueeze(1).long()
        # # cuda?
        # # print('decoder_input shape', decoder_input.shape)
        # # print('decoder_input', decoder_input)

        # loss = 0

        # for i in range(max_seq_len - 1):

        #     # embedded_input: Bx1xE
        #     embedded_input = self.text_embed(decoder_input)
        #     # print('embedded_input', embedded_input.shape)

        #     # after_rnn
        #     after_rnn_out, context_vector = self.decoder_rnn(
        #         embedded_input, context_vector
        #     )

        #     # print('after_rnn_out shape' ,after_rnn_out.shape )
        #     # print('after_rnn_out' ,after_rnn_out)
        #     # print('context_vector shape' ,context_vector.shape )
        #     # print('context_vector' ,context_vector)

        #     # decoder_output
        #     decoder_output = self.fc(after_rnn_out.squeeze())
        #     # print('decoder_output shape' , decoder_output.shape)

        #     pred_idx = torch.argmax(decoder_output, dim=1).unsqueeze(1).detach()

        #     # print('pred_idx shape', pred_idx.shape)

        #     decoder_input = pred_idx

        #     loss += self.criterion(decoder_output, y[:, i])

        # loss = loss / max_seq_len

        # return {"loss": loss}

    def prediction(self, batch, batch_idx):
        x, y = batch

        batch_size = x.shape[0]
        max_seq_len = x.shape[1]

        embedded = self.num_embed(x)

        _, context_vector = self.encoder_rnn(embedded)

        decoder_input = torch.Tensor(batch_size * [1]).unsqueeze(1).long()  # cuda?

        loss = 0

        dec_outputs = []

        for i in range(max_seq_len - 1):
            embedded_input = self.text_embed(decoder_input)

            after_rnn_out, context_vector = self.decoder_rnn(
                embedded_input, context_vector
            )

            # decoder_output
            decoder_output = self.fc(after_rnn_out.squeeze())

            pred_idx = (
                torch.argmax(decoder_output, dim=1).unsqueeze(1).detach()
            )  # MB: how about teacher forcing?

            dec_outputs.append(pred_idx)

            decoder_input = pred_idx

            loss += self.criterion(decoder_output, y[:, i])

        loss = loss / max_seq_len

        return dec_outputs

    # def validation_epoch_end(self, outputs):
    #     losses = torch.stack([x["loss"] for x in outputs])

    #     mean_loss = torch.mean(losses)
    #     print(mean_loss)
    #     return {"log": {"mean_val_loss": mean_loss}}

    def forward(self, x):
        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def train_dataloader(self):
        self.dset = MathDataset()
        train_loader = DataLoader(
            self.dset, batch_size=64, shuffle=True, collate_fn=self.dset.collate,
        )
        return train_loader

    # def val_dataloader(self):
    #     reversed_dataset = ReverseDataset(val_samples, tokenizer)
    #     loader = DataLoader(
    #         reversed_dataset,
    #         batch_size=16,
    #         shuffle=False,
    #         collate_fn=tokenizer.collate_sequences,
    #     )
    #     return loader


if __name__ == "__main__":

    model = SimpleSeq2Seq()
    trainer = pl.Trainer(
        gpus=[0], gradient_clip_val=1.0, max_epochs=50, fast_dev_run=False,
    )
    trainer.fit(model)
