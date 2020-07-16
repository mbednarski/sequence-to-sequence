import torch
import torch.nn as nn
import pytorch_lightning as pl
from seq2seq.dataset.math_dataset import MathDataset
from torch.utils.data.dataloader import DataLoader

from pytorch_lightning.logging import NeptuneLogger


class Encoder(pl.LightningModule):
    def __init__(self, context_size, embedding_size, input_vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=input_vocab_size, embedding_dim=embedding_size, padding_idx=0
        )

        self.encoder_rnn = nn.GRU(
            input_size=embedding_size,
            hidden_size=context_size,
            batch_first=True,
            bidirectional=True,
        )

        self.fc = nn.Linear(2 * context_size, context_size)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_sequence):
        embedded = self.embedding(input_sequence)
        _, encoded = self.encoder_rnn(embedded)

        encoded = torch.cat((encoded[0, :, :], encoded[1, :, :]), dim=1)

        encoded = self.fc(encoded)
        encoded = self.tanh(encoded)
        encoded = self.dropout(encoded)

        encoded = encoded.unsqueeze(0)

        return encoded

class Decoder(pl.LightningModule):
    def __init__(
        self, embedding_size, target_vocab_size, context_size, teacher_force_ratio=0.5
    ):
        super().__init__()
        self.teacher_force_ratio = teacher_force_ratio
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

    def forward(self, context_vector, target_sequence=None):

        assert target_sequence is not None

        batch_size = target_sequence.shape[0]
        # start with <start> token
        decoder_input = torch.Tensor(batch_size * [1]).unsqueeze(1).long().cuda()

        max_seq_len = target_sequence.shape[1]

        outputs = []
        predictions = []

        for t in range(0, max_seq_len):

            # embedded_input: Bx1xE
            embedded_input = self.embed(decoder_input)
            # print('embedded_input', embedded_input.shape)

            # after_rnn
            after_rnn_out, context_vector = self.decoder_rnn(
                embedded_input, context_vector
            )

            decoder_output = self.fc(after_rnn_out.squeeze())

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

        return outputs, predictions


class SimpleSeq2Seq(pl.LightningModule):
    def __init__(self):
        super(SimpleSeq2Seq, self).__init__()
        context_size = 128
        embedding_size = 32

        self.encoder = Encoder(
            context_size=context_size, embedding_size=64, input_vocab_size=17
        )

        self.decoder = Decoder(
            embedding_size=64, target_vocab_size=41, context_size=context_size
        )

        # loss
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, x, y):
        batch_size = x.shape[0]
        max_target_seq_len = y.shape[1]

        encoded = self.encoder(x)
        outputs, predictions = self.decoder(encoded, y)

        loss = 0
        for i, o in enumerate(outputs):
            loss += self.criterion(o, y[:, i])

        loss /= max_target_seq_len

        return {"loss": loss, "output": outputs, "predictions": predictions}

    def training_step(self, batch, batch_idx):
        x, y = batch

        results = self.forward(x, y)

        if self.global_step % 100 == 0:
            print("Example 0 x   :", self.dset.number_tokenizer.decode_clean(x[0, :]))
            print("Example 0 y   :", self.dset.text_tokenizer.decode_clean(y[0, :]))
            predictions = torch.cat([x[0] for x in results["predictions"]])
            print(
                "Example 0 pred:", self.dset.text_tokenizer.decode_clean(predictions),
            )

        return {"loss": results["loss"], "log": {"train-loss": results["loss"]}}

    # def training_step(self, batch, batch_idx):
    #     x, y = batch
    #     max_target_seq_len = y.shape[1]

    #     encoded = self.encoder(x)
    #     outputs, predictions = self.decoder(encoded, y)

    #     loss = 0
    #     for i, o in enumerate(outputs):
    #         loss += self.criterion(o, y[:, i])

    #     loss /= max_target_seq_len

    #     if self.global_step % 100 == 0:
    #         print("Example 0 x   :", self.dset.number_tokenizer.decode_clean(x[0, :]))
    #         print("Example 0 y   :", self.dset.text_tokenizer.decode_clean(y[0, :]))
    #         predictions = torch.cat([x[0] for x in predictions])
    #         print(
    #             "Example 0 pred:", self.dset.text_tokenizer.decode_clean(predictions),
    #         )

    #     return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        max_target_seq_len = y.shape[1]

        results = self.forward(x, y)

        loss = 0
        for i, o in enumerate(results["output"]):
            loss += self.criterion(o, y[:, i])

        loss /= max_target_seq_len

        for i in range(10):
            print(
                "VAL Example 0 x   :",
                self.val_dset.number_tokenizer.decode_clean(x[i, :]),
            )
            print(
                "VAL Example 0 y   :",
                self.val_dset.text_tokenizer.decode_clean(y[i, :]),
            )
            predictions = torch.cat([x[i] for x in results["predictions"]])
            print(
                "VAL Example 0 pred:",
                self.val_dset.text_tokenizer.decode_clean(predictions),
            )

        return {"loss": loss}

    def validation_epoch_end(self, outputs):
        loss = torch.stack([x["loss"] for x in outputs]).mean()

        return {"log": {"val-loss": loss}}

    # def validation_epoch_end(self, outputs):
    #     losses = torch.stack([x["loss"] for x in outputs])

    #     mean_loss = torch.mean(losses)
    #     print(mean_loss)
    #     return {"log": {"mean_val_loss": mean_loss}}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def train_dataloader(self):
        self.dset = MathDataset()
        train_loader = DataLoader(
            self.dset, batch_size=64, shuffle=True, collate_fn=self.dset.collate,
        )
        return train_loader

    def train_dataloader(self):
        self.dset = MathDataset("data/raw/math.train")
        train_loader = DataLoader(
            self.dset, batch_size=64, shuffle=True, collate_fn=self.dset.collate,
        )
        return train_loader

    def val_dataloader(self):
        dset = MathDataset("data/raw/math.val")
        self.val_dset = dset
        loader = DataLoader(
            dset, batch_size=64, shuffle=False, collate_fn=dset.collate,
        )
        return loader

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
        gpus=[0],
        gradient_clip_val=1.0,
        max_epochs=50,
        fast_dev_run=False,
        logger=NeptuneLogger(project_name="mbednarski/seq2seq"),
    )
    trainer.fit(model)
