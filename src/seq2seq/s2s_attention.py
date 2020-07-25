from argparse import ArgumentParser
from os import stat

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.logging import NeptuneLogger
from torch.utils.data.dataloader import DataLoader

from seq2seq import Attention, Decoder, Encoder
from seq2seq.dataset.math_dataset import MathDataset
from seq2seq.tokenizers import NumberTokenizer, TextTokenizer


class SimpleSeq2Seq(pl.LightningModule):
    def __init__(self, hparams):
        super(SimpleSeq2Seq, self).__init__()
        self.hparams = hparams

        self.source_tokenizer = NumberTokenizer()
        self.target_tokenizer = TextTokenizer(pd.read_csv("data/raw/math.csv")["text"])

        self.encoder = Encoder(
            context_size=hparams.context_size,
            embedding_size=hparams.encoder_embedding_size,
            input_vocab_size=self.source_tokenizer.get_vocab_size(),
            padding_idx=self.source_tokenizer.get_padding_idx(),
            dropout=hparams.encoder_dropout,
        )

        self.attention = Attention(self.encoder.get_hidden_size(), hparams.context_size)

        self.decoder = Decoder(
            embedding_size=hparams.decoder_embedding_size,
            target_vocab_size=self.target_tokenizer.get_vocab_size(),
            context_size=hparams.context_size,
            start_token_idx=self.target_tokenizer.get_start_idx(),
            attention=self.attention,
        )

        # loss
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, x, y):
        batch_size = x.shape[0]
        max_target_seq_len = y.shape[1]

        encoder_results = self.encoder(x)
        decoder_results = self.decoder.forward(
            context_vector=encoder_results["context"],
            encoder_outputs=encoder_results["outputs"],
            target_sequence=y,
        )

        loss = self.criterion(decoder_results["outputs"].transpose(1, 2), y)

        return {
            "loss": loss,
            "output": decoder_results["outputs"],
            "predictions": decoder_results["predictions"],
            "attention": decoder_results["attention"],
        }

    def training_step(self, batch, batch_idx):
        x, y = batch

        results = self.forward(x, y)

        # if self.global_step % 100 == 0:
        #     print("Example 0 x   :", self.dset.number_tokenizer.decode_clean(x[0, :]))
        #     print("Example 0 y   :", self.dset.text_tokenizer.decode_clean(y[0, :]))
        #     predictions = torch.cat([x[0] for x in results["predictions"]])
        #     print(
        #         "Example 0 pred:", self.dset.text_tokenizer.decode_clean(predictions),
        #     )

        return {"loss": results["loss"], "log": {"train-loss": results["loss"]}}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        max_target_seq_len = y.shape[1]

        results = self.forward(x, y)

        # for i in range(10):
        #     print(
        #         "VAL Example 0 x   :",
        #         self.val_dset.number_tokenizer.decode_clean(x[i, :]),
        #     )
        #     print(
        #         "VAL Example 0 y   :",
        #         self.val_dset.text_tokenizer.decode_clean(y[i, :]),
        #     )
        #     predictions = torch.cat([x[i] for x in results["predictions"]])
        #     print(
        #         "VAL Example 0 pred:",
        #         self.val_dset.text_tokenizer.decode_clean(predictions),
        #     )

        return {"loss": results["loss"]}

    def validation_epoch_end(self, outputs):
        loss = torch.stack([x["loss"] for x in outputs]).mean()

        return {"log": {"val-loss": loss}}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--context-size", type=int, default=128)
        parser.add_argument("--encoder_embedding_size", type=int, default=32)
        parser.add_argument("--encoder_dropout", type=float, default=0.1)
        parser.add_argument("--decoder_embedding_size", type=int, default=32)
        parser.add_argument("--batch_size", type=int, default=64)

        return parser

    def train_dataloader(self):
        dset = MathDataset(
            "data/raw/math.train",
            number_tokenizer=self.source_tokenizer,
            text_tokenizer=self.target_tokenizer,
        )
        loader = DataLoader(
            dset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            collate_fn=dset.collate,
        )
        return loader

    def val_dataloader(self):
        dset = MathDataset(
            "data/raw/math.val",
            number_tokenizer=self.source_tokenizer,
            text_tokenizer=self.target_tokenizer,
        )
        loader = DataLoader(
            dset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            collate_fn=dset.collate,
        )
        return loader


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = SimpleSeq2Seq.add_model_specific_args(parser)
    args = parser.parse_args()
    model = SimpleSeq2Seq(args)

    trainer = pl.Trainer(
        gpus=[0],
        gradient_clip_val=1.0,
        max_epochs=50,
        fast_dev_run=False,
        logger=NeptuneLogger(project_name="mbednarski/seq2seq"),
    )
    trainer.fit(model)
