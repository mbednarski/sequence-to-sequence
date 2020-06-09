class SimpleSeq2Seq(pl.LightningModule):
    def __init__(self, context_size=128, embedding_size=24):
        super(SimpleSeq2Seq, self).__init__()

        # embedding
        self.embed = nn.Embedding(
            num_embeddings=tokenizer.get_vocab_size(),
            embedding_dim=embedding_size,
            padding_idx=0,
        )

        # encoder
        self.encoder_rnn = nn.GRU(
            input_size=embedding_size, hidden_size=context_size, batch_first=True
        )

        # decoder
        self.decoder_rnn = nn.GRU(
            input_size=embedding_size, hidden_size=context_size, batch_first=True
        )
        self.fc = nn.Linear(context_size, tokenizer.get_vocab_size())

        # loss
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

    def training_step(self, batch, batch_idx):
        x, y = batch

        # x: BxL
        # y: Bx(L-1)

        # print('x shape', x.shape)
        # print('y shape', y.shape)

        batch_size = x.shape[0]
        max_seq_len = x.shape[1]

        # print('batch_size', batch_size)
        # print('max_seq_len', max_seq_len)

        # encode
        embedded = self.embed(x)
        # print('embedded', embedded.shape)

        # embedded: BxLxE
        _, context_vector = self.encoder_rnn(embedded)

        # context_vector: BxC ! LSTM 2xC
        # print('context_vector shape', context_vector.shape)

        # decoder_input: Bx1
        decoder_input = (
            torch.Tensor(batch_size * [1]).unsqueeze(1).long().cuda()
        )  # cuda?
        # print('decoder_input shape', decoder_input.shape)
        # print('decoder_input', decoder_input)

        loss = 0

        for i in range(max_seq_len - 1):

            # embedded_input: Bx1xE
            embedded_input = self.embed(decoder_input)
            # print('embedded_input', embedded_input.shape)

            # after_rnn
            after_rnn_out, context_vector = self.decoder_rnn(
                embedded_input, context_vector
            )

            # print('after_rnn_out shape' ,after_rnn_out.shape )
            # print('after_rnn_out' ,after_rnn_out)
            # print('context_vector shape' ,context_vector.shape )
            # print('context_vector' ,context_vector)

            # decoder_output
            decoder_output = self.fc(after_rnn_out.squeeze())
            # print('decoder_output shape' , decoder_output.shape)

            pred_idx = torch.argmax(decoder_output, dim=1).unsqueeze(1).detach()

            # print('pred_idx shape', pred_idx.shape)

            decoder_input = pred_idx

            loss += self.criterion(decoder_output, y[:, i])

        loss = loss / max_seq_len

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch

        batch_size = x.shape[0]
        max_seq_len = x.shape[1]

        embedded = self.embed(x)

        _, context_vector = self.encoder_rnn(embedded)

        decoder_input = (
            torch.Tensor(batch_size * [1]).unsqueeze(1).long().cuda()
        )  # cuda?

        loss = 0

        for i in range(max_seq_len - 1):
            embedded_input = self.embed(decoder_input)

            after_rnn_out, context_vector = self.decoder_rnn(
                embedded_input, context_vector
            )

            # decoder_output
            decoder_output = self.fc(after_rnn_out.squeeze())

            pred_idx = (
                torch.argmax(decoder_output, dim=1).unsqueeze(1).detach()
            )  # MB: how about teacher forcing?

            decoder_input = pred_idx

            loss += self.criterion(decoder_output, y[:, i])

        loss = loss / max_seq_len

        return {"loss": loss}

    def validation_epoch_end(self, outputs):
        losses = torch.stack([x["loss"] for x in outputs])

        mean_loss = torch.mean(losses)
        print(mean_loss)
        return {"log": {"mean_val_loss": mean_loss}}

    def forward(self, x):
        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def train_dataloader(self):
        reversed_dataset = ReverseDataset(train_samples, tokenizer)
        train_loader = DataLoader(
            reversed_dataset,
            batch_size=16,
            shuffle=True,
            collate_fn=tokenizer.collate_sequences,
        )
        return train_loader

    def val_dataloader(self):
        reversed_dataset = ReverseDataset(val_samples, tokenizer)
        loader = DataLoader(
            reversed_dataset,
            batch_size=16,
            shuffle=False,
            collate_fn=tokenizer.collate_sequences,
        )
        return loader


model = SimpleSeq2Seq()
trainer = pl.Trainer(gpus=1, gradient_clip=1.0, max_epochs=10, fast_dev_run=False)
trainer.fit(model)
