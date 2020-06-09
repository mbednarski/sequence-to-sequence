# %%
from itertools import chain
from language_models.tokenizer import FixedTokenizer
from language_models.dataset.seq2seq import Seq2SeqDataset
from nltk import wordpunct_tokenize, FreqDist
import pandas as pd
from collections import Counter
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.utils.rnn as ru
from torch.optim import Adam
from torch.utils.data import DataLoader

class Encoder(nn.Module):
    def __init__(self, input_size:int):
        super(Encoder, self).__init__()

        self.rnn = nn.GRU(input_size, 1024)
    
    def forward(self, x):
        out, hidden = self.rnn(x)
        hidden = hidden[:, -1:,:]
        return hidden

class Decoder(nn.Module):
    def __init__(self, input_size, context_size, out_features):
        super(Decoder, self).__init__()

        # assert input_size == out_features

        self.rnn = nn.GRU(input_size=input_size, hidden_size=context_size, batch_first=True)
        self.fc = nn.Linear(in_features=context_size, out_features=out_features)
        # self.softmax = nn.Softmax(dim=2)
    
    def forward(self, x, hidden):
        # print(x)
        out, next_hidden = self.rnn(x, hidden)

        out = self.fc(out)
        # out = self.softmax(out)
        return out, hidden

class Seq2SeqModel(pl.LightningModule):
    def __init__(self, input_vocab_size:int, output_vocab_size:int):
        super(Seq2SeqModel, self).__init__()

        # self.encoder_embed = nn.Embedding.from_pretrained(torch.eye(input_vocab_size), padding_idx=0)
        self.encoder_embed = nn.Embedding(input_vocab_size, 10, padding_idx=0)
        self.encoder = Encoder(10)

        # self.decoder_embed = nn.Embedding.from_pretrained(torch.eye(output_vocab_size), padding_idx=0)
        self.decoder_embed = nn.Embedding(output_vocab_size,10, padding_idx=0)
        self.decoder = Decoder(input_size=10, context_size=1024, out_features=output_vocab_size)

        self.criterion = nn.CrossEntropyLoss()

        self.printer = 0

    def forward(self, x):
        # x: BxL

        e = self.embed(x)
        # e: BxLxVin

        context = self.encoder(e)

        return context

    def training_step(self, batch, batch_idx):
        x, y = batch

        e = self.encoder_embed(x)
        context = self.encoder(e)

        start_token = torch.LongTensor([self.textual_tokenizer.tok2idx[self.textual_tokenizer.SOS_TOKEN]]).unsqueeze(0).cuda()
        input = self.decoder_embed(start_token)

        predictions = ''

        loss = 0

        for i in range(y.shape[1]):
            out, context = self.decoder(input, context)
            # if self.printer % 500 == 0:
            #     print(out.mean())

            pred = torch.argmax(out.squeeze()).detach()

            step_loss = self.criterion(out[:,-1,:], y[:, i])

            loss += step_loss

            input = self.decoder_embed(y[:,i]).unsqueeze(0)

            predictions += self.textual_tokenizer.decode(pred.unsqueeze(0)) + ' '


        if self.printer % 500 == 0:
            print(self.numerical_tokenizer.decode(x.squeeze()))
            print(predictions)

        self.printer += 1

        return {'loss': loss}

        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    @pl.data_loader
    def train_dataloader(self):
        df = pd.read_csv('data/raw/equations.csv', sep=';')
        df['textual_tokenized'] = df['textual'].apply(wordpunct_tokenize)
        textual_counter = Counter(chain(*df['textual_tokenized'].tolist()))
        textual_tokens = list(sorted(textual_counter.keys()))
        self.textual_tokenizer = FixedTokenizer(allowed_tokens=textual_tokens)

        df['numerical_tokenized'] = df['numerical'].apply(lambda x: list(x))
        numerical_counter = Counter(chain(*df['numerical_tokenized'].tolist()))
        numerical_tokens = list(sorted(numerical_counter.keys()))

        self.numerical_tokenizer = FixedTokenizer(allowed_tokens=numerical_tokens)

        dset = Seq2SeqDataset(self.numerical_tokenizer, df['numerical_tokenized'].tolist(),
        self.textual_tokenizer, df['textual_tokenized'].tolist()
        )

        return DataLoader(dset, batch_size=1, shuffle=False)

if __name__ == '__main__':
    model = Seq2SeqModel(20, 41)

    trainer = pl.Trainer(gpus=[0], gradient_clip=1.0, accumulate_grad_batches=32, max_epochs=10000)
    trainer.fit(model)

    print(dset[32])

