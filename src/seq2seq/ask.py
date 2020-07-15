import pytorch_lightning as pl
from seq2seq.basic_s2s import SimpleSeq2Seq
from seq2seq.dataset.math_dataset import MathDataset
from torch.utils.data.dataloader import DataLoader
import torch

model = SimpleSeq2Seq.load_from_checkpoint(
    "lightning_logs/version_14/checkpoints/epoch=50.ckpt", map_location="cpu"
)

model.freeze()

dset = MathDataset()
train_loader = DataLoader(dset, batch_size=16, shuffle=False, collate_fn=dset.collate,)

x, y = next(iter(train_loader))
preds = model.prediction((x, y), 0)
zero_decoded = dset.text_tokenizer.decode(torch.cat([x[0] for x in preds]))
print(zero_decoded)
print(dset.df["text"][0])
pass
